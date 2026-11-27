#!/usr/bin/env python3
"""Teacher pairs: rag_base pass-1, then train3 constrained rewrite.

The SFT target is chosen later (assemble_sft.py) from the guard. This script
only writes pass-1 and the constrained rewrite. Resumable by id.

Pass-1 that already exists in train2/data/pass1_train.jsonl is reused when the
question id and user text match (same stock Qwen3-4B + RAG settings). New
questions are generated here with the adapter disabled.

Inference: eos 151645/151643, temperature 0.7, top_p 0.95, max_new_tokens 200,
seed 1, repetition_penalty 1.15. Coverage decoding omits no_repeat_ngram_size.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import prompts  # noqa: E402
from chat import render  # noqa: E402
from protect import extract_protected, generate_constrained, missing_spans  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

T2_PASS1 = C.ROOT / "train2" / "data" / "pass1_train.jsonl"
TEACHER = C.ROOT / "train3" / "lora_v_rewrite"
RAW = C.DATA / "teacher_raw.jsonl"


def seed_all(torch):
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)


def gen_plain(model, tok, torch, sysm, user):
    ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": user}], True)
    if len(ids) > 3600:
        raise RuntimeError(f"prompt too long: {len(ids)} tokens")
    p = {"input_ids": torch.tensor([ids], device="cuda")}
    p["attention_mask"] = torch.ones_like(p["input_ids"])
    seed_all(torch)
    kw = C.generate_kwargs(penalty=True, pad_token_id=tok.pad_token_id)
    # Optional n-gram block. Omitted here: it stalls batch-1 decode and can
    # fight a later coverage pass. repetition_penalty stays.
    kw.pop("no_repeat_ngram_size", None)
    with model.disable_adapter():
        with torch.inference_mode():
            o = model.generate(**p, **kw)
    torch.cuda.synchronize()
    ans = tok.decode(o[0][p["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return ans, int(o.shape[1] - p["input_ids"].shape[1])


def _left_pad(tok, torch, encoded):
    max_len = max(len(x) for x in encoded)
    pad_id = tok.pad_token_id
    ids, mask = [], []
    for row in encoded:
        pad = max_len - len(row)
        ids.append([pad_id] * pad + row)
        mask.append([0] * pad + [1] * len(row))
    p = {
        "input_ids": torch.tensor(ids, device="cuda"),
        "attention_mask": torch.tensor(mask, device="cuda"),
    }
    return p, max_len


def _decode_new(tok, sequences, prompt_len):
    out = []
    for row in sequences:
        gen = row[prompt_len:]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        # Count non-pad new tokens approximately from the decoded side.
        n = int((gen != tok.pad_token_id).sum().item()) if hasattr(gen, "sum") else len(gen)
        out.append((text, n))
    return out


def gen_plain_batch(model, tok, torch, pairs):
    """pairs: list of (sys, user). Adapter off. Returns list of (text, ntok)."""
    encoded = []
    for sysm, user in pairs:
        ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": user}], True)
        if len(ids) > 3600:
            raise RuntimeError(f"prompt too long: {len(ids)} tokens")
        encoded.append(ids)
    p, prompt_len = _left_pad(tok, torch, encoded)
    seed_all(torch)
    kw = C.generate_kwargs(penalty=True, pad_token_id=tok.pad_token_id)
    kw.pop("no_repeat_ngram_size", None)
    with model.disable_adapter():
        with torch.inference_mode():
            o = model.generate(**p, **kw)
    torch.cuda.synchronize()
    return _decode_new(tok, o, prompt_len)


def gen_rewrite_batch(model, tok, torch, jobs):
    """jobs: list of (sys, user, spans). Constrained, one processor per row."""
    from protect import ProtectedSpanProcessor
    from transformers import LogitsProcessorList

    encoded = []
    for sysm, user, spans in jobs:
        prompts.assert_no_retrieval(sysm, user)
        ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": user}], True)
        encoded.append((ids, spans))
    p, prompt_len = _left_pad(tok, torch, [e[0] for e in encoded])
    procs = [
        ProtectedSpanProcessor(tok, spans or [], prompt_len)
        for _, spans in encoded
    ]

    class _BatchSpans:
        def __call__(self, input_ids, scores):
            import torch as _t

            parts = []
            for b, proc in enumerate(procs):
                parts.append(proc(input_ids[b : b + 1], scores[b : b + 1]))
            return _t.cat(parts, dim=0)

    seed_all(torch)
    kw = C.generate_kwargs(penalty=True, pad_token_id=tok.pad_token_id)
    kw.pop("no_repeat_ngram_size", None)
    if any(spans for _, spans in encoded):
        kw["logits_processor"] = LogitsProcessorList([_BatchSpans()])
    model.set_adapter("voice")
    with torch.inference_mode():
        o = model.generate(**p, **kw)
    torch.cuda.synchronize()
    return _decode_new(tok, o, prompt_len)


def gen_rewrite(model, tok, torch, sysm, user, spans):
    prompts.assert_no_retrieval(sysm, user)
    ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": user}], True)
    p = {"input_ids": torch.tensor([ids], device="cuda")}
    p["attention_mask"] = torch.ones_like(p["input_ids"])
    seed_all(torch)
    model.set_adapter("voice")
    with torch.inference_mode():
        o = generate_constrained(
            model,
            tok,
            p["input_ids"],
            p["attention_mask"],
            spans,
            penalty=True,
            pad_token_id=tok.pad_token_id,
        )
    torch.cuda.synchronize()
    ans = tok.decode(o[0][p["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return ans, int(o.shape[1] - p["input_ids"].shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reserve-min", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--voice", default=str(TEACHER))
    a = ap.parse_args()
    qs = C.load_jsonl(C.DATA / "questions.jsonl")
    cache = {r["id"]: r for r in C.load_jsonl(C.DATA / "rag_cache.jsonl")}
    have = {r.get("id") for r in C.load_jsonl(RAW)}
    todo = [q for q in qs if q.get("id") not in have and q.get("id") in cache]
    missing_cache = [q.get("id") for q in qs if q.get("id") not in cache and q.get("id") not in have]
    if missing_cache:
        print("missing rag cache", len(missing_cache), "e.g.", missing_cache[:5], flush=True)
    if a.limit:
        todo = todo[: a.limit]
    print("teacher todo", len(todo), "have", len(have), "questions", len(qs), flush=True)
    if not todo:
        print("teacher skip, already complete", flush=True)
        return

    reused = {}
    for r in C.load_jsonl(T2_PASS1):
        if r.get("id") and r.get("pass1"):
            reused[r["id"]] = r

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    C.unload_ollama()
    print("loading", C.BASE, "plus", a.voice, flush=True)
    tok = AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        C.BASE, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    model = PeftModel.from_pretrained(base, a.voice, adapter_name="voice")
    model.eval()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    n_reuse = 0
    bs = max(1, a.batch)
    with RAW.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), bs):
            if C.past_deadline(a.reserve_min) and n > 0:
                print("deadline near, stopping teacher after", n, flush=True)
                break
            chunk = todo[start : start + bs]
            try:
                plain_idx = []
                plain_pairs = []
                pass1 = [None] * len(chunk)
                origins = [None] * len(chunk)
                for i, q in enumerate(chunk):
                    hit = reused.get(q["id"])
                    if hit and (hit.get("user") or "") == q["user"] and hit.get("char") == q["char"]:
                        pass1[i] = ((hit.get("pass1") or "").strip(), int(hit.get("toks") or 0))
                        origins[i] = "train2_pass1"
                        n_reuse += 1
                    else:
                        plain_idx.append(i)
                        plain_pairs.append((cache[q["id"]]["sys"], q["user"]))
                        origins[i] = "train4_pass1"
                t_p = time.time()
                if plain_pairs:
                    got = gen_plain_batch(model, tok, torch, plain_pairs) if len(plain_pairs) > 1 else [
                        gen_plain(model, tok, torch, plain_pairs[0][0], plain_pairs[0][1])
                    ]
                    for i, ans in zip(plain_idx, got):
                        pass1[i] = ans
                p1_sec = time.time() - t_p
                jobs = []
                span_lists = []
                for i, q in enumerate(chunk):
                    ans1 = pass1[i][0]
                    spans = extract_protected(ans1, q["char"])
                    span_lists.append(spans)
                    jobs.append((C.sysprompt(q["char"], int(q["reader_pos"])), prompts.rewrite_user(q["user"], ans1), spans))
                t_r = time.time()
                if len(jobs) > 1:
                    rewrites = gen_rewrite_batch(model, tok, torch, jobs)
                else:
                    ans2, toks2 = gen_rewrite(model, tok, torch, jobs[0][0], jobs[0][1], jobs[0][2])
                    rewrites = [(ans2, toks2)]
                rw_sec = time.time() - t_r
            except Exception as e:
                print("FAIL batch", [q.get("id") for q in chunk], type(e).__name__, e, flush=True)
                if "CUDA" in type(e).__name__ or "out of memory" in str(e).lower():
                    raise
                continue
            for i, q in enumerate(chunk):
                ans1, toks1 = pass1[i]
                ans2, toks2 = rewrites[i]
                spans = span_lists[i]
                miss = missing_spans(ans2, spans)
                src = cache[q["id"]]
                row = dict(
                    id=q["id"],
                    char=q["char"],
                    user=q["user"],
                    reader_pos=int(q["reader_pos"]),
                    unit_id=q.get("unit_id") or src.get("unit_id") or "",
                    source=q.get("source") or "",
                    sys=src["sys"],
                    pass1=ans1,
                    pass1_toks=toks1,
                    pass1_origin=origins[i],
                    rewrite=ans2,
                    rewrite_toks=toks2,
                    protected_spans=spans,
                    missing_spans=miss,
                    span_ok=not miss,
                )
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if n <= 4 or n % 20 == 0:
                    rate = n / max(time.time() - t0, 1e-6)
                    print(
                        f"teacher {n}/{len(todo)} {time.time() - t0:.0f}s {rate:.2f}/s "
                        f"p1 {p1_sec:.1f}s rw {rw_sec:.1f}s",
                        q["char"],
                        q["id"],
                        "span_ok" if row["span_ok"] else "span_miss",
                        (ans2 or "")[:70].replace("\n", " "),
                        flush=True,
                    )
            fh.flush()
    print("teacher wrote", n, "reuse", n_reuse, "sec", round(time.time() - t0), "total", len(have) + n, flush=True)


if __name__ == "__main__":
    main()
