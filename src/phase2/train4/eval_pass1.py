#!/usr/bin/env python3
"""Single-pass eval: RAG-only base, facts LoRA, stacked facts+voice, or a merged model.

Writes runs/<cond>.json in the exp4 {meta, results} shape.
Lock lines are used when the character is not yet on the page (no model call).
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
from chat import render  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_items(chars: list[str]):
    items = []
    for q in C.must_questions():
        for char in chars:
            items.append(
                dict(
                    kind="must",
                    id=q["id"],
                    set="M",
                    char=char,
                    probe=q["question"],
                    expected=q["expected_answer"],
                    reader_pos=q["reader_pos"],
                    unit_id=q["unit_id"],
                    rubric=C.MUST_RUBRIC.get(q["id"], r"$a"),
                )
            )
    for p in C.leak_probes():
        for char in chars:
            items.append(
                dict(
                    kind="leak",
                    id=p["id"],
                    set=p["kind"],
                    char=char,
                    probe=p["question"],
                    expected=p["expected_answer"],
                    reader_pos=p["asked_at_pos"],
                    unit_id=p["asked_at_unit"],
                    fail_if=p["fail_if_mentions"],
                    probe_full=p,
                )
            )
    return items


def load_local(spec):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["base"], local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["base"], dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    if spec.get("merged"):
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        return tok, model
    if spec.get("facts") and spec.get("voice"):
        model = PeftModel.from_pretrained(model, spec["facts"], adapter_name="facts")
        model.load_adapter(spec["voice"], adapter_name="voice")
        model.base_model.set_adapter(["facts", "voice"], inference_mode=True)
    elif spec.get("facts"):
        model = PeftModel.from_pretrained(model, spec["facts"], adapter_name="facts")
        model.base_model.set_adapter(["facts"], inference_mode=True)
    elif spec.get("voice"):
        model = PeftModel.from_pretrained(model, spec["voice"], adapter_name="voice")
        model.base_model.set_adapter(["voice"], inference_mode=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    return tok, model


def make_gen(spec):
    import torch

    tok, model = load_local(spec)

    def gen(sysm, q):
        ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": q}], True)
        p = {"input_ids": torch.tensor([ids], device="cuda")}
        p["attention_mask"] = torch.ones_like(p["input_ids"])
        t0 = time.time()
        with torch.no_grad():
            o = model.generate(
                **p,
                **C.generate_kwargs(penalty=False, pad_token_id=tok.pad_token_id),
            )
        torch.cuda.synchronize()
        n = o.shape[1] - p["input_ids"].shape[1]
        return tok.decode(o[0][p["input_ids"].shape[1] :], skip_special_tokens=True).strip(), int(n), time.time() - t0

    return gen, torch


def unlocked(char: str, pos: int) -> bool:
    if char == "narrator":
        return True
    return pos >= C.CHAT_FROM.get(char, 99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base", default=C.BASE)
    ap.add_argument("--facts", default="")
    ap.add_argument("--voice", default="")
    ap.add_argument("--merged", default="")
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--chars", default="victor,creature")
    ap.add_argument("--backend", default="bge")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    chars = [c.strip() for c in a.chars.split(",") if c.strip()]
    items = build_items(chars)
    if a.limit:
        items = items[: a.limit]
    spec = dict(
        base=a.merged or a.base,
        facts=a.facts,
        voice=a.voice,
        merged=bool(a.merged),
        rag=a.rag,
    )
    C.unload_ollama()
    # RAG still needs bge-m3; load it after the 4B is up? Retrieval is called per item
    # before generate. Keep ollama for bge only.
    gen, torch = make_gen(spec)
    res = []
    t0 = time.time()
    import torch as T

    T.manual_seed(1)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(1)
    for it in items:
        pos = int(it["reader_pos"])
        if not unlocked(it["char"], pos):
            ans = C.LOCK_LINES[it["char"]]
            r = dict(
                it,
                sample=0,
                n_samples_note="lock_line",
                answer=ans,
                toks=0,
                sec=0,
                locked=True,
                fail=C.fail_hits_probe(ans, it.get("probe_full") or {"fail_if_mentions": []}),
                voice=C.voice_flags(ans),
            )
            if it["kind"] == "must":
                r["correct"] = C.must_correct(it["id"], ans)
            # drop bulky probe_full
            r.pop("probe_full", None)
            res.append(r)
            print(a.cond, it["id"], it["char"], "LOCK", ans[:80], flush=True)
            continue
        sysm = C.rag_sysprompt(it["char"], it["probe"], pos, backend=a.backend) if a.rag else C.sysprompt(it["char"], pos)
        for s in range(a.n):
            ans, nt, sec = gen(sysm, it["probe"])
            probe = it.get("probe_full") or {"fail_if_mentions": []}
            r = dict(
                it,
                sample=s,
                answer=ans,
                toks=nt,
                sec=round(sec, 3),
                locked=False,
                fail=C.fail_hits_probe(ans, probe),
                voice=C.voice_flags(ans),
            )
            if it["kind"] == "must":
                r["correct"] = C.must_correct(it["id"], ans)
            r.pop("probe_full", None)
            res.append(r)
            print(a.cond, it["id"], it["char"], s, "FAIL" if r["fail"] else "", ans[:100].replace("\n", " "), flush=True)
    meta = dict(
        cond=a.cond,
        n=a.n,
        rag=a.rag,
        base=spec["base"],
        facts=a.facts,
        voice=a.voice,
        merged=a.merged,
        chars=chars,
        backend=a.backend,
        gpu_after_mb=C.gpu_used_mb(),
        wall=round(time.time() - t0),
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        penalty_note="off: smoke 0/20 loops and 0 near-names without it; penalty induced Agrappa, Morritz, Cliversal",
        eos_token_id=[151645, 151643],
        torch_peak_gb=round(T.cuda.max_memory_allocated() / 2**30, 2) if T.cuda.is_available() else 0,
        n_results=len(res),
    )
    out = a.out or str(C.TRAIN / "runs" / f"{a.cond}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=meta, results=res), open(out, "w"), indent=1)
    print("META", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
