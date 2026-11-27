#!/usr/bin/env python3
"""Labeled proxy: train7 best merge loaded 4-bit nf4 (bitsandbytes).

Same chapter-locked bge RAG harness as train5 merged_e1_rag: unlocked n=3,
both characters, repetition penalty off. This is NOT an MLC q4f16_1 score.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from chat import render  # noqa: E402
from eval_pass1 import build_items, unlocked  # noqa: E402


def load_bnb(merged: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(str(merged), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(merged),
        quantization_config=bnb,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    return tok, model


def make_gen(merged: Path):
    import torch

    tok, model = load_bnb(merged)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chars", default="victor,creature")
    ap.add_argument("--backend", default="bge")
    ap.add_argument("--merged", default=str(TRAIN / "merged_mixa"))
    ap.add_argument("--cond", default="merged_mixa_bnb4_proxy")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    merged = Path(a.merged)
    chars = [c.strip() for c in a.chars.split(",") if c.strip()]
    items = build_items(chars)
    if a.limit:
        items = items[: a.limit]
    report = json.loads((merged / "merge_report.json").read_text(encoding="utf-8"))
    adapter = str(report.get("adapter") or "")
    if "/phase2-frankenstein/train7/" not in adapter:
        raise SystemExit(f"refusing proxy of a non-train7 merge: {report}")
    C.unload_ollama()
    gen, torch = make_gen(merged)
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
            r.pop("probe_full", None)
            res.append(r)
            print("bnb4", it["id"], it["char"], "LOCK", ans[:80], flush=True)
            continue
        sysm = C.rag_sysprompt(it["char"], it["probe"], pos, backend=a.backend)
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
            print("bnb4", it["id"], it["char"], s, "FAIL" if r["fail"] else "", ans[:100].replace("\n", " "), flush=True)
    meta = dict(
        cond=a.cond,
        proxy=True,
        proxy_note=(
            "train7 merge in bitsandbytes NF4, compute bfloat16. "
            "Same harness as the full merge. Not an MLC q4f16_1 score."
        ),
        n=a.n,
        rag=True,
        base=str(merged),
        merged=str(merged),
        adapter=report.get("adapter"),
        chars=chars,
        backend=a.backend,
        gpu_after_mb=C.gpu_used_mb(),
        wall=round(time.time() - t0),
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        penalty_note="off, same as the train7 full-merge harness",
        eos_token_id=[151645, 151643],
        temperature=0.7,
        top_p=0.95,
        torch_peak_gb=round(T.cuda.max_memory_allocated() / 2**30, 2) if T.cuda.is_available() else 0,
        n_results=len(res),
        bitsandbytes="nf4 4-bit, compute bfloat16",
    )
    out = a.out or str(TRAIN / "runs" / f"{a.cond}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=meta, results=res), open(out, "w"), indent=1)
    print("META", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
