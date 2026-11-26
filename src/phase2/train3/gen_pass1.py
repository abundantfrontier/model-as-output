#!/usr/bin/env python3
"""Stock Qwen3-4B + chapter-locked RAG answers for the rewrite-pair question pool.

Needs the GPU lock. Resumable: skips ids already in data/pass1_train.jsonl.
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


def already(path: Path) -> set[str]:
    return {r.get("id") for r in C.load_jsonl(path) if r.get("id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(C.DATA / "questions.jsonl"))
    ap.add_argument("--out", default=str(C.DATA / "pass1_train.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", default="bge")
    a = ap.parse_args()
    qs = C.load_jsonl(a.questions)
    if a.limit:
        qs = qs[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = already(out)
    todo = [q for q in qs if q.get("id") not in have]
    print("pass1 questions", len(qs), "have", len(have), "todo", len(todo), flush=True)
    if not todo:
        print("pass1 skip, already complete", flush=True)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    C.unload_ollama()
    tok = AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        C.BASE, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    model.eval()
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    t0 = time.time()
    n = 0
    with out.open("a", encoding="utf-8") as fh:
        for q in todo:
            if C.past_deadline(120):
                print("deadline near, stopping pass1 after", n, flush=True)
                break
            sysm = C.rag_sysprompt(q["char"], q["user"], int(q["reader_pos"]), backend=a.backend)
            ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": q["user"]}], True)
            p = {"input_ids": torch.tensor([ids], device="cuda")}
            p["attention_mask"] = torch.ones_like(p["input_ids"])
            with torch.no_grad():
                o = model.generate(
                    **p,
                    **C.generate_kwargs(penalty=True, pad_token_id=tok.pad_token_id),
                )
            ans = tok.decode(o[0][p["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            row = dict(
                q,
                pass1=ans,
                toks=int(o.shape[1] - p["input_ids"].shape[1]),
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n == 1 or n % 25 == 0:
                print(
                    f"pass1 {n}/{len(todo)} {time.time() - t0:.0f}s",
                    q["char"],
                    q["id"],
                    ans[:80].replace("\n", " "),
                    flush=True,
                )
    print("pass1 wrote", n, "total", len(already(out)), "sec", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
