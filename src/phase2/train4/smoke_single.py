#!/usr/bin/env python3
"""Early loop and name smoke. Single-pass + RAG, both characters, ~20 unlocked items.

Runs WITH and WITHOUT the repetition penalty. Abort if the loop rate with the
penalty is above 15%. Spot-check canonical names (the penalty is a safety net).
Uses holdout questions as eval only. Does not write training pairs.
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
import names  # noqa: E402
from chat import render  # noqa: E402
from summarize import is_loop  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Prefer items whose gist covers the train3 drift examples.
PREFER = ("m05", "m07", "m08", "m09", "m13", "m15", "m20", "m14", "m16", "m11")


def pick(n_each: int = 10) -> list[dict]:
    must = {q["id"]: q for q in C.must_questions()}
    picked = []
    for char in ("victor", "creature"):
        pool = []
        for qid in PREFER:
            q = must.get(qid)
            if not q:
                continue
            pos = int(q["reader_pos"])
            if pos < C.CHAT_FROM.get(char, 99):
                continue
            pool.append(q)
        for q in C.must_questions():
            if q in pool:
                continue
            pos = int(q["reader_pos"])
            if pos < C.CHAT_FROM.get(char, 99):
                continue
            pool.append(q)
        for q in pool[:n_each]:
            picked.append(
                dict(
                    id=q["id"],
                    char=char,
                    probe=q["question"],
                    reader_pos=int(q["reader_pos"]),
                    unit_id=q["unit_id"],
                )
            )
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(C.TRAIN / "lora_sft"))
    ap.add_argument("--merged", default="")
    ap.add_argument("--out", default=str(C.TRAIN / "smoke" / "loop.json"))
    ap.add_argument("--n-each", type=int, default=10)
    a = ap.parse_args()
    rows = pick(a.n_each)
    if len(rows) < 16:
        raise SystemExit(f"smoke pool too small: {len(rows)}")
    print("smoke items", len(rows), flush=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    C.unload_ollama()
    base_name = a.merged or C.BASE
    tok = AutoTokenizer.from_pretrained(base_name, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    if not a.merged:
        model = PeftModel.from_pretrained(model, a.adapter, adapter_name="voice")
        model.base_model.set_adapter(["voice"], inference_mode=True)
    model.eval()

    def gen(sysm, user, penalty: bool):
        ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": user}], True)
        p = {"input_ids": torch.tensor([ids], device="cuda")}
        p["attention_mask"] = torch.ones_like(p["input_ids"])
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)
        t0 = time.time()
        with torch.inference_mode():
            o = model.generate(**p, **C.generate_kwargs(penalty=penalty, pad_token_id=tok.pad_token_id))
        torch.cuda.synchronize()
        ans = tok.decode(o[0][p["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        return ans, int(o.shape[1] - p["input_ids"].shape[1]), time.time() - t0

    # Cache RAG prompts so the two penalty passes see the same context.
    sys_for = {}
    for r in rows:
        sys_for[(r["char"], r["id"])] = C.rag_sysprompt(r["char"], r["probe"], r["reader_pos"], backend="bge")

    results = []
    summary = {}
    for penalty in (True, False):
        loops = 0
        near_hits = []
        chunk = []
        for r in rows:
            ans, nt, sec = gen(sys_for[(r["char"], r["id"])], r["probe"], penalty)
            loop = is_loop(ans)
            loops += int(loop)
            near = names.near_name_tokens(ans)
            if near:
                near_hits.append(dict(id=r["id"], char=r["char"], near=near, answer=ans[:240]))
            row = dict(r, penalty=penalty, answer=ans, toks=nt, sec=round(sec, 3), loop=loop, near=near)
            chunk.append(row)
            results.append(row)
            print(
                "smoke",
                "pen" if penalty else "raw",
                r["id"],
                r["char"],
                "LOOP" if loop else "",
                "NEAR" if near else "",
                ans[:90].replace("\n", " "),
                flush=True,
            )
        rate = loops / max(len(rows), 1)
        summary["with_penalty" if penalty else "without_penalty"] = dict(
            n=len(rows), loops=loops, rate=round(rate, 4), near=len(near_hits), near_hits=near_hits
        )
        print("SMOKE", "penalty" if penalty else "no-penalty", "loops", loops, "/", len(rows), "near", len(near_hits), flush=True)

    abort = summary["with_penalty"]["rate"] > 0.15
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(
        meta=dict(
            adapter=a.adapter,
            merged=a.merged,
            n=len(rows),
            abort_loops=abort,
            eos_token_id=[151645, 151643],
            note="repetition_penalty is a safety net, not the fix",
        ),
        summary=summary,
        results=results,
    )
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    flag = C.TRAIN / "smoke" / "ABORT_LOOPS"
    if abort:
        flag.write_text(json.dumps(summary["with_penalty"], indent=1), encoding="utf-8")
        print("ABORT loops with penalty", summary["with_penalty"], flush=True)
    elif flag.exists():
        flag.unlink()
    print("wrote", out, "abort", abort, flush=True)


if __name__ == "__main__":
    main()
