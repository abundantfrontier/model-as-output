#!/usr/bin/env python3
"""Early loop smoke test on ~20 unlocked items, with and without repetition penalty.

Uses train1 rag_base pass-1 (holdout eval questions — eval only, not train).
Abort rule: unguarded loop rate > 15% even WITH the penalty.
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pick_rows(pass1_path: Path, n_each: int = 10):
    doc = json.loads(pass1_path.read_text())
    picked = []
    for char in ("victor", "creature"):
        got = 0
        for r in doc["results"]:
            if r.get("locked"):
                continue
            if r.get("char") != char:
                continue
            if r.get("sample", 0) != 0:
                continue
            picked.append(r)
            got += 1
            if got >= n_each:
                break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--pass1", default=str(C.ROOT / "train1" / "runs" / "rag_base.json"))
    ap.add_argument("--out", default=str(C.TRAIN / "smoke" / "loop.json"))
    ap.add_argument("--n-each", type=int, default=10)
    a = ap.parse_args()
    rows = pick_rows(Path(a.pass1), a.n_each)
    if len(rows) < 10:
        raise SystemExit(f"smoke pool too small: {len(rows)}")
    print("smoke items", len(rows), "chars", {c: sum(r["char"] == c for r in rows) for c in ("victor", "creature")}, flush=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    C.unload_ollama()
    tok = AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        C.BASE, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    model = PeftModel.from_pretrained(model, a.voice, adapter_name="voice")
    model.base_model.set_adapter(["voice"], inference_mode=True)
    model.eval()

    def gen(sysm, user, penalty: bool):
        prompts.assert_no_retrieval(sysm, user)
        ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": user}], True)
        p = {"input_ids": torch.tensor([ids], device="cuda")}
        p["attention_mask"] = torch.ones_like(p["input_ids"])
        t0 = time.time()
        with torch.no_grad():
            o = model.generate(**p, **C.generate_kwargs(penalty=penalty, pad_token_id=tok.pad_token_id))
        torch.cuda.synchronize()
        ans = tok.decode(o[0][p["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        return ans, int(o.shape[1] - p["input_ids"].shape[1]), time.time() - t0

    results = []
    for penalty in (True, False):
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)
        loops = 0
        for r in rows:
            sysm = C.sysprompt(r["char"], int(r["reader_pos"]))
            user = prompts.rewrite_user(r["probe"], r["answer"])
            ans, nt, sec = gen(sysm, user, penalty)
            loop = C.is_loop(ans)
            loops += int(loop)
            results.append(
                dict(
                    id=r["id"],
                    char=r["char"],
                    kind=r["kind"],
                    penalty=penalty,
                    pass1=r["answer"][:400],
                    answer=ans,
                    toks=nt,
                    sec=round(sec, 3),
                    loop=loop,
                )
            )
            print(
                "smoke",
                "pen" if penalty else "nopen",
                r["id"],
                r["char"],
                "LOOP" if loop else "ok",
                ans[:90].replace("\n", " "),
                flush=True,
            )
        rate = loops / len(rows)
        print(f"SMOKE penalty={penalty} loops {loops}/{len(rows)} rate {rate:.3f}", flush=True)

    by = {True: [x for x in results if x["penalty"]], False: [x for x in results if not x["penalty"]]}
    summary = dict(
        n=len(rows),
        with_penalty_loops=sum(x["loop"] for x in by[True]),
        without_penalty_loops=sum(x["loop"] for x in by[False]),
        with_penalty_rate=sum(x["loop"] for x in by[True]) / len(rows),
        without_penalty_rate=sum(x["loop"] for x in by[False]) / len(rows),
        abort_threshold=0.15,
        abort=sum(x["loop"] for x in by[True]) / len(rows) > 0.15,
        voice=a.voice,
        pass1=a.pass1,
        note="repetition_penalty is a safety net, not the fix. Abort if WITH-penalty rate > 15%.",
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=summary, results=results), open(a.out, "w"), indent=2)
    print("SMOKE_SUMMARY", json.dumps(summary), flush=True)
    if summary["abort"]:
        raise SystemExit("ABORT: unguarded loop rate > 15% even with repetition penalty")


if __name__ == "__main__":
    main()
