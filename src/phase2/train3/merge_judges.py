#!/usr/bin/env python3
"""Average qwen3:14b and gemma3:12b voice scores onto a judged run.

Headline leak-judge stays qwen3:14b (train1/train2 comparable).
Voice headline is the two-judge average. Writes voice_score as the average
and keeps voice_qwen / voice_gemma.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qwen")
    ap.add_argument("gemma")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    q = json.load(open(a.qwen))
    g = json.load(open(a.gemma))
    if len(q["results"]) != len(g["results"]):
        raise SystemExit(f"row mismatch {len(q['results'])} vs {len(g['results'])}")
    for qr, gr in zip(q["results"], g["results"]):
        vq = qr.get("voice_qwen", qr.get("voice_score"))
        vg = gr.get("voice_gemma", gr.get("voice_score"))
        qr["voice_qwen"] = vq
        qr["voice_gemma"] = vg
        if vq is None or vg is None:
            qr["voice_score"] = vq if vg is None else vg
            qr["voice_two_judge"] = None
        else:
            qr["voice_score"] = round((float(vq) + float(vg)) / 2.0, 4)
            qr["voice_two_judge"] = qr["voice_score"]
    meta = dict(q.get("meta") or {})
    meta["voice_judges"] = ["qwen3:14b", "gemma3:12b"]
    meta["voice_score_is"] = "average of qwen3:14b and gemma3:12b"
    meta["leak_judge"] = "qwen3:14b"
    meta["gemma_run"] = str(Path(a.gemma).resolve())
    meta["qwen_run"] = str(Path(a.qwen).resolve())
    q["meta"] = meta
    out = a.out or a.qwen
    json.dump(q, open(out, "w"), indent=1)
    print("merged judges ->", out, flush=True)


if __name__ == "__main__":
    main()
