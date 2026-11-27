#!/usr/bin/env python3
"""Fail if any training user line is too close to a must-answer or leak-probe question."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def main() -> None:
    hold = C.holdout_questions()
    files = [
        C.DATA / "facts.jsonl",
        C.DATA / "voice.jsonl",
        C.DATA / "dpo.jsonl",
        C.DATA / "rewrite_pairs.jsonl",
        C.DATA / "rewrite_train.jsonl",
        C.DATA / "questions.jsonl",
        C.DATA / "sft_pairs.jsonl",
        C.DATA / "teacher_raw.jsonl",
    ]
    n = 0
    bad = []
    for f in files:
        if not f.is_file():
            continue
        for row in C.load_jsonl(f):
            n += 1
            q = row.get("question") or (row.get("user") or "").split("\n\n")[0]
            if C.too_close_to_holdout(q, hold):
                bad.append((f.name, row.get("id"), q[:160]))
    print("checked", n, "rows, holdout hits", len(bad))
    for item in bad[:20]:
        print("HOLD", item)
    if bad:
        raise SystemExit(f"{len(bad)} training rows overlap the eval holdout")
    print("HOLDOUT OK")


if __name__ == "__main__":
    main()
