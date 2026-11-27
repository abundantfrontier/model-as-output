#!/usr/bin/env python3
"""Fail if any training user line is too close to a must-answer or leak-probe question."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def jaccard(user: str, holdouts: list[str]) -> float:
    u_txt = C.norm_q(user)
    u = set(u_txt.split())
    best = 0.0
    if len(u) < 3:
        return 0.0
    for h in holdouts:
        nh = C.norm_q(h)
        if not nh:
            continue
        if u_txt == nh or u_txt in nh or nh in u_txt:
            return 1.0
        v = set(nh.split())
        if not v:
            continue
        j = len(u & v) / len(u | v)
        if j > best:
            best = j
    return best


def main() -> None:
    hold = C.holdout_questions()
    files = [
        C.DATA / "facts.jsonl",
        C.DATA / "voice.jsonl",
        C.DATA / "dpo.jsonl",
        C.DATA / "mixed_a.jsonl",
        C.DATA / "mixed_b.jsonl",
        C.DATA / "ce_recover.jsonl",
        C.DATA / "rewrite_pairs.jsonl",
        C.DATA / "rewrite_train.jsonl",
        C.DATA / "questions.jsonl",
        C.DATA / "sft_pairs.jsonl",
        C.DATA / "sft_pairs_epoch2.jsonl",
        C.DATA / "teacher_raw.jsonl",
    ]
    n = 0
    bad = []
    closest = 0.0
    closest_q = ""
    for f in files:
        if not f.is_file():
            continue
        for row in C.load_jsonl(f):
            n += 1
            q = row.get("question") or (row.get("user") or "").split("\n\n")[0]
            j = jaccard(q, hold)
            if j > closest:
                closest = j
                closest_q = q[:120]
            if C.too_close_to_holdout(q, hold):
                bad.append((f.name, row.get("id"), q[:160]))
    print("checked", n, "rows, holdout hits", len(bad))
    print(f"closest Jaccard {closest:.3f} {closest_q}")
    for item in bad[:20]:
        print("HOLD", item)
    if bad:
        raise SystemExit(f"{len(bad)} training rows overlap the eval holdout")
    print("HOLDOUT OK")


if __name__ == "__main__":
    main()
