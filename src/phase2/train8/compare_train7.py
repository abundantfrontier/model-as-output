#!/usr/bin/env python3
"""Compare a train8 judged run to train7 merged_mixa_rag. Read-only on train7."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

T7 = Path("$HOME/output-model/phase2-frankenstein/train7/runs/merged_mixa_rag.judged.json")


def load(p: Path):
    doc = json.loads(p.read_text(encoding="utf-8"))
    return [r for r in doc["results"] if not r.get("locked")]


def must_rates(rows):
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["kind"] != "must":
            continue
        key = (r["id"], r["char"])
        by[key][1] += 1
        by[key][0] += int(bool(r.get("correct")))
    return by


def leak_cells(rows):
    by = defaultdict(list)
    for r in rows:
        if r["kind"] != "leak":
            continue
        by[(r["id"], r["char"])].append(r)
    out = {}
    for k, rs in by.items():
        out[k] = dict(
            marker=any(r.get("fail") for r in rs),
            judge=any(r.get("judge_leak") for r in rs),
        )
    return out


def main():
    newp = Path(sys.argv[1])
    a = load(T7)
    b = load(newp)
    ma, mb = must_rates(a), must_rates(b)
    print("must moves (train7 merged_mixa_rag -> this run):")
    for k in sorted(set(ma) | set(mb)):
        oa, na = ma.get(k, [0, 0]), mb.get(k, [0, 0])
        if oa != na:
            print(f"  {k[0]} {k[1]}: {oa[0]}/{oa[1]} -> {na[0]}/{na[1]}")
    la, lb = leak_cells(a), leak_cells(b)
    print("leak judge flips:")
    for k in sorted(set(la) | set(lb)):
        ja = la.get(k, {}).get("judge")
        jb = lb.get(k, {}).get("judge")
        if ja != jb:
            print(f"  {k[0]} {k[1]}: {ja} -> {jb}")
    print("still leaking:")
    for k in sorted(lb):
        if lb[k]["judge"]:
            print(f"  {k[0]} {k[1]}")


if __name__ == "__main__":
    main()
