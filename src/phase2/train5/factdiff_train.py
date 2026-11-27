#!/usr/bin/env python3
"""Fact-diff the teacher rewrites that could still be SFT targets.

Same judge as train3: local Ollama qwen3:14b, temp 0, seed 1, JSON dropped/added.
Span failures and name failures do not need the judge; the rewrite cannot be
the target. Resumable by id.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import names  # noqa: E402
from factdiff import one  # noqa: E402

RAW = C.DATA / "teacher_raw.jsonl"
OUT = C.DATA / "factdiff.jsonl"


def already(path: Path) -> set[str]:
    return {r.get("id") for r in C.load_jsonl(path) if r.get("id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reserve-min", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = C.load_jsonl(RAW)
    have = already(OUT)
    todo = [r for r in rows if r.get("id") not in have]
    if a.limit:
        todo = todo[: a.limit]
    print("factdiff todo", len(todo), "have", len(have), "raw", len(rows), flush=True)
    if not todo:
        print("factdiff skip, already complete", flush=True)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    n_llm = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for r in todo:
            if C.past_deadline(a.reserve_min) and n > 0:
                print("deadline near, stopping factdiff after", n, flush=True)
                break
            p1 = r.get("pass1") or ""
            p2 = r.get("rewrite") or ""
            pos = int(r.get("reader_pos") or 0)
            miss = r.get("missing_spans") or []
            near = names.near_name_tokens(p2)
            missing_names = names.missing_canonical(p1, p2)
            called = False
            dropped, added, ok, raw = [], [], True, ""
            if miss or near or missing_names:
                fallback = True
                why = "span" if miss else ("near_name" if near else "missing_canonical")
            else:
                called = True
                n_llm += 1
                dropped, added, ok, raw = one(r.get("user") or "", p1, p2)
                p1_s = set(C.fail_hits_at(p1, pos))
                p2_s = set(C.fail_hits_at(p2, pos))
                new_m = sorted(p2_s - p1_s)
                fallback = (not ok) or bool(dropped) or bool(added) or bool(new_m)
                why = "factdiff" if fallback else "keep"
            row = dict(
                id=r["id"],
                called=called,
                fallback=bool(fallback),
                why=why,
                dropped=dropped,
                added=added,
                n_dropped=len(dropped),
                n_added=len(added),
                parse_ok=bool(ok),
                near=near,
                missing_names=missing_names,
                raw=(raw or "")[:500],
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n == 1 or n % 20 == 0:
                print(
                    f"factdiff {n}/{len(todo)} {time.time() - t0:.0f}s llm {n_llm}",
                    r["id"],
                    why,
                    "fb" if fallback else "keep",
                    flush=True,
                )
    print("factdiff wrote", n, "llm", n_llm, "sec", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
