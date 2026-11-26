#!/usr/bin/env python3
"""Anchor checks against the bge-m3 index. Same six phrases as ../scripts/test_lock.py."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import retrieve  # noqa: E402


def fail(msg: str) -> None:
    raise SystemExit(msg)


def main() -> None:
    idx = retrieve.load_bge()
    if idx is None:
        fail("bge index missing")
    chunks = idx["chunks"]
    anchors = [
        ("St. Petersburgh, Dec. 11th", "vol1-letter-1"),
        ("do not waste your time upon this; it is sad trash", "vol1-ch-1"),
        ("It was on a dreary night of November", "vol1-ch-4"),
        ("William is dead", "vol1-ch-6"),
        ("I shall be with you on your wedding-night", "vol3-ch-3"),
        ("I shall ascend my funeral pile triumphantly", "vol3-ch-7"),
    ]
    for phrase, unit_id in anchors:
        needle = retrieve.loose(phrase)
        hits_pos = [c["pos"] for c in chunks if needle in retrieve.loose(c["text"])]
        if not hits_pos:
            fail(f"anchor not in the index: {phrase}")
        first = min(hits_pos)
        unit_pos = next(c["pos"] for c in chunks if c["unit_id"] == unit_id)
        if first != unit_pos and not any(
            c["unit_id"] == unit_id and needle in retrieve.loose(c["text"]) for c in chunks
        ):
            fail(f"anchor {phrase!r} first appears at pos {first}, not in {unit_id}")
        early = retrieve.retrieve(phrase, first - 1, k=8, backend="bge")
        if any(needle in retrieve.loose(h["text"]) or h["pos"] > first - 1 for h in early):
            fail(f"bge lock failed for {phrase!r} at upto {first - 1}: {[h['unit_id'] for h in early]}")
        late = retrieve.retrieve(phrase, first, k=6, backend="bge")
        if not any(needle in retrieve.loose(h["text"]) for h in late):
            fail(f"bge unlocked retrieval missed {phrase!r}: {[h['unit_id'] for h in late]}")
        if any(h["pos"] > first for h in late):
            fail(f"bge retrieval past the unlock for {phrase!r}")
        print("ok", phrase[:40], "first", first, "unit", unit_id, flush=True)

    probes = [
        "Who is Elizabeth Lavenza?",
        "yellow eye November candle",
        "De Lacey Safie Felix",
        "funeral pile ice raft",
        "Justine ballots",
        "Cornelius Agrippa sad trash",
    ]
    for upto in (-1, 0, 1, 4, 8, 13, 20, 26, 27):
        for query in probes:
            for hit in retrieve.retrieve(query, upto, k=6, backend="bge"):
                if hit["pos"] > upto:
                    fail(f"pos {hit['pos']} returned at upto {upto} for {query!r}")
    print("BGE LOCK ANCHORS OK", flush=True)


if __name__ == "__main__":
    main()
