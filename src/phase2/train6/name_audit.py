#!/usr/bin/env python3
"""Name-drift audit of an eval run against data/canonical_names.txt.

Any invented near-name is a miss. Prints every distinct near-name and the rows
it appeared in. Exit 0 either way; the caller records the list.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import names  # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: name_audit.py runs/<cond>.json [out.json]")
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    by = defaultdict(list)
    n_rows = 0
    n_hit = 0
    for r in doc.get("results") or []:
        if r.get("locked"):
            continue
        n_rows += 1
        near = names.near_name_tokens(r.get("answer") or "")
        if not near:
            continue
        n_hit += 1
        for tok in near:
            by[tok.lower()].append(
                dict(id=r.get("id"), char=r.get("char"), sample=r.get("sample"), answer=(r.get("answer") or "")[:220])
            )
    report = dict(
        file=sys.argv[1],
        unlocked_rows=n_rows,
        rows_with_near_name=n_hit,
        distinct_near_names=sorted(by),
        hits={k: v[:8] for k, v in sorted(by.items())},
        n_hits_total=sum(len(v) for v in by.values()),
    )
    text = json.dumps(report, indent=1)
    print(text)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text + "\n", encoding="utf-8")
    print(
        "NAME AUDIT",
        "rows",
        n_hit,
        "/",
        n_rows,
        "distinct",
        ", ".join(sorted(by)) or "(none)",
        flush=True,
    )


if __name__ == "__main__":
    main()
