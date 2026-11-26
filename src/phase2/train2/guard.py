#!/usr/bin/env python3
"""Guarded fallback: if fact-diff fails a sample, score the pass-1 text. Adapted from exp9/guard.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("factdiff")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    doc = json.load(open(a.run))
    fd = json.load(open(a.factdiff))
    rows = doc["results"]
    items = fd["items"]
    if len(items) != len(rows):
        raise SystemExit(f"factdiff rows {len(items)} != run rows {len(rows)}")
    by_index = {it["index"]: it for it in items}
    out_rows = []
    n_fallback = 0
    for i, src in enumerate(rows):
        it = by_index[i]
        r = dict(src)
        for k in ("judge_leak", "judge_ev", "voice_score"):
            r.pop(k, None)
        use_pass1 = bool(it["fallback"])
        if use_pass1:
            n_fallback += 1
            r["answer"] = r.get("pass1") or r["answer"]
            r["toks"] = r.get("pass1_toks") or 0
            r["sec"] = r.get("pass1_sec") or 0
        else:
            r["answer"] = r.get("pass2") or r["answer"]
        probe = {"fail_if_mentions": r.get("fail_if") or []}
        r["fail"] = C.fail_hits_probe(r["answer"], probe)
        r["voice"] = C.voice_flags(r["answer"])
        r["guard_fallback"] = use_pass1
        if src["kind"] == "must":
            r["correct"] = C.must_correct(src["id"], r["answer"])
        out_rows.append(r)
    meta = dict(doc.get("meta") or {})
    out = a.out
    if not out:
        p = Path(a.run)
        stem = p.stem + "_guarded"
        out = str(p.with_name(stem + ".json"))
    meta.update(
        cond=Path(out).stem,
        parent=doc.get("meta", {}).get("cond"),
        guarded=True,
        n_fallback=n_fallback,
        n_kept=len(rows) - n_fallback,
        factdiff=str(Path(a.factdiff).resolve()),
        scored="pass-1 text when fact-diff fails, otherwise the pass-2 rewrite",
    )
    json.dump(dict(meta=meta, results=out_rows), open(out, "w"), indent=1)
    print("guard wrote", out, "fallback", n_fallback, "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
