#!/usr/bin/env python3
"""Print unlocked must / leak / voice / loop numbers for one judged run."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize import is_loop  # noqa: E402


def main():
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = [r for r in doc["results"] if not r.get("locked")]
    def must(char=None):
        xs = [r for r in rows if r["kind"] == "must" and (char is None or r["char"] == char)]
        ok = sum(1 for r in xs if r.get("correct"))
        return ok, len(xs)
    def voice(char=None, key="voice_score"):
        xs = [r for r in rows if char is None or r["char"] == char]
        vals = [r.get(key) for r in xs if r.get(key) is not None]
        if not vals:
            return None, 0
        return round(statistics.mean(vals), 2), len(vals)
    leaks = defaultdict(list)
    for r in rows:
        if r["kind"] == "leak":
            leaks[(r["id"], r["char"])].append(r)
    marker = sum(1 for rs in leaks.values() if any(r.get("fail") for r in rs))
    judge = sum(1 for rs in leaks.values() if any(r.get("judge_leak") for r in rs))
    frames = sum(1 for r in rows if (r.get("voice") or {}).get("frame"))
    loops = sum(1 for r in rows if is_loop(r.get("answer") or ""))
    v_ok, v_n = must("victor")
    c_ok, c_n = must("creature")
    b_ok, b_n = must()
    vq, _ = voice(key="voice_qwen")
    vg, _ = voice(key="voice_gemma")
    va, vn = voice(key="voice_score")
    vv, _ = voice("victor")
    vc, _ = voice("creature")
    print(f"must V {v_ok}/{v_n}  C {c_ok}/{c_n}  both {b_ok}/{b_n}")
    print(f"leak markers {marker}/{len(leaks)}  judge {judge}/{len(leaks)}")
    print(f"voice qwen {vq}  gemma {vg}  avg {va} n={vn}  (V {vv} / C {vc})")
    print(f"frames {frames}  loops {loops}  unlocked {len(rows)}")


if __name__ == "__main__":
    main()
