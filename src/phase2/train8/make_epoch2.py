#!/usr/bin/env python3
"""Build the epoch-2 SFT file. Does not drop pairs and does not add holdout questions.

Epoch 1 stays on data/sft_pairs.jsonl, same order as train4.

Epoch 2 keeps every pair once, then adds copies:
- weight 3: creature, reader_pos 15..21, first person, not third-person narration,
  not Victor's creator voice
- weight 2: Victor, reader_pos in {12, 18, 19} (the 0/3 must-answer misses)
- weight 1: everything else, including third-person narration (relative downweight)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

SRC = C.DATA / "sft_pairs.jsonl"
DST = C.DATA / "sft_pairs_epoch2.jsonl"
REPORT = C.DATA / "epoch2_report.json"

FIRST = re.compile(r"\b(I|me|my|myself)\b")
# Sentence subject is the creature or Victor. "I am the creature" does not match.
THIRD_NARR = re.compile(
    r"(?i)(?:^|[.!?]\s+)"
    r"(?:the creature|the monster|the daemon|the being|victor)"
    r"\s+"
    r"(?:demands|demand|claims|claim|says|said|describes|described|does|did|"
    r"reads|read|burns|burned|enters|entered|stops|stopped|learns|learned|"
    r"goes|went|asks|asked|vows|vowed|wants|wanted|believes|believed|"
    r"meets|met|promises|promised|feels|felt|thinks|thought|starts|started|"
    r"begins|began|refuses|refused|agrees|agreed|destroys|destroyed|climbs|climbed)\b"
)
VICTOR_VOICE = re.compile(
    r"(?i)\bI (?:created|chose to create)\b"
    r"|\bmy ambition\b"
    r"|\bI made (?:the |a |my )?(?:being|creature|monster)\b"
    r"|\bElizabeth was my\b"
)
VICTOR_UP = {12, 18, 19}
COTTAGE = range(15, 22)


def weight(row: dict) -> tuple[int, str]:
    ans = row.get("answer") or ""
    char = row.get("char")
    pos = int(row.get("reader_pos") or 0)
    first = bool(FIRST.search(ans))
    third = bool(THIRD_NARR.search(ans))
    victor_voice = bool(VICTOR_VOICE.search(ans))
    if (
        char == "creature"
        and pos in COTTAGE
        and first
        and not third
        and not victor_voice
    ):
        return 3, "creature_cottage_first"
    if char == "victor" and pos in VICTOR_UP and first and not third:
        return 2, "victor_miss_pos"
    if third:
        return 1, "third_narration"
    if char == "creature" and victor_voice:
        return 1, "creature_victor_voice"
    return 1, "base"


def main() -> None:
    rows = C.load_jsonl(SRC)
    out = []
    counts = Counter()
    for row in rows:
        w, tag = weight(row)
        counts[tag] += 1
        counts[f"copies_{tag}"] += w
        for _ in range(w):
            out.append(row)
    with DST.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = dict(
        source=str(SRC),
        out=str(DST),
        n_base_pairs=len(rows),
        n_epoch2_rows=len(out),
        extra_copies=len(out) - len(rows),
        dropped=0,
        holdout_added=0,
        rule={
            "creature_reader_pos_15_to_21_clean_first_person": 3,
            "victor_reader_pos_12_18_19_first_person": 2,
            "third_person_narration_and_everyone_else": 1,
        },
        counts=dict(counts),
    )
    REPORT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
