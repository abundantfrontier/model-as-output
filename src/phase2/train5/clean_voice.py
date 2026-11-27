#!/usr/bin/env python3
"""Clean train1 generated voice data and build the pass-1 question pool.

Drops Walton-voice / 'the stranger' as a third person on Letter IV (Victor, pos<=4).
Drops generated rows whose user line talks about the speaker in the third person.
Keeps handwritten seeds as gold. Keeps paraphrases (drop the 3x CE repeat instead).
Does not train on holdout must-answer / leak probes.
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

GEN = C.DATA / "generated"
HAND = C.DATA / "handwritten"

VICTOR_THIRD_USER = re.compile(
    r"\b((did|does|was|were|is|has|had|could|would|should|can)\s+Victor\b|\bVictor'?s\b|\bVictor\s+(think|feel|expect|miss|know|believe|say|do|go|come))",
    re.I,
)
CREATURE_THIRD_USER = re.compile(
    r"\b((did|does|was|were|is|has|had)\s+(the\s+)?(creature|monster|daemon|being)\b)",
    re.I,
)
STRANGER = re.compile(r"\bthe stranger\b|\bthat stranger\b", re.I)
WALTON_TELLS = re.compile(
    r"\bI (thought him|found him|saw him|rescued him|brought him|feared (it|he)|had not expected to see a human)\b",
    re.I,
)


def load(path: Path):
    return C.load_jsonl(path)


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def why_drop_generated(r: dict) -> str | None:
    char = r.get("char")
    user = r.get("user") or ""
    ans = r.get("answer") or ""
    pos = int(r.get("reader_pos") or 0)
    if char not in ("victor", "creature"):
        return "bad_char"
    if pos < C.CHAT_FROM.get(char, 99):
        return "before_unlock"
    if C.too_close_to_holdout(user):
        return "holdout"
    if C.fail_hits_at(user + " " + ans, pos):
        return "leak"
    if C.voice_flags(ans)["frame"]:
        return "frame"
    if C.is_loop(ans):
        return "loop"
    words = ans.split()
    if len(words) < 8 or len(words) > 180:
        return "length"
    if char == "victor":
        if pos <= 4 and STRANGER.search(user + " " + ans):
            return "stranger_third"
        if WALTON_TELLS.search(ans) and pos <= 4:
            return "walton_voice"
        if VICTOR_THIRD_USER.search(user):
            return "user_third_person"
    if char == "creature" and CREATURE_THIRD_USER.search(user):
        return "user_third_person"
    # First-person referent: the speaker should be "I", not narrating the other as self.
    if char == "victor" and re.search(r"\bmy creator\b", ans, re.I):
        return "wrong_person"
    if char == "creature" and re.search(r"\bI created\b|\bI made (him|the being|the creature)\b", ans, re.I):
        return "wrong_person"
    return None


def main():
    hold = C.holdout_questions()
    report = dict(dropped=Counter(), kept=Counter(), notes=[])
    voice_rows = []

    for char in ("victor", "creature"):
        for r in load(HAND / f"{char}.jsonl"):
            r = dict(r)
            r.setdefault("source", "handwritten")
            r.setdefault("kind", "voice")
            r.setdefault("char", char)
            if C.too_close_to_holdout(r.get("user") or "", hold):
                report["dropped"]["handwritten_holdout"] += 1
                continue
            voice_rows.append(r)
            report["kept"]["handwritten"] += 1

    dropped_examples = []
    for fname, src in (("voice_victor.jsonl", "generated"), ("voice_creature.jsonl", "generated")):
        for r in load(GEN / fname):
            r = dict(r)
            r["source"] = src
            why = why_drop_generated(r)
            if why:
                report["dropped"][why] += 1
                if len(dropped_examples) < 40:
                    dropped_examples.append(dict(id=r.get("id"), why=why, user=r.get("user"), char=r.get("char")))
                continue
            voice_rows.append(r)
            report["kept"]["generated"] += 1

    # Paraphrases of handwritten: keep up to 2 per seed (drop 3x CE repeat, not these).
    by_src = Counter()
    for r in load(GEN / "voice_paraphrase.jsonl"):
        r = dict(r)
        r["source"] = "paraphrase"
        sid = r.get("src_id") or ""
        if by_src[sid] >= 2:
            report["dropped"]["paraphrase_cap"] += 1
            continue
        why = why_drop_generated(r) if r.get("char") in ("victor", "creature") else "bad_char"
        # paraphrases reuse handwritten answers; skip stranger filter on the gold answer
        if why in ("stranger_third", "walton_voice", "user_third_person", "wrong_person"):
            # still drop if the USER line is the problem
            if why != "user_third_person":
                why = None
        if C.too_close_to_holdout(r.get("user") or "", hold):
            why = "holdout"
        if why:
            report["dropped"][f"para_{why}"] += 1
            continue
        by_src[sid] += 1
        voice_rows.append(r)
        report["kept"]["paraphrase"] += 1

    dump_jsonl(C.DATA / "voice.jsonl", voice_rows)

    # Question pool for rag_base pass-1 + teacher rewrite.
    seen = set()
    questions = []

    def add_q(char, user, pos, unit_id, gold=None, source="", qid=""):
        user = (user or "").strip()
        if not user or char not in ("victor", "creature"):
            return
        if int(pos) < C.CHAT_FROM[char]:
            return
        if C.too_close_to_holdout(user, hold):
            report["dropped"]["pool_holdout"] += 1
            return
        key = (char, C.norm_q(user))
        if key in seen:
            report["dropped"]["pool_dup"] += 1
            return
        seen.add(key)
        questions.append(
            dict(
                id=qid or f"{source}-{char}-{len(questions):04d}",
                char=char,
                user=user,
                reader_pos=int(pos),
                unit_id=unit_id,
                gold=gold,
                source=source,
            )
        )

    for r in voice_rows:
        add_q(r["char"], r["user"], r.get("reader_pos") or r.get("ch") or 4, r.get("unit_id"), r.get("answer"), r.get("source"), r.get("id"))

    for r in load(GEN / "facts.jsonl"):
        pos = int(r.get("reader_pos") or 0)
        uid = r.get("unit_id")
        for char in ("victor", "creature"):
            add_q(char, r.get("user"), pos, uid, None, "facts", f"{r.get('id')}-{char}")

    dump_jsonl(C.DATA / "questions.jsonl", questions)
    report["kept"]["questions"] = len(questions)
    report["kept"]["voice"] = len(voice_rows)
    report["dropped"] = dict(report["dropped"])
    report["kept"] = dict(report["kept"])
    report["dropped_examples"] = dropped_examples
    report["char_voice"] = dict(Counter(r["char"] for r in voice_rows))
    report["char_questions"] = dict(Counter(r["char"] for r in questions))
    report["note"] = (
        "3x handwritten CE repeat dropped (each seed once). "
        "Paraphrases kept at most 2 per seed. "
        "Victor Letter IV 'the stranger' / Walton-voice generated rows dropped. "
        "Generated third-person-about-Victor user lines dropped."
    )
    (C.DATA / "clean_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("kept", "dropped", "char_voice", "char_questions")}, indent=2), flush=True)
    print("wrote", C.DATA / "voice.jsonl", len(voice_rows), "questions", len(questions), flush=True)


if __name__ == "__main__":
    main()
