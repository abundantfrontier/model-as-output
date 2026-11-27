#!/usr/bin/env python3
"""Build the non-holdout question pool for single-pass distillation.

Reuses train2/data/questions.jsonl (already holdout-filtered) and adds
name-centered questions at each unlocked reader position. Both characters.
No must-answer or leak-probe questions. No Walton third-person 'the stranger'
user lines.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import names  # noqa: E402
from protect import BOOK_NAMES  # noqa: E402

T2 = C.ROOT / "train2" / "data" / "questions.jsonl"
OUT = C.DATA / "questions.jsonl"
CAP = 2400

TEMPLATES = [
    "Who is {name} to you?",
    "What do you remember about {name}?",
    "How did {name} come into your history?",
    "What can you tell me of {name}?",
    "What part did {name} play in what you lived?",
]

OPENERS = [
    "What should I understand about your life from the pages so far?",
    "What has weighed on you most in the story you have told up to now?",
    "Which person from your history still stands clearest to you here?",
]


def norm_key(char: str, pos: int, user: str) -> str:
    return f"{char}|{pos}|{C.norm_q(user)}"


def walton_user(char: str, pos: int, user: str) -> bool:
    if char != "victor" or pos > 4:
        return False
    import re

    return bool(re.search(r"\bthe stranger\b", user, re.I))


def main() -> None:
    names.write_list()
    names.book_vocab.cache_clear()
    names.canonical_tokens.cache_clear()
    canon = names.load_canonical()
    # Phrases of 2+ words, plus single tokens of length >= 4 that are not stopwords.
    askable = []
    for name in canon:
        parts = [names._clean_token(p) for p in names.WORD.findall(name)]
        parts = [p for p in parts if p]
        if not parts or all(p in names.STOP for p in parts):
            continue
        if len(parts) == 1 and (len(parts[0]) < 4 or parts[0] in names.STOP):
            continue
        # Skip gist phrases that are not names (sledge, cousin, yellow eye).
        if name[0].islower():
            continue
        # Accidental bigrams ("Ingolstadt Justine") make bad questions.
        # Ask BOOK_NAMES phrases and single proper nouns only.
        if " " in name and name not in BOOK_NAMES:
            continue
        askable.append(name)

    text_at = {}
    for u in C.units():
        text_at[u["pos"]] = C.chapter_text_through(u["pos"])

    rows = []
    seen = set()
    dropped = {"holdout": 0, "walton": 0, "unlock": 0, "dup": 0, "char": 0, "cap": 0}

    def consider(row: dict) -> dict | None:
        char = row["char"]
        pos = int(row["reader_pos"])
        user = (row.get("user") or "").strip()
        if char not in ("victor", "creature"):
            dropped["char"] += 1
            return None
        if pos < C.CHAT_FROM.get(char, 99):
            dropped["unlock"] += 1
            return None
        if not user:
            return None
        if C.too_close_to_holdout(user):
            dropped["holdout"] += 1
            return None
        if walton_user(char, pos, user):
            dropped["walton"] += 1
            return None
        key = norm_key(char, pos, user)
        if key in seen:
            dropped["dup"] += 1
            return None
        seen.add(key)
        row = dict(row)
        row["user"] = user
        return row

    if T2.is_file():
        for r in C.load_jsonl(T2):
            kept = consider(
                dict(
                    id=r.get("id"),
                    char=r.get("char"),
                    user=r.get("user"),
                    reader_pos=int(r.get("reader_pos") or 0),
                    unit_id=r.get("unit_id") or "",
                    source=r.get("source") or "train2",
                    origin="train2",
                )
            )
            if kept:
                rows.append(kept)

    n_train2 = len(rows)
    buckets: dict[tuple, list] = {}
    for u in C.units():
        pos = int(u["pos"])
        blob = text_at.get(pos) or ""
        present = []
        from protect import span_in

        for name in askable:
            if span_in(blob, name):
                present.append(name)
        # Prefer longer names so "Henry Clerval" is asked, not only "Henry".
        present.sort(key=len, reverse=True)
        # Cap names per position so one chapter does not dominate.
        present = present[:28]
        for char in ("victor", "creature"):
            if pos < C.CHAT_FROM.get(char, 99):
                continue
            for name in present:
                h = hashlib.sha256(f"{char}|{pos}|{name}".encode()).digest()
                t1 = TEMPLATES[h[0] % len(TEMPLATES)]
                t2 = TEMPLATES[h[1] % len(TEMPLATES)]
                if t2 == t1:
                    t2 = TEMPLATES[(h[1] + 1) % len(TEMPLATES)]
                for ti, tmpl in enumerate((t1, t2)):
                    user = tmpl.format(name=name)
                    kept = consider(
                        dict(
                            id=f"q-{char[0]}-{pos:02d}-{hashlib.sha1(user.encode()).hexdigest()[:8]}",
                            char=char,
                            user=user,
                            reader_pos=pos,
                            unit_id=u["unit_id"],
                            source="name_template",
                            origin="train4",
                            name=name,
                        )
                    )
                    if kept:
                        buckets.setdefault((pos, char), []).append(kept)
            for oi, opener in enumerate(OPENERS):
                if hashlib.sha256(f"{char}|{pos}|open|{oi}".encode()).digest()[0] % 2:
                    continue
                kept = consider(
                    dict(
                        id=f"q-{char[0]}-{pos:02d}-o{oi}",
                        char=char,
                        user=opener,
                        reader_pos=pos,
                        unit_id=u["unit_id"],
                        source="opener",
                        origin="train4",
                    )
                )
                if kept:
                    buckets.setdefault((pos, char), []).append(kept)

    # Round-robin across (position, character) so the late book is not cut off.
    order = sorted(buckets)
    while len(rows) < CAP and any(buckets.values()):
        progressed = False
        for key in order:
            bucket = buckets.get(key) or []
            if not bucket:
                continue
            rows.append(bucket.pop(0))
            progressed = True
            if len(rows) >= CAP:
                break
        if not progressed:
            break
    dropped["cap"] = sum(len(v) for v in buckets.values())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_char = {}
    by_src = {}
    for r in rows:
        by_char[r["char"]] = by_char.get(r["char"], 0) + 1
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(
        "questions",
        len(rows),
        "from_train2_kept",
        n_train2,
        "by_char",
        by_char,
        "by_source",
        by_src,
        "dropped",
        dropped,
        flush=True,
    )


if __name__ == "__main__":
    main()
