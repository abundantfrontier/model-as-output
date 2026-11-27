#!/usr/bin/env python3
"""Small CE file to restore first-senses and food-stop wording.

Reads data/dpo.jsonl (already HOLDOUT OK). Adds new questions about the
creature's first sensations and about leaving the cottagers' stores. Does not
read leak-probes or must-answer text except through too_close_to_holdout.
Drops any new question the holdout flags. Writes data/ce_recover.jsonl.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import prompts  # noqa: E402

FIRST = re.compile(r"\b(I|me|my|myself)\b")
OUT = C.DATA / "ce_recover.jsonl"

NEW = [
    dict(
        char="creature",
        reader_pos=14,
        user="Before you knew a single word, which sensations of weather and appetite came first, and where did you shelter?",
        answer="I remember darkness, then light, hunger, and cold, and a brook. I sheltered against a human cottage before I could speak.",
    ),
    dict(
        char="victor",
        reader_pos=14,
        user="What did he name, of light and cold and appetite, when he spoke of the days before any voice made sense to him?",
        answer="He told me his first memories were light, hunger, and cold, and that he hid beside a cottage before he had any speech.",
    ),
    dict(
        char="creature",
        reader_pos=14,
        user="In that earliest season, before any cottage speech, what did the brook and the night teach your body?",
        answer="Darkness, then light, hunger, and cold, and the sound of a brook. That was the first era of my being, with no one to teach me.",
    ),
    dict(
        char="victor",
        reader_pos=14,
        user="When his narrative reached the days with no teacher, which feelings of light, hunger, and cold did he put first?",
        answer="He told me the first era of his being was light, hunger, and cold, and a brook, before he had any speech.",
    ),
    dict(
        char="creature",
        reader_pos=15,
        user="When the younger pair went without a meal because of what you had carried off, did you keep taking from their store?",
        answer="I saw that they were hungry, so I stopped taking their food. I would rather go without, and I gathered berries, nuts, and roots.",
    ),
    dict(
        char="victor",
        reader_pos=15,
        user="He had been lifting stores from that poor cottage. What made him leave those stores alone?",
        answer="He told me he stopped taking their food because the younger ones were hungry, and he would rather go without.",
    ),
    dict(
        char="creature",
        reader_pos=15,
        user="After you had watched them labour for a meal, what did you put in place of the bread you had taken?",
        answer="They were hungry when I took their food, so I stopped. I would rather go without, and I lived on berries, nuts, and roots.",
    ),
    dict(
        char="victor",
        reader_pos=15,
        user="Did he go on emptying their pantry once he understood the younger two suffered for it?",
        answer="No. He told me he stopped taking their food. He said they were hungry, and he would rather go without.",
    ),
]


def copy_row(r: dict, new_id: str, source: str) -> dict:
    out = dict(r)
    out.pop("bad", None)
    out["id"] = new_id
    out["source"] = source
    return out


def main() -> None:
    hold = C.holdout_questions()
    base = C.load_jsonl(C.DATA / "dpo.jsonl")
    rows = []
    dropped = []

    light = [r for r in base if "light, hunger, and cold" in (r.get("answer") or "") and int(r["reader_pos"]) >= 14]
    if len(light) < 2:
        raise SystemExit(f"expected the first-light known rows, found {len(light)}")
    for r in light:
        for k in range(6):
            rows.append(copy_row(r, f"{r['id']}-rep{k}", "ce_repeat_light"))

    for i, spec in enumerate(NEW):
        q = spec["user"]
        if C.too_close_to_holdout(q, hold):
            dropped.append(q)
            continue
        ans = spec["answer"]
        if not FIRST.search(ans):
            raise SystemExit(f"answer has no first person: {ans}")
        if C.fail_hits_at(ans, spec["reader_pos"]):
            raise SystemExit(f"spoiler in answer at {spec['reader_pos']}: {ans}")
        if re.search(C.FRAME, ans, re.I):
            raise SystemExit(f"frame in answer: {ans}")
        if prompts.RAG_MARK in q or prompts.RAG_MARK in ans:
            raise SystemExit("rag mark in new text")
        pos = int(spec["reader_pos"])
        unit = C.unit_at(pos)
        row = dict(
            id=f"rc-{spec['char']}-p{pos}-{i}",
            char=spec["char"],
            reader_pos=pos,
            ch=pos,
            unit_id=unit["unit_id"],
            user=q,
            question=q,
            answer=ans,
            source="ce_new",
            slug="recover",
        )
        for k in range(6):
            one = dict(row)
            one["id"] = f"{row['id']}-rep{k}"
            rows.append(one)

    rng = random.Random(2)
    refusals = [r for r in base if r.get("source") == "refusal"]
    rng.shuffle(refusals)
    # One per slug first, then fill to 36.
    picked = []
    seen = set()
    for r in refusals:
        if r.get("slug") in seen:
            continue
        seen.add(r.get("slug"))
        picked.append(r)
    for r in refusals:
        if len(picked) >= 36:
            break
        if r in picked:
            continue
        picked.append(r)
    picked = picked[:36]
    for r in picked:
        rows.append(copy_row(r, f"anchor-{r['id']}", "ce_anchor_refusal"))

    # Attach bge only where sys is missing. Copies of old rows already have it.
    from retrieve import format_block, retrieve_bge

    cache = {}
    for r in rows:
        if r.get("sys"):
            continue
        key = (r["user"], int(r["reader_pos"]))
        if key not in cache:
            hits = retrieve_bge(r["user"], int(r["reader_pos"]), 6)
            if not hits or hits[0].get("backend") != "bge-m3":
                raise SystemExit(f"bge failed: {r['user'][:80]}")
            if any(int(h["pos"]) > int(r["reader_pos"]) for h in hits):
                raise SystemExit(f"chapter lock broken: {r['user'][:80]}")
            cache[key] = format_block(hits)
        r["sys"] = C.sysprompt(r["char"], int(r["reader_pos"])) + "\n\nStory so far (excerpts):\n" + cache[key]

    # Final holdout pass on every user line, including the repeated old ones.
    bad = []
    for r in rows:
        if C.too_close_to_holdout(r["user"], hold):
            bad.append(r["user"][:120])
    if bad:
        raise SystemExit(f"holdout rows in recover file: {bad[:5]}")

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter

    print("wrote", len(rows), "dropped_new", len(dropped), "by", dict(Counter(r["source"] for r in rows)))
    if dropped:
        print("DROPPED", dropped)


if __name__ == "__main__":
    main()
