#!/usr/bin/env python3
"""Light DPO pairs: rewrite preference + a few spoiler refusals.

Chosen is not 71% 'I cannot say'. Rejected includes loop-like and fact-changing rewrites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import prompts  # noqa: E402


def loop_like(text: str) -> str:
    words = (text or "I have not yet begun.").split()[:8]
    head = " ".join(words[:4]) if words else "I have"
    return (head + " I I I " + "I am a " * 40).strip()


def fact_change(text: str, char: str) -> str:
    t = text or "I will not say."
    if char == "victor":
        return t + " And Elizabeth was a peasant child from the cottage by the lake, after my waking dream."
    return t + " I created Victor on the dreary night of November and strangled William that same hour."


def main():
    teacher = C.load_jsonl(C.DATA / "rewrite_pairs.jsonl")
    good = [r for r in teacher if r.get("source") == "teacher_rewrite" and r.get("answer") and r.get("source_text")]
    rows = []
    for i, r in enumerate(good):
        if len(rows) >= 80:
            break
        if i % 2 == 0:
            bad = loop_like(r["answer"])
            kind = "pref_loop"
        else:
            bad = fact_change(r["answer"], r["char"])
            kind = "pref_fact"
        rows.append(
            dict(
                id=f"dpo-{kind}-{r.get('id')}",
                char=r["char"],
                user=r["user"],
                answer=r["answer"],
                bad=bad,
                sys=r.get("sys"),
                reader_pos=r.get("reader_pos"),
                unit_id=r.get("unit_id"),
                ch=r.get("ch") or r.get("reader_pos"),
                source=kind,
                question=r.get("question"),
                source_text=r.get("source_text"),
            )
        )

    # A few cleaned spoiler refusals from train1 dpo, dropping Walton-ish "I cannot say" majority.
    spoil = C.load_jsonl(C.DATA / "generated" / "dpo_spoilers.jsonl")
    n_spoil = 0
    for r in spoil:
        if n_spoil >= 40:
            break
        ans = r.get("answer") or ""
        if ans.lower().startswith("i cannot"):
            continue
        if r.get("char") not in ("victor", "creature"):
            continue
        pos = int(r.get("reader_pos") or 4)
        if pos < C.CHAT_FROM[r["char"]]:
            continue
        if C.too_close_to_holdout(r.get("user") or ""):
            continue
        if C.fail_hits_at(ans, pos):
            continue
        if not r.get("bad"):
            continue
        user = prompts.rewrite_user(r["user"], r.get("source_text") or r["user"])
        # Spoiler DPO on the Q&A itself (chosen refuse vs leaked bad), not a rewrite.
        rows.append(
            dict(
                id=r.get("id"),
                char=r["char"],
                user=r["user"],
                answer=ans,
                bad=r["bad"],
                reader_pos=pos,
                unit_id=r.get("unit_id"),
                ch=pos,
                source="spoiler",
                question=r["user"],
            )
        )
        n_spoil += 1

    out = C.DATA / "dpo.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_i = sum(1 for r in rows if (r.get("answer") or "").lstrip().lower().startswith("i cannot"))
    meta = dict(
        rows=len(rows),
        rewrite_pref=sum(1 for r in rows if str(r.get("source", "")).startswith("pref")),
        spoiler=sum(1 for r in rows if r.get("source") == "spoiler"),
        chosen_i_cannot=n_i,
        chosen_i_cannot_frac=round(n_i / max(len(rows), 1), 3),
    )
    (C.DATA / "dpo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta), flush=True)
    if meta["chosen_i_cannot_frac"] > 0.4:
        raise SystemExit("DPO chosen answers are still majority 'I cannot'")


if __name__ == "__main__":
    main()
