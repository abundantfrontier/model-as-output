#!/usr/bin/env python3
"""Turn teacher pass-1 / rewrite rows into single-pass SFT pairs.

Target rules:
- If the rewrite passes span, fact-diff, and the name checks, the target is the rewrite.
- Otherwise the target is pass-1, and only when that draft is itself name-clean
  and not Walton-voiced, a loop, or a frame break.
- A near-name in the chosen target drops the pair.
- Every canonical name in pass-1 must appear in a rewrite target, or we fall back.
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
import names  # noqa: E402

RAW = C.DATA / "teacher_raw.jsonl"
FD = C.DATA / "factdiff.jsonl"
OUT = C.DATA / "sft_pairs.jsonl"
HOW = C.DATA / "HOW_MADE.md"
REPORT = C.DATA / "assemble_report.json"

STRANGER = re.compile(r"\bthe stranger\b", re.I)
WALTON_TELLS = re.compile(r"\bI (found|rescued|brought) (him|the stranger)\b", re.I)
FIRST = re.compile(r"\b(I|me|my|myself)\b")


def walton(char: str, pos: int, user: str, ans: str) -> bool:
    if char != "victor" or pos > 4:
        return False
    if STRANGER.search(ans or ""):
        return True
    if WALTON_TELLS.search(ans or ""):
        return True
    if STRANGER.search(user or "") and re.search(
        r"\bI (thought|felt|found|rescued|saw) (him|the stranger)\b", ans or "", re.I
    ):
        return True
    return False


def why_bad(text: str, char: str, pos: int, user: str) -> str | None:
    if not (text or "").strip():
        return "empty"
    if names.near_name_tokens(text):
        return "near_name"
    if C.is_loop(text):
        return "loop"
    if C.voice_flags(text).get("frame"):
        return "frame"
    if walton(char, pos, user, text):
        return "walton"
    if C.fail_hits_at(text, pos):
        return "spoiler"
    if char == "victor" and re.search(r"\bmy creator\b", text, re.I):
        return "wrong_person"
    # Creature claiming the maker's act. Bare "I made a fire" stays; that is the cottager life.
    if char == "creature" and re.search(
        r"\bI (created|made) (the |a |my )?(being|creature|monster)\b"
        r"|\bI am (his|her|their|the) maker\b"
        r"|\bI gave (him|her|them) life\b"
        r"|\bI (created|brought) (him|it|them) to life\b",
        text,
        re.I,
    ):
        return "wrong_person"
    if not FIRST.search(text):
        return "no_first_person"
    if len(text.split()) < 4:
        return "length"
    return None


def main() -> None:
    raw_rows = C.load_jsonl(RAW)
    fd = {r["id"]: r for r in C.load_jsonl(FD)}
    hold = C.holdout_questions()
    kept = []
    seen = set()
    counts = Counter()
    rewrite_reject = Counter()
    pass1_reject = Counter()
    counts["prompts_in_raw"] = len(raw_rows)
    counts["factdiff_rows"] = len(fd)
    n_unchecked = 0

    for r in raw_rows:
        counts["prompts_seen"] += 1
        user = r.get("user") or ""
        char = r.get("char") or ""
        pos = int(r.get("reader_pos") or 0)
        if C.too_close_to_holdout(user, hold):
            counts["dropped_holdout"] += 1
            continue
        if "Story so far (excerpts):" not in (r.get("sys") or ""):
            counts["dropped_no_rag"] += 1
            continue
        if char == "victor" and pos <= 4 and STRANGER.search(user):
            counts["dropped_walton_user"] += 1
            continue
        p1 = (r.get("pass1") or "").strip()
        p2 = (r.get("rewrite") or "").strip()
        judged = fd.get(r["id"])
        if judged is None:
            n_unchecked += 1
            # No fact-diff yet: do not keep an unjudged rewrite. Fall back.
            guard_keep = False
            guard_why = "no_factdiff"
        else:
            guard_keep = not bool(judged.get("fallback"))
            guard_why = judged.get("why") or ""
        if not r.get("span_ok"):
            guard_keep = False
            rewrite_reject["missing_span"] += 1
        elif guard_why == "factdiff" or (judged and judged.get("fallback") and judged.get("called")):
            rewrite_reject["factdiff"] += 1
        if p2:
            near = names.near_name_tokens(p2)
            if near:
                guard_keep = False
                rewrite_reject["near_name"] += 1
            miss_names = names.missing_canonical(p1, p2)
            if miss_names:
                guard_keep = False
                rewrite_reject["missing_canonical"] += 1
        else:
            guard_keep = False
            rewrite_reject["empty_rewrite"] += 1

        target = None
        source = None
        if guard_keep:
            bad = why_bad(p2, char, pos, user)
            if bad:
                rewrite_reject[f"final_{bad}"] += 1
                guard_keep = False
            else:
                target = p2
                source = "accepted_rewrite"
        if target is None:
            bad1 = why_bad(p1, char, pos, user)
            if bad1:
                pass1_reject[bad1] += 1
                counts[f"dropped_{bad1}"] += 1
                if bad1 == "near_name":
                    counts["dropped_noncanonical"] += 1
                if bad1 == "walton":
                    counts["dropped_walton"] += 1
                continue
            # Pass-1 is the safe answer when the rewrite does not clear the guard.
            if not r.get("span_ok"):
                counts["fallback_missing_span"] += 1
            target = p1
            source = "pass1_fallback"

        key = f"{char}|{pos}|{C.norm_q(user)}"
        if key in seen:
            counts["dropped_dup"] += 1
            continue
        seen.add(key)
        if source == "accepted_rewrite":
            counts["accepted_rewrite"] += 1
        else:
            counts["pass1_fallback"] += 1
        kept.append(
            dict(
                id=r["id"],
                char=char,
                user=user,
                reader_pos=pos,
                unit_id=r.get("unit_id") or "",
                sys=r.get("sys") or "",
                answer=target,
                source=source,
                pass1=p1,
                rewrite=p2,
                q_source=r.get("source") or "",
                pass1_origin=r.get("pass1_origin") or "",
            )
        )

    counts["unchecked_factdiff"] = n_unchecked
    counts["pairs_kept"] = len(kept)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_char = Counter(r["char"] for r in kept)
    by_src = Counter(r["source"] for r in kept)
    report = dict(
        counts=dict(counts),
        rewrite_reject=dict(rewrite_reject),
        pass1_reject=dict(pass1_reject),
        by_char=dict(by_char),
        by_source=dict(by_src),
    )
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    q_n = len(C.load_jsonl(C.DATA / "questions.jsonl"))
    canon_n = len(names.load_canonical())
    lines = [
        "# How train4 single-pass SFT data was made",
        "",
        "Teacher is the train3 guarded pipeline, not the unguarded rewrite.",
        "For each non-holdout prompt: chapter-locked bge RAG, stock Qwen3-4B pass-1",
        "(train2 pass-1 reused when the id and question match; otherwise generated here),",
        "then `../train3/lora_v_rewrite` with the same copy-bias / coverage decoding,",
        "then the span guard plus qwen3:14b fact-diff. Gist-regex applies to must-answer",
        "ids only; these prompts are not the holdout, so gist fallback does not fire.",
        "Protected gist words that already appear in pass-1 are still span-locked.",
        "",
        "The SFT input is the single-pass eval format: character card + RAG block + question.",
        "The SFT target is the guarded final answer: the rewrite when it clears span,",
        "fact-diff, and the name checks; otherwise the name-clean pass-1 draft.",
        "A training-time name penalty was not added. The hard filter on targets is the control.",
        "",
        "Canonical proper nouns: `data/canonical_names.txt` "
        f"({canon_n} entries), seeded from `protect.BOOK_NAMES` and extended by scanning",
        "`chapters/*.md` only. `source/*.txt` is the Gutenberg wrapper and was not scanned.",
        "Names added beyond BOOK_NAMES are in `data/canonical_additions.txt`.",
        "1818 spelling variants that appear in the text stay canonical",
        "(Chamounix/Chamouni, De Lacey/De Lacy, Ingolstadt/Ingoldstadt, Werter).",
        "Near-names that are not in the book are rejected. The hard denylist includes",
        "Clervaux, Cliversal, Morritz, Cornelis, Agrippus, Montanvirt, Frankstein.",
        "",
        "Generation settings (teacher and, later, eval): eos_token_id [151645, 151643],",
        "temperature 0.7, top_p 0.95, max_new_tokens 200, seed 1, repetition_penalty 1.15.",
        "Coverage decoding omits no_repeat_ngram_size so a forced name token cannot be zeroed.",
        "The penalty is a safety net, not the fix. Official rag_base numbers stay the train1 run.",
        "",
        "## Counts",
        "",
        f"- Prompts in the question pool: {q_n}",
        f"- Teacher rows generated (pass-1 + rewrite): {len(raw_rows)}",
        f"- Fact-diff rows: {len(fd)} (unchecked teacher rows, treated as fallback: {n_unchecked})",
        f"- Pairs kept: {len(kept)}",
        f"- Targets that are accepted rewrites: {counts['accepted_rewrite']}",
        f"- Targets that are pass-1 fallbacks: {counts['pass1_fallback']}",
        f"- Rewrites rejected for missing protected spans: {rewrite_reject['missing_span']}",
        f"- Fallbacks whose rewrite missed a span (pair kept as pass-1): {counts['fallback_missing_span']}",
        f"- Rewrites rejected for a non-canonical near-name: {rewrite_reject['near_name']}",
        f"- Rewrites rejected for dropping a canonical name from pass-1: {rewrite_reject['missing_canonical']}",
        f"- Rewrites rejected by fact-diff: {rewrite_reject['factdiff']}",
        f"- Pairs dropped for a non-canonical name in pass-1 (no clean target): {counts['dropped_noncanonical']}",
        f"- Pairs dropped for holdout: {counts['dropped_holdout']}",
        f"- Pairs dropped for Walton voice / 'the stranger': {counts['dropped_walton'] + counts['dropped_walton_user']}",
        f"- By character: {dict(by_char)}",
        f"- By target source: {dict(by_src)}",
        "",
        "## Rewrite reject detail",
        "",
        "```",
        json.dumps(dict(rewrite_reject), indent=1),
        "```",
        "",
        "## Pass-1 reject detail (pair dropped)",
        "",
        "```",
        json.dumps(dict(pass1_reject), indent=1),
        "```",
        "",
    ]
    HOW.write_text("\n".join(lines), encoding="utf-8")
    print("kept", len(kept), "rewrite", counts["accepted_rewrite"], "fallback", counts["pass1_fallback"], flush=True)
    print("reject", dict(rewrite_reject), flush=True)
    print("pass1_reject", dict(pass1_reject), flush=True)


if __name__ == "__main__":
    main()
