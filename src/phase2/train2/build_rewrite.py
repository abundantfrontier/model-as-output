#!/usr/bin/env python3
"""Assemble rewrite CE pairs: gold style_flatten + teacher rewrites of rag_base pass-1.

Target 500–1500. No 3x handwritten repeat. Holdout filtered. Writes HOW_MADE.md.
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
import prompts  # noqa: E402

CAP = 1500
MIN_PAIRS = 500


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def flatten(text):
    parts = sentences(text)
    if not parts:
        return None
    return " ".join(f"{i}. {s}" for i, s in enumerate(parts, 1))


def ok_text(text, pos):
    if not text or not text.strip():
        return False
    words = text.split()
    if len(words) < 8 or len(words) > 180:
        return False
    if C.fail_hits_at(text, pos):
        return False
    if C.voice_flags(text)["frame"]:
        return False
    if C.is_loop(text):
        return False
    return True


def make_pair(char, question, source_text, target, kind, pos, unit_id, qid=""):
    if not ok_text(source_text, pos) or not ok_text(target, pos):
        return None
    if source_text.strip() == target.strip():
        return None
    user = prompts.rewrite_user(question, source_text)
    sysm = C.sysprompt(char, pos)
    if prompts.RAG_MARK in sysm or prompts.RAG_MARK in user:
        return None
    if not user.endswith(prompts.REWRITE_INSTR):
        raise RuntimeError("rewrite instruction missing")
    if C.too_close_to_holdout(question):
        return None
    return dict(
        id=qid or f"{kind}-{char}-{pos}",
        char=char,
        user=user,
        answer=target.strip(),
        sys=sysm,
        reader_pos=int(pos),
        unit_id=unit_id,
        ch=int(pos),
        source=kind,
        question=question,
        source_text=source_text,
    )


def main():
    voice = C.load_jsonl(C.DATA / "voice.jsonl")
    teacher = C.load_jsonl(C.DATA / "rewrite_teacher.jsonl")
    rejected = C.load_jsonl(C.DATA / "rewrite_reject.jsonl")
    rej_n = len(rejected)
    rej_why = dict(Counter((r.get("why") or "?").split(":")[0] for r in rejected))
    flat = []
    for r in voice:
        char = r.get("char")
        if char not in ("victor", "creature"):
            continue
        pos = int(r.get("reader_pos") or r.get("ch") or 4)
        src = flatten(r.get("answer") or "")
        pair = make_pair(
            char,
            r.get("user") or "",
            src or "",
            r.get("answer") or "",
            "style_flatten",
            pos,
            r.get("unit_id"),
            f"flat-{r.get('id')}",
        )
        if pair:
            # Prefer handwritten flatten; skip extra paraphrases if we are over cap later.
            pair["gold_source"] = r.get("source")
            flat.append(pair)

    # Prefer handwritten flatten, then generated, then paraphrase.
    rank = {"handwritten": 0, "generated": 1, "paraphrase": 2}
    flat.sort(key=lambda r: (rank.get(r.get("gold_source"), 9), r["char"], r["question"]))

    rows = []
    seen = set()

    def add(p):
        key = (p["char"], C.norm_q(p["question"]), C.norm_q(p["source_text"][:80]))
        if key in seen:
            return False
        seen.add(key)
        rows.append(p)
        return True

    # Teacher rewrites are the main objective (eval-matched task).
    n_teacher = 0
    for r in teacher:
        if r.get("char") not in ("victor", "creature"):
            continue
        if not r.get("user") or not r.get("answer"):
            continue
        if C.too_close_to_holdout(r.get("question") or ""):
            continue
        if not ok_text(r["answer"], int(r.get("reader_pos") or 4)):
            continue
        try:
            prompts.assert_no_retrieval(r.get("sys") or C.sysprompt(r["char"], int(r["reader_pos"])), r["user"])
        except Exception:
            continue
        if add(r):
            n_teacher += 1
        if len(rows) >= CAP:
            break

    n_flat = 0
    for p in flat:
        if len(rows) >= CAP:
            break
        # Keep flatten as the small gold mix, not the whole set.
        if n_flat >= max(80, n_teacher // 8) and p.get("gold_source") != "handwritten":
            # still keep remaining handwritten
            continue
        if add(p):
            n_flat += 1

    # Always include remaining handwritten flatten even near cap (they're few).
    for p in flat:
        if p.get("gold_source") != "handwritten":
            continue
        if len(rows) >= CAP:
            break
        add(p)

    if len(rows) < 50:
        raise SystemExit(f"too few rewrite pairs: {len(rows)}")

    out = C.DATA / "rewrite_pairs.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    src_counts = dict(Counter(r.get("source") for r in rows))
    char_counts = dict(Counter(r["char"] for r in rows))
    meta = dict(
        path=str(out),
        rows=len(rows),
        teacher=n_teacher,
        style_flatten=n_flat,
        sources=src_counts,
        chars=char_counts,
        cap=CAP,
        min_pairs=MIN_PAIRS,
        instruction=prompts.REWRITE_INSTR,
        retrieval_in_rewrite=False,
        below_target=len(rows) < MIN_PAIRS,
    )
    (C.DATA / "rewrite_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    clean = {}
    cr = C.DATA / "clean_report.json"
    if cr.is_file():
        clean = json.loads(cr.read_text())
    how = C.DATA / "HOW_MADE.md"
    how.write_text(
        "\n".join(
            [
                "# How train2 rewrite data was made",
                "",
                "Spec: `../dev1/VOICE_DIAGNOSIS.md` (concrete retrain plan). "
                "Eval is a style-only rewrite of a rag_base pass-1, so CE is trained as rewrite, not Q&A.",
                "",
                "## Cleaned voice Q&A (small gold mix, not the main objective)",
                "",
                f"- Handwritten seeds: {clean.get('kept', {}).get('handwritten', '?')} (copied from `../voice/`; no 3× CE repeat).",
                f"- Generated kept: {clean.get('kept', {}).get('generated', '?')}. Dropped Walton-voice / 'the stranger' as a third person on Letter IV, and generated user lines that talk about Victor in the third person.",
                f"- Paraphrases kept: {clean.get('kept', {}).get('paraphrase', '?')} (at most 2 per handwritten seed; paraphrases kept, 3× repeat dropped).",
                f"- voice.jsonl total: {clean.get('kept', {}).get('voice', '?')}.",
                f"- Drop reasons: `{json.dumps(clean.get('dropped', {}))}`",
                "",
                "## Question pool and pass-1",
                "",
                f"- Unique in-scope (char, question) pairs: {clean.get('kept', {}).get('questions', '?')} (cleaned voice + generated facts questions for unlocked characters).",
                "- Holdout: `../questions/must-answer.json` and `leak-probes.json`, Jaccard 0.62 / substring, same as train1 `check_holdout.py`.",
                "- Pass-1: stock Qwen3-4B + chapter-locked bge RAG (`scripts/gen_pass1.py`), temperature 0.7, top_p 0.95, max_new_tokens 200, seed 1, repetition_penalty 1.15, no_repeat_ngram_size 3, eos_token_id [151645, 151643].",
                "- Unlock: Victor reader_pos ≥ 4, creature ≥ 13. No training line before that.",
                "",
                "## Teacher rewrite",
                "",
                "- Local Ollama `qwen3:14b`, JSON mode, think off, temp 0.4.",
                "- Instruction: style-only, keep the same facts, first person, no stutter.",
                "- Rejected: token loops, frame breaks, wrong-person narration, cheap proper-name fact drift, unchanged copies.",
                f"- Teacher pairs kept in assemble: {n_teacher}.",
                f"- Teacher file rows: {len(teacher)}. Reject file: {rej_n} ({rej_why}).",
                "",
                "## Assembled rewrite_pairs.jsonl",
                "",
                f"- rows: {len(rows)} (cap {CAP}, target {MIN_PAIRS}–{CAP})",
                f"- by source: {src_counts}",
                f"- by char: {char_counts}",
                "- Train user message matches pass-2 eval (`prompts.rewrite_user` + `REWRITE_INSTR`). No retrieval block.",
                "- Style-flatten of gold answers is a small mix; teacher rewrite of rag_base pass-1 is the main objective.",
                "",
                "## DPO",
                "",
                "- Optional, only if the loop smoke test passes and time remains. See `dpo.jsonl` / `HOW_MADE.md` update after that step.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2), flush=True)
    if len(rows) < MIN_PAIRS:
        print("WARNING below target", len(rows), "<", MIN_PAIRS, flush=True)


if __name__ == "__main__":
    main()
