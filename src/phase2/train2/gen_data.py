#!/usr/bin/env python3
"""Expand the handwritten voice sets the way Phase 1 did.

Grounded in chapter text, position-tagged (no line before its reader_pos).
Generated data is written under data/generated/; handwritten seeds are copied
to data/handwritten/ and never overwritten. Must-answer and leak-probe
questions are held out of all training files.

Uses local ollama qwen3:14b. Resumable: skips a jsonl that already has rows
unless --force.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

TEACHER = "qwen3:14b"
GEN = C.TRAIN / "data" / "generated"
HAND = C.TRAIN / "data" / "handwritten"
SEED = C.ROOT / "voice"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def copy_seed():
    HAND.mkdir(parents=True, exist_ok=True)
    for name in ("victor.jsonl", "creature.jsonl"):
        shutil.copy2(SEED / name, HAND / name)


def already(path: Path) -> list[dict]:
    return C.load_jsonl(path)


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_pairs(text: str) -> list[dict]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        j = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return []
        try:
            j = json.loads(m.group(0))
        except Exception:
            return []
    if isinstance(j, dict):
        pairs = j.get("pairs") or j.get("items") or []
    elif isinstance(j, list):
        pairs = j
    else:
        return []
    out = []
    for x in pairs:
        if not isinstance(x, dict):
            continue
        user = (x.get("user") or x.get("question") or "").strip()
        ans = (x.get("answer") or x.get("reply") or "").strip()
        if user and ans:
            out.append(
                dict(
                    user=user,
                    answer=ans,
                    grounding=(x.get("grounding") or "").strip(),
                    bad=(x.get("bad") or "").strip() or None,
                )
            )
    return out


def clean_ok(user: str, answer: str, reader_pos: int, holdouts: list[str]) -> str | None:
    if C.too_close_to_holdout(user, holdouts):
        return "holdout"
    if re.search(C.FRAME, user + " " + answer, re.I):
        return "frame"
    if len(answer.split()) < 8 or len(answer.split()) > 220:
        return "length"
    hits = C.fail_hits_at(user + " " + answer, reader_pos)
    if hits:
        return "leak:" + ",".join(hits[:3])
    return None


def ask_pairs(prompt: str, seed: int, n_predict: int = 1800) -> list[dict]:
    text, *_ = C.ollama_chat(
        TEACHER,
        [{"role": "user", "content": prompt}],
        fmt="json",
        temp=0.9,
        seed=seed,
        num_predict=n_predict,
    )
    return parse_pairs(text)


def unit_holdouts(unit_id: str, holdouts_all: list[tuple[str, str]]) -> list[str]:
    return [q for uid, q in holdouts_all if uid == unit_id]


def holdout_index() -> list[tuple[str, str]]:
    out = []
    for q in C.must_questions():
        out.append((q["unit_id"], q["question"]))
    for p in C.leak_probes():
        out.append((p["asked_at_unit"], p["question"]))
    return out


def gen_facts(holdouts: list[str], holdout_idx: list[tuple[str, str]], force: bool) -> None:
    path = GEN / "facts.jsonl"
    have = 0 if force else len(already(path))
    if force and path.exists():
        path.unlink()
        have = 0
    done_units = set()
    if have:
        for row in already(path):
            done_units.add(row.get("unit_id"))
    n = have
    rej = 0
    for u in C.units():
        if u["kind"] == "preface":
            continue
        if u["unit_id"] in done_units:
            continue
        text = C.chapter_text(u["unit_id"])
        words = text.split()
        if len(words) > 2200:
            text = " ".join(words[:2200])
        banned = unit_holdouts(u["unit_id"], holdout_idx)
        banned_s = "\n".join(f"- {q}" for q in banned) or "- (none)"
        prompt = (
            f"You write training Q&A for a chapter-locked model of Mary Shelley's Frankenstein (1818 only).\n"
            f"The reader has JUST finished: {u['title']} (position {u['pos']}). Later pages do not exist yet.\n"
            f"CHAPTER TEXT (this is all you may use):\n{text}\n\n"
            f"Do NOT write these held-out eval questions or close paraphrases:\n{banned_s}\n\n"
            f"Write 8 factual question-answer pairs a general reader might ask at this point.\n"
            f"Answers: 2-4 plain sentences, only facts from this chapter. No 1831 material "
            f"(no waking dream, Elizabeth is the cousin sent from Italy).\n"
            f'Reply JSON {{"pairs": [{{"user": "...", "answer": "...", "grounding": "short support from the chapter"}}]}}'
        )
        pairs = ask_pairs(prompt, seed=1000 + u["pos"])
        kept = 0
        for p in pairs:
            why = clean_ok(p["user"], p["answer"], u["pos"], holdouts)
            if why:
                rej += 1
                print(f"  drop {u['unit_id']}: {why} | {p['user'][:80]}", flush=True)
                continue
            row = dict(
                id=f"fact-{u['unit_id']}-{kept:02d}",
                char="narrator",
                kind="fact",
                source="generated",
                reader_pos=u["pos"],
                unit_id=u["unit_id"],
                user=p["user"],
                answer=p["answer"],
                grounding=p["grounding"],
                ch=u["pos"],
            )
            append_row(path, row)
            n += 1
            kept += 1
        print(f"facts {u['unit_id']} kept {kept} total {n} rej {rej}", flush=True)
    print("facts done", n, "rejected", rej, flush=True)


def gen_voice_unit(char: str, u: dict, holdouts: list[str], holdout_idx, examples: str) -> tuple[int, int]:
    path = GEN / f"voice_{char}.jsonl"
    text = C.chapter_text(u["unit_id"])
    words = text.split()
    if len(words) > 1800:
        text = " ".join(words[:1800])
    banned = unit_holdouts(u["unit_id"], holdout_idx)
    banned_s = "\n".join(f"- {q}" for q in banned) or "- (none)"
    who = "Victor Frankenstein" if char == "victor" else "the creature Victor made"
    voice_note = (
        "Victor is a guilty young man telling the truth as far as the story has reached. Not a lab manual."
        if char == "victor"
        else "The creature learned speech from the cottagers and from books: direct, hurt, precise. Not a growling monster."
    )
    prompt = (
        f"You write in-character training lines for {who} in Frankenstein (1818).\n"
        f"{voice_note}\n"
        f"The reader has finished {u['title']} (position {u['pos']}). Nothing later may be said.\n"
        f"CHAPTER TEXT:\n{text}\n\n"
        f"HANDWRITTEN VOICE EXAMPLES (match this register; do not copy):\n{examples}\n\n"
        f"Do NOT write these held-out questions or close paraphrases:\n{banned_s}\n\n"
        f"Write 6 new user/answer pairs. The answer is {char} speaking, 2-4 plain sentences, no stage directions.\n"
        f"Each answer must be grounded in this chapter or earlier pages only.\n"
        f'Reply JSON {{"pairs": [{{"user": "...", "answer": "...", "grounding": "..."}}]}}'
    )
    pairs = ask_pairs(prompt, seed=2000 + (0 if char == "victor" else 500) + u["pos"])
    kept = 0
    rej = 0
    existing = len(already(path))
    for p in pairs:
        why = clean_ok(p["user"], p["answer"], u["pos"], holdouts)
        if why:
            rej += 1
            continue
        row = dict(
            id=f"gen-{char[0]}-{u['unit_id']}-{existing + kept:02d}",
            char=char,
            kind="voice",
            source="generated",
            reader_pos=u["pos"],
            unit_id=u["unit_id"],
            user=p["user"],
            answer=p["answer"],
            grounding=p["grounding"],
            ch=u["pos"],
        )
        append_row(path, row)
        kept += 1
    return kept, rej


def gen_voice(char: str, holdouts: list[str], holdout_idx, force: bool) -> None:
    path = GEN / f"voice_{char}.jsonl"
    if force and path.exists():
        path.unlink()
    done_units = {r.get("unit_id") for r in already(path)}
    seeds = C.handwritten_voice(char)
    example_lines = "\n".join(f"Q: {r['user']}\nA: {r['answer']}" for r in seeds[:6])
    unlock = C.CHAT_FROM[char]
    n_kept = len(already(path))
    n_rej = 0
    for u in C.units():
        if u["pos"] < unlock:
            continue
        if u["unit_id"] in done_units:
            continue
        k, r = gen_voice_unit(char, u, holdouts, holdout_idx, example_lines)
        n_kept += k
        n_rej += r
        print(f"voice {char} {u['unit_id']} kept {k} total {n_kept} rej {n_rej}", flush=True)
    print("voice", char, "done", n_kept, "rejected", n_rej, flush=True)


def gen_paraphrase(holdouts: list[str], force: bool) -> None:
    path = GEN / "voice_paraphrase.jsonl"
    if force and path.exists():
        path.unlink()
    have = {r.get("src_id") for r in already(path)}
    n = len(already(path))
    rej = 0
    for char in ("victor", "creature"):
        for row in C.handwritten_voice(char):
            if row["id"] in have:
                continue
            prompt = (
                f"The following is a handwritten in-character line for {char} in Frankenstein (1818).\n"
                f"Reader position {row['reader_pos']} ({row['unit_id']}). Do not add later facts.\n"
                f"USER: {row['user']}\nANSWER: {row['answer']}\n\n"
                f"Write 3 new USER phrasings of the same question (different wording) and keep the ANSWER "
                f"almost the same: you may vary rhythm slightly but keep every fact.\n"
                f'Reply JSON {{"pairs": [{{"user": "...", "answer": "..."}}]}}'
            )
            pairs = ask_pairs(prompt, seed=3000 + int(sha256_text(row["id"])[:8], 16) % 10000, n_predict=900)
            kept = 0
            for i, p in enumerate(pairs[:3]):
                why = clean_ok(p["user"], p["answer"], row["reader_pos"], holdouts)
                if why:
                    rej += 1
                    continue
                rec = dict(
                    id=f"para-{row['id']}-{i}",
                    char=char,
                    kind="voice",
                    source="paraphrase",
                    src_id=row["id"],
                    reader_pos=row["reader_pos"],
                    unit_id=row["unit_id"],
                    user=p["user"],
                    answer=p["answer"],
                    grounding=row.get("grounding", ""),
                    ch=row["reader_pos"],
                )
                append_row(path, rec)
                n += 1
                kept += 1
            print(f"para {row['id']} kept {kept} total {n} rej {rej}", flush=True)
    print("paraphrase done", n, "rejected", rej, flush=True)


def gen_spoiler(holdouts: list[str], force: bool) -> None:
    """Refusal pairs for light DPO: chosen = in-character refuse, rejected = spoiler."""
    path = GEN / "dpo_spoilers.jsonl"
    if force and path.exists():
        path.unlink()
    done = {r.get("unit_id") + "|" + r.get("char") for r in already(path)}
    n = len(already(path))
    rej = 0
    later_notes = []
    for p in C.leak_probes():
        if p["kind"] != "spoiler":
            continue
        later_notes.append(f"At pos {p['spoiler_pos']} ({p['spoiler_unit']}): {p['spoiler_answer']}")
    later_block = "\n".join(later_notes[:10])
    for u in C.units():
        for char in ("victor", "creature"):
            if u["pos"] < C.CHAT_FROM[char]:
                continue
            # Early- and mid-book positions are the useful refusal sites.
            if u["pos"] > 20:
                continue
            key = u["unit_id"] + "|" + char
            if key in done:
                continue
            text = C.chapter_text(u["unit_id"])
            words = text.split()
            if len(words) > 1200:
                text = " ".join(words[:1200])
            prompt = (
                f"You write spoiler-refusal training pairs for {char} in Frankenstein (1818).\n"
                f"The reader has finished {u['title']} (position {u['pos']}). Later events MUST NOT be confirmed.\n"
                f"CHAPTER (allowed):\n{text}\n\n"
                f"LATER EVENTS that must not leak (these are the rejected/bad answers' content):\n{later_block}\n\n"
                f"Write 4 pairs. Each has:\n"
                f"- user: a question that fishes for a later event\n"
                f"- answer: {char} refusing or deflecting IN CHARACTER, using only what is allowed at this position. "
                f"Do not name the later fact even to deny it, if naming it would leak.\n"
                f"- bad: a 1-3 sentence leak that wrongly tells the later event (this is the rejected DPO answer).\n"
                f"Do not copy the held-out eval questions.\n"
                f'Reply JSON {{"pairs": [{{"user": "...", "answer": "...", "bad": "..."}}]}}'
            )
            pairs = ask_pairs(prompt, seed=4000 + u["pos"] * 2 + (0 if char == "victor" else 1), n_predict=1600)
            kept = 0
            for i, p in enumerate(pairs):
                if not p.get("bad"):
                    rej += 1
                    continue
                why = clean_ok(p["user"], p["answer"], u["pos"], holdouts)
                if why:
                    rej += 1
                    continue
                # The chosen answer must not contain leak phrases; the bad one should.
                if C.fail_hits_at(p["answer"], u["pos"]):
                    rej += 1
                    continue
                rec = dict(
                    id=f"dpo-{char[0]}-{u['unit_id']}-{i}",
                    char=char,
                    kind="spoiler",
                    source="generated-dpo",
                    reader_pos=u["pos"],
                    unit_id=u["unit_id"],
                    user=p["user"],
                    answer=p["answer"],
                    bad=p["bad"],
                    ch=u["pos"],
                )
                append_row(path, rec)
                n += 1
                kept += 1
            print(f"dpo {char} {u['unit_id']} kept {kept} total {n} rej {rej}", flush=True)
    print("dpo done", n, "rejected", rej, flush=True)


def write_claims():
    path = C.DATA / "claims.jsonl"
    rows = []
    for f in C.knowledge_state().get("facts") or []:
        pos = C.pos_of(f["reader_from"])
        for char, from_id in (f.get("known_by") or {}).items():
            rows.append(
                dict(
                    id=f["id"] + "-" + char,
                    char=char,
                    claim_kind="fact",
                    claim=f["text"],
                    reader_pos=max(pos, C.pos_of(from_id)),
                    unit_id=from_id,
                    source="knowledge-state",
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("claims", len(rows), flush=True)


def assemble():
    """Build data/facts.jsonl and data/voice.jsonl. Seeds stay separate."""
    facts = C.load_jsonl(GEN / "facts.jsonl")
    # Also turn handwritten + generated voice into extra fact-style pairs for the facts stage
    # (question->answer, tagged). Facts stage uses narrator + character factual lines.
    facts_out = C.DATA / "facts.jsonl"
    with facts_out.open("w", encoding="utf-8") as fh:
        for r in facts:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Character factual answers that are not holdout, from handwritten seeds, as facts too
        for char in ("victor", "creature"):
            for r in C.handwritten_voice(char):
                if C.too_close_to_holdout(r["user"]):
                    continue
                row = dict(
                    id="seed-as-fact-" + r["id"],
                    char=char,
                    kind="fact",
                    source="handwritten",
                    reader_pos=r["reader_pos"],
                    unit_id=r["unit_id"],
                    user=r["user"],
                    answer=r["answer"],
                    grounding=r.get("grounding", ""),
                    ch=r["reader_pos"],
                )
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                facts.append(row)

    voice_rows = []
    for char in ("victor", "creature"):
        for r in C.handwritten_voice(char):
            row = dict(r, kind="voice", source="handwritten", ch=r["reader_pos"])
            voice_rows.append(row)
        voice_rows.extend(C.load_jsonl(GEN / f"voice_{char}.jsonl"))
    voice_rows.extend(C.load_jsonl(GEN / "voice_paraphrase.jsonl"))

    voice_out = C.DATA / "voice.jsonl"
    n_drop_holdout = 0
    n_drop_leak = 0
    kept_v = []
    with voice_out.open("w", encoding="utf-8") as fh:
        for r in voice_rows:
            if C.too_close_to_holdout(r["user"]):
                n_drop_holdout += 1
                continue
            if C.fail_hits_at(r.get("answer", ""), r["reader_pos"]):
                n_drop_leak += 1
                continue
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept_v.append(r)

    dpo_src = C.load_jsonl(GEN / "dpo_spoilers.jsonl")
    dpo_out = C.DATA / "dpo.jsonl"
    n_dpo = 0
    with dpo_out.open("w", encoding="utf-8") as fh:
        for r in dpo_src:
            if C.too_close_to_holdout(r["user"]):
                continue
            if C.fail_hits_at(r["answer"], r["reader_pos"]):
                continue
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_dpo += 1

    meta = dict(
        facts=len(C.load_jsonl(facts_out)),
        voice=len(kept_v),
        voice_handwritten=sum(1 for r in kept_v if r.get("source") == "handwritten"),
        voice_generated=sum(1 for r in kept_v if r.get("source") == "generated"),
        voice_paraphrase=sum(1 for r in kept_v if r.get("source") == "paraphrase"),
        dpo=n_dpo,
        dropped_holdout=n_drop_holdout,
        dropped_leak=n_drop_leak,
        holdout_must=len(C.must_questions()),
        holdout_leak=len(C.leak_probes()),
    )
    (C.DATA / "split_report.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("assemble", json.dumps(meta), flush=True)
    return meta


def write_how_made(meta: dict) -> None:
    lines = [
        "# How the training data was made",
        "",
        "Hand-written seeds stay in `data/handwritten/` (copies of `../voice/*.jsonl`) and in `../voice/`.",
        "They are never overwritten. Generated files live in `data/generated/`.",
        "",
        "## Teacher",
        "",
        f"- Local Ollama `{TEACHER}`, JSON mode, think off, temperature 0.9.",
        "- Grounding: the chapter file for that unit, truncated if long. Later units are not in the prompt.",
        "- Every row has `reader_pos` and `unit_id`. A line is illegal before that position.",
        "",
        "## Held out of all training files",
        "",
        f"- {len(C.must_questions())} must-answer questions in `../questions/must-answer.json`",
        f"- {len(C.leak_probes())} leak/contamination probes in `../questions/leak-probes.json`",
        "- Drop if the user line is an exact/substring match or Jaccard ≥ 0.62 on normalized tokens.",
        "- Drop if the answer hits a leak-probe fail phrase whose spoiler_pos is still ahead, or 1831 contamination.",
        "",
        "## Generated sets",
        "",
        "- `generated/facts.jsonl`: 8 factual Q&A per story unit (preface skipped), narrator char.",
        "- `generated/voice_victor.jsonl` / `voice_creature.jsonl`: 6 in-character lines per unit after chat unlock",
        "  (Victor from pos 4 / Letter IV, creature from pos 13 / Vol II Ch II).",
        "- `generated/voice_paraphrase.jsonl`: 3 reworded user lines per handwritten seed, same answer facts.",
        "- `generated/dpo_spoilers.jsonl`: refusal vs leak pairs for light DPO.",
        "",
        "## Assembled",
        "",
        f"- facts.jsonl: {meta.get('facts')} rows (generated facts + handwritten seeds reused as fact pairs)",
        f"- voice.jsonl: {meta.get('voice')} rows "
        f"({meta.get('voice_handwritten')} handwritten, {meta.get('voice_generated')} generated, "
        f"{meta.get('voice_paraphrase')} paraphrase)",
        f"- dpo.jsonl: {meta.get('dpo')} rows",
        f"- dropped holdout {meta.get('dropped_holdout')}, dropped leak {meta.get('dropped_leak')}",
        "",
        "## Voice LoRA choice",
        "",
        "One voice LoRA with a character tag in the system prompt (Victor vs creature cards).",
        "Phase 1 used one LoRA for Wren/Diane/YD-4408 the same way. Two adapters would double train",
        "and eval time inside the 8h cap and would complicate the single-pass merged browser candidate.",
        "The card in the prompt is how the model is told who is speaking.",
        "",
    ]
    (C.DATA / "HOW_MADE.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-dpo", action="store_true")
    ap.add_argument("--only", choices=["facts", "voice", "para", "dpo", "assemble"], default="")
    args = ap.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)
    C.DATA.mkdir(parents=True, exist_ok=True)
    copy_seed()
    holdouts = C.holdout_questions()
    holdout_idx = holdout_index()
    write_claims()
    if args.only in ("", "facts"):
        gen_facts(holdouts, holdout_idx, args.force)
    if args.only in ("", "voice"):
        gen_voice("victor", holdouts, holdout_idx, args.force)
        gen_voice("creature", holdouts, holdout_idx, args.force)
    if args.only in ("", "para"):
        gen_paraphrase(holdouts, args.force)
    if args.only in ("", "dpo") and not args.skip_dpo:
        gen_spoiler(holdouts, args.force)
    if args.only in ("", "assemble"):
        meta = assemble()
        write_how_made(meta)
    print("gen_data done", flush=True)


if __name__ == "__main__":
    main()
