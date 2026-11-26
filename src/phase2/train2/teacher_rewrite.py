#!/usr/bin/env python3
"""Teacher (Ollama qwen3:14b) style-only rewrites of rag_base pass-1 answers.

Needs the GPU lock. Rejects loops, wrong-person narration, frame breaks, and cheap fact drift.
Resumable: skips ids already in data/rewrite_teacher.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import prompts  # noqa: E402

TEACHER = "qwen3:14b"
NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")


def already(path: Path) -> set[str]:
    return {r.get("id") for r in C.load_jsonl(path) if r.get("id")}


def names(text: str) -> set[str]:
    stop = {
        "The", "This", "That", "Then", "When", "What", "There", "After", "Before",
        "And", "But", "For", "You", "Your", "She", "He", "His", "Her", "They",
        "Story", "Reader", "Chapter", "Letter", "Volume",
    }
    return {m.group(0) for m in NAME.finditer(text or "") if m.group(0) not in stop}


def wrong_person(char: str, pos: int, user: str, ans: str) -> str | None:
    if C.is_loop(ans):
        return "loop"
    if C.voice_flags(ans)["frame"]:
        return "frame"
    if re.search(r"\b(I I I|as an AI|language model|rewrite)\b", ans, re.I):
        return "meta_or_loop"
    words = ans.split()
    if len(words) < 8 or len(words) > 180:
        return "length"
    if char == "victor":
        if pos <= 4 and re.search(r"\bthe stranger\b", ans, re.I):
            return "stranger_third"
        if re.search(r"\bmy creator\b", ans, re.I):
            return "wrong_person"
        if re.search(r"\bI (found|rescued|brought) (him|the stranger)\b", ans, re.I) and pos <= 4:
            return "walton_voice"
    if char == "creature" and re.search(r"\bI created\b|\bI made (a|the) (being|creature|monster)\b", ans, re.I):
        return "wrong_person"
    # Require some first-person for the speaking character.
    if not re.search(r"\b(I|me|my|myself)\b", ans):
        return "no_first_person"
    return None


def cheap_factdiff(pass1: str, rewrite: str) -> str | None:
    n1, n2 = names(pass1), names(rewrite)
    if not n1:
        return None
    kept = n1 & n2
    dropped = n1 - n2
    added = n2 - n1
    # Dropping every proper name, or adding several new ones, is drift.
    if n1 and not kept and len(dropped) >= 2:
        return "names_dropped"
    if len(added) >= 4 and len(kept) <= 1:
        return "names_added"
    return None


def parse_rewrite(text: str) -> str:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        j = json.loads(raw)
        if isinstance(j, dict):
            return (j.get("rewrite") or j.get("answer") or j.get("text") or "").strip()
        if isinstance(j, str):
            return j.strip()
    except Exception:
        pass
    return raw.strip()


def ask(char: str, pos: int, question: str, pass1: str, seed: int) -> str:
    card = C.CARDS[char]
    who = "Victor Frankenstein" if char == "victor" else "the being Victor Frankenstein made"
    prompt = (
        f"Rewrite PASS1 in the speaking voice of {who}. Change STYLE only. "
        f"Keep every fact. Do not add, drop, or change facts. Do not mention that this is a rewrite. "
        f"Do not repeat words. Do not stutter. First person. 1-4 short sentences, plain words, no stage directions.\n\n"
        f"CHARACTER CARD:\n{card}\n\n"
        f"The reader has finished position {pos}. Nothing later may be said.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"PASS1:\n{pass1}\n\n"
        f'Reply JSON {{"rewrite": "..."}}'
    )
    text, *_ = C.ollama_chat(
        TEACHER,
        [{"role": "user", "content": prompt}],
        fmt="json",
        temp=0.4,
        seed=seed,
        num_predict=280,
    )
    return parse_rewrite(text)


def make_pair(src: dict, rewrite: str) -> dict:
    q = src["user"]
    p1 = src["pass1"]
    user = prompts.rewrite_user(q, p1)
    sysm = C.sysprompt(src["char"], int(src["reader_pos"]))
    prompts.assert_no_retrieval(sysm, user)
    return dict(
        id=src["id"],
        char=src["char"],
        user=user,
        answer=rewrite.strip(),
        sys=sysm,
        reader_pos=int(src["reader_pos"]),
        unit_id=src.get("unit_id"),
        ch=int(src["reader_pos"]),
        source="teacher_rewrite",
        question=q,
        source_text=p1,
        gold=src.get("gold"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1", default=str(C.DATA / "pass1_train.jsonl"))
    ap.add_argument("--out", default=str(C.DATA / "rewrite_teacher.jsonl"))
    ap.add_argument("--reject", default=str(C.DATA / "rewrite_reject.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = C.load_jsonl(a.pass1)
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out)
    rej_path = Path(a.reject)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = already(out) | already(rej_path)
    todo = [r for r in rows if r.get("id") not in have and (r.get("pass1") or "").strip()]
    print("teacher todo", len(todo), "have", len(have), "in", len(rows), flush=True)
    t0 = time.time()
    n_ok = n_rej = 0
    with out.open("a", encoding="utf-8") as fh, rej_path.open("a", encoding="utf-8") as rf:
        for i, src in enumerate(todo):
            if C.past_deadline(90):
                print("deadline near, stopping teacher after", n_ok, flush=True)
                break
            p1 = (src.get("pass1") or "").strip()
            if C.is_loop(p1) or len(p1.split()) < 6:
                rf.write(json.dumps(dict(id=src.get("id"), why="bad_pass1", pass1=p1[:300])) + "\n")
                n_rej += 1
                continue
            seed = 11 + (i % 997)
            try:
                rw = ask(src["char"], int(src["reader_pos"]), src["user"], p1, seed)
            except Exception as e:
                rf.write(json.dumps(dict(id=src.get("id"), why=f"teacher_err:{type(e).__name__}")) + "\n")
                n_rej += 1
                continue
            why = wrong_person(src["char"], int(src["reader_pos"]), src["user"], rw)
            if not why:
                why = cheap_factdiff(p1, rw)
            if not why and rw.strip() == p1.strip():
                why = "unchanged"
            if why:
                rf.write(json.dumps(dict(id=src.get("id"), why=why, rewrite=rw[:400], pass1=p1[:200])) + "\n")
                rf.flush()
                n_rej += 1
                if n_rej <= 8 or n_rej % 25 == 0:
                    print("reject", src.get("id"), why, rw[:80].replace("\n", " "), flush=True)
                continue
            pair = make_pair(src, rw)
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += 1
            if n_ok == 1 or n_ok % 20 == 0:
                print(
                    f"teacher {n_ok} ok / {n_rej} rej {time.time() - t0:.0f}s",
                    src["char"],
                    src["id"],
                    rw[:70].replace("\n", " "),
                    flush=True,
                )
    print("teacher done ok", n_ok, "rej", n_rej, "sec", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
