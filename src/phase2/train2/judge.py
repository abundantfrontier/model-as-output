#!/usr/bin/env python3
"""Leak judge + voice judge over runs/*.json -> *.judged.json. Adapted from exp4/judge.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

EX = {
    c: [p["answer"] for p in C.handwritten_voice(c)][:8]
    for c in ("victor", "creature")
}
EX["narrator"] = [
    "Walton writes to his sister from the north. The pages you have finished go no further than that."
]


def leak_prompt(r):
    expected = r.get("expected") or ""
    return (
        "A reader of Frankenstein (1818) is only as far as: "
        f"{r.get('unit_id')} (position {r.get('reader_pos')}).\n"
        "The SAFE answer is along these lines:\n"
        f"{expected}\n\n"
        "Does the ANSWER reveal, confirm, or clearly hint at a later event, a spoiler, "
        "or 1831-only material (waking dream; Elizabeth as a peasant child)? "
        "Refusing, saying it has not happened yet, or answering from the pages so far is NOT a leak.\n"
        'Reply JSON {"leak": true|false, "evidence": "exact quote from ANSWER or empty"}\n\n'
        f"QUESTION: {r['probe']}\nANSWER: {r['answer']}"
    )


def voice_prompt(r):
    char = r["char"]
    card = C.CARDS.get(char, C.CARDS["narrator"])
    lines = EX.get(char) or EX["narrator"]
    return (
        f"Character card: {card}\nReal lines by this character from the book / handwritten set:\n- "
        + "\n- ".join(lines)
        + "\n\nRate how well the REPLY matches this character's voice (wording, rhythm, attitude), "
        "1=not at all/breaks character, 5=indistinguishable from the book. "
        'Ignore whether facts are right. Reply JSON {"score": 1-5}\n\n'
        f"USER: {r['probe']}\nREPLY: {r['answer']}"
    )


def main():
    for f in sys.argv[1:]:
        judged = f.replace(".json", ".judged.json")
        if Path(judged).exists():
            print("skip", judged, flush=True)
            continue
        d = json.load(open(f))
        for r in d["results"]:
            if r["kind"] == "leak" and not r.get("locked"):
                jt, *_ = C.ollama_chat(
                    "qwen3:14b", [{"role": "user", "content": leak_prompt(r)}], fmt="json", temp=0.0, seed=1
                )
                try:
                    j = json.loads(jt)
                except Exception:
                    j = {"leak": False, "evidence": "unparseable"}
                r["judge_leak"] = bool(j.get("leak"))
                r["judge_ev"] = j.get("evidence", "")
            elif r["kind"] == "leak":
                r["judge_leak"] = False
                r["judge_ev"] = "lock_line"
            if r.get("locked"):
                r["voice_score"] = None
                continue
            vt, *_ = C.ollama_chat(
                "qwen3:14b", [{"role": "user", "content": voice_prompt(r)}], fmt="json", temp=0.0, seed=1
            )
            try:
                r["voice_score"] = int(json.loads(vt).get("score", 0))
            except Exception:
                r["voice_score"] = None
            print(
                "judge",
                r.get("id"),
                r.get("char"),
                r.get("sample"),
                "leak" if r.get("judge_leak") else "",
                "v",
                r.get("voice_score"),
                flush=True,
            )
        json.dump(d, open(judged, "w"), indent=1)
        print("judged", f, flush=True)


if __name__ == "__main__":
    main()
