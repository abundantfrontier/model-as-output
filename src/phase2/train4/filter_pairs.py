#!/usr/bin/env python3
"""Filter train2 rewrite pairs into a clean train3 set.

Reject a pair when the rewrite drops a protected span from pass-1, swaps a
gist word, is Walton's voice, loops, hits holdout, or is an unchanged copy.
Does not regenerate a teacher unless the kept set is too small.
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
from protect import extract_protected, missing_spans, span_ok  # noqa: E402

T2 = C.ROOT / "train2" / "data"
MIN_KEEP = 200


def walton_voice(row: dict) -> str | None:
    char = row.get("char")
    pos = int(row.get("reader_pos") or row.get("ch") or 4)
    ans = row.get("answer") or ""
    user = row.get("question") or (row.get("user") or "").split("\n\n")[0]
    src = row.get("source_text") or ""
    blob = f"{user}\n{ans}\n{src}"
    if char == "victor" and pos <= 4:
        if re.search(r"\bthe stranger\b", ans, re.I):
            return "stranger_third"
        if re.search(r"\bI (found|rescued|brought) (him|the stranger)\b", ans, re.I):
            return "walton_voice"
        if re.search(r"\bthe stranger\b", user, re.I) and re.search(
            r"\bI (thought|felt|found|rescued|saw) (him|the stranger)\b", ans, re.I
        ):
            return "walton_voice"
    if char == "victor" and re.search(r"\bmy creator\b", ans, re.I):
        return "wrong_person"
    if char == "creature" and re.search(
        r"\bI created\b|\bI made (a|the) (being|creature|monster)\b", ans, re.I
    ):
        return "wrong_person"
    if not re.search(r"\b(I|me|my|myself)\b", ans):
        return "no_first_person"
    if C.is_loop(ans) or C.is_loop(src):
        return "loop"
    if C.voice_flags(ans)["frame"]:
        return "frame"
    return None


def gist_swap(pass1: str, rewrite: str) -> str | None:
    """Catch the exact train2 failures even if extract missed them."""
    checks = [
        ("sledge", "sled"),
        ("clerval", "cliversal"),
        ("agrippa", "agrippus"),
        ("montanvert", "montanvirt"),
        ("cornelius", "cornelis"),
    ]
    p = (pass1 or "").lower()
    r = (rewrite or "").lower()
    for good, bad in checks:
        if re.search(r"\b" + good + r"\b", p) and re.search(r"\b" + bad + r"\b", r):
            if not re.search(r"\b" + good + r"\b", r):
                return f"gist_swap:{good}->{bad}"
    # cousin / father's sister must not become only "niece"
    if re.search(r"\bcousin\b", p) and not re.search(r"\bcousin\b", r):
        return "gist_swap:cousin"
    if re.search(r"father['’]s sister", p) and not re.search(r"father['’]s sister", r):
        if not re.search(r"\bcousin\b", r):
            return "gist_swap:fathers_sister"
    return None


def main() -> None:
    src_path = T2 / "rewrite_pairs.jsonl"
    rows = C.load_jsonl(src_path)
    hold = C.holdout_questions()
    kept = []
    dropped = Counter()
    examples = {k: [] for k in ("missing_span", "gist_swap", "walton", "holdout", "unchanged", "other")}
    seen = set()

    for row in rows:
        char = row.get("char")
        p1 = row.get("source_text") or ""
        rw = row.get("answer") or ""
        q = row.get("question") or (row.get("user") or "").split("\n\n")[0]
        if char not in ("victor", "creature"):
            dropped["bad_char"] += 1
            continue
        why = walton_voice(row)
        if why:
            dropped[why] += 1
            if why in ("stranger_third", "walton_voice") and len(examples["walton"]) < 5:
                examples["walton"].append((row.get("id"), why, rw[:120]))
            continue
        if C.too_close_to_holdout(q, hold):
            dropped["holdout"] += 1
            continue
        if (rw or "").strip() == (p1 or "").strip():
            dropped["unchanged"] += 1
            continue
        words = rw.split()
        if len(words) < 8 or len(words) > 180:
            dropped["length"] += 1
            continue
        swap = gist_swap(p1, rw)
        if swap:
            dropped[swap.split(":")[0]] += 1
            if len(examples["gist_swap"]) < 8:
                examples["gist_swap"].append((row.get("id"), swap, p1[:80], rw[:80]))
            continue
        ok, spans, miss = span_ok(p1, rw, char)
        if not ok:
            dropped["missing_span"] += 1
            if len(examples["missing_span"]) < 12:
                examples["missing_span"].append((row.get("id"), miss, p1[:100], rw[:100]))
            continue
        key = (char, C.norm_q(q), C.norm_q(p1[:80]))
        if key in seen:
            dropped["dup"] += 1
            continue
        seen.add(key)
        pos = int(row.get("reader_pos") or row.get("ch") or 4)
        user = prompts.rewrite_user(q, p1)
        sysm = C.sysprompt(char, pos)
        try:
            prompts.assert_no_retrieval(sysm, user)
        except Exception:
            dropped["rag_leak"] += 1
            continue
        out = dict(row)
        out["user"] = user
        out["sys"] = sysm
        out["question"] = q
        out["protected_spans"] = spans
        kept.append(out)

    outp = C.DATA / "rewrite_pairs.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    src_counts = dict(Counter(r.get("source") for r in kept))
    char_counts = dict(Counter(r["char"] for r in kept))
    n_spans = [len(r.get("protected_spans") or []) for r in kept]
    meta = dict(
        source=str(src_path),
        in_rows=len(rows),
        kept=len(kept),
        dropped=dict(dropped),
        sources=src_counts,
        chars=char_counts,
        mean_protected=round(sum(n_spans) / max(len(n_spans), 1), 2),
        max_protected=max(n_spans) if n_spans else 0,
        below_min=len(kept) < MIN_KEEP,
        min_keep=MIN_KEEP,
        examples={k: v for k, v in examples.items() if v},
    )
    (C.DATA / "filter_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in meta if k != "examples"}, indent=2), flush=True)
    for k, xs in examples.items():
        for x in xs[:3]:
            print("EX", k, x, flush=True)
    if len(kept) < MIN_KEEP:
        print("FILTER_TOO_SMALL", len(kept), "<", MIN_KEEP, flush=True)
        sys.exit(2)
    print("FILTER_OK", len(kept), flush=True)


if __name__ == "__main__":
    main()
