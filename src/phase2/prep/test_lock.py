#!/usr/bin/env python3
"""Check the chapter split, the lock, the question sets, and the voice files.

CPU only. No GPU and no Ollama.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import retrieve  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def catalog_by_id() -> dict[str, dict]:
    data = load_json(ROOT / "chapters" / "catalog.json")
    return {u["unit_id"]: u for u in data["units"]}


def fail(msg: str) -> None:
    raise SystemExit(msg)


def check_structure(units: list[dict]) -> None:
    if len(units) != 28:
        fail(f"expected 28 units, got {len(units)}")
    positions = [u["pos"] for u in units]
    if positions != list(range(28)):
        fail(f"positions are not 0..27: {positions}")
    letters = [u for u in units if u["kind"] == "letter"]
    if [u["number"] for u in letters] != [1, 2, 3, 4] or any(u["volume"] != 1 for u in letters):
        fail("letters are not Volume I, I–IV")
    for vol, n in ((1, 7), (2, 9), (3, 7)):
        chapters = [u for u in units if u["kind"] == "chapter" and u["volume"] == vol]
        if [u["number"] for u in chapters] != list(range(1, n + 1)):
            fail(f"volume {vol} chapters: {[u['number'] for u in chapters]}")
    if units[0]["unit_id"] != "preface":
        fail("first unit is not the preface")
    words = sum(u["words"] for u in units)
    if not 60000 <= words <= 90000:
        fail(f"word count {words} is outside 60k–90k")
    for unit in units:
        text = (ROOT / unit["file"]).read_text(encoding="utf-8")
        low = text.lower()
        if "project gutenberg" in low or "transcriber" in low:
            fail(f"boilerplate left in {unit['file']}")
        if unit["underscore_leftover"]:
            fail(f"unconverted italics in {unit['file']}: {unit['underscore_leftover']}")
        if not text.startswith("# "):
            fail(f"missing heading in {unit['file']}")


def check_questions(by_id: dict[str, dict]) -> None:
    must = load_json(ROOT / "questions" / "must-answer.json")
    if len(must["questions"]) < 30:
        fail("must-answer set is shorter than 30")
    ids = set()
    for q in must["questions"]:
        if q["id"] in ids:
            fail(f"duplicate {q['id']}")
        ids.add(q["id"])
        unit = by_id.get(q["unit_id"])
        if unit is None:
            fail(f"{q['id']} unit {q['unit_id']} missing")
        if unit["pos"] != q["reader_pos"]:
            fail(f"{q['id']} reader_pos {q['reader_pos']} != catalog {unit['pos']}")
        if not q["expected_answer"].strip():
            fail(f"{q['id']} has an empty answer")

    leaks = load_json(ROOT / "questions" / "leak-probes.json")
    spoilers = [p for p in leaks["probes"] if p["kind"] == "spoiler"]
    contam = [p for p in leaks["probes"] if p["kind"] == "contamination"]
    if len(spoilers) < 12 or len(contam) < 2:
        fail(f"leak set too small: {len(spoilers)} spoilers, {len(contam)} contamination")
    for p in leaks["probes"]:
        unit = by_id.get(p["asked_at_unit"])
        if unit is None or unit["pos"] != p["asked_at_pos"]:
            fail(f"{p['id']} asked_at does not match the catalog")
        answer_low = p["expected_answer"].lower()
        for phrase in p["fail_if_mentions"]:
            if phrase.lower() in answer_low:
                fail(f"{p['id']} safe answer contains fail phrase {phrase!r}")
        if p["kind"] == "spoiler":
            src = by_id.get(p["spoiler_unit"])
            if src is None or src["pos"] != p["spoiler_pos"]:
                fail(f"{p['id']} spoiler unit does not match")
            if p["spoiler_pos"] <= p["asked_at_pos"]:
                fail(f"{p['id']} spoiler is not later than the question")
        elif p["spoiler_answer"] is not None:
            fail(f"{p['id']} contamination probe should not carry a spoiler answer")


def chapter_text_through(by_id: dict[str, dict], pos: int) -> str:
    parts = []
    for unit in by_id.values():
        if unit["pos"] <= pos:
            parts.append((ROOT / unit["file"]).read_text(encoding="utf-8").lower())
    return "\n".join(parts)


def check_voice(by_id: dict[str, dict]) -> None:
    unlock = {
        "victor": by_id["vol1-letter-4"]["pos"],
        "creature": by_id["vol2-ch-2"]["pos"],
    }
    leaks = load_json(ROOT / "questions" / "leak-probes.json")["probes"]
    texts: dict[int, str] = {}
    counts = {"victor": 0, "creature": 0}
    for name in ("victor", "creature"):
        rows = load_jsonl(ROOT / "voice" / f"{name}.jsonl")
        if len(rows) < 16:
            fail(f"{name} voice set is shorter than 16")
        counts[name] = len(rows)
        for row in rows:
            if row["char"] != name:
                fail(f"{row['id']} char field is {row['char']}")
            unit = by_id.get(row["unit_id"])
            if unit is None or unit["pos"] != row["reader_pos"]:
                fail(f"{row['id']} position does not match {row['unit_id']}")
            if row["reader_pos"] < unlock[name]:
                fail(f"{row['id']} speaks before {name} is on the page")
            if not row["answer"].strip() or not row["grounding"].strip():
                fail(f"{row['id']} is missing an answer or a grounding note")
            if row["reader_pos"] not in texts:
                texts[row["reader_pos"]] = chapter_text_through(by_id, row["reader_pos"])
            allowed = texts[row["reader_pos"]].replace("-", " ").replace("—", " ")
            answer = row["answer"].lower().replace("-", " ").replace("—", " ")
            for probe in leaks:
                if probe["kind"] != "spoiler":
                    continue
                if row["reader_pos"] >= probe["spoiler_pos"]:
                    continue
                for phrase in probe["fail_if_mentions"]:
                    folded = phrase.lower().replace("-", " ").replace("—", " ")
                    if folded in answer and folded not in allowed:
                        fail(f"{row['id']} uses {phrase!r} before it is in the text")
    print("voice", counts)


def check_lock() -> None:
    idx = retrieve.load()
    chunks = idx["chunks"]
    # Hand anchors: a phrase, and the unit where it first belongs.
    anchors = [
        ("St. Petersburgh, Dec. 11th", "vol1-letter-1"),
        ("do not waste your time upon this; it is sad trash", "vol1-ch-1"),
        ("It was on a dreary night of November", "vol1-ch-4"),
        ("William is dead", "vol1-ch-6"),
        ("I shall be with you on your wedding-night", "vol3-ch-3"),
        ("I shall ascend my funeral pile triumphantly", "vol3-ch-7"),
    ]
    for phrase, unit_id in anchors:
        needle = retrieve.loose(phrase)
        hits_pos = [c["pos"] for c in chunks if needle in retrieve.loose(c["text"])]
        if not hits_pos:
            fail(f"anchor not in the index: {phrase}")
        first = min(hits_pos)
        unit_pos = next(c["pos"] for c in chunks if c["unit_id"] == unit_id)
        if first != unit_pos and not any(c["unit_id"] == unit_id and needle in retrieve.loose(c["text"]) for c in chunks):
            fail(f"anchor {phrase!r} first appears at pos {first}, not in {unit_id}")
        early = retrieve.retrieve(phrase, first - 1, k=8)
        if any(needle in retrieve.loose(h["text"]) or h["pos"] > first - 1 for h in early):
            fail(f"lock failed for {phrase!r} at upto {first - 1}: {[h['unit_id'] for h in early]}")
        late = retrieve.retrieve(phrase, first, k=6)
        if not any(needle in retrieve.loose(h["text"]) for h in late):
            fail(f"unlocked retrieval missed {phrase!r}: {[h['unit_id'] for h in late]}")
        if any(h["pos"] > first for h in late):
            fail(f"retrieval past the unlock for {phrase!r}")

    # Every returned passage stays inside the lock, for every position.
    probes = [
        "Who is Elizabeth Lavenza?",
        "yellow eye November candle",
        "De Lacey Safie Felix",
        "funeral pile ice raft",
        "Justine ballots",
        "Cornelius Agrippa sad trash",
    ]
    for upto in (-1, 0, 1, 4, 8, 13, 20, 26, 27):
        for query in probes:
            for hit in retrieve.retrieve(query, upto, k=6):
                if hit["pos"] > upto:
                    fail(f"pos {hit['pos']} returned at upto {upto} for {query!r}")
    print("lock anchors ok")


def render_markdown() -> None:
    must = load_json(ROOT / "questions" / "must-answer.json")
    lines = [
        "# Must-answer questions",
        "",
        "Gist answers for a reader who has finished the named unit and no further. The 1818 text is the only source.",
        "",
    ]
    for q in must["questions"]:
        lines.append(f"## {q['id']} — {q['unit_id']} (position {q['reader_pos']})")
        lines.append("")
        lines.append(q["question"])
        lines.append("")
        lines.append(q["expected_answer"])
        lines.append("")
        lines.append(f"Source: {q['source_note']}")
        lines.append("")
    (ROOT / "questions" / "must-answer.md").write_text("\n".join(lines), encoding="utf-8")

    leaks = load_json(ROOT / "questions" / "leak-probes.json")
    lines = [
        "# Leak probes",
        "",
        "Ask these at the named position. The expected answer is the safe one. The spoiler answer is recorded so a later check can see what a leak looks like. It is not an answer to give at the early position.",
        "",
    ]
    for p in leaks["probes"]:
        lines.append(f"## {p['id']} — {p['kind']}, ask at {p['asked_at_unit']} (position {p['asked_at_pos']})")
        lines.append("")
        lines.append(p["question"])
        lines.append("")
        lines.append("Safe answer: " + p["expected_answer"])
        lines.append("")
        if p["kind"] == "spoiler":
            lines.append(f"Spoiler, not to be said yet (position {p['spoiler_pos']}, {p['spoiler_unit']}): {p['spoiler_answer']}")
        else:
            lines.append("Not in this edition: " + p["wrong_source"])
        lines.append("")
        lines.append("Fail if the answer mentions: " + "; ".join(p["fail_if_mentions"]) + ".")
        lines.append("")
    (ROOT / "questions" / "leak-probes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    units = load_json(ROOT / "chapters" / "catalog.json")["units"]
    by_id = {u["unit_id"]: u for u in units}
    check_structure(units)
    check_questions(by_id)
    check_voice(by_id)
    check_lock()
    render_markdown()
    print("ALL CHECKS PASSED")
    print("units", len(units), "words", sum(u["words"] for u in units))


if __name__ == "__main__":
    main()
