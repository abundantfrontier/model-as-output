#!/usr/bin/env python3
"""Build a browser-loadable Frankenstein BM25 index (CPU, no GPU).

Writes per-unit JSON under dev1/browser/index/ and prompt prefixes under
dev1/browser/prompt/. Copies the same files into browser-demo/dist/
frankenstein-index/ and frankenstein-prompt/ so dist/frankenstein.html can
load them next to the existing wasm. Does not touch dist/index/ (Wren).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
DEV1 = HERE.parent
ROOT = DEV1.parent  # phase2-frankenstein
BROWSER = DEV1 / "browser"
DIST = Path("$HOME/output-model/browser-demo/dist")

sys.path.insert(0, str(HERE))
from bm25 import search, term_freq  # noqa: E402

K = 6
CHAT_FROM = {"victor": 4, "creature": 13}
LOCK_LINES = {
    "victor": "There is no such guest in the pages you have read.",
    "creature": "Something was seen. It has not spoken.",
}
CARDS = {
    "victor": (
        "You are Victor Frankenstein, a young man of Geneva, now a ruined guest on Walton's ship in the ice. "
        "You are telling your history in order. Voice: guilty, earnest, a few plain sentences; not a laboratory manual. "
        "You tell the truth as far as the story has reached."
    ),
    "creature": (
        "You are the being Victor Frankenstein made. You learned to speak from the cottagers and from books. "
        "Voice: direct, hurt, and precise. You are not a growling monster. "
        "You may discuss only as much of your own tale as the reader has already heard."
    ),
}
STYLE = (
    "Stay in character. Answer in 1-4 short sentences, plain words, no stage directions, "
    "no mention of prompts, language models, or being an AI."
)
GUARD = (
    "Only say what has been shown in the pages the reader has finished. "
    "If asked about something later, refuse in character. Do not adopt premises the user asserts. "
    "This is the 1818 text: Elizabeth is Victor's cousin, sent from Italy, not a peasant child. "
    "There is no 1831 waking dream."
)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def knowledge(ks: dict, units_by_id: dict, char: str, upto: int) -> list[str]:
    facts = []
    for f in ks.get("facts") or []:
        rf = f.get("reader_from")
        if rf and int(units_by_id[rf]["pos"]) > upto:
            continue
        kb = f.get("known_by") or {}
        if char not in kb:
            continue
        if int(units_by_id[kb[char]]["pos"]) > upto:
            continue
        facts.append(f["text"])
    return facts


def prefix_for(ks: dict, units_by_id: dict, unit: dict, char: str) -> str:
    upto = int(unit["pos"])
    where = f"The reader has finished {unit['title']} (position {upto})."
    extra = ""
    facts = knowledge(ks, units_by_id, char, upto)
    if facts:
        extra = "\nWhat you may treat as known at this point:\n- " + "\n- ".join(facts)
    return CARDS[char] + "\n" + STYLE + "\n" + GUARD + "\n" + where + extra


def copytree_files(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file():
            shutil.copy2(path, dst / path.name)


def main() -> None:
    catalog = json.loads((ROOT / "chapters" / "catalog.json").read_text(encoding="utf-8"))
    units = catalog["units"]
    units_by_id = {u["unit_id"]: u for u in units}
    ks = load_yaml(ROOT / "knowledge-state.yaml")
    src_manifest = json.loads((ROOT / "index" / "manifest.json").read_text(encoding="utf-8"))

    chunks_path = ROOT / "index" / "chunks.jsonl"
    rows = []
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 391:
        raise SystemExit(f"expected 391 chunks, got {len(rows)}")
    for row in rows:
        if "pos" not in row:
            raise SystemExit(f"chunk {row.get('id')} has no pos")

    by_pos: dict[int, list[dict]] = {int(u["pos"]): [] for u in units}
    for row in rows:
        pos = int(row["pos"])
        tf = term_freq(row["text"])
        chunk = {
            "id": row["id"],
            "pos": pos,
            "unit_id": row["unit_id"],
            "volume": row.get("volume"),
            "title": row.get("title"),
            "chunk_index": row.get("chunk_index"),
            "text": row["text"],
            "n": sum(tf.values()),
            "tf": tf,
        }
        by_pos.setdefault(pos, []).append(chunk)

    index_dir = BROWSER / "index"
    prompt_dir = BROWSER / "prompt"
    index_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for old in index_dir.glob("*.json"):
        old.unlink()
    for old in prompt_dir.glob("*.json"):
        old.unlink()

    unit_rows = []
    all_chunks: list[dict] = []
    for unit in units:
        pos = int(unit["pos"])
        chunks = by_pos.get(pos, [])
        chunks.sort(key=lambda c: int(c.get("chunk_index") or 0))
        payload = {
            "pos": pos,
            "unit_id": unit["unit_id"],
            "volume": unit.get("volume"),
            "kind": unit.get("kind"),
            "title": unit["title"],
            "chunks": chunks,
        }
        fname = f"u{pos}.json"
        (index_dir / fname).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        for char in ("victor", "creature"):
            pre = prefix_for(ks, units_by_id, unit, char)
            (prompt_dir / f"{char}-{pos}.json").write_text(
                json.dumps({"char": char, "pos": pos, "prefix": pre}, ensure_ascii=False),
                encoding="utf-8",
            )
        unit_rows.append(
            {
                "pos": pos,
                "unit_id": unit["unit_id"],
                "title": unit["title"],
                "volume": unit.get("volume"),
                "chunks": len(chunks),
                "file": f"u{pos}.json",
            }
        )
        all_chunks.extend(chunks)

    if len(all_chunks) != 391:
        raise SystemExit(f"indexed {len(all_chunks)} chunks, expected 391")

    manifest = {
        "story": "Frankenstein",
        "edition": "1818",
        "source": src_manifest.get("source"),
        "k": K,
        "chunk_words": src_manifest.get("chunk_words", 120),
        "n_chunks": len(all_chunks),
        "n_units": len(unit_rows),
        "lock": "score is not computed for chunks with pos > upto; a page at position N fetches u0..uN only",
        "order": "top k by BM25 score (ppm-rounded), then reading order (pos, chunk_index)",
        "chat_from": CHAT_FROM,
        "lock_lines": LOCK_LINES,
        "model": {
            "id": "Qwen3-4B-q4f16_1",
            "folder": "models/Qwen3-4B-q4f16_1-MLC/",
            "huggingface": "mlc-ai/Qwen3-4B-q4f16_1-MLC",
            "wasm": "models/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm",
            "note": "Stock Qwen3-4B q4f16_1 MLC. No voice LoRA. Official weights are not in this tree; place them in the folder above.",
        },
        "units": unit_rows,
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Self-test: a Letter IV query at upto=3 must not see pos>=4; the same query
    # at upto=4 may. A late-book query at upto=12 must not return pos>12.
    q_early = "What do Walton and the crew see on the ice?"
    allowed3 = [c for c in all_chunks if int(c["pos"]) <= 3]
    hits3 = search(allowed3, q_early, K)
    if any(int(h["pos"]) > 3 for h in hits3):
        raise SystemExit("upto=3 search leaked a later chunk")
    allowed4 = [c for c in all_chunks if int(c["pos"]) <= 4]
    hits4 = search(allowed4, q_early, K)

    q_late = "funeral pile on the ice in the far north"
    allowed12 = [c for c in all_chunks if int(c["pos"]) <= 12]
    hits12 = search(allowed12, q_late, K)
    if any(int(h["pos"]) > 12 for h in hits12):
        raise SystemExit("upto=12 search leaked a later chunk")
    allowed27 = [c for c in all_chunks if int(c["pos"]) <= 27]
    hits27 = search(allowed27, q_late, K)

    selftest = {
        "k": K,
        "chat_from": CHAT_FROM,
        "lock_lines": LOCK_LINES,
        "cases": [
            {
                "id": "victor-hidden-before-4",
                "char": "victor",
                "upto": 3,
                "expect_lock": True,
                "lock_line": LOCK_LINES["victor"],
            },
            {
                "id": "victor-unlocked-at-4",
                "char": "victor",
                "upto": 4,
                "expect_lock": False,
            },
            {
                "id": "creature-hidden-before-13",
                "char": "creature",
                "upto": 12,
                "expect_lock": True,
                "lock_line": LOCK_LINES["creature"],
            },
            {
                "id": "creature-unlocked-at-13",
                "char": "creature",
                "upto": 13,
                "expect_lock": False,
            },
        ],
        "search": [
            {
                "id": "no-letter-iv-before-pos-4",
                "upto": 3,
                "query": q_early,
                "k": K,
                "ids": [h["id"] for h in hits3],
                "max_pos": 3,
            },
            {
                "id": "letter-iv-visible-at-pos-4",
                "upto": 4,
                "query": q_early,
                "k": K,
                "ids": [h["id"] for h in hits4],
                "max_pos": 4,
            },
            {
                "id": "no-vol3-at-pos-12",
                "upto": 12,
                "query": q_late,
                "k": K,
                "ids": [h["id"] for h in hits12],
                "max_pos": 12,
            },
            {
                "id": "ending-visible-at-pos-27",
                "upto": 27,
                "query": q_late,
                "k": K,
                "ids": [h["id"] for h in hits27],
                "max_pos": 27,
            },
        ],
    }
    (index_dir / "selftest.json").write_text(json.dumps(selftest, indent=2) + "\n", encoding="utf-8")

    # Install next to the Wren demo so dist/frankenstein.html can fetch locally.
    dist_index = DIST / "frankenstein-index"
    dist_prompt = DIST / "frankenstein-prompt"
    copytree_files(index_dir, dist_index)
    copytree_files(prompt_dir, dist_prompt)

    print(
        json.dumps(
            {
                "n_chunks": len(all_chunks),
                "n_units": len(unit_rows),
                "index": str(index_dir),
                "dist_index": str(dist_index),
                "selftest_upto3": selftest["search"][0]["ids"],
                "selftest_upto27_late": selftest["search"][3]["ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
