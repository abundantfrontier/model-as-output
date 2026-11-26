#!/usr/bin/env python3
"""Write the chapter-locked static index and Wren's prompt prefixes.

Each chapter is its own file. A page at chapter N fetches chapters 1..N
and prompt/wren-N.json only. Later files stay on disk and are not requested.

The prompt prefix is the locked pass-1 system text (card, style, guard,
day, knowledge, beliefs) with the excerpt block left off. Knowledge comes
from exp10/common.py, which already drops facts the reader has not reached.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXP10 = Path("$HOME/output-model/exp10")
READER = Path("$HOME/output-model/reader")
sys.path.insert(0, str(EXP10))
sys.path.insert(0, str(READER))

import common as C  # noqa: E402
import prompting  # noqa: E402

from bm25 import search, term_freq  # noqa: E402

MARKER = "\n\nStory so far (excerpts):\n"
K = 6


def title_of(text: str, n: int) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return f"Chapter {n}"


def prefix_for(chapter: int) -> str:
    full = prompting.assemble_pass1("wren", chapter, "")
    if not full.endswith(MARKER):
        raise SystemExit(f"chapter {chapter} prompt did not end with the excerpt marker")
    return full[: -len(MARKER)]


def main() -> None:
    index_dir = DIST / "index"
    prompt_dir = DIST / "prompt"
    index_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    chapters = []
    all_chunks: dict[int, list[dict]] = {}
    nums = []
    data = Path(C.D) / "chapters"
    for path in sorted(data.glob("ch*.md")):
        stem = path.stem[2:]
        if stem.isdigit():
            nums.append(int(stem))
    for n in nums:
        text = C.chapter_text(n)
        chunks = []
        for j, piece in enumerate(C.chunks(text)):
            tf = term_freq(piece)
            chunks.append({
                "id": f"ch{n}-c{j}",
                "chapter": n,
                "text": piece,
                "n": sum(tf.values()),
                "tf": tf,
            })
        all_chunks[n] = chunks
        payload = {"chapter": n, "title": title_of(text, n), "chunks": chunks}
        (index_dir / f"ch{n}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        pre = prefix_for(n)
        (prompt_dir / f"wren-{n}.json").write_text(
            json.dumps({"chapter": n, "prefix": pre}, ensure_ascii=False),
            encoding="utf-8",
        )
        chapters.append({"n": n, "title": payload["title"], "chunks": len(chunks), "file": f"index/ch{n}.json"})
        hits = C.fail_hits(pre)
        if hits:
            raise SystemExit(f"wren-{n} prompt contains a leak marker: {hits}")

    manifest = {
        "story": "No Return",
        "character": "wren",
        "k": K,
        "chunk_words": 120,
        "chapters": chapters,
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Fixed query so the page can check its BM25 against this file.
    query = "What did Diane say about the air today?"
    allowed = []
    for n in range(1, 4):
        allowed.extend(all_chunks[n])
    hits = search(allowed, query, K)
    expect = {
        "chapter": 3,
        "query": query,
        "k": K,
        "ids": [h["id"] for h in hits],
    }
    (index_dir / "selftest.json").write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chapters": chapters, "selftest_ids": expect["ids"]}, indent=2))


if __name__ == "__main__":
    main()
