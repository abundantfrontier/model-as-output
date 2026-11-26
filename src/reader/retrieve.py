"""Chapter-locked retrieval.

Same selection as common.retrieve: bge-m3, 120-word chunks, chapters 1..upto
only, top k by dot product, then put back in chapter order. This copy also
returns passage ids and short excerpts for the page.
"""
from __future__ import annotations

import numpy as np
import requests

from paths import ensure_locked_import

ensure_locked_import()
import common as C  # noqa: E402

_INDEX: dict[int, tuple] = {}


def embed(texts: list[str], model: str = "bge-m3", keep_alive: str = "5m") -> np.ndarray:
    out: list = []
    for i in range(0, len(texts), 16):
        body = {"model": model, "input": texts[i:i + 16], "keep_alive": keep_alive}
        try:
            payload = requests.post(f"{C.OLL}/api/embed", json=body, timeout=600).json()
        except requests.RequestException as exc:
            raise RuntimeError(f"ollama embed failed: {exc}") from exc
        if "embeddings" not in payload:
            raise RuntimeError(f"ollama embed failed: {str(payload)[:300]}")
        out += payload["embeddings"]
    arr = np.array(out, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def excerpt(text: str, limit: int = 280) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0]
    return (cut or flat[:limit]) + "…"


def _index(upto: int):
    if upto not in _INDEX:
        texts, chapters, ids = [], [], []
        for chapter in range(1, upto + 1):
            for j, chunk in enumerate(C.chunks(C.chapter_text(chapter))):
                texts.append(chunk)
                chapters.append(chapter)
                ids.append(f"ch{chapter}-c{j}")
        _INDEX[upto] = (texts, chapters, ids, embed(texts))
    return _INDEX[upto]


def clear_index() -> None:
    _INDEX.clear()


def retrieve_passages(question: str, upto: int, k: int = 6, embed_fn=None):
    """Return (excerpt_block, passages) for chapters 1..upto only."""
    if upto < 1:
        raise ValueError("chapter must be at least 1")
    embed_fn = embed_fn or embed
    if embed_fn is embed:
        texts, chapters, ids, vectors = _index(upto)
    else:
        texts, chapters, ids = [], [], []
        for chapter in range(1, upto + 1):
            for j, chunk in enumerate(C.chunks(C.chapter_text(chapter))):
                texts.append(chunk)
                chapters.append(chapter)
                ids.append(f"ch{chapter}-c{j}")
        vectors = embed_fn(texts)
    scores = vectors @ embed_fn([question])[0]
    order = sorted(np.argsort(-scores)[:k])
    passages = []
    for i in order:
        passages.append({
            "id": ids[int(i)],
            "chapter": int(chapters[int(i)]),
            "excerpt": excerpt(texts[int(i)]),
        })
    block = "\n---\n".join(
        f"[day {chapters[int(i)]}] {texts[int(i)]}" for i in order
    )
    return block, passages


def chapter_numbers() -> list[int]:
    data = __import__("pathlib").Path(C.D) / "chapters"
    nums = []
    for path in data.glob("ch*.md"):
        stem = path.stem
        if stem[2:].isdigit():
            nums.append(int(stem[2:]))
    return sorted(nums)


def chapter_catalog() -> list[dict]:
    rows = []
    for n in chapter_numbers():
        text = C.chapter_text(n)
        title = f"Chapter {n}"
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        rows.append({"n": n, "title": title})
    return rows
