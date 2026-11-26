"""BM25 over chapter chunks. The browser copy in dist/app.js must match.

Scores are rounded to parts per million before ranking so Python and JS
agree. Selected hits are returned in story order, same as the locked
retriever (top k by score, then ascending chunk index).
"""
from __future__ import annotations

import math
import re

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
K1 = 1.2
B = 0.75


def tokenize(text: str) -> list[str]:
    return TOKEN.findall((text or "").lower())


def term_freq(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    return counts


def search(chunks: list[dict], query: str, k: int = 6) -> list[dict]:
    """Return up to k chunks. `chunks` must already be limited to chapters 1..N."""
    if not chunks or k < 1:
        return []
    total = len(chunks)
    avgdl = sum(int(c["n"]) for c in chunks) / total
    if avgdl <= 0:
        avgdl = 1.0
    df: dict[str, int] = {}
    for chunk in chunks:
        for tok in chunk["tf"]:
            df[tok] = df.get(tok, 0) + 1
    qtf = term_freq(query)
    scored: list[tuple[int, int]] = []
    for index, chunk in enumerate(chunks):
        raw = 0.0
        length = int(chunk["n"]) or 1
        for tok, qf in qtf.items():
            freq = int(chunk["tf"].get(tok, 0))
            if not freq:
                continue
            n_docs = df[tok]
            idf = math.log(1.0 + (total - n_docs + 0.5) / (n_docs + 0.5))
            denom = freq + K1 * (1.0 - B + B * length / avgdl)
            raw += idf * (freq * (K1 + 1.0)) / denom * qf
        scored.append((int(round(raw * 1_000_000)), index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    picked = [index for _score, index in scored[:k]]
    picked.sort()
    return [chunks[i] for i in picked]
