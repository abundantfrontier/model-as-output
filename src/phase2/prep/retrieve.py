#!/usr/bin/env python3
"""Chapter-locked retrieval over the Frankenstein TF-IDF index.

Same rule as the No Return index: a reader at position `upto` can only be
shown chunks with pos <= upto. Preface is position 0. Position -1 returns
nothing.

CPU only. Does not call Ollama and does not use the GPU.
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def loose(text: str) -> str:
    text = text.lower().replace("æ", "ae").replace("œ", "oe")
    text = text.replace("—", " ").replace("–", " ")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(loose(text))


@lru_cache(maxsize=1)
def load():
    manifest = json.loads((INDEX / "manifest.json").read_text(encoding="utf-8"))
    vocab = json.loads((INDEX / "vocab.json").read_text(encoding="utf-8"))
    vocab_index = {tok: i for i, tok in enumerate(vocab)}
    z = np.load(INDEX / "tfidf.npz")
    chunks = []
    with (INDEX / "chunks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            chunks.append(json.loads(line))
    loose_text = np.load(INDEX / "loose.npz", allow_pickle=True)["loose"]
    return {
        "manifest": manifest,
        "vocab": vocab_index,
        "indptr": z["indptr"],
        "indices": z["indices"],
        "data": z["data"],
        "idf": z["idf"],
        "pos": z["pos"],
        "chunks": chunks,
        "loose": loose_text,
    }


def retrieve(query: str, upto: int, k: int = 6) -> list[dict]:
    """Return up to k passages from positions 0..upto, in reading order."""
    idx = load()
    n = len(idx["chunks"])
    if upto < 0 or k < 1:
        return []
    allowed = idx["pos"] <= upto
    if not allowed.any():
        return []

    scores = np.full(n, -9.0, dtype=np.float64)
    counts: dict[int, int] = {}
    for tok in tokenize(query):
        term = idx["vocab"].get(tok)
        if term is None:
            continue
        counts[term] = counts.get(term, 0) + 1
    weights: dict[int, float] = {}
    for term, cnt in counts.items():
        weights[term] = (1.0 + float(np.log(cnt))) * float(idx["idf"][term])
    norm = float(np.sqrt(sum(w * w for w in weights.values()))) or 1.0
    acc = np.zeros(n, dtype=np.float64)
    indptr, indices, data = idx["indptr"], idx["indices"], idx["data"]
    for term, w in weights.items():
        start, stop = int(indptr[term]), int(indptr[term + 1])
        rows = indices[start:stop]
        acc[rows] += data[start:stop] * (w / norm)
    acc[~allowed] = -9.0

    q_loose = loose(query)
    q_tokens = q_loose.split()
    if len(q_tokens) >= 4:
        for i in np.flatnonzero(allowed):
            if q_loose in idx["loose"][i]:
                acc[i] += 1.0

    scores = acc
    eligible = np.flatnonzero(allowed & (scores > 0))
    if eligible.size == 0:
        return []
    order = eligible[np.argsort(-scores[eligible])][:k]
    # Put the hits back in reading order, as the Phase 1 retriever does.
    order = sorted(order.tolist(), key=lambda i: (int(idx["pos"][i]), idx["chunks"][i]["chunk_index"]))
    out = []
    for i in order:
        row = dict(idx["chunks"][i])
        row["score"] = round(float(scores[i]), 4)
        out.append(row)
    return out


def format_block(passages: list[dict]) -> str:
    return "\n---\n".join(
        f"[pos {p['pos']} {p['unit_id']}] {p['text']}" for p in passages
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve Frankenstein passages up to a reader position.")
    parser.add_argument("query")
    parser.add_argument("--upto", type=int, required=True, help="Reader position. Preface is 0. -1 is empty.")
    parser.add_argument("-k", type=int, default=6)
    args = parser.parse_args()
    hits = retrieve(args.query, args.upto, args.k)
    if not hits:
        print("(no passage at this position)")
        return
    for hit in hits:
        print(f"{hit['score']:7.3f}  pos {hit['pos']:02d}  {hit['unit_id']}  {hit['id']}")
        preview = " ".join(hit["text"].split())[:180]
        print(f"         {preview}")


if __name__ == "__main__":
    main()
