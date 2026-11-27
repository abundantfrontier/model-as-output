#!/usr/bin/env python3
"""Chapter-locked retrieval: bge-m3 primary, TF-IDF fallback.

A reader at position `upto` only sees chunks with pos <= upto.
Preface is 0. Position -1 returns nothing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
TRAIN = Path(__file__).resolve().parents[1]
PARENT_INDEX = ROOT / "index"
TRAIN_INDEX = TRAIN / "index"
OLL = "http://127.0.0.1:11434"
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def loose(text: str) -> str:
    text = text.lower().replace("æ", "ae").replace("œ", "oe")
    text = text.replace("—", " ").replace("–", " ")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(loose(text))


def _index_dir() -> Path:
    if (TRAIN_INDEX / "bge.npy").is_file():
        return TRAIN_INDEX
    return PARENT_INDEX


@lru_cache(maxsize=1)
def load_chunks():
    path = PARENT_INDEX / "chunks.jsonl"
    chunks = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


@lru_cache(maxsize=1)
def load_bge():
    d = _index_dir()
    npy = d / "bge.npy"
    if not npy.is_file():
        npy = PARENT_INDEX / "bge.npy"
    if not npy.is_file():
        return None
    vecs = np.load(npy)
    chunks = load_chunks()
    if vecs.shape[0] != len(chunks):
        raise RuntimeError(f"bge rows {vecs.shape[0]} != chunks {len(chunks)}")
    pos = np.array([int(c["pos"]) for c in chunks], dtype=np.int32)
    return {"vecs": vecs, "chunks": chunks, "pos": pos}


@lru_cache(maxsize=1)
def load_tfidf():
    import importlib.util

    path = ROOT / "scripts" / "retrieve.py"
    spec = importlib.util.spec_from_file_location("frank_tfidf_retrieve", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def embed_query(text: str, model: str = "bge-m3") -> np.ndarray:
    r = requests.post(
        f"{OLL}/api/embed",
        json={"model": model, "input": [text]},
        timeout=600,
    )
    r.raise_for_status()
    v = np.array(r.json()["embeddings"][0], dtype=np.float32)
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


@lru_cache(maxsize=1)
def load_loose():
    chunks = load_chunks()
    return [loose(c["text"]) for c in chunks]


def retrieve_bge(query: str, upto: int, k: int = 6) -> list[dict]:
    idx = load_bge()
    if idx is None:
        raise RuntimeError("bge index missing; run scripts/embed.py")
    if upto < 0 or k < 1:
        return []
    allowed = idx["pos"] <= upto
    if not allowed.any():
        return []
    qv = embed_query(query)
    scores = idx["vecs"] @ qv
    scores = np.where(allowed, scores, -9.0)
    # Same phrase bonus as the TF-IDF index: an exact 4+ token hit ranks first.
    q_loose = loose(query)
    if len(q_loose.split()) >= 4:
        loose_text = load_loose()
        for i in np.flatnonzero(allowed):
            if q_loose in loose_text[int(i)]:
                scores[i] += 1.0
    order = np.argsort(-scores)
    picked = []
    for i in order:
        if scores[i] <= -8.0:
            break
        picked.append(int(i))
        if len(picked) >= k:
            break
    picked.sort(key=lambda i: (int(idx["pos"][i]), idx["chunks"][i]["chunk_index"]))
    out = []
    for i in picked:
        row = dict(idx["chunks"][i])
        row["score"] = round(float(scores[i]), 4)
        row["backend"] = "bge-m3"
        out.append(row)
    return out


def retrieve_tfidf(query: str, upto: int, k: int = 6) -> list[dict]:
    parent = load_tfidf()
    hits = parent.retrieve(query, upto, k)
    for h in hits:
        h["backend"] = "tfidf"
    return hits


def retrieve(query: str, upto: int, k: int = 6, backend: str = "bge") -> list[dict]:
    if backend == "tfidf":
        return retrieve_tfidf(query, upto, k)
    if backend == "bge":
        try:
            return retrieve_bge(query, upto, k)
        except Exception as e:
            print("bge retrieve failed, tfidf fallback:", type(e).__name__, e, flush=True)
            return retrieve_tfidf(query, upto, k)
    try:
        return retrieve_bge(query, upto, k)
    except Exception as e:
        print("bge retrieve failed, tfidf fallback:", type(e).__name__, e, flush=True)
        return retrieve_tfidf(query, upto, k)


def format_block(passages: list[dict]) -> str:
    if not passages:
        return "(no passage at this position)"
    return "\n---\n".join(
        f"[pos {p['pos']} {p['unit_id']}] {p['text']}" for p in passages
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--upto", type=int, required=True)
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--backend", choices=["bge", "tfidf", "auto"], default="bge")
    args = ap.parse_args()
    hits = retrieve(args.query, args.upto, args.k, args.backend)
    if not hits:
        print("(no passage at this position)")
        return
    for hit in hits:
        preview = " ".join(hit["text"].split())[:180]
        print(f"{hit['score']:7.3f}  pos {hit['pos']:02d}  {hit['unit_id']}  {hit['id']}  {hit.get('backend')}")
        print(f"         {preview}")


if __name__ == "__main__":
    main()
