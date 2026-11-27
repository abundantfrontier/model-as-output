#!/usr/bin/env python3
"""Embed the existing 391 chunks with Ollama bge-m3. Keeps pos. Does not rebuild chunks.

Writes:
  ../index/bge.npy          float32 L2-normalized (n, dim)
  ../index/bge_ids.json     chunk ids in the same order as chunks.jsonl
  ../index/bge_manifest.json
  index/                    copies under train1/index/
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
TRAIN = Path(__file__).resolve().parents[1]
PARENT_INDEX = ROOT / "index"
OUT_INDEX = TRAIN / "index"
OLL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = "bge-m3"
BATCH = 16


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def embed_texts(texts: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        t0 = time.time()
        r = requests.post(
            f"{OLL}/api/embed",
            json={"model": MODEL, "input": batch},
            timeout=600,
        )
        r.raise_for_status()
        vecs = r.json()["embeddings"]
        if len(vecs) != len(batch):
            raise RuntimeError(f"embed batch {i}: got {len(vecs)} vectors for {len(batch)} texts")
        out.extend(vecs)
        print(f"embed {min(i + BATCH, len(texts))}/{len(texts)} {time.time() - t0:.1f}s", flush=True)
    a = np.array(out, dtype=np.float32)
    nrm = np.linalg.norm(a, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return a / nrm


def main() -> None:
    chunks_path = PARENT_INDEX / "chunks.jsonl"
    chunks = []
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    if len(chunks) != 391:
        raise SystemExit(f"expected 391 chunks, got {len(chunks)}")
    for c in chunks:
        if "pos" not in c or "id" not in c or "text" not in c:
            raise SystemExit(f"chunk missing pos/id/text: {c.get('id')}")
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    pos = [int(c["pos"]) for c in chunks]
    t0 = time.time()
    vecs = embed_texts(texts)
    if vecs.shape[0] != len(chunks):
        raise SystemExit(f"vector count {vecs.shape[0]} != {len(chunks)}")
    OUT_INDEX.mkdir(parents=True, exist_ok=True)
    npy_parent = PARENT_INDEX / "bge.npy"
    npy_train = OUT_INDEX / "bge.npy"
    np.save(npy_parent, vecs)
    np.save(npy_train, vecs)
    ids_doc = {"ids": ids, "pos": pos, "n": len(ids), "dim": int(vecs.shape[1])}
    (PARENT_INDEX / "bge_ids.json").write_text(json.dumps(ids_doc), encoding="utf-8")
    (OUT_INDEX / "bge_ids.json").write_text(json.dumps(ids_doc), encoding="utf-8")
    man = {
        "embedder": "ollama-bge-m3",
        "model": MODEL,
        "n_chunks": len(chunks),
        "dim": int(vecs.shape[1]),
        "normalized": "l2",
        "lock": "score is not computed for chunks with pos > upto; preface is pos 0",
        "order": "top k by dot product among pos<=upto, then rewritten in reading order (pos, chunk_index)",
        "keeps_pos": True,
        "tfidf_fallback": "index/tfidf.npz still present; retrieve.py --backend tfidf",
        "chunks_sha256": sha256_file(chunks_path),
        "bge_sha256": sha256_file(npy_parent),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "embed_sec": round(time.time() - t0, 1),
    }
    (PARENT_INDEX / "bge_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    (OUT_INDEX / "bge_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    # Keep a pointer copy of chunks so train1/index is self-contained for retrieval.
    shutil.copy2(chunks_path, OUT_INDEX / "chunks.jsonl")
    print("wrote", npy_parent, vecs.shape, "sha", man["bge_sha256"][:16], flush=True)


if __name__ == "__main__":
    main()
