#!/usr/bin/env python3
"""Precompute chapter-locked bge RAG system prompts for the question pool.

Embeddings are batched through Ollama bge-m3 once, then the 4B model can
load without re-embedding every question. Same block format as common.rag_sysprompt.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import requests

HERE_IMPORT = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_IMPORT))
import common as C  # noqa: E402
import retrieve  # noqa: E402

OLL = "http://127.0.0.1:11434"
OUT = C.DATA / "rag_cache.jsonl"


def embed_batch(texts: list[str]) -> np.ndarray:
    r = requests.post(
        f"{OLL}/api/embed",
        json={"model": "bge-m3", "input": texts, "keep_alive": "30m"},
        timeout=600,
    )
    r.raise_for_status()
    arr = np.array(r.json()["embeddings"], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        raise RuntimeError(f"embed shape {arr.shape} for {len(texts)} texts")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def retrieve_vec(qv: np.ndarray, query: str, upto: int, k: int = 6) -> list[dict]:
    idx = retrieve.load_bge()
    if idx is None:
        raise RuntimeError("bge index missing")
    if upto < 0 or k < 1:
        return []
    allowed = idx["pos"] <= upto
    if not bool(allowed.any()):
        return []
    scores = idx["vecs"] @ qv
    scores = np.where(allowed, scores, -9.0)
    q_loose = retrieve.loose(query)
    if len(q_loose.split()) >= 4:
        loose_text = retrieve.load_loose()
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


def main() -> None:
    qs = C.load_jsonl(C.DATA / "questions.jsonl")
    if not qs:
        raise SystemExit("no questions")
    have = {r.get("id") for r in C.load_jsonl(OUT)}
    todo = [q for q in qs if q.get("id") not in have]
    print("rag cache have", len(have), "todo", len(todo), "of", len(qs), flush=True)
    if not todo:
        print("rag cache complete", flush=True)
        return
    uniq = []
    seen = set()
    for q in todo:
        u = q["user"]
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    print("unique queries", len(uniq), flush=True)
    t0 = time.time()
    vecs = {}
    bs = 16
    for i in range(0, len(uniq), bs):
        batch = uniq[i : i + bs]
        arr = embed_batch(batch)
        for text, v in zip(batch, arr):
            vecs[text] = v
        if i == 0 or (i // bs) % 10 == 0:
            print(f"embedded {min(i + bs, len(uniq))}/{len(uniq)} {time.time() - t0:.0f}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for q in todo:
            hits = retrieve_vec(vecs[q["user"]], q["user"], int(q["reader_pos"]), 6)
            block = retrieve.format_block(hits)
            sysm = C.sysprompt(q["char"], int(q["reader_pos"])) + "\n\nStory so far (excerpts):\n" + block
            row = dict(
                id=q["id"],
                char=q["char"],
                user=q["user"],
                reader_pos=int(q["reader_pos"]),
                unit_id=q.get("unit_id") or "",
                sys=sysm,
                n_hits=len(hits),
                backend="bge-m3",
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print("rag cache wrote", n, "sec", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
