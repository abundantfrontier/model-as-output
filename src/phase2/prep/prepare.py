#!/usr/bin/env python3
"""Split the 1818 Frankenstein text and build a CPU chapter-locked index.

No GPU. No Ollama. No network. Reads source/pg41445.txt, which is Project
Gutenberg eBook 41445 (a photo-reprint of the 1818 edition).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "pg41445.txt"
CHAPTERS = ROOT / "chapters"
INDEX = ROOT / "index"

ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
}
ROMAN_OUT = {n: r for r, n in ROMAN.items()}
HEADING = re.compile(r"^(PREFACE\.|LETTER ([IVX]+)\.|CHAPTER ([IVX]+)\.)$")
VOLUME = re.compile(r"^VOL\. ([IVX]+)\.$")
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
CHUNK_WORDS = 120

# Gutenberg boilerplate and the transcriber note must not enter chapter files.
DROP_LINE = re.compile(
    r"project gutenberg|transcriber|^\*\*\*|^\s*\[",
    re.I,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def slice_body(text: str) -> list[str]:
    start = text.index("*** START")
    end = text.index("*** END")
    lines = text[start:end].splitlines()
    return [ln for ln in lines if not DROP_LINE.search(ln)]


def unwrap(lines: list[str]) -> list[str]:
    paras: list[str] = []
    buf: list[str] = []

    def emit() -> None:
        if not buf:
            return
        text = " ".join(x.strip() for x in buf if x.strip())
        buf.clear()
        if not text or re.fullmatch(r"[*\s]+", text):
            return
        text = re.sub(r"_([^_]+)_", r"*\1*", text)
        text = re.sub(r" +", " ", text).strip()
        paras.append(text)

    for line in lines:
        if line.strip() == "":
            emit()
        else:
            buf.append(line)
    emit()
    return paras


def split_units(lines: list[str]) -> tuple[list[dict], list[str]]:
    units: list[dict] = []
    front: list[str] = []
    cur: dict | None = None
    vol: int | None = None

    def flush() -> None:
        nonlocal cur
        if cur and any(ln.strip() for ln in cur["lines"]):
            units.append(cur)
        cur = None

    for line in lines:
        s = line.strip()
        mvol = VOLUME.match(s)
        if mvol and cur is None:
            vol = ROMAN[mvol.group(1)]
            front.append(line)
            continue
        if s.startswith("END OF VOL"):
            flush()
            continue
        if s in {"FRANKENSTEIN;", "THE END."} or s.startswith("THE END"):
            flush()
            front.append(line)
            continue
        mh = HEADING.match(s)
        if mh:
            flush()
            if s.startswith("PREFACE"):
                kind, number = "preface", 0
                volume = 0
            elif mh.group(2):
                kind, number = "letter", ROMAN[mh.group(2)]
                volume = vol
            else:
                kind, number = "chapter", ROMAN[mh.group(3)]
                volume = vol
            cur = {
                "kind": kind,
                "number": number,
                "volume": volume,
                "original_heading": s,
                "lines": [],
            }
            continue
        if cur is None:
            front.append(line)
        else:
            cur["lines"].append(line)
    flush()
    return units, front


def title_of(unit: dict) -> str:
    if unit["kind"] == "preface":
        return "Preface"
    if unit["kind"] == "letter":
        return f"Volume {ROMAN_OUT[unit['volume']]}, Letter {ROMAN_OUT[unit['number']]}"
    return f"Volume {ROMAN_OUT[unit['volume']]}, Chapter {ROMAN_OUT[unit['number']]}"


def unit_id(unit: dict) -> str:
    if unit["kind"] == "preface":
        return "preface"
    if unit["kind"] == "letter":
        return f"vol{unit['volume']}-letter-{unit['number']}"
    return f"vol{unit['volume']}-ch-{unit['number']}"


def words(text: str) -> int:
    return len(text.split())


def chunks_of(text: str, size: int = CHUNK_WORDS) -> list[str]:
    """Same packing as the No Return pipeline: paragraphs, about 120 words."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    cur: list[str] = []
    for para in paras:
        cur.append(para)
        if sum(len(x.split()) for x in cur) >= size:
            out.append("\n\n".join(cur))
            cur = []
    if cur:
        out.append("\n\n".join(cur))
    return out


def loose(text: str) -> str:
    text = text.lower().replace("æ", "ae").replace("œ", "oe")
    text = text.replace("—", " ").replace("–", " ")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(loose(text))


def write_chapters(units: list[dict]) -> list[dict]:
    if CHAPTERS.exists():
        for old in CHAPTERS.glob("*.md"):
            old.unlink()
    CHAPTERS.mkdir(parents=True, exist_ok=True)
    catalog = []
    for pos, unit in enumerate(units):
        paras = unwrap(unit["lines"])
        body = "\n\n".join(paras).strip() + "\n"
        uid = unit_id(unit)
        title = title_of(unit)
        fname = f"{pos:02d}-{uid}.md"
        text = f"# {title}\n\n{body}"
        (CHAPTERS / fname).write_text(text, encoding="utf-8")
        leftover = text.count("_")
        catalog.append({
            "pos": pos,
            "unit_id": uid,
            "volume": unit["volume"],
            "kind": unit["kind"],
            "number": unit["number"],
            "title": title,
            "original_heading": unit["original_heading"],
            "file": f"chapters/{fname}",
            "words": words(body),
            "paragraphs": len(paras),
            "opening": paras[0][:240] if paras else "",
            "underscore_leftover": leftover,
        })
    (CHAPTERS / "catalog.json").write_text(
        json.dumps({"edition": "1818", "units": catalog}, indent=2) + "\n",
        encoding="utf-8",
    )
    return catalog


def build_index(catalog: list[dict]) -> dict:
    chunk_rows = []
    for unit in catalog:
        body = (ROOT / unit["file"]).read_text(encoding="utf-8")
        # Drop the H1 so retrieval matches prose, not the heading we added.
        body = re.sub(r"^# .+\n+", "", body)
        parts = chunks_of(body)
        for i, part in enumerate(parts):
            chunk_rows.append({
                "id": f"{unit['unit_id']}-c{i:02d}",
                "pos": unit["pos"],
                "unit_id": unit["unit_id"],
                "volume": unit["volume"],
                "kind": unit["kind"],
                "number": unit["number"],
                "title": unit["title"],
                "chunk_index": i,
                "words": words(part),
                "text": part,
                "loose": loose(part),
            })

    vocab: dict[str, int] = {}
    token_rows: list[list[int]] = []
    for row in chunk_rows:
        ids = []
        for tok in tokenize(row["text"]):
            if tok not in vocab:
                vocab[tok] = len(vocab)
            ids.append(vocab[tok])
        token_rows.append(ids)

    n = len(chunk_rows)
    v = len(vocab)
    df = np.zeros(v, dtype=np.int32)
    counts: list[Counter] = []
    for ids in token_rows:
        c = Counter(ids)
        counts.append(c)
        for term in c:
            df[term] += 1
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    # CSC-like columns, one posting list per term: chunk id, weight.
    posts: list[list[tuple[int, float]]] = [[] for _ in range(v)]
    for i, c in enumerate(counts):
        weights = {t: (1.0 + np.log(cnt)) * float(idf[t]) for t, cnt in c.items()}
        norm = float(np.sqrt(sum(w * w for w in weights.values()))) or 1.0
        for t, w in weights.items():
            posts[t].append((i, w / norm))
    for term_posts in posts:
        indices.extend(i for i, _ in term_posts)
        data.extend(w for _, w in term_posts)
        indptr.append(len(indices))

    INDEX.mkdir(parents=True, exist_ok=True)
    # jsonl without the loose key (it is only a retrieval aid, stored beside vectors).
    with (INDEX / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for row in chunk_rows:
            public = {k: val for k, val in row.items() if k != "loose"}
            fh.write(json.dumps(public, ensure_ascii=False) + "\n")
    vocab_list = [""] * v
    for tok, i in vocab.items():
        vocab_list[i] = tok
    (INDEX / "vocab.json").write_text(json.dumps(vocab_list), encoding="utf-8")
    np.savez_compressed(
        INDEX / "tfidf.npz",
        indptr=np.asarray(indptr, dtype=np.int32),
        indices=np.asarray(indices, dtype=np.int32),
        data=np.asarray(data, dtype=np.float32),
        idf=idf.astype(np.float32),
        pos=np.asarray([r["pos"] for r in chunk_rows], dtype=np.int16),
        df=df,
    )
    # Loose text is large but lets the lock test and the phrase bonus run
    # without re-tokenizing. Kept next to the vectors, not in git-unfriendly logs.
    np.savez_compressed(
        INDEX / "loose.npz",
        loose=np.asarray([r["loose"] for r in chunk_rows], dtype=object),
    )
    manifest = {
        "edition": "1818",
        "source": "Project Gutenberg eBook #41445",
        "source_file": "source/pg41445.txt",
        "source_sha256": sha256(SOURCE),
        "embedder": "tfidf-sublinear-cosine-cpu",
        "not_bge_m3": (
            "bge-m3 in the Phase 1 pipeline is served by Ollama and uses the "
            "GPU. another GPU job held ~/gpu.lock, so this index is a CPU TF-IDF index "
            "with the same chapter lock: chunks whose pos is greater than "
            "the reader position are excluded."
        ),
        "chunk_words": CHUNK_WORDS,
        "lock": "score is not computed for chunks with pos > upto; preface is pos 0",
        "phrase_bonus": "if a query of at least 4 tokens occurs inside an allowed chunk, that chunk is ranked first",
        "order": "top k by score, then rewritten in reading order (pos, chunk_index)",
        "n_units": len(catalog),
        "n_chunks": n,
        "vocab": v,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (INDEX / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_source_note(catalog: list[dict], manifest: dict, front: list[str]) -> None:
    front_path = ROOT / "source" / "front-matter.txt"
    front_path.write_text("\n".join(front).strip() + "\n", encoding="utf-8")
    total_words = sum(u["words"] for u in catalog)
    lines = [
        "# Source",
        "",
        "Project Gutenberg eBook **#41445**, *Frankenstein; Or, The Modern Prometheus*.",
        "",
        "- Edition: the 1818 text, from a photo-reprint. Three volumes. Letters I–IV, then chapters restarting in each volume.",
        "- Publisher on the title page: Lackington, Hughes, Harding, Mavor, & Jones, London, 1818.",
        "- This is not the 1831 revision. Elizabeth Lavenza is the daughter of Victor's father's sister. The 1831 introduction (the waking dream) is not in this file. The preface here is the 1818 preface.",
        "- Downloaded 2026-11-20 from the University of Waterloo Gutenberg mirror because gutenberg.org returned 503:",
        "  https://mirror.csclub.uwaterloo.ca/gutenberg/4/1/4/4/41445/41445-0.txt",
        "- Catalog record: https://www.gutenberg.org/ebooks/41445",
        f"- SHA-256 of `source/pg41445.txt`: `{manifest['source_sha256']}`",
        "",
        "The raw file still has the Project Gutenberg header, transcriber note, and license. Those are stripped before the chapter split. Title pages, the dedication to William Godwin, and the *Paradise Lost* epigraph are kept in `source/front-matter.txt` and are not part of the retrieval index.",
        "",
        f"Chapter files: {len(catalog)}. Words in the chapter bodies: {total_words}.",
        "",
    ]
    (ROOT / "source" / "SOURCE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raw = SOURCE.read_text(encoding="utf-8")
    units, front = split_units(slice_body(raw))
    catalog = write_chapters(units)
    manifest = build_index(catalog)
    write_source_note(catalog, manifest, front)
    kinds = Counter((u["volume"], u["kind"]) for u in catalog)
    print("units", len(catalog), "chunks", manifest["n_chunks"], "vocab", manifest["vocab"])
    print("words", sum(u["words"] for u in catalog))
    for key in sorted(kinds):
        print(f"  vol{key[0]} {key[1]}: {kinds[key]}")
    bad = [u["unit_id"] for u in catalog if u["underscore_leftover"] % 2]
    print("odd underscores", bad)


if __name__ == "__main__":
    main()
