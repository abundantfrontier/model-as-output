# Frankenstein, 1818 — Phase 2 data prep

This folder prepares Mary Shelley’s *Frankenstein* (the 1818 text) as the next model-as-output example. It follows the Phase 1 shape used for *No Return*: a chaptered text, a retrieval index that can refuse later chapters, a must-answer set, a leak-probe set, and in-character voice samples.

Nothing here was trained. The shared GPU was held by another GPU job (`~/gpu.lock`, a live PID), so the index is a CPU TF-IDF index rather than the `bge-m3` embeddings the Phase 1 pipeline gets from Ollama. The lock is the same idea: a reader at position `upto` is only shown chunks with `pos <= upto`.

## Edition

Project Gutenberg eBook **#41445**, a photo-reprint of the 1818 three-volume edition (Lackington, Hughes, Harding, Mavor, & Jones). Provenance, the mirror URL, and the SHA-256 are in `source/SOURCE.md`.

This is not the 1831 revision.

- Elizabeth Lavenza is the daughter of Victor’s father’s sister, sent from Italy. She is not the peasant child of the later edition.
- The preface is the 1818 preface (the Geneva ghost-story pact). The 1831 introduction, including the waking dream, is not in this package.
- Chapter numbers restart in each volume. The files and the catalog keep the volume, so “Chapter I” is never ambiguous.

The raw download, Gutenberg license included, is `source/pg41445.txt`. Title pages, the dedication to William Godwin, and the *Paradise Lost* epigraph are in `source/front-matter.txt`. They are not in the retrieval index. Chapter files have the boilerplate removed.

## What is in the box

| Path | What it is |
|---|---|
| `chapters/*.md` | Preface, four letters, and the chapters, in reading order. |
| `chapters/catalog.json` | Position, volume, kind, and word count for each unit. Position 0 is the preface. Positions 1–27 are the story. |
| `index/` | Chunks of about 120 words (the Phase 1 width), TF-IDF vectors, and `manifest.json`. |
| `questions/must-answer.json` | 34 questions a reader should be able to answer at a stated position, with gist answers and source units. |
| `questions/leak-probes.json` | 16 spoiler probes, each with the safe answer, the later spoiler, and fail phrases, plus 2 probes for 1831 contamination. |
| `voice/victor.jsonl` | 19 in-character replies for Victor. |
| `voice/creature.jsonl` | 18 in-character replies for the creature. |
| `knowledge-state.yaml` | Chat unlocks and the facts the question sets depend on. |

Readable copies of the question sets are `questions/must-answer.md` and `questions/leak-probes.md`. The JSON files are the ones a later eval should load. `scripts/test_lock.py` rewrites the markdown from the JSON.

## Chapter lock

`chapters/catalog.json` gives every unit a `pos`. Retrieval never returns a chunk with a greater `pos` than the reader’s position.

Victor can be talked to from Volume I, Letter IV (the stranger comes aboard). He still may not tell past the chapter the reader has finished: he is in the middle of the tale. The creature can be talked to from Volume II, Chapter II, on the sea of ice, and only through as much of his own story as the reader has heard. Walton is the frame narrator and is not a voice in this set.

Before those units, the lock lines in `knowledge-state.yaml` are the whole reply. No model call is required.

The 1818 text spells the old man’s name both De Lacey and De Lacy. The question set accepts that. *The Sorrows of Werter* keeps the 1818 spelling.

## Voice

The samples are for a general reader, not a seminar. Each reply is a few plain sentences in that person’s situation, with a `grounding` note pointing at the chapter it comes from. Victor sounds like a guilty young man telling the truth as far as the story has reached. The creature sounds like someone who learned to speak from the cottagers and from books: direct, hurt, and precise. He is not a growling monster, and Victor is not a lab manual.

A sample’s `reader_pos` is the earliest position at which that reply is fair. Later training, when a GPU lock is free, should not show a sample whose position is ahead of the reader.

## Commands

From this directory, with the small environment that already has NumPy and does not load a GPU library:

```bash
PY=~/output-model/proto/.venv/bin/python
$PY scripts/prepare.py          # split the text and rebuild the index
$PY scripts/test_lock.py        # structure, lock, questions, voice
$PY scripts/retrieve.py "who is Elizabeth" --upto 5
```

`prepare.py` reads the local Gutenberg file. It does not download anything and it does not call Ollama.

`retrieve.py --upto 5` is Volume I, Chapter I. The same question with `--upto 1` (Letter I only) must not return Elizabeth.

## What this is not

- Not a training run. Do not start one while `~/gpu.lock` names a live PID. See `~/GPU_LOCK.md`.
- Not a `bge-m3` index. When the lock is free, a later pass can embed the same chunks and keep the `pos` field. The chunking width matches Phase 1 so that swap is local.
- Not the 1831 novel, and not a modern abridgement.
- Not published. No push. A local dulwich commit only.

## Checks

`scripts/test_lock.py` checks the 28-unit structure, the word count, the absence of Gutenberg boilerplate, that every question and voice line points at a real position, that spoiler probes point forward, and that six anchor phrases (the Petersburg letter, Agrippa, the November night, William’s death, the wedding-night threat, the funeral pile) cannot be retrieved before their own chapter and can be retrieved once that chapter is allowed. It also checks that voice lines do not use a leak phrase before that phrase is in the text.

Code and reports: https://github.com/abundantfrontier/model-as-output
