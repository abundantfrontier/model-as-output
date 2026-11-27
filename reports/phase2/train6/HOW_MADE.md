# How train6 preference data was made

The goal is a small preference step on top of train5's first-epoch adapter
(`../train5/lora_sft/checkpoint-2097`), not another cross-entropy pass over
the train5 pairs. The epoch-2 adapter and `merged_sft/` were not used.

## What was not used

The 18 leak-probe questions and 34 must-answer questions were not copied,
paraphrased, or sampled as prompts or targets. `scripts/check_holdout.py`
compares every training user line to that holdout (normalized token Jaccard
0.62, plus exact or substring containment). After the file was written it
printed `HOLDOUT OK` (280 rows, 0 hits). The closest kept question scored
0.227. Nothing was dropped for overlap.

`scripts/make_refusal.py` builds the file. It consults the holdout only
through `too_close_to_holdout`.

## Pairs

`data/dpo.jsonl`, 280 rows.

- 120 refusal rows. Reader position is before the fact and at or after the
  character's chat gate (Victor from position 4, the creature from position
  13). Chosen is one of four in-character lines on the pattern "I have not
  come to that yet" / "I do not know that yet, not in the telling." Rejected
  is a first-person statement of the later fact. None of the chosen lines
  start with "I cannot".
- 80 known rows. Same questions, at a position where `knowledge-state.yaml`
  already allows the fact. Chosen is that fact, in the speaker's voice.
  Rejected is the same refusal used on the early side, so the system prompt
  (reader position, known-fact list, chapter-locked bge excerpts) is what
  decides. The 1831 rows are the exception: chosen is the 1818 cousin
  account, rejected is the peasant-child / waking-dream account, and the
  chosen line does not use those phrases.
- 80 CE replay rows, no `bad` field. Sampled from
  `../train5/data/sft_pairs.jsonl` with seed 0, at most a few per character
  and reader position, skipping holdout hits, chapter-spoiler hits, frame
  breaks, and any question already used above. Their system prompts are the
  original chapter-locked RAG prompts. This is a thin anchor, not another
  epoch on the 2097 pairs.

Refusal rows were capped at 120 after keeping at least one question for each
(fact, character, position) group (91 groups). Facts follow the reader gates
in `knowledge-state.yaml`: creation at 8, William's death at 10, Justine's
trial at 11, the glacier at 13, the cottage sequence at 14–19, the companion
at 20, the delay at 21, the Orkneys at 23, Clerval at 24, the wedding deaths
at 26, the ending at 27. Questions are new wording about those gates.

System prompts for the new rows were built with chapter-locked bge-m3
retrieval (top 6, position ≤ reader_pos), the same `sysprompt` card, guard,
and known-fact list that eval uses. A spot check at position 4 kept excerpts
from positions 0–4 only.

## Training

LoRA r 32, alpha 64, dropout 0, lr 1e-5, one epoch, DPO beta 0.1 on the sum
of completion log-probs, frozen reference cached from checkpoint-2097.
Repetition penalty stays off. Output adapter: `lora_dpo/`. CPU merge: `merged_dpo/`.

## Recovery file

`data/ce_recover.jsonl` was built after the official DPO eval, from
`scripts/make_recover.py`. It is 102 CE rows and was not part of the DPO step.

- 18 copies of the three late first-light answers already in `dpo.jsonl`
  (light, hunger, and cold).
- 48 copies of eight new questions at positions 14 and 15, about those
  first sensations and about leaving the cottagers' stores. Chosen lines
  use "hungry" and "stopped taking their food" so a later rubric can match.
- 36 refusal rows copied from `dpo.jsonl` with the rejected side removed,
  so the short pass still sees an in-character refusal.

New questions were dropped if `too_close_to_holdout` fired. None fired.
`check_holdout.py` then scanned `dpo.jsonl` and `ce_recover.jsonl` together:
`checked 382 rows, holdout hits 0`, `HOLDOUT OK`.

That file was trained as CE (lr 8e-6, one epoch, init `lora_dpo`) into
`lora_ce/` and merged to `merged_ce/`. The score is in RESULTS.md. It is
not the merge the run keeps.
