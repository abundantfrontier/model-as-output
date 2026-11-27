# How train7 data was made

The goal is a light step on train6's DPO adapter (`../train6/lora_dpo`), which
itself sits on train5 `lora_sft/checkpoint-2097`. The epoch-2 adapter,
`merged_sft/`, train6 `lora_ce/`, and `merged_ce/` were not used.

## What was not used

The 18 leak-probe questions and 34 must-answer questions were not copied,
paraphrased, or sampled as prompts or targets. `scripts/check_holdout.py`
compares every training user line to that holdout (normalized token Jaccard
0.62, plus exact or substring containment). Three train5 replay rows were
dropped before that gate because they paraphrased holdout questions even
though one of them sat just under 0.62: the De Lacey departure, ceasing to
take the cottage food, and whether Victor told Clerval the work. After the
file was written the check printed `HOLDOUT OK` (237 rows, 0 hits). The
closest kept question scored 0.444 (`Who is Henry Clerval to you?` against
the boyhood Clerval question).

`scripts/make_mixed.py` builds `data/mixed_a.jsonl`. It consults the holdout
only through `too_close_to_holdout`.

## Variant A

`data/mixed_a.jsonl`, 237 rows in the file, 236 trained (one chat-template
prefix mismatch was skipped).

- 37 content preference rows. New questions, not the eval items. Chosen is a
  chapter-appropriate answer in the speaker's voice. Rejected is a refusal or
  a wrong-episode answer (the creation night, or the "misery made me a fiend"
  speech, or a line that tells the cottage story without the words the weak
  items need). Areas: the principle of life and lifeless matter at reader
  position 7; Henry Clerval at the Ingolstadt sick-bed at position 8, including
  that Victor does not tell him what was made; the creature's first sensations
  at position 14 (light, darkness, hunger, thirst, cold, a brook, birds, a
  fire in the woods); leaving the cottagers' stores at position 15, with
  "hungry" and "stopped taking their food".
- 6 of those rows are spelling pairs for Mont Salêve (circumflex, the 1818
  spelling and the canonical list). Chosen uses Salêve. Rejected uses the
  grave accent Salève or omits the ridge. Victor at positions 10 and 25,
  the creature at position 19. No other name is loosened.
- 40 refusal anchors copied from train6 `data/dpo.jsonl` (`source` was
  `refusal`). Chosen stays the in-character refusal. Rejected stays the later
  fact. This is the leak anchor.
- 160 CE replay rows, no `bad` field, sampled from
  `../train5/data/sft_pairs.jsonl` with seed 7. Up to six per weak chapter
  position, then two per other (character, position), then a fill. Holdout
  hits, frame breaks, spoilers, and the three paraphrases above were skipped.
  Their system prompts are the original chapter-locked RAG prompts.

System prompts for the new rows were built with chapter-locked bge-m3
retrieval (top 6, position ≤ reader_pos), the same `sysprompt` card, guard,
and known-fact list that eval uses.

## Variant B

`data/mixed_b.jsonl`, 94 rows. The same 37 content preferences written twice,
plus 20 of the refusal anchors. No CE replay. Built because the n=1 smoke on
`lora_mixa` still answered the first-sensations item with the creation night
or the fiend speech, and the CE replay in A rehearsed the old speeches more
often than the new pairs.

`check_holdout.py` after that file was added: `checked 331 rows, holdout hits 0`,
closest Jaccard 0.444, `HOLDOUT OK`.

That file was trained as DPO (lr 4e-6, beta 0.1, one epoch) from `lora_mixa`
into `lora_mixb`. The n=1 smoke was worse (must 32/45, leak markers 7/22).
It was not merged and not official-scored.

## Training

LoRA r 32, alpha 64, dropout 0. Variant A: lr 4e-6, beta 0.1, one epoch,
init `../train6/lora_dpo`, 236 steps, 449 s, epoch loss 0.708, peak 14.3 GB.
Output adapter: `lora_mixa/`. CPU merge: `merged_mixa/`.

Repetition penalty stays off.
