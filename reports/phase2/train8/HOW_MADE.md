# How train8 data was made

The goal is a light step on train7's adapter (`../train7/lora_mixa`), which
itself sits on train6 `lora_dpo` on train5 `lora_sft/checkpoint-2097`. The
pass is QLoRA DPO: the base is NF4, compute is bf16, the adapter stays bf16,
and double quant is off. train6 `lora_ce` / `merged_ce`, the epoch-2
`lora_sft`, and train7 `lora_mixb` were not used.

## What was not used

The leak-probe questions and the must-answer questions were not copied,
paraphrased, or sampled as prompts or targets. `scripts/check_holdout.py`
compares every training user line to that holdout. After the file was
written the check printed `HOLDOUT OK` (239 rows, 0 hits, closest Jaccard
0.333). The same check was run again before the weight average, on the same
file: `HOLDOUT OK`, 239 rows, 0 hits, closest Jaccard 0.333. Max Jaccard
against train7 `mixed_a.jsonl` and `mixed_b.jsonl` is 0.333. Those train7
rows are allowed as replay. They were not copied back in as new content.

`scripts/make_quanta.py` builds `data/mixed_qa.jsonl`. The report is
`data/mixed_qa_report.json`.

## Variant A

`data/mixed_qa.jsonl`, 239 rows, all 239 trained (0 prefix skips).

- 61 content preference rows (`content_dpo`). New questions on the fragile
  areas from `analysis/nf4_loss_diff.md`, not the eval items. Counts by
  `reader_pos`: 14 has 16, 15 has 9, 19 has 7, 18 has 5, and 4, 10, 12, 21,
  22, 24, and 27 have 3 or 4 each. Chosen is the chapter-appropriate answer.
  Rejected is a refusal, a wrong episode, or a generic answer.
- 18 content replay rows from train7, including the Mont Salêve spelling
  pairs. The `answer` field has the circumflex (8 occurrences, 6 rows) and
  no grave accent. The grave accent is only in the rejected `bad` field
  (7 occurrences).
- 40 refusal anchors. Chosen stays the in-character refusal.
- 120 CE replay rows, no second content copy of the new questions. 31 CE
  rows at reader positions 14 and 15 were dropped because they rehearse the
  NF4 failure mode on the first sensations and the cottage food. 0 CE rows
  were dropped by the holdout gate.

System prompts for the new rows use chapter-locked bge retrieval with the
same reader-position rule as train7.

## Variant B

No new file. `lora_qb` is the weight average
`(1 - 0.5) * lora_mixa + 0.5 * lora_qa` (504 tensors). The holdout check
was run again before that average and printed `HOLDOUT OK` on the same 239
rows. This was the second lever, used because the QLoRA merge missed the
bar on both precisions. It was not another training pass.

## Training

LoRA r 32, alpha 64, dropout 0. Variant A: lr 4e-6, beta 0.1, one epoch,
init `../train7/lora_mixa`, NF4 base, 239 steps, 801 s, epoch loss 0.724,
peak 12.93 GB. Output adapter: `lora_qa/`. CPU merge: `merged_qa/`.
The weight average is `lora_qb/`, CPU merge `merged_qb/`.

Repetition penalty stays off.
