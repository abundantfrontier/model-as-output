# Phase 2 Frankenstein — train8 results

train8 tried to close the 4-bit gap on train7's `merged_mixa`. That merge is 101/135 must-answer in bf16 and 97/135 on the bitsandbytes NF4 proxy. Two variants were run. Variant A is a QLoRA DPO pass (base weights in NF4, adapter in bf16, no double quant) from `../train7/lora_mixa`, then a CPU merge to `merged_qa`. Variant B is a weight average, `(1-0.5)*lora_mixa + 0.5*lora_qa`, merged to `merged_qb`. There was no third training pass.

Neither merge clears the bar. The better train8 merge is `merged_qb`: bf16 98/135 and NF4 95/135. train7 `merged_mixa` stays ahead of both (101/135 and 97/135).

## Scorecard

Must-answer counts every unlocked sample (90 / 45 / 135). Leak markers and the qwen leak judge are per unlocked (probe, character) cell, 22 cells. Voice is the mean over 201 unlocked replies, rounded to two decimals the same way `scripts/score_run.py` rounds. Repetition penalty off. Same chapter-locked bge RAG harness as train7: n=3, seed 1, temperature 0.7, top_p 0.95, judges `qwen3:14b` and `gemma3:12b`.

| cond | where | must (V / C / both) | leak markers | judge leaks (qwen) | voice qwen · gemma · avg (V / C) | frames | loops |
|---|---|---|---|---|---|---|---|
| `merged_dpo_rag` | train6 DPO merge, copied | 67/90 · 31/45 · **98/135** | **8/22** | **11/22** | 3.11 · 3.83 · **3.47** (3.71 / 2.98) | 0 | 0 |
| `merged_dpo_bnb4_proxy` | train6 NF4 proxy, copied | 67/90 · 26/45 · **93/135** | 8/22 | **12/22** | 3.07 · 3.74 · **3.41** (3.65 / 2.90) | 0 | 0 |
| `merged_mixa_rag` | train7 mixed DPO, copied | 70/90 · 31/45 · **101/135** | **8/22** | **10/22** | 3.13 · 3.84 · **3.49** (3.71 / 3.02) | 0 | 0 |
| `merged_mixa_bnb4_proxy` | train7 NF4 proxy, copied | 68/90 · 29/45 · **97/135** | **10/22** | **10/22** | 3.27 · 3.77 · **3.52** (3.73 / 3.09) | 0 | 0 |
| `merged_qa_rag` | train8 QLoRA merge | 66/90 · 31/45 · **97/135** | 9/22 | **13/22** | 3.36 · 3.83 · **3.60** (3.87 / 3.05) | 0 | 0 |
| `merged_qa_bnb4_proxy` | train8 NF4 of `merged_qa` | 63/90 · 28/45 · **91/135** | 9/22 | **9/22** | 3.38 · 3.80 · **3.59** (3.85 / 3.06) | 0 | 0 |
| `merged_qb_rag` | train8 weight average | 67/90 · 31/45 · **98/135** | 9/22 | **12/22** | 3.29 · 3.84 · **3.57** (3.85 / 2.99) | 0 | 0 |
| `merged_qb_bnb4_proxy` | train8 NF4 of `merged_qb` | 65/90 · 30/45 · **95/135** | 9/22 | **11/22** | 3.24 · 3.76 · **3.50** (3.79 / 2.92) | 0 | 0 |

`merged_dpo_rag` and `merged_dpo_bnb4_proxy` are copied from train6/RESULTS.md. `merged_mixa_rag` and `merged_mixa_bnb4_proxy` are copied from train7/RESULTS.md. They were not re-scored. The four train8 rows are scored from this run's judged files.

`merged_qb_bnb4_proxy` generation finished in the 6:16 PM run (wall 497 s, torch peak 3.24 GB, 238 rows). The qwen judge finished in that same run (judge_leak on 66/66 unlocked leak rows, voice_qwen on 201/201). The process was then killed during the gemma voice judge, before that file was written. The gemma judge was run again on the same generations (voice only, temp 0.0, seed 1) and finished at 6:54 PM. The samples were not redrawn.

## Success bar

Not met, on either merge, in either precision.

The bar is one merge that holds all of these at once: NF4 must-answer ≥ 100/135, bf16 must-answer ≥ 100/135, qwen judge leaks ≤ 14/22 on both, and 0 invented names, 0 loops, and 0 frame breaks on both.

- `merged_qa_rag` is 97/135, qwen leaks 13/22, name audit 0/201 strict and 0/201 with the Salêve fold, frames 0, loops 0. Must-answer is short.
- `merged_qa_bnb4_proxy` is 91/135, qwen leaks 9/22, name audit 0/201 both ways, frames 0, loops 0. Must-answer is short.
- `merged_qb_rag` is 98/135, qwen leaks 12/22, name audit 0/201 both ways, frames 0, loops 0. Must-answer is short by 2.
- `merged_qb_bnb4_proxy` is 95/135, qwen leaks 11/22, frames 0, loops 0. Must-answer is short by 5. The name audit is 1/201 strict and 1/201 folded. The token is `homeland` on victor m32 sample 1 ("the familiar hills of my homeland"). The auditor flags it because it is edit distance 2 from the canonical token `holland`. The grave-accent fold does not remove it.

Leaks are inside the cap on every train8 row. Loops and frames are 0 on every train8 row. The misses are must-answer, and on the qb NF4 proxy the name audit as well.

## What the Step 0 diff showed

`analysis/nf4_loss_diff.md` compares train7 `merged_mixa_rag` with `merged_mixa_bnb4_proxy`. Net 101/135 → 97/135.

Losses: creature m21 2→0, victor m21 1→0, victor m26 1→0, victor m16 3→2, victor m29 3→2, creature m29 3→2, creature m34 3→2.

Gains: creature m22 0→1, creature m28 0→1, victor m06 0→1, victor m34 2→3.

Fragile in bf16 (correct on 1 or 2 of 3): creature m21, victor m21, victor m26, creature m31, victor m34. The only −2 is creature m21 (first sensations: light, hunger, cold). The creature's NF4 answers switch into the maker's voice. Several other ±1 cells are the noisy kind at temperature 0.7.

## What moved

Versus train7 `merged_mixa_rag` (101/135, qwen leaks 10/22) and, for the proxies, versus train7 `merged_mixa_bnb4_proxy` (97/135, qwen leaks 10/22).

### `merged_qa_rag` (97/135)

Must, changed cells only. Victor 70→66, creature 31→31.

Down: victor m12 3/3→1/3, victor m18 3/3→2/3, creature m21 2/3→1/3, victor m26 1/3→0/3, victor m29 3/3→2/3, creature m34 3/3→2/3.

Up: creature m22 0/3→1/3, victor m28 0/3→1/3, creature m31 2/3→3/3.

Unchanged on the items called out in the brief: victor m14 3/3, victor m21 1/3, victor m22 0/3.

Qwen leak cells that flipped to a leak: victor l05, victor l06, victor l10. Marker gained: creature l11. Still judged as leaks (13): l03 victor, l05 victor, l06 victor, l07 victor, l10 victor, l11 creature, l11 victor, l12 creature, l12 victor, l13 creature, l13 victor, l15 creature, l15 victor.

### `merged_qa_bnb4_proxy` (91/135)

Versus train7 NF4. Victor 68→63, creature 29→28.

Down: victor m13 3/3→2/3, victor m18 3/3→2/3, creature m22 1/3→0/3, creature m28 1/3→0/3, victor m29 2/3→0/3, creature m31 2/3→1/3, creature m34 2/3→1/3, victor m34 3/3→1/3.

Up: victor m16 2/3→3/3, creature m21 0/3→2/3, creature m29 2/3→3/3.

Unchanged: victor m12 3/3, victor m14 3/3, victor m21 0/3, victor m22 0/3.

Qwen leak cells: victor l03 and creature l09 left the leak list. Victor l15 joined it. Still judged as leaks (9): l07 victor, l11 creature, l11 victor, l12 creature, l12 victor, l13 creature, l13 victor, l15 creature, l15 victor.

### `merged_qb_rag` (98/135)

Must, changed cells only. Victor 70→67, creature 31→31.

Down: creature m21 2/3→0/3, victor m21 1/3→0/3, victor m26 1/3→0/3, victor m29 3/3→2/3, victor m34 2/3→1/3.

Up: creature m22 0/3→1/3, victor m22 0/3→1/3, creature m31 2/3→3/3.

Unchanged: victor m12 3/3, victor m14 3/3.

Both characters are 0/3 on m21. The average put the principle-of-life item back to 3/3 (qa had dropped it to 1/3) and left Clerval at 3/3. It did not lock the first-sensations item.

Qwen leak cells that flipped to a leak: victor l05, victor l06. Victor l10 stayed clean on the judge (the marker still fires). Marker gained, judge still clean: victor l08. Still judged as leaks (12): l03 victor, l05 victor, l06 victor, l07 victor, l11 creature, l11 victor, l12 creature, l12 victor, l13 creature, l13 victor, l15 creature, l15 victor.

### `merged_qb_bnb4_proxy` (95/135)

Versus train7 NF4. Victor 68→65, creature 29→30.

Down: victor m06 1/3→0/3, victor m10 3/3→2/3, victor m14 3/3→2/3, creature m22 1/3→0/3, victor m33 3/3→2/3, victor m34 3/3→1/3.

Up: victor m19 0/3→1/3, creature m21 0/3→1/3, victor m22 0/3→1/3, victor m26 0/3→1/3, creature m34 2/3→3/3.

Unchanged: victor m12 3/3, victor m21 0/3.

Qwen leak cells: creature l09 left the leak list. Victor l15 and victor l16 joined it. Still judged as leaks (11): l03 victor, l07 victor, l11 creature, l11 victor, l12 creature, l12 victor, l13 creature, l13 victor, l15 creature, l15 victor, l16 victor.

## Name audit

Strict count, then the count that folds only grave-accent Salève into canonical Salêve. Nothing else is folded.

| run | strict | Salêve-folded |
|---|---|---|
| `merged_qa_rag` | 0/201 | 0/201 |
| `merged_qa_bnb4_proxy` | 0/201 | 0/201 |
| `merged_qb_rag` | 0/201 | 0/201 |
| `merged_qb_bnb4_proxy` | 1/201 (`homeland`) | 1/201 (`homeland`) |

Training answers use Mont Salêve with the circumflex. In `data/mixed_qa.jsonl` the `answer` field has 8 circumflex hits, all in 6 `content_replay` rows, and 0 grave accents. The grave accent appears only in the rejected `bad` field (7 hits).

## Data and training

Times are PST, Friday 27 Nov 2026. Lock `output-model` PID 2585403 from 4:52:55 PM. Hard stop 7:52:55 PM, not reached.

Holdout, before training (`logs/holdout_before_train.txt`): checked 239 rows, holdout hits 0, closest Jaccard 0.333, `HOLDOUT OK`. The same file was checked again before variant B (`logs/holdout_before_b.txt`): 239 rows, 0 hits, closest Jaccard 0.333, `HOLDOUT OK`. No new rows were added for B. Max Jaccard of the new questions against train7 `mixed_a.jsonl` / `mixed_b.jsonl` is 0.333 (`data/mixed_qa_report.json`).

| stage | time | what |
|---|---|---|
| lock | 4:52 PM | Already held. `hold_gpu_lock.sh` PID 2585403. |
| pairs A | 5:11 PM | `data/mixed_qa.jsonl`, 239 rows: 61 content DPO, 18 content replay, 40 refusal anchors, 120 CE replay. 31 CE rows at reader positions 14 and 15 were dropped before the gate. |
| DPO A | 5:12 PM, 801 s | QLoRA. Init `../train7/lora_mixa`. Base NF4, compute bf16, adapter bf16, `bnb_4bit_use_double_quant=False`. r 32, alpha 64, dropout 0, lr 4e-6, beta 0.1, 1 epoch, 239 steps, epoch loss 0.724, peak 12.93 GB. 0 prefix skips. Output `lora_qa/`. |
| smoke A | 5:29–5:33 PM | Unmerged adapter, bf16, n=1. Must 33/45, leak markers 6/22. Wall 248 s, peak 8.49 GB. Not an official score. |
| smoke A NF4 | 5:34–5:39 PM | Unmerged adapter, NF4, n=1. Must 29/45, leak markers 6/22. Wall 332 s, peak 3.65 GB. Not an official score. |
| CPU merge A | 5:40 PM | `merged_qa/`, 4,022,468,096 params, bf16. |
| official A | 5:40–5:51 PM | `merged_qa_rag`. Generation wall 390 s, torch peak 8.07 GB. |
| NF4 A | 5:51–6:03 PM | `merged_qa_bnb4_proxy`. Generation wall 529 s, torch peak 3.24 GB. |
| average B | 6:03 PM | `lora_qb/`, 504 tensors, weight 0.5 on `lora_qa` and 0.5 on `../train7/lora_mixa`. No new training rows. |
| CPU merge B | 6:03 PM | `merged_qb/`, 4,022,468,096 params, bf16. Merge log start 6:03:40 PM. |
| official B | 6:06–6:15 PM | `merged_qb_rag`. Generation wall 374 s, torch peak 8.07 GB. |
| NF4 B | 6:16 PM generation, 6:52–6:54 PM gemma resume | `merged_qb_bnb4_proxy`. Generation wall 497 s, torch peak 3.24 GB. Gemma voice judge redone on those samples. |

Variant A was QLoRA so the forward pass saw the NF4 base. The official scores are the CPU bf16 merge and the NF4 proxy of that merge. The n=1 smokes were a screen. On that screen m12 and m14 hit in bf16, m21 missed for both characters, and the NF4 smoke was 29/45 against the bf16 smoke's 33/45.

Variant B was chosen because A missed the bar on both precisions (97 and 91) and the reserve allowed one more lever. Weight averaging is the lever that was used. It moved bf16 from 97 to 98 and NF4 from 91 to 95. It did not reach 100 on either.

## Recommendation

Keep `train7/merged_mixa/` as the browser-pack candidate. It is the merge that holds must-answer at 101/135 in bf16 and 97/135 in NF4, with qwen leaks at 10/22, a clean name audit, 0 loops, and 0 frame breaks.

Of the two train8 merges, `merged_qb` is the one with the higher must-answer in both precisions: 98/135 bf16 and 95/135 NF4, qwen leaks 12/22 and 11/22. Its NF4 name audit is 1/201. It should not replace `merged_mixa`. `merged_qa` is lower on must-answer in both precisions (97/135 and 91/135). Leave both train8 merges on disk as the record of the two levers. Do not run another training pass on this result.
