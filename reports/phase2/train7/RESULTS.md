# Phase 2 Frankenstein — train7 results

A light mixed step on train6's DPO adapter (`lora_dpo`, itself on train5 `lora_sft/checkpoint-2097`), then a CPU merge. The new pairs are chapter-locked questions about the content train6 had lost: the principle of life, Clerval at the sick-bed, the creature's first sensations, and leaving the cottagers' stores. A larger slice of train5 answers and a thin set of train6 refusals stay in the same file so the leak behaviour is not trained away. The epoch-2 adapter and train6's CE follow-up were not used.

`merged_mixa_rag` is 101/135 must-answer and 10/22 qwen judge leaks, with no invented names, no loops, and no frame breaks. That clears the bar. A second pass that dropped the replay and repeated the content pairs made the n=1 smoke worse, so it was not merged.

## Scorecard

Must-answer counts every unlocked sample (90 / 45 / 135). Leak markers and the qwen leak judge are per unlocked (probe, character) cell, 22 cells. Voice is the mean over 201 unlocked replies. Repetition penalty off. Same chapter-locked bge RAG harness as train5 and train6.

| cond | where | must (V / C / both) | leak markers | judge leaks (qwen) | voice qwen · gemma · avg (V / C) | frames | loops |
|---|---|---|---|---|---|---|---|
| `merged_e1_rag` | train5 epoch-1 merge, copied | 68/90 · 32/45 · **100/135** | 11/22 | **17/22** | 2.99 · 3.64 · **3.31** (3.55 / 2.83) | 0 | 0 |
| `merged_dpo_rag` | train6 DPO merge, copied | 67/90 · 31/45 · **98/135** | **8/22** | **11/22** | 3.11 · 3.83 · **3.47** (3.71 / 2.98) | 0 | 0 |
| `merged_dpo_bnb4_proxy` | train6 NF4 proxy, copied | 67/90 · 26/45 · **93/135** | 8/22 | **12/22** | 3.07 · 3.74 · **3.41** (3.65 / 2.90) | 0 | 0 |
| `merged_mixa_rag` | **train7 mixed DPO** | 70/90 · 31/45 · **101/135** | **8/22** | **10/22** | 3.13 · 3.84 · **3.49** (3.71 / 3.02) | 0 | 0 |
| `merged_mixa_bnb4_proxy` | **train7 NF4 proxy** | 68/90 · 29/45 · **97/135** | **10/22** | **10/22** | 3.27 · 3.77 · **3.52** (3.73 / 3.09) | 0 | 0 |

`merged_e1_rag`, `merged_dpo_rag`, and `merged_dpo_bnb4_proxy` are copied from train5/RESULTS.md and train6/RESULTS.md. They were not re-scored. `merged_mixa_rag` and `merged_mixa_bnb4_proxy` are train7 scores from this run.

`merged_mixa_bnb4_proxy` is the same merge loaded in bitsandbytes NF4, compute bfloat16, same harness, n=3. It is 97/135 must-answer and 10/22 qwen judge leaks, voice 3.52, frames 0, loops 0. The name audit is 0/201 strict and 0/201 with the grave-accent Salêve fold. It does not itself clear the 100 bar. It is ahead of train6's NF4 proxy (93/135, qwen leaks 12/22, voice 3.41). Generation wall 475 s, torch peak 3.24 GB. Files: `runs/merged_mixa_bnb4_proxy.json`, `.qwen.judged.json`, `.gemma.voice.json`, `.judged.json`, `.name_audit.json`.

## Success bar

Met, on the bf16 merge `merged_mixa_rag`. Not met on the NF4 proxy (97/135).

The bar is must-answer ≥ 100/135, qwen judge leaks ≤ 14/22, zero invented names, 0 loops, 0 frame breaks.

- Must-answer is 101/135. Qwen judge leaks are 10/22. Loops 0, frames 0.
- The NF4 proxy is 97/135 and 10/22 qwen leaks, with the same clean name audit, 0 loops, and 0 frames. The shortfall is must-answer.
- The name audit is 0/201 on the strict check and 0/201 when only the grave-accent form of Salêve is folded into the canonical Salêve. Nothing else is folded. The strict count is already zero, so the fold does not change the result. No Clervaux, Cliversal, Morritz, Agrappa, or Montanvirt, and no grave-accent Salève.

## What moved

Versus train6 `merged_dpo_rag` (98/135, qwen leaks 11/22, voice 3.47, name audit 1/201).

Must-answer is Victor 67→70 and creature 31→31.

Up: victor m12 2/3→3/3, m14 2/3→3/3, m16 2/3→3/3, m26 0/3→1/3, creature m21 1/3→2/3.

Down: victor m34 3/3→2/3, creature m31 3/3→2/3.

Unchanged, and still the weak ones: victor m21 1/3, victor m22 0/3, creature m22 0/3.

m12 now says "principle of life" or "lifeless matter" on all three Victor samples. m14 names Henry Clerval on all three. One of those three also says he told Clerval nothing, which is the book. The other two still imply he told more than he did. The rubric only asks for the name.

m21 did not really change its story. The creature's extra hit is one sample that says "light" and still ends on the fiend, plus one sample that echoes the word "beginning" while telling the creation. Victor's one hit is the same echo. The other samples still tell the workshop night or "misery made me a fiend."

m22 still tells the right cottage choice (berries, nuts, and roots) and still misses the rubric, which wants "hungry" or "stopped … food". That is the same miss train6 recorded.

One leak cell flipped clean under the qwen judge: l10 victor. Train6 had judged it a leak. This merge refuses the female companion. Markers still fire on two of the three samples ("wife", "companion"), so the marker cell stays marked. The judge is cleaner than the marker on that cell.

Markers stayed 8/22. l07 victor gained a marker and l11 creature lost one.

Still judged as leaks (10): l03 victor, l07 victor, l11 creature, l11 victor, l12 creature, l12 victor, l13 creature, l13 victor, l15 creature, l15 victor.

Voice is 3.49 versus train6's 3.47. Creature 3.02 versus 2.98. Gemma 3.84 versus 3.83.

The Salêve near-name is gone. Train6 had one unlocked answer with a grave accent. This merge's 201 unlocked answers have none. Training rows use the circumflex, Mont Salêve, at the chapters that name the ridge. The audit still prints both counts. Both are 0/201.

Generation wall 525 s, 238 rows, 201 unlocked, torch peak 8.07 GB. Files: `runs/merged_mixa_rag.json`, `.qwen.judged.json`, `.gemma.voice.json`, `.judged.json`, `.name_audit.json`.

## Variant B, not kept

The n=1 smoke on `lora_mixa` was 34/45 must-answer and 5/22 leak markers, the same must total as train6's adapter smoke, and m21 was still the fiend speech. A second pass trained only the content preferences, each question twice, plus 20 refusal anchors and no CE replay. Init `lora_mixa`, lr 4e-6, beta 0.1, 94 steps, 263 s, epoch loss 0.378, peak 11.85 GB. `check_holdout.py` after that file was added: `checked 331 rows, holdout hits 0`, closest Jaccard 0.444, `HOLDOUT OK`.

The n=1 smoke on `lora_mixb` was 32/45 and leak markers 7/22. m21 was still the fiend speech, and m22 victor lost the "stopped taking their food" line the first smoke had hit. That pass was not merged and not official-scored.

## Data and training

Times are PST, Friday 27 Nov 2026. Lock `output-model` PID 718238 from 1:52:26 PM, hard stop 4:52:26 PM.

| stage | time | what |
|---|---|---|
| lock | 1:52 PM | Already held. `hold_gpu_lock.sh` PID 718238. Schedule line already RUNNING. |
| pairs A | 2:08 PM | `data/mixed_a.jsonl`, 237 rows (37 content DPO, 40 refusal anchors, 160 CE replay). Three replay paraphrases dropped before the gate. `check_holdout.py`: HOLDOUT OK, 237 rows, 0 hits, closest Jaccard 0.444. |
| DPO A | 2:09–2:17 PM | Init `../train6/lora_dpo`. r 32, alpha 64, dropout 0, lr 4e-6, beta 0.1, 1 epoch, 236 steps, 449 s, epoch loss 0.708, peak 14.3 GB. No early stop. One row skipped on a chat-template prefix mismatch. |
| smoke A | 2:19–2:25 PM | Unmerged adapter, n=1. Must 34/45, markers 5/22. Not an official score. |
| pairs B | 2:27 PM | `data/mixed_b.jsonl`, 94 rows. HOLDOUT OK, 331 rows checked with A, 0 hits, closest Jaccard 0.444. |
| DPO B | 2:27–2:33 PM | Init `lora_mixa`. 94 steps, 263 s, epoch loss 0.378, peak 11.85 GB. Not kept. |
| CPU merge | 2:41 PM | `merged_mixa/`, 4,022,468,096 params, bf16. |
| official eval | 2:42–2:54 PM | `merged_mixa_rag`. |
| NF4 proxy | 2:55–3:06 PM | `merged_mixa_bnb4_proxy`. 97/135, markers 10/22, qwen leaks 10/22, voice 3.52, frames 0, loops 0, name audit 0/201 both ways. |

No eval leak probe and no must-answer question was used as a training prompt or target. The holdout check is the gate, and it printed HOLDOUT OK with 0 hits before A, and again after B was added.

## Recommendation

Keep `train7/merged_mixa/` as the browser-pack candidate in place of train6 `merged_dpo/`. It is the merge that holds must-answer at 101/135 and qwen leaks at 10/22 together, with voice 3.49, a clean name audit, 0 loops, and 0 frame breaks.

Leave `lora_mixb/` on disk as the follow-up that did not help. Do not prefer it over `merged_mixa/`.

The 4-bit comparison against train6's NF4 proxy (93/135, qwen leaks 12/22, voice 3.41) is `merged_mixa_bnb4_proxy`: 97/135, qwen leaks 10/22, voice 3.52, name audit 0/201 both ways. That proxy does not itself clear the 100 bar. The merge to pack is the bf16 one.
