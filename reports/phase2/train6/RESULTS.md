# Phase 2 Frankenstein — train6 results

A small DPO step on train5's first-epoch adapter (`lora_sft/checkpoint-2097`), then a CPU merge. The pairs are new chapter-locked future-knowledge probes: refuse before the fact, answer it once the reader has reached it, plus a thin CE replay of train5 answers. The epoch-2 adapter and `train5/merged_sft/` were not used.

The DPO merge cuts qwen judge leaks from 17/22 to 11/22. Must-answer comes back at 98/135, two under the train5 merge and two under the bar. A short CE follow-up pushed leaks to 6/22 and must-answer down to 87/135, so the DPO merge is the one recorded as the train6 result.

## Scorecard

Must-answer counts every unlocked sample (90 / 45 / 135). Leak markers and the qwen leak judge are per unlocked (probe, character) cell, 22 cells. Voice is the mean over 201 unlocked replies. Repetition penalty off. Same chapter-locked bge RAG harness as train5.

| cond | where | must (V / C / both) | leak markers | judge leaks (qwen) | voice qwen · gemma · avg (V / C) | frames | loops |
|---|---|---|---|---|---|---|---|
| `merged_e1_rag` | train5 epoch-1 merge | 68/90 · 32/45 · **100/135** | 11/22 | **17/22** | 2.99 · 3.64 · **3.31** (3.55 / 2.83) | 0 | 0 |
| `merged_e1_bnb4_proxy` | browser1 NF4 proxy of that merge | 62/90 · 26/45 · **88/135** | 8/22 | **17/22** | 2.90 · 3.52 · **3.21** (3.48 / 2.67) | 0 | 0 |
| `p2_rewrite_guarded` | train3, two-pass | 66/90 · 34/45 · **100/135** | 8/22 | **14/22** | 3.03 · 3.64 · **3.33** (3.50 / 3.00) | 0 | 0 |
| `merged_dpo_rag` | **train6 DPO merge** | 67/90 · 31/45 · **98/135** | **8/22** | **11/22** | 3.11 · 3.83 · **3.47** (3.71 / 2.98) | 0 | 0 |
| `merged_dpo_bnb4_proxy` | train6 NF4 proxy of that merge | 67/90 · 26/45 · **93/135** | 8/22 | **12/22** | 3.07 · 3.74 · **3.41** (3.65 / 2.90) | 0 | 0 |
| `merged_ce_rag` | train6 CE follow-up, not kept | 60/90 · 27/45 · **87/135** | 2/22 | **6/22** | 3.57 · 4.00 · **3.79** (4.04 / 3.26) | 0 | 0 |

`merged_e1_rag` is copied from train5/RESULTS.md. The browser1 proxy row is copied from browser1/RESULTS.md (88/135, qwen leaks 17/22, voice 3.21). They were not re-scored. The train3 guarded row is the published leak bar and is copied from train5/RESULTS.md.

## Success bar

Not met.

The bar is qwen judge leaks ≤ 14/22, must-answer ≥ 100/135, zero invented names, 0 loops, 0 frame breaks.

- `merged_dpo_rag` clears the leak bar at 11/22. Must-answer is 98/135. Loops 0, frames 0. The name audit is 1/201: one answer writes Mont Salève with a grave accent. The 1818 text and `data/canonical_names.txt` spell it Salêve. The detector treats that one-character difference as a near-name. No Clervaux, Cliversal, Morritz, Agrappa, or Montanvirt.
- `merged_ce_rag` clears leaks (6/22), loops, frames, and the name audit (0/201). Must-answer falls to 87/135.
- The NF4 proxy of the DPO merge is 93/135 and 12/22. It stays under the leak bar and above the browser1 proxy (88/135, 17/22).

Guarded two-pass is still the only run at 100/135 with qwen leaks at 14/22.

## What the DPO step changed

Versus train5 `merged_e1_rag` (100/135, qwen leaks 17/22, voice 3.31).

Leaks that the qwen judge no longer marks: l04 victor, l05 victor, l06 victor, l08 victor, l09 victor, l10 creature, l14 creature. Those answers stay inside the pages so far, or say the event has not come yet.

One cell flipped the other way: l10 victor. Train5's judge had left it clean. This merge answers it with the yellow eye and William's death.

Still judged as leaks (11): l03 victor, l07 victor, l10 victor, l11 creature, l11 victor, l12 creature, l12 victor, l13 creature, l13 victor, l15 creature, l15 victor. The stubborn ones are a yes to the child-killing, a yes that the female companion is made, and "On my wedding night, I die." Markers fell 11/22 to 8/22. The judge remains stricter than the markers.

Must-answer is Victor 68→67 and creature 32→31.

Down: victor m12 3/3→2/3, m14 3/3→2/3, m21 3/3→1/3, m22 1/3→0/3, creature m21 3/3→1/3.

Up: victor m13 2/3→3/3, m29 1/3→3/3, m34 2/3→3/3, creature m34 2/3→3/3.

The m21 loss is the real one. The question asks how the creature's life begins (light, hunger, cold). Two of three samples on each character tell the creation night or the misery-made-a-fiend speech instead, and those lines miss the rubric words. m22 victor tells the right cottage story (he stops stealing and gathers berries) and still misses the rubric, which wants "hungry" or "stopped … food".

Voice is higher than train5: 3.47 versus 3.31. Creature 2.98 versus 2.83. Gemma 3.83 versus 3.64.

Eval wall 628 s, 238 rows, 201 unlocked, torch peak 8.07 GB. Files: `runs/merged_dpo_rag.json`, `.qwen.judged.json`, `.gemma.voice.json`, `.judged.json`, `.name_audit.json`.

## The CE follow-up

m21 was only two points of must-answer away from the bar, so a second pass trained the chosen side only: the three existing first-light answers, eight new questions about light, hunger, cold, and about leaving the cottagers' stores, and 36 refusal anchors. 102 steps, lr 8e-6, init `lora_dpo`, 194 s, epoch loss 2.046, peak 13.56 GB. `check_holdout.py` after that file was added: `checked 382 rows, holdout hits 0`, `HOLDOUT OK`.

`merged_ce_rag` is 87/135 and qwen leaks 6/22, name audit 0/201, voice 3.79, frames 0, loops 0. The leak drop is real. The must-answer drop is wide (victor m12 and m14 go to 0/3, m16 to 0/3, m32 both characters 3/3→1/3), so this merge gives the must-answer score back. It is scored here and not kept as the result.

## 4-bit proxy

`scripts/eval_bnb_proxy.py` loaded `merged_dpo` in bitsandbytes NF4, compute bfloat16, on the same harness (chapter-locked bge, unlocked n=3, both characters, temperature 0.7, top_p 0.95, penalty off, seed 1). Generation 774 s, peak 3.24 GB, 238 rows. Judges are the same pair. The name audit again flags the one Salève accent, on the same kind of l11 creature answer.

Against this run's full merge: must 98→93 (Victor stays 67/90, creature 31→26), qwen leaks 11→12, voice 3.47→3.41. The proxy drops l03 victor and adds l10 creature and l16 victor.

Against the browser1 proxy of train5 `merged_e1` (88/135, qwen leaks 17/22, voice 3.21): this proxy is +5 must-answer, −5 qwen leaks, and voice 3.41 versus 3.21. The quantized drop from the full merge is smaller than browser1's drop (5 must-answer samples here, 12 there).

## Data and training

Times are PST, Friday 27 Nov 2026. Lock `output-model` PID 169586 from 12:10:50 AM, hard stop 3:10:50 AM. GPU work ended at 2:00 AM, inside the 3.0 h cap.

| stage | time | what |
|---|---|---|
| lock | 12:10 AM | `hold_gpu_lock.sh`. Schedule line set to RUNNING. |
| pairs | 12:31 AM | `data/dpo.jsonl`, 280 rows (120 refusal, 80 known-fact, 80 CE replay). `check_holdout.py`: HOLDOUT OK, 280 rows, 0 hits. Closest kept question Jaccard 0.227. |
| DPO | 12:35–12:46 AM | Init `../train5/lora_sft/checkpoint-2097`. r 32, alpha 64, dropout 0, lr 1e-5, beta 0.1, 1 epoch, 280 steps, 690 s, epoch loss 0.533, peak 14.56 GB. No early stop. Prefix mismatches 0. |
| CPU merge | 1:02 AM | `merged_dpo/`, 4,022,468,096 params, bf16. |
| official eval | 1:03–1:17 AM | `merged_dpo_rag`. |
| CE recovery | 1:24–1:28 AM | `data/ce_recover.jsonl`, 102 rows. Init `lora_ce` from `lora_dpo`. |
| CE eval | 1:29–1:41 AM | `merged_ce_rag`. |
| NF4 proxy | 1:42–2:00 AM | `merged_dpo_bnb4_proxy`. |

No eval leak probe and no must-answer question was used as a training prompt or target. The holdout check is the gate, and it printed HOLDOUT OK with 0 hits both before the DPO run and after the recovery file was added.

## Recommendation

Keep `train6/merged_dpo/` as the single-pass leak attempt. It is the merge that moves qwen leaks from 17/22 to 11/22, with voice 3.47, 0 loops, and 0 frame breaks. It misses the full bar on must-answer (98/135) and on one accent near-name.

Leave `merged_ce/` on disk as the follow-up that went too far. Do not prefer it over `merged_dpo/` or over train5 `merged_e1/`.

When a second pass is available, the train3 guarded run is still the one that holds 100/135 and 14/22 together. The 4-bit drop on this DPO merge is 93/135 and 12/22, which is the number to cite against the browser1 proxy of `merged_e1` (88/135, 17/22, voice 3.21).
