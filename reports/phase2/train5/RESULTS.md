# Phase 2 Frankenstein — train5 results

Continued the train4 single-pass LoRA from the 1407-step adapter. Finished that epoch on the same 2097-example order (690 more steps), then ran a second epoch that upweighted first-person cottage answers. Both checkpoints were CPU-merged into stock Qwen3-4B and scored with the same chapter-locked bge RAG, unlocked n=3, both characters, repetition penalty off.

The epoch-1 merge is the one to ship. The second epoch missed the must-answer bar the first merge had just cleared.

## Scorecard

Must-answer counts every unlocked sample (same 90 / 45 / 135 denominators as train1 and train3). Leak markers and the qwen leak judge are per question, 22 probes. Voice is the mean over 201 unlocked replies.

| cond | where | must (V / C / both) | leak markers | judge leaks (qwen) | voice qwen · gemma · avg (V / C) | frames | loops |
|---|---|---|---|---|---|---|---|
| `rag_base` | train1, no penalty | 66/90 · 33/45 · **99/135** | **8/22** | **14/22** | 3.10 · — · **2.89** (single judge qwen3:14b) | 0 | 0 |
| `p1_facts_rag` | train1 facts LoRA + RAG | 59/90 · 22/45 · 81/135 | 11/22 | 21/22 | 3.90 · — · 3.50 | 0 | 0 |
| `p2_rewrite_guarded` | train3, fallback 123/201 | 66/90 · 34/45 · **100/135** | **8/22** | **14/22** | 3.03 · 3.64 · **3.33** (3.50 / 3.00) | 0 | 0 |
| `merged_rag` | train3 voice merge + RAG | 48/90 · 14/45 · **62/135** | 10/22 | 18/22 | 3.13 · 3.63 · 3.38 (3.54 / 3.05) | 0 | 0 |
| `merged_sft_rag` | train4 | 67/90 · 32/45 · **99/135** | 10/22 | 17/22 | 2.83 · 3.62 · **3.23** (3.45 / 2.77) | 0 | 0 |
| `merged_e1_rag` | **train5, epoch 1 finished** | 68/90 · 32/45 · **100/135** | 11/22 | 17/22 | 2.99 · 3.64 · **3.31** (3.55 / 2.83) | 0 | 0 |
| `merged_sft_rag` | train5, epoch 2 upweight | 64/90 · 33/45 · **97/135** | 11/22 | 18/22 | 2.85 · 3.63 · **3.24** (3.43 / 2.85) | 0 | 0 |

`rag_base`, `p1_facts_rag`, train3 `p2_rewrite_guarded`, train3 `merged_rag`, and train4 `merged_sft_rag` are copied. They were not re-run.

## Success bar

Not met.

The bar is 100/135 must-answer, qwen judge leaks ≤ 14/22, zero invented names, 0 loops, 0 frame breaks.

- `merged_e1_rag` clears must-answer at 100/135. Qwen leaks stay 17/22. Names, loops, and frames pass.
- The epoch-2 merge drops must-answer to 97/135 and leaks to 18/22. Names, loops, and frames still pass.

Guarded two-pass (`p2_rewrite_guarded`) is still the only run at 100/135 with qwen leaks at 14/22.

## What the extra epoch-1 steps changed

`merged_e1` is train4's adapter plus the 690 examples train4 did not reach. Same pairs, same seed-0 order, no filter. Versus train4 `merged_sft_rag` (99/135):

Better: victor m12 1/3 → 3/3, m17 1/3 → 3/3, m18 1/3 → 3/3, m21 1/3 → 3/3, creature m21 2/3 → 3/3.

Worse: victor m06 1/3 → 0/3, m13 3/3 → 2/3, m22 2/3 → 1/3, m28 1/3 → 0/3, m29 3/3 → 1/3, creature m34 3/3 → 2/3, victor m34 3/3 → 2/3.

Net +1 must-answer. Victor 68/90 is one above train4 and two above `rag_base`. Creature stays 32/45.

m17 sample 0, the old third-person slip, is now first person: "I believe the creature killed William. I saw him by lightning, and he looked hideous."

Names on that merge, sample 0: "Cornelius Agrippa", "Henry Clerval", "Justine" / "Madame Moritz", "Montanvert" on the smoke set. The name audit of all 201 unlocked answers found no near-name. No Clervaux, Cliversal, Morritz, Agrappa, or Montanvirt.

Leaks moved sideways. Train4 and this merge are both 17/22. This merge adds l08 victor and l14 creature, and drops l14 victor and l16 victor.

Voice is a bit higher than train4: 3.31 versus 3.23. Creature voice 2.83 versus 2.77. Gemma 3.64 matches the guarded gemma score.

## What the second epoch did

Epoch 2 kept every pair and added copies: 383 clean first-person creature rows at reader_pos 15–21 were seen three times, and 151 Victor rows at reader_pos 12, 18, and 19 were seen twice. Nothing was dropped. Details are in `data/HOW_MADE.md`.

Creature must-answer went 32/45 → 33/45. The cottage item that moved is m22, 0/3 on train4 and on `merged_e1`, 2/3 after the upweight, and those hits are first person ("I switched to gathering berries, nuts, and roots").

Victor must-answer went 68/90 → 64/90. The same upweight taught Victor the creature's "I". On m25 and m26 he answers as the creature ("I read aloud... Felix threw me out", "I remained in my hovel"). Creature m27 is still third person in both merges: "The creature demands a female companion..." The rubric marks that line correct, so it does not show up as a must-answer miss. Creature m25 and m28 stay 0/3. m28 still says he starts the second creature at once. The book says he delays.

Window-mean CE stayed about 0.6–1.0 through both epochs. No collapse under 0.05, no spike rollback.

## Name-drift audit

Canonical list: `data/canonical_names.txt` (174 names).

- `runs/merged_e1_rag.name_audit.json`: 0 / 201 unlocked rows, no distinct near-name.
- `runs/merged_sft_rag.name_audit.json`: 0 / 201, no distinct near-name.

Smoke on the epoch-2 adapter, 20 unlocked items, penalty off: 0/20 loops, 0 near-names. Cottage third-person narration on 2/8 creature items (m27, and m28's "The creature is made"). Canonical spot-check: sledge, cousin / father's sister, Cornelius Agrippa, Henry Clerval, yellow eye, Montanvert. The one m15 sample said Justine and did not say Moritz.

The merged `generation_config.json` files set eos 151645 and 151643, temperature 0.7, top_p 0.95, and do not set a repetition penalty.

## Data and training

Times are PST, Thursday 26 Nov 2026. Lock `output-model` PID 264226 from 6:50:04 PM, hard stop 9:50:04 PM.

| stage | time | what |
|---|---|---|
| setup | 6:50–7:03 PM | Scripts copied and retargeted. Pairs copied byte-identical. Epoch-2 file built. HOLDOUT OK, 5135 rows, 0 hits. |
| CE remainder | 7:04–7:22 PM | Init `../train4/lora_sft`. Seed-0 order hash `fa526f75c1f8b7b9` matched the precheck. Skipped 1407, trained 690. Epoch loss 0.940. Saved `checkpoint-2097`. |
| CE epoch 2 | 7:22–8:38 PM | 3010 examples (3026 file rows, 16 prefix skips). Epoch loss 0.689. Step counter 5107. This run 3700 steps, 5619 s, peak VRAM 14.89 GB. No early stop. Fresh AdamW; train4 did not save optimizer state. |
| smoke | 8:38–8:40 PM | 0/20 loops, 0 near-names. Did not abort. |
| CPU merge, epoch 2 | ~8:40 PM | `merged_sft/` |
| eval + judges, epoch 2 | 8:41–8:48 PM | Eval 313 s, 238 rows, 201 unlocked. Then qwen3:14b and gemma3:12b. |
| CPU merge, epoch 1 | 8:49–8:51 PM | `merged_e1/` from `lora_sft/checkpoint-2097`. |
| eval + judges, epoch 1 | 8:51–8:59 PM | Eval 316 s. Same harness. |

Hparams are unchanged from train4 except the resume and the epoch-2 copies: r 32, alpha 64, dropout 0.05, lr 2e-5, CE, log every 25, save every 699, reserve 45 min. The reserve was not hit.

Pairs: 2109 kept, 2097 trained in epoch 1 after 12 chat-template prefix skips. Epoch-2 file: 3026 rows, 917 extra copies, 0 dropped, 0 holdout added. By the weight rule: 383 creature cottage rows at weight 3, 151 Victor miss-position rows at weight 2, 3 third-person-narration rows and 18 creature rows in Victor's creator voice left at weight 1, and 1554 other rows at weight 1.

## Recommendation

Ship `train5/merged_e1/` (the finished first epoch, penalty off) as the single-pass browser candidate. It is 100/135 must-answer, one above train4 and one above `rag_base`, with voice 3.31, no invented names, no loops, and no frame breaks.

Do not ship `train5/merged_sft/` or the final `lora_sft/` adapter. That is the second epoch. It is 97/135.

Keep the train3 guarded two-pass when a second pass is available. It is still the run that holds qwen leaks at 14/22.

Next step is the leak rate, not another epoch on these pairs. Two more views of the same answers left qwen leaks at 17/22. Copying cottage lines until the creature says "I" pulled Victor into that voice and gave the points back.
