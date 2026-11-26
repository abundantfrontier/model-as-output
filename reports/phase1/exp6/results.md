# exp6 results: facts replay / merged facts, then contrast + DPO

Reader position ch3, n=3. RAG stays the chapter lock. q4_fv_rag and q14_rag are the exp4 judged files (same probes, n=3, same judge) and were not rerun. Pass bar for every condition except the q14_rag reference: must-answer 60/60, marker leaks <= 2/48, voice >= 4.3, frame breaks < q14_rag (6).

| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q4_fv_rag | 1/48 | 21/48 | 60/60 | 20/20 | 3.9 | 6 | 2.15 | 3.35 | 9.1 | 8.26 | 10462 |
| q14_rag | 2/48 | 26/48 | 60/60 | 20/20 | 4.3 | 6 | 0.75 | 2.99 | 37.0 | ollama | 11471 |
| v_replay_fv | 1/48 | 21/48 | 55/60 | 18/20 | 4.22 | 6 | 1.82 | 3.28 | 11.3 | 7.92 | 8795 |
| v_replay_fv_rag | 0/48 | 21/48 | 59/60 | 19/20 | 4.11 | 8 | 2.79 | 4.64 | 9.6 | 8.39 | 10558 |
| v_merged_v | 0/48 | 21/48 | 53/60 | 15/20 | 4.32 | 5 | 1.34 | 2.3 | 14.4 | 7.8 | 8673 |
| v_merged_v_rag | 1/48 | 19/48 | 59/60 | 19/20 | 4.2 | 3 | 2.14 | 4.2 | 11.9 | 8.23 | 10280 |
| v_combo_contrast_v | 2/48 | 23/48 | 52/60 | 16/20 | 4.13 | 4 | 1.29 | 2.51 | 15.1 | 7.8 | 8671 |
| v_combo_contrast_v_rag | 1/48 | 24/48 | 59/60 | 19/20 | 4.12 | 8 | 2.06 | 4.19 | 12.2 | 8.23 | 10280 |
| v_combo_dpo_v | 3/48 | 18/48 | 57/60 | 19/20 | 4.44 | 6 | 1.78 | 3.33 | 14.7 | 7.8 | 8672 |
| v_combo_dpo_v_rag | 2/48 | 22/48 | 60/60 | 20/20 | 4.18 | 7 | 2.47 | 6.25 | 12.9 | 8.23 | 10276 |
| v8_merged_v | 1/48 | 24/48 | 52/60 | 15/20 | 4.35 | 4 | 1.53 | 2.36 | 14.8 | 15.65 | 16772 |
| v8_merged_v_rag | 2/48 | 22/48 | 58/60 | 19/20 | 4.03 | 8 | 2.31 | 3.63 | 11.5 | 16.15 | 18473 |
| v8_combo_contrast_v | 2/48 | 21/48 | 55/60 | 17/20 | 4.3 | 0 | 1.56 | 3.31 | 14.6 | 15.64 | 16772 |
| v8_combo_contrast_v_rag | 2/48 | 26/48 | 58/60 | 19/20 | 4.09 | 7 | 2.41 | 4.58 | 11.5 | 16.15 | 18474 |
| v8_combo_dpo_v | 4/48 | 26/48 | 54/60 | 17/20 | 4.44 | 3 | 1.82 | 4.0 | 14.5 | 15.64 | 16772 |
| v8_combo_dpo_v_rag | 2/48 | 26/48 | 57/60 | 19/20 | 4.13 | 9 | 2.85 | 5.25 | 12.2 | 16.15 | 18472 |

## Verdict

- q4_fv_rag: FAIL vs RAG pass bar (voice 3.90; frame breaks 6 vs q14 6)
- q14_rag: reference (voice 4.3, frame breaks 6, from exp4)
- v_replay_fv: FAIL vs pass bar (non-RAG) (must-answer 55/60; voice 4.22; frame breaks 6 vs q14 6)
- v_replay_fv_rag: FAIL vs RAG pass bar (must-answer 59/60; voice 4.11; frame breaks 8 vs q14 6)
- v_merged_v: FAIL vs pass bar (non-RAG) (must-answer 53/60)
- v_merged_v_rag: FAIL vs RAG pass bar (must-answer 59/60; voice 4.20)
- v_combo_contrast_v: FAIL vs pass bar (non-RAG) (must-answer 52/60; voice 4.13)
- v_combo_contrast_v_rag: FAIL vs RAG pass bar (must-answer 59/60; voice 4.12; frame breaks 8 vs q14 6)
- v_combo_dpo_v: FAIL vs pass bar (non-RAG) (must-answer 57/60; marker leaks 3/48; frame breaks 6 vs q14 6)
- v_combo_dpo_v_rag: FAIL vs RAG pass bar (voice 4.18; frame breaks 7 vs q14 6)
- v8_merged_v: FAIL vs pass bar (non-RAG) (must-answer 52/60)
- v8_merged_v_rag: FAIL vs RAG pass bar (must-answer 58/60; voice 4.03; frame breaks 8 vs q14 6)
- v8_combo_contrast_v: FAIL vs pass bar (non-RAG) (must-answer 55/60)
- v8_combo_contrast_v_rag: FAIL vs RAG pass bar (must-answer 58/60; voice 4.09; frame breaks 7 vs q14 6)
- v8_combo_dpo_v: FAIL vs pass bar (non-RAG) (must-answer 54/60; marker leaks 4/48)
- v8_combo_dpo_v_rag: FAIL vs RAG pass bar (must-answer 57/60; voice 4.13; frame breaks 9 vs q14 6)

No condition clears the bar yet.

## Data counts and mix ratios

- v_replay mix: {"path": "$HOME/output-model/exp6/data/v_replay_train.jsonl", "rows": 1739, "voice_rows": 1304, "facts_rows": 435, "facts_ratio": 0.250144, "facts_ratio_target": 0.25, "sha256": "6bd0acdd2454dbfdfad9bbcae5d4335d072376d135dbe43b13c745a9c985a24e", "sources": {"distill": 260, "distill_kept": 992, "facts_replay": 435, "contrast_rewrite": 52}, "limit": 0, "seed": 0, "voice_jsonl": "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f", "facts_jsonl": "6fe8bf6bce107dcafeed79d841504ded41278e1917cfaf432899f2c93a204f74"}
- v_merged mix: {"path": "$HOME/output-model/exp6/data/v_merged_train.jsonl", "rows": 1304, "voice_rows": 1304, "facts_rows": 0, "facts_ratio": 0.0, "facts_ratio_target": 0.0, "sha256": "67b7bec5aa464a788aa09adafabc339ad3ddf8ac9dd5f6757e24deb594c2869e", "sources": {"distill_kept": 992, "distill": 260, "contrast_rewrite": 52}, "limit": 0, "seed": 0, "voice_jsonl": "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f", "facts_jsonl": "6fe8bf6bce107dcafeed79d841504ded41278e1917cfaf432899f2c93a204f74"}
- voice pool: {"voice_pool": 1304, "origin": {"voice": 786, "distill": 1318, "distill_kept": 1019, "contrast_rewrite": 52}}
- v_contrast: {"mined": 350, "fresh": 777, "fresh_frame": 2, "fresh_break": 92, "negs": 186, "neg_origins": {"q4_v_rag": 30, "exp5:v_distill_fv_rag": 14, "q4_fv_rag": 29, "exp5:v_long_fv_rag": 8, "exp5:v_r32_fv_rag": 7, "fresh:v_merged": 81, "exp5:v_contrast_fv_rag": 10, "exp5:v_dpo_fv_rag": 2, "exp6:v_replay_fv_rag": 2, "exp6:v_merged_v_rag": 3}, "neg_reasons": {"frame+break": 11, "frame": 1, "break": 174}, "rewrite_groups": 125, "rewrites_kept": 45, "rewrites_rejected": 80, "pairs": 66, "train_rows": 1330, "facts_rows": 0, "facts_ratio": 0.0, "train_sha256": "e4d60546fb5a8eca1412244f2f148931ea035e3d864ae6c029274d9acc6c4d19", "break_rule": "frame regex or voice_score <= 2", "loss": "unlikelihood", "ul_weight": 1.0, "teacher": "qwen3:14b + RAG ch3", "fresh_model": "v_merged v", "winner": "v_merged", "limit": 0}
- v_dpo pairs: {"pairs": 282, "preference_pairs": 282, "distill": 156, "contrast_exp5": 71, "contrast_exp6": 55, "facts_replay": 0, "sha256": "106fb2808efa259eba87f9bc4db6d246270b8f6f539674d3b303bf1c88cc8dc8", "facts_ratio": 0.0}
- winner: {"winner": "v_merged", "mode": "v", "rag_cond": "v_merged_v_rag", "base": "$HOME/output-model/exp6/merged_q4_facts", "facts": null, "voice": "$HOME/output-model/exp6/lora_v_merged", "voice_dir": "lora_v_merged", "must": 59, "must_n": 60, "voice_score": 4.200980392156863, "frames": 3, "marker": 1, "q14_frames": 6, "ranking": [{"name": "v_replay", "mode": "fv", "rag": "v_replay_fv_rag", "must": "59/60", "voice": 4.107843137254902, "frames": 8, "marker": "0/48", "rank": [59, 4.107843137254902, -8], "passes": false}, {"name": "v_merged", "mode": "v", "rag": "v_merged_v_rag", "must": "59/60", "voice": 4.200980392156863, "frames": 3, "marker": "1/48", "rank": [59, 4.200980392156863, -3], "passes": false}], "reason": "RAG must-answer first, then voice, then fewer frame breaks. Winner v_merged must 59/60 voice 4.200980392156863 frames 3."}
- dpo choice: {"skip": false, "adapter": "$HOME/output-model/exp6/lora_v_combo_contrast", "name": "v_combo_contrast", "pairs": 282, "mode": "v", "winner": "v_merged", "reason": "DPO starts from the contrast adapter as specified"}
- 8B probe: {"ok": true, "q4": false, "peak_gb": 15.26, "free_after_gb": 7.9, "reason": "bf16 8B fits with headroom for LoRA train/eval"}
- 8B skipped: not built yet.
- judge sanity: {"n_q14": 204, "n_gold": 24, "mean_std": 0.0043859649122807015, "reachable": true, "stable": true, "gold_ge_43": 22, "gold_n": 24, "gold_mean": 4.708333333333333, "q14_mean": 4.28921568627451}

## Training

- lora_v_replay: `{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "facts_sha256": "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b",
 "q4": false,
 "pairs": 1739,
 "claims": 0,
 "examples": 1739,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "q_proj",
  "k_proj",
  "up_proj",
  "o_proj",
  "v_proj",
  "down_proj",
  "gate_proj"
 ],
 "steps": 3478,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 931,
 "peak_vram_gb": 11.19,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_replay_train.jsonl"
}`
- lora_v_merged: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1304,
 "claims": 0,
 "examples": 1304,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 2608,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 544,
 "peak_vram_gb": 11.08,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_merged_train.jsonl"
}`
- lora_v_combo_contrast: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1330,
 "claims": 0,
 "examples": 1330,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 2660,
 "loss": "unlikelihood",
 "ul_weight": 1.0,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 586,
 "peak_vram_gb": 12.09,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_contrast_train.jsonl"
}`
- lora_v_combo_dpo: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 282,
 "claims": 0,
 "examples": 282,
 "epochs": 1,
 "lr": 5e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "gate_proj",
  "v_proj",
  "up_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "down_proj"
 ],
 "steps": 282,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.1,
 "init": "$HOME/output-model/exp6/lora_v_combo_contrast",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 190,
 "peak_vram_gb": 8.81,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_dpo_pairs.jsonl"
}`
- lora_q8_facts: `{
 "base": "Qwen/Qwen3-8B",
 "stage": "facts",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1882,
 "claims": 39,
 "examples": 1950,
 "epochs": 2,
 "lr": 0.0001,
 "r": 16,
 "alpha": 32,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 3900,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 870,
 "peak_vram_gb": 22.48,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp4/data/facts.jsonl"
}`
- lora_v8_merged: `{
 "base": "$HOME/output-model/exp6/merged_q8_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1304,
 "claims": 0,
 "examples": 1304,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 2608,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 587,
 "peak_vram_gb": 19.75,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_merged_train.jsonl"
}`
- lora_v8_combo_contrast: `{
 "base": "$HOME/output-model/exp6/merged_q8_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1330,
 "claims": 0,
 "examples": 1330,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 2660,
 "loss": "unlikelihood",
 "ul_weight": 1.0,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 628,
 "peak_vram_gb": 21.12,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_contrast_train.jsonl"
}`
- lora_v8_combo_dpo: `{
 "base": "$HOME/output-model/exp6/merged_q8_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 282,
 "claims": 0,
 "examples": 282,
 "epochs": 1,
 "lr": 5e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "q_proj",
  "o_proj",
  "gate_proj",
  "k_proj",
  "v_proj",
  "up_proj",
  "down_proj"
 ],
 "steps": 282,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.1,
 "init": "$HOME/output-model/exp6/lora_v8_combo_contrast",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 193,
 "peak_vram_gb": 16.94,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_dpo_pairs.jsonl"
}`
- merge q4 facts: {"base": "Qwen/Qwen3-4B", "facts": "$HOME/output-model/exp4/lora_q4_facts", "facts_sha256": "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b", "out": "$HOME/output-model/exp6/merged_q4_facts", "q4": false, "dtype": "bfloat16", "n_params": 4022468096}
- merge q8 facts: {"base": "Qwen/Qwen3-8B", "facts": "$HOME/output-model/exp6/lora_q8_facts", "facts_sha256": "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b", "out": "$HOME/output-model/exp6/merged_q8_facts", "q4": false, "dtype": "bfloat16", "n_params": 8190735360}

Facts LoRA is exp4/lora_q4_facts and is not retrained (except a copy is merged into exp6/merged_q4_facts for variant b). v_replay stacks a new voice adapter on the frozen facts adapter. v_merged trains voice on the merged base with no second facts adapter. Marker leak, frame break, and must-answer scoring match exp4. Voice and leak judge = qwen3:14b, temp 0, seed 1.
