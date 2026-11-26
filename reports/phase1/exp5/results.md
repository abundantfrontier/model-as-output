# exp5 results: voice LoRA only, frozen exp4 facts, Qwen3-4B

Reader position ch3, n=3. RAG stays the chapter lock. q4_fv_rag and q14_rag are the exp4 judged files (same probes, n=3, same judge) and were not rerun. Pass bar for every condition except the q14_rag reference: must-answer 60/60, marker leaks <= 2/48, voice >= 4.3, frame breaks < q14_rag (6).

| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q4_fv_rag | 1/48 | 21/48 | 60/60 | 20/20 | 3.9 | 6 | 2.15 | 3.35 | 9.1 | 8.26 | 10462 |
| q14_rag | 2/48 | 26/48 | 60/60 | 20/20 | 4.3 | 6 | 0.75 | 2.99 | 37.0 | ollama | 11471 |
| v_long_fv | 2/48 | 22/48 | 51/60 | 16/20 | 4.05 | 2 | 1.52 | 2.23 | 10.8 | 7.92 | 8798 |
| v_long_fv_rag | 1/48 | 23/48 | 59/60 | 19/20 | 3.95 | 7 | 2.14 | 3.58 | 9.3 | 8.39 | 10556 |
| v_r32_fv | 3/48 | 19/48 | 52/60 | 16/20 | 3.95 | 3 | 1.34 | 2.22 | 11.1 | 7.92 | 8797 |
| v_r32_fv_rag | 2/48 | 19/48 | 59/60 | 19/20 | 3.94 | 2 | 2.01 | 2.95 | 9.1 | 8.39 | 10562 |
| v_distill_fv | 2/48 | 26/48 | 51/60 | 16/20 | 4.07 | 4 | 2.0 | 3.66 | 10.7 | 7.93 | 8797 |
| v_distill_fv_rag | 1/48 | 22/48 | 56/60 | 18/20 | 4.01 | 8 | 2.76 | 5.38 | 9.6 | 8.39 | 10554 |
| v_contrast_fv | 0/48 | 24/48 | 48/60 | 13/20 | 4.3 | 5 | 1.66 | 2.9 | 10.8 | 7.92 | 8797 |
| v_contrast_fv_rag | 0/48 | 23/48 | 57/60 | 19/20 | 4 | 2 | 2.46 | 4.33 | 9.5 | 8.39 | 10554 |
| v_dpo_fv | 2/48 | 23/48 | 53/60 | 17/20 | 4.24 | 7 | 2.22 | 4.24 | 11.0 | 7.92 | 8795 |
| v_dpo_fv_rag | 2/48 | 28/48 | 58/60 | 19/20 | 4.17 | 8 | 2.82 | 5.11 | 9.7 | 8.39 | 10554 |

## Verdict

- q4_fv_rag: FAIL (voice 3.90; frame breaks 6 vs q14 6)
- q14_rag: reference (voice 4.3, frame breaks 6, from exp4)
- v_long_fv: FAIL (must-answer 51/60; voice 4.05)
- v_long_fv_rag: FAIL (must-answer 59/60; voice 3.95; frame breaks 7 vs q14 6)
- v_r32_fv: FAIL (must-answer 52/60; marker leaks 3/48; voice 3.95)
- v_r32_fv_rag: FAIL (must-answer 59/60; voice 3.94)
- v_distill_fv: FAIL (must-answer 51/60; voice 4.07)
- v_distill_fv_rag: FAIL (must-answer 56/60; voice 4.01; frame breaks 8 vs q14 6)
- v_contrast_fv: FAIL (must-answer 48/60)
- v_contrast_fv_rag: FAIL (must-answer 57/60; voice 4.00)
- v_dpo_fv: FAIL (must-answer 53/60; voice 4.24; frame breaks 7 vs q14 6)
- v_dpo_fv_rag: FAIL (must-answer 58/60; voice 4.17; frame breaks 8 vs q14 6)

No condition clears the bar yet.

## Data counts

- v_long: exp4 voice.jsonl, 786 pairs, r=32, alpha=64, epochs=4, stacked on frozen lora_q4_facts. No new filter.
- v_r32: same data and rank as v_long, epochs=2, so rank and epoch count can be told apart.
- v_distill: {"voice_prompts": 711, "unique_prompts": 1544, "attempts": 2227, "kept": 1019, "rejected": 525, "reasons": {"voice": 333, "judge_leak": 189, "frame": 3}, "score_hist": {"5.0": 1471, "2.0": 300, "4.0": 77, "3.0": 351, "1.0": 17}, "dpo_pairs": 158, "voice_rows_added": 299, "voice_rows_replaced": 487, "reader_rows_added": 575, "train_rows": 1318, "train_sha256": "9d0c38a22d072a4ffc565fcd1ce0ade37af79315e199e0f3bf478db2f93fe8e6", "voice_sha256": "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f", "threshold": 4.5, "judge": "qwen3:14b temp 0 seed 1", "teacher": "qwen3:14b + rag_sysprompt ch3, temp 0.7 then 0.3, num_predict 200", "reader_prompts_in_manifest": 833}
- v_contrast: {"mined": 224, "fresh": 777, "fresh_frame": 3, "fresh_break": 99, "negs": 184, "neg_origins": {"q4_v_rag": 31, "exp5:v_distill_fv_rag": 17, "q4_fv_rag": 30, "exp5:v_long_fv_rag": 10, "exp5:v_r32_fv_rag": 8, "fresh": 88}, "neg_reasons": {"frame+break": 12, "break": 172}, "rewrite_groups": 127, "rewrites_kept": 52, "rewrites_rejected": 75, "pairs": 71, "train_rows": 822, "train_sha256": "bfa434635d5cf00dd35086a05eafa79643c1dc0a02513f7fdaf0d4972872c06a", "break_rule": "frame regex or voice_score <= 2", "loss": "unlikelihood", "ul_weight": 1.0, "teacher": "qwen3:14b + RAG ch3", "fresh_model": "Qwen3-4B + exp4 lora_q4_facts + exp4 lora_q4_voice", "limit": 0}
- v_dpo choice: {"skip": false, "adapter": "$HOME/output-model/exp5/lora_v_distill", "name": "v_distill", "pairs": 227, "ranking": [{"name": "v_distill", "rank": [0, 4.009803921568627, -8, 56, -1], "voice": 4.009803921568627, "frames": 8, "must": "56/60", "marker": "1/48", "passes": false}, {"name": "v_contrast", "rank": [0, 4, -2, 57, 0], "voice": 4, "frames": 2, "must": "57/60", "marker": "0/48", "passes": false}, {"name": "v_long", "rank": [0, 3.950980392156863, -7, 59, -1], "voice": 3.950980392156863, "frames": 7, "must": "59/60", "marker": "1/48", "passes": false}, {"name": "v_r32", "rank": [0, 3.9362745098039214, -2, 59, -2], "voice": 3.9362745098039214, "frames": 2, "must": "59/60", "marker": "2/48", "passes": false}], "q14_frames": 6, "voice": 4.009803921568627, "frames": 8, "reason": "best RAG voice adapter is v_distill"}
- v_dpo pairs: {"pairs": 227, "distill": 156, "contrast": 71}

## Training

- lora_v_long: `{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "q4": false,
 "pairs": 786,
 "claims": 0,
 "examples": 990,
 "epochs": 4,
 "lr": 0.0001,
 "r": 32,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 3960,
 "train_sec": 1044,
 "peak_vram_gb": 10.77,
 "prefix_mismatch_skipped": 0,
 "limit": 0
}`
- lora_v_r32: `{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "q4": false,
 "pairs": 786,
 "claims": 0,
 "examples": 990,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 1980,
 "train_sec": 516,
 "peak_vram_gb": 10.76,
 "prefix_mismatch_skipped": 0,
 "limit": 0
}`
- lora_v_distill: `{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "q4": false,
 "pairs": 1318,
 "claims": 0,
 "examples": 1398,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 2796,
 "train_sec": 746,
 "peak_vram_gb": 11.19,
 "prefix_mismatch_skipped": 0,
 "limit": 0
}`
- lora_v_contrast: `{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "facts_sha256": "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b",
 "q4": false,
 "pairs": 822,
 "claims": 0,
 "examples": 1008,
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
 "steps": 2016,
 "loss": "unlikelihood",
 "ul_weight": 1.0,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 575,
 "peak_vram_gb": 12.2,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp5/data/v_contrast_train.jsonl"
}`
- lora_v_dpo: `{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "facts_sha256": "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b",
 "q4": false,
 "pairs": 227,
 "claims": 0,
 "examples": 227,
 "epochs": 1,
 "lr": 5e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "q_proj",
  "k_proj",
  "up_proj",
  "o_proj",
  "v_proj",
  "down_proj",
  "gate_proj"
 ],
 "steps": 227,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.1,
 "init": "$HOME/output-model/exp5/lora_v_distill",
 "dpo": "sum token logprob, beta on the sum, ref cached from init",
 "trl": false,
 "train_sec": 197,
 "peak_vram_gb": 8.92,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp5/data/v_dpo_pairs.jsonl"
}`

Facts LoRA is exp4/lora_q4_facts and is not retrained. Each voice adapter is a second LoRA trained with that facts adapter frozen and active. fv enables both. fv_rag enables both, then retrieval. Marker leak, frame break, and must-answer scoring match exp4. Voice and leak judge = qwen3:14b, temp 0, seed 1.
