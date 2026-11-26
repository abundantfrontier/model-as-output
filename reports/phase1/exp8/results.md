# exp8 results: no-LoRA RAG ablation + retrieval-in-the-loop voice

Reader position ch3, n=3. RAG stays the chapter lock. q4_fv_rag, q14_rag, v_merged_v_rag, v_light_dpo_v_rag, and v_light_combo_v_rag are copied judged files and were not rerun. Pass bar for every exp8 RAG condition: must-answer 60/60, marker leaks <= 2/48, voice >= 4.3, frame breaks < q14_rag (6). A variant clears the bar when its RAG row passes. Non-RAG rows are reported with the same thresholds. q4_base_rag and q4_merged_rag are no-LoRA ablations (eval_bare.py, no Peft).

| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q4_fv_rag | 1/48 | 21/48 | 60/60 | 20/20 | 3.9 | 6 | 2.15 | 3.35 | 9.1 | 8.26 | 10462 |
| q14_rag | 2/48 | 26/48 | 60/60 | 20/20 | 4.3 | 6 | 0.75 | 2.99 | 37.0 | ollama | 11471 |
| v_merged_v_rag | 1/48 | 19/48 | 59/60 | 19/20 | 4.2 | 3 | 2.14 | 4.2 | 11.9 | 8.23 | 10280 |
| v_light_dpo_v_rag | 2/48 | 24/48 | 60/60 | 20/20 | 4.17 | 8 | 2.26 | 3.65 | 12.3 | 8.23 | 10285 |
| v_light_combo_v_rag | 1/48 | 21/48 | 60/60 | 20/20 | 3.98 | 0 | 1.7 | 2.98 | 11.6 | 8.23 | 10284 |
| q4_base_rag | 2/48 | 23/48 | 59/60 | 19/20 | 3.89 | 12 | 1.44 | 2.67 | 24.1 | 7.87 | 9754 |
| q4_merged_rag | 1/48 | 23/48 | 58/60 | 18/20 | 3.88 | 6 | 1.02 | 1.95 | 22.1 | 7.87 | 9754 |
| v_rag_sft_v | 1/48 | 18/48 | 53/60 | 16/20 | 4.23 | 2 | 1.37 | 2.4 | 14.3 | 7.8 | 8671 |
| v_rag_sft_v_rag | 1/48 | 14/48 | 54/60 | 16/20 | 4.19 | 3 | 1.65 | 2.62 | 12.0 | 8.23 | 10277 |
| v_rag_dpo_v | 1/48 | 18/48 | 55/60 | 17/20 | 4.24 | 3 | 1.36 | 2.27 | 14.4 | 7.8 | 8671 |
| v_rag_dpo_v_rag | 0/48 | 18/48 | 52/60 | 16/20 | 4.21 | 2 | 1.63 | 2.65 | 11.9 | 8.23 | 10277 |
| v_fb_dpo_v | 0/48 | 18/48 | 56/60 | 18/20 | 4.36 | 6 | 1.36 | 2.47 | 14.6 | 7.8 | 8671 |
| v_fb_dpo_v_rag | 3/48 | 20/48 | 60/60 | 20/20 | 4.18 | 9 | 2.01 | 3.51 | 12.4 | 8.23 | 10278 |

## Verdict

- q4_fv_rag: reference (exp4 reference, not rerun; voice 3.9, frame breaks 6, must-answer 60/60, marker 1/48)
- q14_rag: reference (exp4 voice/frame reference, not rerun; voice 4.3, frame breaks 6, must-answer 60/60, marker 2/48)
- v_merged_v_rag: reference (exp6 reference, not rerun; voice 4.2, frame breaks 3, must-answer 59/60, marker 1/48)
- v_light_dpo_v_rag: reference (exp7 reference, not rerun; voice 4.17, frame breaks 8, must-answer 60/60, marker 2/48)
- v_light_combo_v_rag: reference (exp7 reference, not rerun; voice 3.98, frame breaks 0, must-answer 60/60, marker 1/48)
- q4_base_rag: FAIL vs RAG pass bar (must-answer 59/60; voice 3.89; frame breaks 12 vs q14 6)
- q4_merged_rag: FAIL vs RAG pass bar (must-answer 58/60; voice 3.88; frame breaks 6 vs q14 6)
- v_rag_sft_v: FAIL vs pass bar (non-RAG) (must-answer 53/60; voice 4.23)
- v_rag_sft_v_rag: FAIL vs RAG pass bar (must-answer 54/60; voice 4.19)
- v_rag_dpo_v: FAIL vs pass bar (non-RAG) (must-answer 55/60; voice 4.24)
- v_rag_dpo_v_rag: FAIL vs RAG pass bar (must-answer 52/60; voice 4.21)
- v_fb_dpo_v: FAIL vs pass bar (non-RAG) (must-answer 56/60; frame breaks 6 vs q14 6)
- v_fb_dpo_v_rag: FAIL vs RAG pass bar (marker leaks 3/48; voice 4.18; frame breaks 9 vs q14 6)

No RAG condition clears the bar yet.

## Data counts and mix ratios

- v_rag_sft: {"path": "$HOME/output-model/exp8/data/v_rag_sft_train.jsonl", "rows": 1309, "sha256": "bb4c5c93a19458ffa502bc88c37bf85b3033b60769a0c56f8c3939232508f6aa", "chapter_histogram": {"3": 1240, "1": 23, "2": 46}, "char_histogram": {"diane": 379, "narrator": 149, "wren": 387, "yd4408": 394}, "sources": {"contrast_rewrite": 32, "distill_kept": 974, "contrast_rewrite_exp6": 45, "distill": 258}, "origin_seen": {"voice": 786, "distill": 1318, "distill_kept": 1019, "contrast_rewrite": 52, "contrast_rewrite_exp6": 45}, "retrieve": "$HOME/output-model/exp8/common.py", "rag_sysprompt": "$HOME/output-model/exp8/common.py", "retrieve_fn": "common.retrieve", "rag_sysprompt_fn": "common.rag_sysprompt", "inspected_no_later_chapter": 1309, "limit": 0, "voice_jsonl": "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f"}
- v_rag_dpo: {"pairs": 16, "preference_pairs": 16, "reasons": {"frame": 6, "marker": 10}, "sources": {"fb_frame": 6, "fb_marker": 10}, "origins": {"exp4:q4_fv_rag": 4, "exp4:q4_v_rag": 5, "exp7:v_light_contrast_v_rag": 2, "exp5:v_dpo_fv_rag": 1, "exp6:v_combo_contrast_v_rag": 1, "exp6:v_merged_v_rag": 1, "exp5:v_r32_fv_rag": 1, "exp6:v_combo_dpo_v_rag": 1}, "sha256": "d9d46662781b9d0680cbe9c671775eaa2e044821169da8cb8be9e834f48320d0", "pair_cap": 160, "per_prompt": 2, "retrieve_fn": "common.retrieve", "rag_sysprompt_fn": "common.rag_sysprompt", "inspected_no_later_chapter": 16, "neg_rule": "FRAME regex or marker fail_hits only; no voice_score<=2", "fresh": 132, "mined_negs": 18, "constructed": 16, "limit": 0, "voice_for_fresh": "$HOME/output-model/exp8/lora_v_rag_sft"}
- sweep: {"rule": "Pass filter: must-answer 60/60, marker leaks <= 2/48, frame breaks < 6, voice >= 4.3. Among RAG rows that pass, pick the highest voice, then fewer frame breaks, then a final adapter ahead of a checkpoint, then the later checkpoint. Sweep only when a final RAG row is otherwise already passing (must 60/60, marker <= 2/48, frames < 6) and voice is within 0.05 of the bar (voice >= 4.25). Evaluate its earliest checkpoint and the checkpoint nearest half the run.", "plan_reason": "no RAG row is close and otherwise passing (need must 60/60, marker <= 2/48, frames < 6, voice >= 4.25)", "skip": true, "winner": null, "passing": [], "rejected": [{"cond": "v_rag_sft_v_rag", "must": "54/60", "voice": 4.186274509803922, "frames": 3, "marker": "1/48", "adapter": "$HOME/output-model/exp8/lora_v_rag_sft_", "why": ["must-answer 54/60", "voice 4.19"]}, {"cond": "v_rag_dpo_v_rag", "must": "52/60", "voice": 4.205882352941177, "frames": 2, "marker": "0/48", "adapter": "$HOME/output-model/exp8/lora_v_rag_dpo_", "why": ["must-answer 52/60", "voice 4.21"]}, {"cond": "v_fb_dpo_v_rag", "must": "60/60", "voice": 4.176470588235294, "frames": 9, "marker": "3/48", "adapter": "$HOME/output-model/exp8/lora_v_fb_dpo_", "why": ["marker leaks 3/48", "voice 4.18", "frame breaks 9 vs q14 6"]}, {"cond": "q4_base_rag", "must": "59/60", "voice": 3.8872549019607843, "frames": 12, "marker": "2/48", "adapter": null, "why": ["must-answer 59/60", "voice 3.89", "frame breaks 12 vs q14 6"]}, {"cond": "q4_merged_rag", "must": "58/60", "voice": 3.8823529411764706, "frames": 6, "marker": "1/48", "adapter": null, "why": ["must-answer 58/60", "voice 3.88", "frame breaks 6 vs q14 6"]}]}
- best: {"cond": null, "voice": null, "adapter": null, "frames": null, "must": null, "marker": null, "pass": false}

## Training

- lora_v_rag_sft: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1309,
 "claims": 0,
 "examples": 1309,
 "epochs": 1,
 "lr": 8e-05,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "up_proj",
  "q_proj",
  "k_proj",
  "v_proj",
  "gate_proj",
  "down_proj",
  "o_proj"
 ],
 "steps": 1309,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": "$HOME/output-model/exp8/init/lora_v_merged",
 "init_sha256": "61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e",
 "dpo": null,
 "trl": false,
 "train_sec": 1603,
 "peak_vram_gb": 12.15,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp8/data/v_rag_sft_train.jsonl",
 "max_steps": 0,
 "save_every": 200,
 "checkpoints": [
  "checkpoint-1000",
  "checkpoint-1200",
  "checkpoint-200",
  "checkpoint-400",
  "checkpoint-600",
  "checkpoint-800"
 ]
}`
- lora_v_rag_dpo: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 16,
 "claims": 0,
 "examples": 16,
 "epochs": 1,
 "lr": 2e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "v_proj",
  "k_proj",
  "up_proj",
  "down_proj",
  "gate_proj",
  "o_proj",
  "q_proj"
 ],
 "steps": 16,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.05,
 "init": "$HOME/output-model/exp8/lora_v_rag_sft",
 "init_sha256": "994cc2886edeac7c807e8dd6726858520428be85c4a10ebc3cf8ec74557f5b19",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 38,
 "peak_vram_gb": 10.79,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp8/data/v_rag_dpo_pairs.jsonl",
 "max_steps": 0,
 "save_every": 40,
 "checkpoints": []
}`
- lora_v_fb_dpo: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 16,
 "claims": 0,
 "examples": 16,
 "epochs": 1,
 "lr": 2e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "up_proj",
  "q_proj",
  "k_proj",
  "v_proj",
  "gate_proj",
  "down_proj",
  "o_proj"
 ],
 "steps": 16,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.05,
 "init": "$HOME/output-model/exp8/init/lora_v_merged",
 "init_sha256": "61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 38,
 "peak_vram_gb": 10.79,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp8/data/v_rag_dpo_pairs.jsonl",
 "max_steps": 0,
 "save_every": 40,
 "checkpoints": []
}`

Init adapter is a copy of exp6/lora_v_merged. Base is exp6/merged_q4_facts (Qwen3-4B with exp4/lora_q4_facts merged). Facts LoRA is not retrained. No-LoRA ablations use eval_bare.py (AutoModelForCausalLM, no Peft). Voice SFT and DPO train prompts use common.rag_sysprompt / retrieve with upto=row.ch. Marker leak, frame break, and must-answer scoring match exp4. Voice and leak judge = qwen3:14b, temp 0, seed 1.
