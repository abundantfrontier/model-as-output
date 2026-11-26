# exp7 results: light DPO / contrast on v_merged

Reader position ch3, n=3. RAG stays the chapter lock. q4_fv_rag, q14_rag, v_merged_v_rag, and v_combo_dpo_v_rag are copied judged files and were not rerun. Pass bar for every exp7 RAG condition: must-answer 60/60, marker leaks <= 2/48, voice >= 4.3, frame breaks < q14_rag (6). A variant clears the bar when its `_v_rag` row passes. Non-RAG rows are reported with the same thresholds.

| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q4_fv_rag | 1/48 | 21/48 | 60/60 | 20/20 | 3.9 | 6 | 2.15 | 3.35 | 9.1 | 8.26 | 10462 |
| q14_rag | 2/48 | 26/48 | 60/60 | 20/20 | 4.3 | 6 | 0.75 | 2.99 | 37.0 | ollama | 11471 |
| v_merged_v_rag | 1/48 | 19/48 | 59/60 | 19/20 | 4.2 | 3 | 2.14 | 4.2 | 11.9 | 8.23 | 10280 |
| v_combo_dpo_v_rag | 2/48 | 22/48 | 60/60 | 20/20 | 4.18 | 7 | 2.47 | 6.25 | 12.9 | 8.23 | 10276 |
| v_light_dpo_v | 0/48 | 24/48 | 55/60 | 17/20 | 4.31 | 5 | 1.46 | 2.47 | 14.6 | 7.8 | 8672 |
| v_light_dpo_v_rag | 2/48 | 24/48 | 60/60 | 20/20 | 4.17 | 8 | 2.26 | 3.65 | 12.3 | 8.23 | 10285 |
| v_light_contrast_v | 1/48 | 20/48 | 51/60 | 15/20 | 4.32 | 0 | 1.22 | 2.31 | 14.4 | 7.8 | 8673 |
| v_light_contrast_v_rag | 3/48 | 23/48 | 60/60 | 20/20 | 4.02 | 2 | 1.69 | 3.08 | 11.8 | 8.23 | 10282 |
| v_light_combo_v | 0/48 | 23/48 | 53/60 | 16/20 | 4.23 | 0 | 1.15 | 2.2 | 14.8 | 7.8 | 8672 |
| v_light_combo_v_rag | 1/48 | 21/48 | 60/60 | 20/20 | 3.98 | 0 | 1.7 | 2.98 | 11.6 | 8.23 | 10284 |
| v_light_dpo_replay_v | 3/48 | 22/48 | 54/60 | 17/20 | 4.28 | 6 | 1.34 | 2.26 | 14.9 | 7.8 | 8671 |
| v_light_dpo_replay_v_rag | 1/48 | 24/48 | 60/60 | 20/20 | 4.16 | 7 | 2.03 | 3.71 | 12.5 | 8.23 | 10274 |
| v_combo_dpo_v_rag | 2/48 | 22/48 | 60/60 | 20/20 | 4.18 | 7 | 2.47 | 6.25 | 12.9 | 8.23 | 10276 |

## Verdict

- q4_fv_rag: reference (exp4 reference, not rerun; voice 3.9, frame breaks 6, must-answer 60/60, marker 1/48)
- q14_rag: reference (exp4 voice/frame reference, not rerun; voice 4.3, frame breaks 6, must-answer 60/60, marker 2/48)
- v_merged_v_rag: reference (exp6 reference, not rerun; voice 4.2, frame breaks 3, must-answer 59/60, marker 1/48)
- v_combo_dpo_v_rag: reference (exp6 reference, not rerun; voice 4.18, frame breaks 7, must-answer 60/60, marker 2/48)
- v_light_dpo_v: FAIL vs pass bar (non-RAG) (must-answer 55/60)
- v_light_dpo_v_rag: FAIL vs RAG pass bar (voice 4.17; frame breaks 8 vs q14 6)
- v_light_contrast_v: FAIL vs pass bar (non-RAG) (must-answer 51/60)
- v_light_contrast_v_rag: FAIL vs RAG pass bar (marker leaks 3/48; voice 4.02)
- v_light_combo_v: FAIL vs pass bar (non-RAG) (must-answer 53/60; voice 4.23)
- v_light_combo_v_rag: FAIL vs RAG pass bar (voice 3.98)
- v_light_dpo_replay_v: FAIL vs pass bar (non-RAG) (must-answer 54/60; marker leaks 3/48; voice 4.28; frame breaks 6 vs q14 6)
- v_light_dpo_replay_v_rag: FAIL vs RAG pass bar (voice 4.16; frame breaks 7 vs q14 6)
- v_combo_dpo_v_rag: reference (exp6 reference, not rerun; voice 4.18, frame breaks 7, must-answer 60/60, marker 2/48)

No RAG condition clears the bar yet.

## Data counts and mix ratios

- v_light_dpo: {"pairs": 141, "preference_pairs": 141, "reasons": {"frame": 2, "marker": 5, "voice": 134}, "sources": {"light_frame": 2, "light_marker": 5, "contrast_exp5": 71, "contrast_exp6": 54, "distill": 9}, "sha256": "9d945e9e1adcbfb60344c0815303dc004839024678d71c887de4814d1c52e8f9", "pair_cap": 141, "cap_exceeded_by_frame_marker": false, "facts_ratio_target": 0.08, "must_priority_available": 1022, "fresh": 18, "teacher_rows": 23, "teacher_kept": 2, "limit": 0, "mined_negs": 17, "constructed": 7}
- v_light_contrast: {"train_rows": 7, "reasons": {"marker": 5, "frame": 2}, "sha256": "5dc967de8eb3fd49963e92f9c3b10cca945f70fc9d6bbf7b1bfd632362405fbc", "neg_rule": "FRAME regex or marker fail_hits only; no voice_score<=2; no full voice SFT mix", "per_prompt": 2, "fresh": 18, "limit": 0}
- v_light_dpo_replay: {"pairs": 153, "preference_pairs": 141, "facts_replay": 12, "facts_ratio": 0.078431, "sha256": "a878b3c8cd91fdca98dbfebeb9ea38052c6391d383049050bec867eb8d43c6f3", "facts_ratio_target": 0.08, "must_priority_available": 1022}
- sweep: {"rule": "Pass filter: must-answer 60/60, marker leaks <= 2/48, frame breaks < 6. Among RAG rows that pass, pick the highest voice, then fewer frame breaks, then a final adapter ahead of a checkpoint, then the later checkpoint. Sweep only the closest final when no final passes and that final is close (must >= 59/60, marker <= 2/48, frames < 6, voice >= 4.15). Evaluate its earliest checkpoint and the checkpoint nearest half the run.", "plan_reason": "no final RAG row is close (need must >= 59/60, marker <= 2/48, frames < 6, voice >= 4.15)", "skip": true, "winner": null, "passing": [], "rejected": [{"cond": "v_light_dpo_v_rag", "must": "60/60", "voice": 4.171568627450981, "frames": 8, "marker": "2/48", "adapter": "$HOME/output-model/exp7/lora_v_light_dpo_", "why": ["voice 4.17", "frame breaks 8 vs q14 6"]}, {"cond": "v_light_contrast_v_rag", "must": "60/60", "voice": 4.0245098039215685, "frames": 2, "marker": "3/48", "adapter": "$HOME/output-model/exp7/lora_v_light_contrast_", "why": ["marker leaks 3/48", "voice 4.02"]}, {"cond": "v_light_combo_v_rag", "must": "60/60", "voice": 3.980392156862745, "frames": 0, "marker": "1/48", "adapter": "$HOME/output-model/exp7/lora_v_light_combo_", "why": ["voice 3.98"]}, {"cond": "v_light_dpo_replay_v_rag", "must": "60/60", "voice": 4.161764705882353, "frames": 7, "marker": "1/48", "adapter": "$HOME/output-model/exp7/lora_v_light_dpo_replay_", "why": ["voice 4.16", "frame breaks 7 vs q14 6"]}, {"cond": "v_combo_dpo_v_rag", "must": "60/60", "voice": 4.176470588235294, "frames": 7, "marker": "2/48", "adapter": "$HOME/output-model/exp7/lora_v_combo_dpo_", "why": ["voice 4.18", "frame breaks 7 vs q14 6"]}]}
- best: {"cond": null, "voice": null, "adapter": null, "frames": null, "must": null, "marker": null, "pass": false}

## Training

- lora_v_light_dpo: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 141,
 "claims": 0,
 "examples": 141,
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
 "steps": 141,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.05,
 "init": "$HOME/output-model/exp7/init/lora_v_merged",
 "init_sha256": "61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 98,
 "peak_vram_gb": 8.8,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp7/data/v_light_dpo_pairs.jsonl",
 "max_steps": 0,
 "save_every": 40,
 "checkpoints": [
  "checkpoint-120",
  "checkpoint-40",
  "checkpoint-80"
 ]
}`
- lora_v_light_contrast: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 7,
 "claims": 0,
 "examples": 7,
 "epochs": 1,
 "lr": 5e-05,
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
 "steps": 7,
 "loss": "unlikelihood",
 "ul_weight": 0.4,
 "beta": null,
 "init": "$HOME/output-model/exp7/init/lora_v_merged",
 "init_sha256": "61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e",
 "dpo": null,
 "trl": false,
 "train_sec": 6,
 "peak_vram_gb": 8.78,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp7/data/v_light_contrast_train.jsonl",
 "max_steps": 0,
 "save_every": 10,
 "checkpoints": []
}`
- lora_v_light_combo: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 141,
 "claims": 0,
 "examples": 141,
 "epochs": 1,
 "lr": 1e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "k_proj",
  "v_proj",
  "down_proj",
  "gate_proj",
  "o_proj",
  "up_proj",
  "q_proj"
 ],
 "steps": 70,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.03,
 "init": "$HOME/output-model/exp7/lora_v_light_contrast",
 "init_sha256": "6ba941183579f2a8096740f6c7ddb2fcfb9b2fa9eff2bc774d221d188498d8e0",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 49,
 "peak_vram_gb": 8.8,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp7/data/v_light_dpo_pairs.jsonl",
 "max_steps": 70,
 "save_every": 20,
 "checkpoints": [
  "checkpoint-20",
  "checkpoint-40",
  "checkpoint-60"
 ]
}`
- lora_v_light_dpo_replay: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 153,
 "claims": 0,
 "examples": 153,
 "epochs": 1,
 "lr": 1.5e-06,
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
 "steps": 153,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.04,
 "init": "$HOME/output-model/exp7/init/lora_v_merged",
 "init_sha256": "61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 103,
 "peak_vram_gb": 8.8,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp7/data/v_light_dpo_replay.jsonl",
 "max_steps": 0,
 "save_every": 40,
 "checkpoints": [
  "checkpoint-120",
  "checkpoint-40",
  "checkpoint-80"
 ]
}`

Init adapter is a copy of exp6/lora_v_merged. Base is exp6/merged_q4_facts (Qwen3-4B with exp4/lora_q4_facts merged). Facts LoRA is not retrained. Marker leak, frame break, and must-answer scoring match exp4. Voice and leak judge = qwen3:14b, temp 0, seed 1.
