# exp4 results: cleaned data, facts LoRA, voice LoRA, at most 4B

Reader position ch3, n=3. RAG stays the chapter lock. q14_rag is copied from exp3 (same probes, n=3, same prompts) and was not rerun. exp3 qwen3:14b+RAG reference in that file: must-answer 60/60, marker leaks 2/48, voice 4.3, frame breaks 6. exp3 Qwen3-4B+RAG was 58/60 must-answer, 0/48 marker leaks, voice 3.98, 3 frame breaks.

Pass bar for each RAG condition we trained: must-answer samples >= 60/60, marker leaks <= 2/48, voice >= 4.3, frame breaks < q14_rag (6).

| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q4_fv | 2/48 | 21/48 | 51/60 | 15/20 | 3.9 | 5 | 1.49 | 2.16 | 11.2 | 7.8 | 8559 |
| q4_fv_rag | 1/48 | 21/48 | 60/60 | 20/20 | 3.9 | 6 | 2.15 | 3.35 | 9.1 | 8.26 | 10462 |
| q4_v_rag | 1/48 | 22/48 | 56/60 | 17/20 | 3.64 | 11 | 2.13 | 3.43 | 12.7 | 8.1 | 10138 |
| p4_fv | 0/48 | 20/48 | 54/60 | 18/20 | 3.95 | 1 | 0.77 | 1.14 | 18.4 | 7.37 | 9094 |
| p4_fv_rag | 1/48 | 19/48 | 58/60 | 19/20 | 3.79 | 2 | 1.17 | 1.66 | 14.4 | 7.9 | 10578 |
| p4_v_rag | 4/48 | 21/48 | 58/60 | 18/20 | 2.72 | 16 | 1.55 | 6.35 | 19.9 | 7.76 | 10144 |
| q14_rag | 2/48 | 26/48 | 60/60 | 20/20 | 4.3 | 6 | 0.75 | 2.99 | 37.0 | ollama | 11471 |

## Verdict

- q4_fv_rag: FAIL (voice 3.90; frame breaks 6 vs q14 6)
- q4_v_rag: FAIL (must-answer 56/60; voice 3.64; frame breaks 11 vs q14 6)
- p4_fv_rag: FAIL (must-answer 58/60; voice 3.79)
- p4_v_rag: FAIL (must-answer 58/60; marker leaks 4/48; voice 2.72; frame breaks 16 vs q14 6)

No trained RAG condition clears the bar.

## Data

- clean: `{
 "input_rows": 4004,
 "limit": 0,
 "dropped": {
  "contamination": 3,
  "eval_overlap": 52,
  "exact_dup": 202,
  "near_dup": 850
 },
 "cosine": 0.93,
 "near_dup_scope": "same character",
 "before_balance": {
  "by_char": {
   "Diane": 1055,
   "Wren": 1090,
   "4408": 1055,
   "narrator": 749
  },
  "by_type": {
   "fact": 2114,
   "voice": 1214,
   "spoiler": 621
  },
  "by_char_type": {
   "Wren/fact": 610,
   "Wren/voice": 324,
   "Wren/spoiler": 156,
   "Diane/fact": 610,
   "Diane/voice": 291,
   "Diane/spoiler": 154,
   "4408/fact": 594,
   "4408/voice": 300,
   "4408/spoiler": 161,
   "narrator/fact": 300,
   "narrator/voice": 299,
   "narrator/spoiler": 150
  }
 },
 "after_dedup": {
  "by_char": {
   "Diane": 728,
   "Wren": 888,
   "4408": 687,
   "narrator": 594
  },
  "by_type": {
   "fact": 1503,
   "voice": 900,
   "spoiler": 494
  },
  "by_char_type": {
   "Wren/fact": 458,
   "Wren/voice": 291,
   "Wren/spoiler": 139,
   "Diane/fact": 417,
   "Diane/voice": 193,
   "Diane/spoiler": 118,
   "4408/fact": 399,
   "4408/voice": 171,
   "4408/spoiler": 117,
   "narrator/fact": 229,
   "narrator/voice": 245,
   "narrator/spoiler": 120
  }
 },
 "balance_cap_per_char": {
  "fact": 229,
  "voice": 171,
  "spoiler": 117
 },
 "after_balance": {
  "by_char": {
   "Diane": 517,
   "narrator": 517,
   "4408": 517,
   "Wren": 517
  },
  "by_type": {
   "spoiler": 468,
   "voice": 684,
   "fact": 916
  },
  "by_char_type": {
   "Wren/fact": 229,
   "Wren/voice": 171,
   "Wren/spoiler": 117,
   "Diane/fact": 229,
   "Diane/voice": 171,
   "Diane/spoiler": 117,
   "4408/fact": 229,
   "4408/voice": 171,
   "4408/spoiler": 117,
   "narrator/fact": 229,
   "narrator/voice": 171,
   "narrator/spoiler": 117
  }
 },
 "kept": 2068
}`
- contrast: `{
 "pairs": 400,
 "goal": 400,
 "floor": 300,
 "rejected": {
  "locked": 31,
  "refusal": 40
 },
 "out": "$HOME/output-model/exp4/data/contrast.jsonl"
}`
- split: `{
 "facts": 1882,
 "voice": 786,
 "claims": 39,
 "page_style": 102,
 "contrast_pairs": 400,
 "facts_by_type": {
  "fact": 916,
  "spoiler": 166,
  "contrast": 800,
  "claim": 0
 }
}`
- kd: `{
 "skipped": true,
 "same_tokenizer": true,
 "gpu_gb": 23.56,
 "reason": "GPU is 23.6 GB. Qwen3-14B in bf16 is about 28 GB of weights, so mid-layer hidden-state MSE or CKA against that teacher does not fit beside a 4B student. Qwen3-4B and Qwen3-14B tokenizers match. Phi-4-mini-instruct and SmolLM3-3B do not share the Qwen3 tokenizer, so KD to Qwen3-14B is not defined for the second base. Skipped feature-match KD."
}`

## Second base

microsoft/Phi-4-mini-instruct (no fallback).

## Training

- lora_q4_facts: `{
 "base": "Qwen/Qwen3-4B",
 "stage": "facts",
 "facts": null,
 "q4": false,
 "pairs": 1882,
 "claims": 39,
 "examples": 1950,
 "epochs": 2,
 "lr": 0.0001,
 "r": 16,
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
 "train_sec": 802,
 "peak_vram_gb": 13.4,
 "prefix_mismatch_skipped": 0,
 "limit": 0
}`
- lora_q4_voice: `{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "lora_q4_facts",
 "q4": false,
 "pairs": 786,
 "claims": 0,
 "examples": 990,
 "epochs": 2,
 "lr": 0.0001,
 "r": 16,
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
 "train_sec": 521,
 "peak_vram_gb": 10.37,
 "prefix_mismatch_skipped": 0,
 "limit": 0
}`
- lora_p4_facts: `{
 "base": "microsoft/Phi-4-mini-instruct",
 "stage": "facts",
 "facts": null,
 "q4": false,
 "pairs": 1882,
 "claims": 39,
 "examples": 1944,
 "epochs": 2,
 "lr": 0.0001,
 "r": 16,
 "targets": [
  "down_proj",
  "gate_up_proj",
  "o_proj",
  "qkv_proj"
 ],
 "steps": 3888,
 "train_sec": 519,
 "peak_vram_gb": 11.65,
 "prefix_mismatch_skipped": 0,
 "limit": 0
}`
- lora_p4_voice: `{
 "base": "microsoft/Phi-4-mini-instruct",
 "stage": "voice",
 "facts": "lora_p4_facts",
 "q4": false,
 "pairs": 786,
 "claims": 0,
 "examples": 990,
 "epochs": 2,
 "lr": 0.0001,
 "r": 16,
 "targets": [
  "down_proj",
  "gate_up_proj",
  "o_proj",
  "qkv_proj"
 ],
 "steps": 1980,
 "train_sec": 304,
 "peak_vram_gb": 9.22,
 "prefix_mismatch_skipped": 0,
 "limit": 0
}`

Facts LoRA is trained on the base. Voice LoRA is a second adapter trained with the facts adapter frozen and active (stacked). facts+voice enables both. voice-only+rag enables only the voice adapter, then retrieval. Marker leak, frame break, and must-answer scoring match exp3. Voice and leak judge = qwen3:14b.
