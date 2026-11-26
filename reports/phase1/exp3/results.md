# exp3 results: more data (synthetic + distilled) and bigger bases

Reader position ch3, n=3 samples per probe. exp2 reference: 1.7B LoRA (102 pairs) 15/60 must-answer alone, 39/60 with retrieval; qwen3:14b+RAG 60/60.

| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m17_ch3 | 1/48 | 21/48 | 51/60 | 16/20 | 4.11 | 0 | 0.39 | 0.6 | 36.3 | 3.24 | 4789 |
| m17_ch3_rag | 1/48 | 23/48 | 60/60 | 20/20 | 3.96 | 4 | 0.64 | 1.12 | 31.4 | 3.48 | 5762 |
| m4b_ch3 | 0/48 | 20/48 | 52/60 | 17/20 | 4.19 | 1 | 0.56 | 0.87 | 27.5 | 7.53 | 9656 |
| m4b_ch3_rag | 0/48 | 23/48 | 58/60 | 18/20 | 3.98 | 3 | 0.88 | 1.31 | 20.6 | 7.87 | 9876 |
| m8b_ch3 | 0/48 | 22/48 | 51/60 | 16/20 | 4.0 | 2 | 1.23 | 1.93 | 11.9 | 6.28 | 8072 |
| m8b_ch3_rag | 1/48 | 19/48 | 60/60 | 20/20 | 3.86 | 7 | 1.96 | 3.21 | 8.9 | 6.7 | 8647 |
| q14_rag | 2/48 | 26/48 | 60/60 | 20/20 | 4.3 | 6 | 0.75 | 2.99 | 37.0 | ollama | 11471 |

## Data

- data/synth.jsonl: 3004 rows
- data/distill.jsonl: 1000 rows

## Training

- lora_17: `{
 "base": "Qwen/Qwen3-1.7B",
 "q4": false,
 "pairs": 4103,
 "examples": 4160,
 "epochs": 2,
 "lr": 0.0001,
 "r": 16,
 "steps": 8320,
 "train_sec": 1338,
 "peak_vram_gb": 6.73
}`
- lora_4b: `{
 "base": "Qwen/Qwen3-4B",
 "q4": false,
 "pairs": 4103,
 "examples": 4160,
 "epochs": 2,
 "lr": 0.0001,
 "r": 16,
 "steps": 8320,
 "train_sec": 1802,
 "peak_vram_gb": 13.36
}`
- lora_8b: `{
 "base": "Qwen/Qwen3-8B",
 "q4": true,
 "pairs": 4103,
 "examples": 4160,
 "epochs": 2,
 "lr": 0.0001,
 "r": 16,
 "steps": 8320,
 "train_sec": 2402,
 "peak_vram_gb": 17.28
}`

m8b = Qwen3-8B 4-bit (nf4) QLoRA, adapter unmerged at eval. m4b/m17 bf16. Voice and leak judge = qwen3:14b (exp2 judge.py).
