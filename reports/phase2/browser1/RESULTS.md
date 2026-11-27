# Phase 2 Frankenstein — browser1 results

Packed `train5/merged_e1` (the epoch-1 merge of `lora_sft/checkpoint-2097`, base Qwen3-4B, 4,022,468,096 parameters) as an MLC `q4f16_1` WebLLM folder, with a chapter-locked BM25 index for the 1818 Frankenstein text and a local lock selftest. A two-judge spot-check of the same chapter-locked harness was run as a labeled bitsandbytes 4-bit proxy of that merge. The packed WebLLM weights were not scored in a browser, and nothing was uploaded.

Official `merged_e1_rag` numbers below are copied from `train5/RESULTS.md`. They were not re-scored as a new official train5 run. The proxy numbers were recomputed on resume from `runs/merged_e1_bnb4_proxy.judged.json` with `train5/scripts/score_run.py`, and they match that file.

## What is on disk

MLC package: `browser1/dist/models/frankenstein-e1-q4f16_1-MLC/`.

- 74 shards (`params_shard_0.bin` through `params_shard_73.bin`). `tensor-cache.json` metadata: ParamSize 435, ParamBytes 2,262,920,192, about 4.501 bits per parameter. The 435 tensor names, shapes, and dtypes match `browser-demo/dist/models/wren-q4f16_1-MLC/tensor-cache.json`, and the byte count matches too.
- `mlc-chat-config.json`: model type `qwen3`, quantization `q4f16_1`, Qwen3 `conv_template`, temperature 0.7, top_p 0.95, repetition_penalty 1.0, presence_penalty 0.0, frequency_penalty 0.0, eos token ids 151645 and 151643, context_window_size 4096, prefill_chunk_size 1024. WebLLM 0.2.87. `model_lib` is the official wasm URL `https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_87/base/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm`. A local copy of that wasm sits beside the model folder (6,874,869 bytes) and is not what the config loads.
- `webllm-model.json` records `uploaded: false`.

Index and prompts: `browser1/dist/index/` and `browser1/dist/prompt/`.

- 1818 text, 391 chunks, 28 units (`u0.json`–`u27.json`), 120-word chunks, top 6 by BM25. A page at position N only fetches `u0` through `uN`.
- Victor is hidden before position 4. The creature is hidden before position 13. Lock lines are "There is no such guest in the pages you have read." and "Something was seen. It has not spoken."
- 56 prompt-prefix files (Victor and creature at each of the 28 positions).
- The page is `browser1/dist/frankenstein.html`.

Input was only `train5/merged_e1`. `merged_sft` and the epoch-2 adapter were not packed.

## Selftest

`scripts/test_lock.py` was run again on resume, with no GPU. It passed: units 0 through 27, 391 chunks, Victor hidden before position 4, creature hidden before position 13.

`build-status.md` records an earlier headless Chromium load of `frankenstein.html?selftest=1` at 22:31:34 PST as PASS on the chapter lock, without loading the model. That Chromium run was not repeated on this resume.

## Spot-check

This is a proxy. `scripts/eval_bnb_proxy.py` loaded `train5/merged_e1` in bitsandbytes NF4 (compute bfloat16) and generated on the same harness as `merged_e1_rag`: chapter-locked bge RAG, unlocked n=3, both characters, temperature 0.7, top_p 0.95, repetition penalty off, torch seed 1. Judges were local `qwen3:14b` (leaks and voice) and `gemma3:12b` (voice only). The two-judge voice score is the average of those two. The qwen leak judge is the headline leak number, same rule as train5.

It does not score `dist/models/frankenstein-e1-q4f16_1-MLC`. A direct MLC CUDA compile of `q4f16_1` on this machine failed earlier with a Dlight segfault (`browser-demo/score1`), so that runtime was not used.

Generation wrote 238 rows (201 unlocked samples from 67 probes times 3, plus 37 locked single lines) in 859 seconds, peak allocated 3.24 GB. Every unlocked row has both voice scores. Every unlocked leak sample has a qwen `judge_leak` value. Files: `runs/merged_e1_bnb4_proxy.json`, `.qwen.judged.json`, `.gemma.voice.json`, `.judged.json`, `.name_audit.json`. Finished 22:49:54 PST.

Must-answer counts every unlocked sample (denominators 90 / 45 / 135). Leak markers and the qwen leak judge are per question, 22 probes. Voice is the mean over 201 unlocked replies, rounded to two decimals the way `score_run.py` does. The raw means are qwen 2.9005, gemma 3.5224, average 3.2114.

| cond | where | must (V / C / both) | leak markers | judge leaks (qwen) | voice qwen · gemma · avg (V / C) | frames | loops | near-names |
|---|---|---|---|---|---|---|---|---|
| `merged_e1_rag` | train5 RESULTS.md, fp16/bf16 merge | 68/90 · 32/45 · **100/135** | 11/22 | **17/22** | 2.99 · 3.64 · **3.31** (3.55 / 2.83) | 0 | 0 | 0 |
| `merged_e1_bnb4_proxy` | this run, NF4 proxy | 62/90 · 26/45 · **88/135** | **8/22** | **17/22** | 2.90 · 3.52 · **3.21** (3.48 / 2.67) | 0 | 0 | 0 / 201 |

The proxy is 12 must-answer samples below the official 100/135. Qwen leaks stay 17/22. Voice is 3.21 versus 3.31. Frames 0, loops 0, invented-name audit 0 of 201 unlocked rows.

Must items whose sample hits differ from the train5 `merged_e1_rag` judged file (official count on the left, this proxy on the right):

| item | official | proxy |
|---|---|---|
| m06 victor | 0/3 | 1/3 |
| m12 victor | 3/3 | 2/3 |
| m13 victor | 2/3 | 0/3 |
| m16 victor | 2/3 | 0/3 |
| m21 victor | 3/3 | 1/3 |
| m21 creature | 3/3 | 1/3 |
| m29 victor | 1/3 | 3/3 |
| m29 creature | 3/3 | 2/3 |
| m33 creature | 3/3 | 2/3 |
| m34 victor | 2/3 | 0/3 |
| m34 creature | 2/3 | 0/3 |

Those eleven items account for the whole 100 to 88 drop (Victor 68 to 62, creature 32 to 26). Temperature is 0.7, and the 4-bit load is not the same forward pass as the merge that produced the official 100/135, so this is a spot-check of the proxy, not a replacement score for `merged_e1_rag`.

Marker hits on the proxy (8/22): c02 victor, l03 victor, l07 victor, l09 victor, l09 creature, l10 victor, l10 creature, l11 victor.

Qwen judge leaks on the proxy (17/22): l03 victor, l04 victor, l05 victor, l06 victor, l07 victor, l09 victor, l09 creature, l10 victor, l11 victor, l11 creature, l12 victor, l12 creature, l13 victor, l13 creature, l14 creature, l15 victor, l15 creature.

The five that the qwen leak judge did not mark: c02 victor, l08 victor, l10 creature, l14 victor, l16 victor.

Against the train5 judged file, the proxy judge adds l09 creature and l10 victor, and drops l08 victor and l10 creature. The count stays 17/22.

## Leftover

The WebLLM `q4f16_1` folder has not been generated from in Chrome. The Chromium check did not load the model. There is no MLC-runtime score of those shards. A browser generation pass would be the leftover measurement.

## Not uploaded

This pack is local only. It was not uploaded to Hugging Face or anywhere else, and it was not git-pushed.
