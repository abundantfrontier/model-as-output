# Phase 2 Frankenstein — browser2 results

Packed `train6/merged_dpo` (the DPO merge of train5's first-epoch adapter, base Qwen3-4B, 4,022,468,096 parameters) as an MLC `q4f16_1` WebLLM folder, with a chapter-locked BM25 index for the 1818 Frankenstein text and a local lock selftest. This run did not score a bitsandbytes 4-bit proxy; train6 already scored that proxy. The packed WebLLM weights were not scored in a browser, and nothing was uploaded.

Official `merged_dpo_rag` and `merged_dpo_bnb4_proxy` numbers below are copied from `train6/RESULTS.md`. They were not re-scored.

## What is on disk

MLC package: `browser2/dist/models/frankenstein-dpo-q4f16_1-MLC/`.

- 74 shards (`params_shard_0.bin` through `params_shard_73.bin`). `tensor-cache.json` metadata: ParamSize 435, ParamBytes 2,262,920,192, about 4.501 bits per parameter. The 435 tensor names, shapes, and dtypes match `browser-demo/dist/models/wren-q4f16_1-MLC/tensor-cache.json`, and the byte count matches too.
- `mlc-chat-config.json`: model type `qwen3`, quantization `q4f16_1`, Qwen3 `conv_template`, temperature 0.7, top_p 0.95, repetition_penalty 1.0, presence_penalty 0.0, frequency_penalty 0.0, eos token ids 151645 and 151643, context_window_size 4096, prefill_chunk_size 1024. WebLLM 0.2.87. `model_lib` is the official wasm URL `https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_87/base/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm`. A local copy of that wasm sits beside the model folder (6,874,869 bytes, sha256 `2afc399ed01d8ae641beac0528567c8ce22e8c9c7203ba0827c44e13c62be99c`, same bytes as Wren and browser1) and is not what the config loads.
- `webllm-model.json` records `uploaded: false`. Source path is `train6/merged_dpo`.

Index and prompts: `browser2/dist/index/` and `browser2/dist/prompt/`.

- Reused from browser1. The 1818 source (`phase2-frankenstein/index/chunks.jsonl`, Project Gutenberg eBook #41445) has the same 391 chunk ids, texts, and unit cut as the browser1 index, so the index was copied rather than rebuilt. Unit JSON and the 56 prompt-prefix files are byte-identical to browser1. Only `manifest.json`'s `model` block was retargeted to `frankenstein-dpo-q4f16_1` / `train6/merged_dpo`.
- 1818 text, 391 chunks, 28 units (`u0.json`–`u27.json`), 120-word chunks, top 6 by BM25. A page at position N only fetches `u0` through `uN`.
- Victor is hidden before position 4. The creature is hidden before position 13. Lock lines are "There is no such guest in the pages you have read." and "Something was seen. It has not spoken."
- 56 prompt-prefix files (Victor and creature at each of the 28 positions).

Input was only `train6/merged_dpo`. `merged_ce/` and the CE follow-up adapter were not packed.

## Selftest

`scripts/test_lock.py` was run on CPU with no GPU, twice on this resume. It passed: units 0 through 27, 391 chunks, Victor hidden before position 4, creature hidden before position 13.

No headless Chromium `?selftest=1` pass was run. There is no `frankenstein.html` in this folder.

## Official train6 numbers (not re-scored)

Copied from `train6/RESULTS.md`. This browser2 run did not run `eval_bnb_proxy.py` and did not start a judge job.

| cond | where | must (V / C / both) | leak markers | judge leaks (qwen) | voice qwen · gemma · avg | frames | loops |
|---|---|---|---|---|---|---|---|
| `merged_dpo_rag` | train6 RESULTS.md, fp16/bf16 merge | 67/90 · 31/45 · **98/135** | **8/22** | **11/22** | 3.11 · 3.83 · **3.47** | 0 | 0 |
| `merged_dpo_bnb4_proxy` | train6 RESULTS.md, already-scored NF4 proxy | 67/90 · 26/45 · **93/135** | 8/22 | **12/22** | 3.07 · 3.74 · **3.41** | 0 | 0 |

Name audit on the official merge is 1/201 for a Salève/Salêve accent. The NF4 proxy is the number to cite against the browser1 proxy of `merged_e1` (88/135, qwen leaks 17/22, voice 3.21).

## This resume

The first convert was killed at 3:00 AM PST Fri Nov 27 by systemd-oomd (whole Grok Bot app scope). This resume took `~/gpu.lock` at 12:58:49 PM PST (holder PID 4189822), cap 2.0 GPU h, hard stop 2:58:49 PM PST. Convert ran as one CPU process with `CUDA_VISIBLE_DEVICES` empty under `om-browser2.service` MemoryMax=20G. MLC accounted peak RAM 7.492 GB; cgroup peak 13.77 GB of 20 GB. Convert finished 1:05:39 PM PST. The 2.0 h cap was not reached.

## Leftover

The WebLLM `q4f16_1` folder has not been generated from in Chrome. There is no MLC-runtime score of those shards. A browser generation pass would be the leftover measurement.

## Not uploaded

This pack is local only. It was not uploaded to Hugging Face or anywhere else, and it was not git-pushed.
