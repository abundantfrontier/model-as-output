# Phase 2 Frankenstein — browser3 results

Packed `train7/merged_mixa` (the mixed-DPO merge of train6's DPO adapter, base Qwen3-4B, 4,022,468,096 parameters, bf16) as an MLC `q4f16_1` WebLLM folder, with the chapter-locked BM25 index reused from browser1 and a local lock selftest. This run did not score a bitsandbytes 4-bit proxy; train7 already scored that proxy. The packed WebLLM weights were not scored in a browser, and nothing was uploaded.

Official `merged_mixa_rag` and `merged_mixa_bnb4_proxy` numbers below are copied from `train7/RESULTS.md`. They were not re-scored.

## What is on disk

MLC package: `browser3/dist/models/frankenstein-mixa-q4f16_1-MLC/`.

- 74 shards (`params_shard_0.bin` through `params_shard_73.bin`), contiguous, summing to 2,262,920,192 bytes. `tensor-cache.json` metadata: ParamSize 435, ParamBytes 2,262,920,192, BitsPerParam 4.50056. The 435 tensor names, shapes, and dtypes match `browser-demo/dist/models/wren-q4f16_1-MLC/tensor-cache.json` and the browser2 `frankenstein-dpo-q4f16_1-MLC` cache. `mlc-chat-config.json` matches the browser2 chat config byte for byte.
- `mlc-chat-config.json`: model type `qwen3`, quantization `q4f16_1`, Qwen3 `conv_template`, temperature 0.7, top_p 0.95, repetition_penalty 1.0, presence_penalty 0.0, frequency_penalty 0.0, eos token ids 151645 and 151643, context_window_size 4096, prefill_chunk_size 1024. WebLLM 0.2.87. `model_lib` is the official wasm URL `https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_87/base/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm`. A local copy of that wasm sits beside the model folder (6,874,869 bytes, sha256 `2afc399ed01d8ae641beac0528567c8ce22e8c9c7203ba0827c44e13c62be99c`, same bytes as Wren, browser1, and browser2) and is not what the config loads.
- `webllm-model.json` records `uploaded: false`. Source path is `train7/merged_mixa`.

Index and prompts: `browser3/dist/index/` and `browser3/dist/prompt/`.

- Reused from browser1. Unit JSON, `selftest.json`, and the 56 prompt-prefix files are byte-identical to browser1. Only `manifest.json`'s `model` block was retargeted to `frankenstein-mixa-q4f16_1` / `train7/merged_mixa` from `train7/lora_mixa`.
- 1818 text, 391 chunks, 28 units (`u0.json`–`u27.json`), 120-word chunks, top 6 by BM25. A page at position N only fetches `u0` through `uN`.
- Victor is hidden before position 4. The creature is hidden before position 13. Lock lines are "There is no such guest in the pages you have read." and "Something was seen. It has not spoken."
- 56 prompt-prefix files (Victor and creature at each of the 28 positions).

Input was only `train7/merged_mixa`. `lora_mixb` was not packed. MLC's weight view is a symlink plus a header-built `model.safetensors.index.json` under `browser3/work/hf-src/`. No index file was written into `train7/merged_mixa/`.

## Selftest

`scripts/test_lock.py` was run on CPU with no GPU, twice (before the convert, and again after it). Both passed: units 0 through 27, 391 chunks, Victor hidden before position 4, creature hidden before position 13.

No headless Chromium `?selftest=1` pass was run. There is no `frankenstein.html` in this folder.

## Official train7 numbers (not re-scored)

Copied from `train7/RESULTS.md`. This browser3 run did not run `eval_bnb_proxy.py` and did not start a judge job.

| cond | where | must (V / C / both) | leak markers | judge leaks (qwen) | voice qwen · gemma · avg (V / C) | frames | loops |
|---|---|---|---|---|---|---|---|
| `merged_mixa_rag` | train7 RESULTS.md, bf16 merge | 70/90 · 31/45 · **101/135** | **8/22** | **10/22** | 3.13 · 3.84 · **3.49** (3.71 / 3.02) | 0 | 0 |
| `merged_mixa_bnb4_proxy` | train7 RESULTS.md, already-scored NF4 proxy | 68/90 · 29/45 · **97/135** | **10/22** | **10/22** | 3.27 · 3.77 · **3.52** (3.73 / 3.09) | 0 | 0 |

Name audit on the official merge is 0/201 strict and 0/201 with the Salêve grave-accent fold. The NF4 proxy has the same name audit, 0/201 both ways. The proxy does not clear the 100 must-answer bar. It is the number to cite against train6's NF4 proxy (93/135, qwen leaks 12/22, voice 3.41).

## This run

Lock `output-model` PID 574339 from Fri Nov 27 04:24:43 PM PST, hard stop Fri Nov 27 05:24:43 PM PST (1.0 GPU h). The holder was already running in `om-browser3-lock`. nvidia-smi compute apps stayed at PID 4187666 `/usr/bin/ptyxis` only. Convert ran as one CPU process with `CUDA_VISIBLE_DEVICES` empty under `om-browser3.service` MemoryMax=20G. Convert started 4:30 PM PST and finished 4:33 PM PST. Cgroup peak was 13.2 GiB of 20 GiB. The 1.0 h cap was not reached.

## Leftover

The WebLLM `q4f16_1` folder has not been generated from in Chrome. There is no MLC-runtime score of these shards. A browser generation pass would be the leftover measurement.

## Not uploaded

This pack is local only. It was not uploaded to Hugging Face or anywhere else, and it was not git-pushed.
