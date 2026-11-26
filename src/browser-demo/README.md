# Wren in the browser

No Return is a short invented story. This demo runs Wren, the night-watch officer on the ship Meridian, entirely in the browser. The site is static. Nothing is published from here.

## Run it

From this directory:

```bash
cd ~/output-model/browser-demo/dist
python3 serve.py
```

The server prints a URL, `http://127.0.0.1:8093` when that port is free. It binds to loopback only. If 8093 is taken it tries the next ports and skips 8090 and 8091. Set `PORT` to choose one.

Open that URL. Pick a chapter, then click **Load model**, then ask. The page does not load the weights until you click. `http://127.0.0.1:8093/?selftest=1` checks the chapter lock and does not load the model. A passing check shows `PASS chapter lock`.

Opening `index.html` as a file will not work. WebGPU needs cross-origin isolation, and `serve.py` sends the required `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers.

An offline check of the same lock, with no browser and no GPU:

```bash
cd ~/output-model/browser-demo
$HOME/output-model/exp3/.venv/bin/python scripts/test_lock.py
```

## Size

`dist/` is about 2.2 GB.

| piece | size |
|---|---|
| `dist/models/wren-q4f16_1-MLC/` (q4 weights, tokenizer, config) | 2.2 GB |
| `dist/models/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm` | 6.6 MB |
| `dist/vendor/web-llm.js` (WebLLM 0.2.87) | 6.5 MB |
| `dist/index/` (chapters 1–5) | 88 KB |
| `dist/prompt/` | 24 KB |

The weight files hold 2,262,920,192 parameter bytes, the same count as the official `mlc-ai/Qwen3-4B-q4f16_1-MLC` cache. On disk that is 74 shards plus an 11 MB tokenizer.

## Browser

Use a current Chrome or Edge with WebGPU. The page asks for about 3.5 GB of GPU memory and caps the context at 4096 tokens, with a prefill chunk of 1024, matching the `cs1k` wasm. A longer context is not what this page requests.

Loading the model needs a free GPU. On this machine the 3090 is in use by another job, so **Load model** was not exercised here. The chapter picker, the search, and `?selftest=1` do not need the GPU.

## What the model is

One Qwen3-4B answerer. The facts adapter is already in `exp6/merged_q4_facts`. The pass-1 voice adapter `exp6/lora_v_merged` is folded into those weights, then packed as MLC `q4f16_1`. The 435 tensor names, shapes, and dtypes match the official Qwen3-4B `q4f16_1` cache, so the prebuilt wasm can load them.

The page builds the same pass-1 prompt the reader uses, with thinking turned off, and samples at temperature 0.7, top_p 0.95, 200 new tokens, seed 1. Search is BM25 over chapters 1 through the chapter you picked. Later chapter files are not fetched.

It is not the locked two-pass pipeline. The light DPO rewrite and the 14B fact-diff judge stay on disk. Search is not bge-m3. The base is not a smaller model.

## Scores

Smaller bases that were already scored do not hold the Phase 1 bar. Qwen3-1.7B with retrieval is 60/60 must-answer, 1/48 leaks, and 4 frame breaks, with voice 3.96, under the 4.2 voice bar. The locked `p2_light_dpo_guarded` reference is 59/60, 1/48, voice 4.22, 3 frames. The full table is in `results.md`.

No new GPU eval was run. The closest recorded score for these weights, before q4 and before BM25, is pass-1 `v_merged_v_rag`: 59/60, 1/48, 3 frames, two-judge average 4.27. That is not a score of the file in `dist/models/`.

## Checked here

`scripts/test_lock.py` passed. Headless Chromium, desktop and mobile, loaded `?selftest=1`, switched chapters, and refused to answer before Load model. Those runs did not fetch chapter 4 or 5, and they did not fetch the weights. WebGPU generation was not run.

## Hugging Face-hosted weights

The same quantized Wren build is published at <https://huggingface.co/AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC>. To load those weights in the browser instead of `./models/`, use `dist/app.huggingface.js` (a copy of `app.js`) or the model record in `dist/hf-model-config.js`. The default local `app.js` is unchanged and still reads the on-disk folder.
The WebGPU wasm is the official WebLLM 0.2.87 library; the Hugging Face config points at <https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_87/base/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm> rather than the local copy.
