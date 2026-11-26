---
license: apache-2.0
base_model: Qwen/Qwen3-4B
tags:
  - mlc
  - webllm
  - webgpu
  - model-as-output
---

# Wren (No Return) — browser model

This is the in-browser model for Wren, a character from *No Return,* a short invented story. Wren is the night-watch officer on the ship Meridian. You can talk with Wren in your own browser after you pick how far you have read.

The work is part of **Stories you can step into**, a model-as-output research project by Boyd Wilson, Abundant Frontier Institute and Omnibond Systems. The Lab write-up is at [abundantfrontier.org/lab/](https://abundantfrontier.org/lab/).

Hugging Face repository: [AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC](https://huggingface.co/AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC)

Code, data and reports: <https://github.com/abundantfrontier/model-as-output> (this GitHub repo may still be filling in).

## What you are downloading

A quantized [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) chat model, packed as MLC `q4f16_1` for [WebLLM](https://github.com/mlc-ai/web-llm) 0.2.87 on WebGPU. The Phase 1 facts adapter and the pass-1 voice adapter are already folded into these weights. The page, not this repo, adds a chapter-locked search over the story.

On disk the weight folder is about 2.2 GB: 74 shards plus an 11 MB tokenizer. The first visit downloads that into the browser. The page asks for about 3.5 GB of GPU memory and caps the context at 4,096 tokens.

The model runs on the visitor's computer. It does not need a server GPU, and the story does not leave the machine.

This repository contains the quantized weights, tokenizer, `tensor-cache.json`, and `mlc-chat-config.json`. It does not include the WebGPU runtime library (`.wasm`). That file is the official WebLLM 0.2.87 library for this Qwen3-4B `q4f16_1` / `cs1k` layout. Point `model_lib` at:

<https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_87/base/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm>

Example `model_list` entry for WebLLM 0.2.87:

```js
{
  model: "https://huggingface.co/AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC",
  model_id: "wren-q4f16_1",
  model_lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_87/base/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm",
  overrides: { context_window_size: 4096, prefill_chunk_size: 1024 },
  vram_required_MB: 3500,
}
```

## This build is looser than the locked model

The research prototype that we locked, `p2_light_dpo_guarded`, is a two-pass pipeline: a retrieval-grounded answer, then a voice rewrite, then a fact-diff guard that can fall back to the first answer. On the held-out *No Return* probes that locked model scored **59/60** must-answer, **1/48** leak probes, voice **4.22**, and **3** frame breaks.

This browser build is a single pass. It does not run the second-pass rewrite or the fact-diff guard. Search in the demo page is a lexical index (BM25), not the embedding index used in the locked pipeline.

The quantized file in this folder has not itself been scored. WebGPU generation was not run in the build that produced these weights. The closest recorded score for the same merged weights, before this quantization and before the BM25 swap, is the pass-1 condition `v_merged_v_rag`: **59/60** must-answer, **1/48** leak probes, **3** frame breaks, and a two-judge voice average of **4.27**. Quantization and the lexical index can move those answers. Treat the browser chat as looser than the locked model.

## Intended use

This model is for a public, in-character demo: a reader of *No Return* asks Wren about the story so far, in the browser, with later chapters held back. It is a research demo, not a general assistant, and not a substitute for reading the story.

## Limits

Wren may say something the story never said. Wren may also lean later than the chapter you picked, even though the page tries not to fetch or prompt with later chapters. The single-pass build has no second-pass rewrite and no fact guard to catch that. Answers can be wrong, oddly phrased, or out of character. The model should not be used for advice, and it should not be treated as a complete or spoiler-safe account of the text.

Use a current Chrome or Edge with WebGPU. Opening the demo as a local file will not work; the page needs the cross-origin isolation headers that a small local server can send.

## License

The Qwen3-4B base is Apache 2.0, and this packed model is released under Apache 2.0 as well. *No Return* is a short invented story; the demo text and this card are part of the same research project. Please keep the byline with any reuse: Boyd Wilson, Abundant Frontier Institute and Omnibond Systems.
