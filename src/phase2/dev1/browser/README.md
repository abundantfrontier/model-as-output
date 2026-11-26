# Frankenstein facts+RAG browser fallback

Stock Qwen3-4B q4f16_1 (MLC) plus a chapter-locked BM25 index of the 1818 text. **No voice LoRA.** Train1’s voice adapter looped; the shippable row is `rag_base` (`train1/RESULTS.md`).

## Files

- `index/` — 28 unit files `u0.json`…`u27.json` (391 chunks, `pos` kept), `manifest.json`, `selftest.json`
- `prompt/` — `{victor,creature}-{pos}.json` prefixes (knowledge-state facts for that speaker and position only)
- `frankenstein.html` / `frankenstein.js` — local copy of the page

The runnable copy next to the wasm is:

`~/output-model/browser-demo/dist/frankenstein.html`

Wren stays at `browser-demo/dist/index.html`. This page does not overwrite `dist/index/`.

## Model weights (not downloaded here)

Place the official MLC build here, then load the model in the page:

```
~/output-model/browser-demo/dist/models/Qwen3-4B-q4f16_1-MLC/
```

Hugging Face repo: `mlc-ai/Qwen3-4B-q4f16_1-MLC` (~2.2 GB). Reuse the existing wasm:

```
~/output-model/browser-demo/dist/models/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm
```

Do not put the failed train1 voice merge in that folder.

## Lock

- Reader at position `upto` may only see chunks with `pos <= upto`. The page fetches `u0`…`u{upto}` only.
- Victor speaks from Volume I, Letter IV (`vol1-letter-4`, pos 4). Before that the reply is: `There is no such guest in the pages you have read.`
- The creature speaks from Volume II, Chapter II (`vol2-ch-2`, pos 13). Before that: `Something was seen. It has not spoken.`
- Lock lines are the whole reply. No model call.

## Checks

```
CUDA_VISIBLE_DEVICES= python3 ../scripts/test_lock.py
```

In the browser, `frankenstein.html?selftest=1` runs the same lock fixture (no model).

Rebuild:

```
CUDA_VISIBLE_DEVICES= python3 ../scripts/build_index.py
```
