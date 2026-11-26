# Browser demo: smaller base vs the locked pipeline

No new GPU eval was run. another GPU job holds `~/gpu.lock` (PID 3401149) and the 3090 is full. A two-judge rescore is a 2–4 hour block, so it was not started. See `build-status.md` and `~/GPU_SCHEDULE.md`.

The numbers below are the recorded runs on the same probe set (reader at chapter 3, 3 samples). They are not a new measurement of the q4 file in `dist/models/`.

## Bar

Must-answer 60/60, with 59/60 accepted when the only miss is the known M16. Marker leaks ≤ 2/48. Frame breaks < 6. Two-judge voice ≥ 4.2.

Locked `p2_light_dpo_guarded` (exp9 judged file): **59/60, 1/48, voice 4.22 (4.2157) on qwen3:14b, 3 frame breaks.** Same-session two-judge average later: qwen3:14b 4.1569, gemma3:12b 4.4020, **average 4.2794**. Source: `exp10/results.md`, `exp10/relative_bar.md`.

## Smaller bases that were actually trained

These are earlier recipes (exp3, exp4), not the locked two-pass stack, and the voice number is the single qwen3:14b judge. No gemma3:12b score was recorded for them, so the two-judge average does not exist.

| model | must | marker leaks | voice (qwen3:14b) | frames | holds the bar? |
|---|---|---|---|---|---|
| Qwen3-1.7B + RAG, exp3 `m17_ch3_rag` | 60/60 | 1/48 | 3.96 | 4 | No. Voice is under 4.2. Must, leaks, and frames pass. |
| Qwen3-1.7B, no RAG, exp3 `m17_ch3` | 51/60 | 1/48 | 4.11 | 0 | No. Must-answer misses. |
| Phi-4-mini facts+voice+RAG, exp4 `p4_fv_rag` | 58/60 | 1/48 | 3.79 | 2 | No. Must and voice miss. |
| Phi-4-mini voice+RAG, exp4 `p4_v_rag` | 58/60 | 4/48 | 2.72 | 16 | No. Leaks, voice, and frames miss. |
| Locked Qwen3-4B `p2_light_dpo_guarded` | 59/60 | 1/48 | 4.22 (two-judge 4.28) | 3 | Yes. This is the reference, not a smaller base. |

Qwen3-1.7B with retrieval is the smallest base that was fully scored. It reaches the must-answer, leak, and frame bars, including a clean 60/60 rather than the locked 59/60. It does not reach voice 4.2. The gap versus the locked qwen3:14b score is 3.96 against 4.22, about 0.26, larger than the 0.03 judge-jitter tolerance used in exp10. That is not a two-judge result. It is the only voice number those runs have.

No Qwen3-0.6B run exists in this tree. SmolLM3-3B was a fallback in exp4 and does not have a row in `exp4/results.md`.

## What the browser ships

The page runs one model: Qwen3-4B with the facts adapter already merged (`exp6/merged_q4_facts`) and the pass-1 voice adapter (`exp6/lora_v_merged`) folded in, then quantized to MLC `q4f16_1`. That is the same base as the lock, not a smaller one. The pass-2 light DPO rewrite and the qwen3:14b fact-diff judge are not in the browser. Retrieval in the page is BM25 over chapters 1..N, not bge-m3.

The closest recorded score for these weights, before q4 and before the BM25 swap, is pass-1 `v_merged_v_rag`: **59/60, 1/48, 3 frames, two-judge average 4.2696** (qwen3:14b 4.1373, gemma3:12b 4.4020). That clears the bar. It is not a score of the quantized browser file. Quantization and the lexical index can move the answers. Those were not re-judged, because that job needs the GPU another GPU job is using.

The q4 file is now on disk at `dist/models/wren-q4f16_1-MLC`. It has 435 tensors. Every name, shape, and dtype matches `mlc-ai/Qwen3-4B-q4f16_1-MLC`, and the parameter byte count is the same 2,262,920,192. That is a layout check against the prebuilt wasm, not a quality score.
