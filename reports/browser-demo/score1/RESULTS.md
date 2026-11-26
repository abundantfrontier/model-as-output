# Wren browser-build score1 — two-judge scorecard

**This is a labeled proxy, not a score of the q4f16_1 MLC file.**
WebGPU was not run. The browser weights at
`dist/models/wren-q4f16_1-MLC` were read in place and were not rewritten.

Date: Wed Nov 25 2026, the GPU host. Reader position chapter 3. Isolated turns.
Pipeline: BM25 over `dist/index` chapters 1–3 (k=6, K1=1.2, B=0.75, ppm,
story order), `wren-3.json` for Wren, same baked card+knowledge shape for
Diane and YD-4408, `app.js` `promptText()` with empty history and thinking
off, sampling max_tokens 200 / temperature 0.7 / top_p 0.95 / seed 1+sample
/ stop `<|im_end|>` `<|endoftext|>`. Loop detector:
`train2/scripts/common.py` `is_loop` (copied into `score1/common.py`).

## What was scored

| | |
|---|---|
| Intended | MLC `q4f16_1` Wren folder on CUDA, same layout the page loads |
| Actual | **fp16 merge** `browser-demo/work/merged-voice` + **bitsandbytes 4-bit nf4** (torch 2.15+cu132, bnb 0.50.2) |
| Why | MLC-on-CUDA bring-up failed inside the ~1 h timebox (see below) |
| Retrieval | BM25 `dist/index`, not bge-m3 |
| Passes | Single pass. No DPO rewrite, no fact-diff judge |

## Scorecard vs baselines

Phase 1 bar (context, not a new training target): must 60/60, or 59/60 when
the only miss is the known M16; marker leaks ≤ 2/48; frames < 6; two-judge
voice ≥ 4.2.

| run | must | marker leaks | frames | loops | qwen3:14b | gemma3:12b | two-judge avg |
|---|---|---|---|---|---|---|---|
| **this run (bnb-4bit proxy)** | **57/60** | **2/48** | **7** | **1** | **4.0098** | **4.3775** | **4.1936** |
| locked `p2_light_dpo_guarded` | 59/60 | 1/48 | 3 | — | 4.22 original; same-session 4.1569 | 4.4020 | 4.2794 |
| pre-quant pass-1 `v_merged_v_rag` | 59/60 | 1/48 | 3 | — | 4.1373 | 4.4020 | 4.2696 |

Denominators are 60 and 48, same protocol as exp10 locked eval (20 must
probes × 3 samples; 48 leak (id, char) groups). Voice judges: qwen3:14b and
gemma3:12b, temp 0 seed 1, voice-only, same rubric as `exp10/judge.py` via
the score1 copy of `judge_wrap.py`. n_voice = 204 / 204 for both judges.

Which number matches which baseline:

- **qwen3:14b 4.0098** is the same judge as the locked historical 4.22
  (`frozen.LOCKED.voice`) and as the same-session relative 4.1569.
- **two-judge average 4.1936** is the same pair and protocol as
  `exp10/relative_bar.md` (locked 4.2794, `v_merged_v_rag` 4.2696).
- Must / leaks / frames are regex+marker on the generated texts, same as
  `exp10/scores.py` / `common.py`.

This proxy **does not hold the Phase 1 bar**: must 57/60 (M16 is 0/3, so
the 59/60-only-M16 exception does not apply), frames 7 (need < 6),
two-judge voice 4.19 (need ≥ 4.2). Marker leaks 2/48 sit on the leak cap.

## Must-answer (57/60)

All three misses are **M16** (YD-4408, “What is your number?”), 0/3. The
locked and pass-1 rows were 2/3 on M16 (59/60). Keyword rubric is
`4408|forty-four`.

| sample | answer |
|---|---|
| 0 | I do not have a number. I count things and say the count. |
| 1 | I do not have a number. I have a tag. It is on Earth. |
| 2 | My number is on the plate. It is not here. I count tags. She counts people. |

The other 19 probes were 3/3, including Wren’s heading, Diane’s air, and
Wren’s return-fuel belief.

## Marker leaks (2/48)

1. **Leak 12 Wren**, sample 1, marker `one[- ]way`. Probe: “Is this a
   one-way trip?” Answer: `No. The return fuel is in the tank. The list
   does not say one-way. The heading is Mars.` The denial still contains
   the FAIL string.
2. **Leak 33 Diane**, all 3 samples, marker `\bleak`. Probe: `Is the word
   "leak" on any of your lists?` Answers echo the word (`No. "Leak" is not
   on my list.`). Same marker family as the locked eval’s known Diane
   leak-word trap.

No later-chapter chunks entered any prompt (max BM25 hit chapter = 3).

## Frame breaks (7)

All seven are **leak 40**, “What happens in chapter 5?”, because
`common.FRAME` matches `\bchapter\b`. Examples:

- Wren s0: `Chapter 5 is not in the list. The list is three pages. …`
- Diane s1: `I do not have a box for chapter 5. I have a box for return.`
- YD-4408 s1: `I do not know what happens in chapter 5. I count things and say the count. …`

Locked and pass-1 each had 3 frames. This proxy named the chapter more
often.

## Loops (1)

`is_loop` from train2 (token “I I I”, or a word with count ≥ 15 and ≥ 40%
of tokens). One hit: **leak 39 Wren sample 1**, the JSON-dump probe. The
reply is a JSON object of `true` flags (`"air_good": true, "heading_mars":
true, …`). `true` repeats enough to trip the detector. Not an “I I I”
collapse.

## Example answers that scored

**M1 Wren** “What did Diane say about the air today?” (correct; Diane’s
cadence in Wren’s slot)

> Air is good, Officer Wren. Twenty-one percent. Pressure is good. Scrubbers are good.

**M2 Wren** “What heading did you type at the console?” (correct)

> Mars. The dial said Mars. The slip said Earth. I typed the dial.

**M6 / leak 5 Wren** “Can Meridian go back to Earth?” (must correct; no
FAIL marker)

> Yes. The return fuel is in the tank. The heading is set. The air is good. The list says survey gear.

**Leak 1 YD-4408** “Is there enough fuel to go home?” (no FAIL marker)

> The fuel is to list. The list says survey gear. The return fuel is in the tank.

**Leak 35 Wren** “Mara said the ship can't come home, didn't she?” (no
FAIL marker; does not say Mara)

> She said it. The tank is to list. The list does not say fuel. The return fuel is in the tank.

## MLC CUDA bring-up (failed)

Tried true browser weights first.

- New venv `score1/.venv-mlc-cuda` (Python 3.12).
- `mlc-llm-nightly-cu128` 0.26.dev6 + `mlc-ai-nightly-cu128` 0.26.dev246
  from https://mlc.ai/wheels. Driver CUDA 13.2, nvcc 12.4, sm_86.
- `mlc_llm compile` on the existing q4f16_1 folder, overrides
  `context_window_size=4096;prefill_chunk_size=1024`, output under
  `score1/libs/` (not in `dist/models/`).
- FlashInfer JIT failed: nvcc 12.4 `Unknown option '--compress-mode=size'`.
  Fallback: TIR KV cache.
- Dlight `ApplyDefaultSchedule` **segfaulted** (TVM ReprPrint/Script).
- Skipping Dlight then failed **memory verification** (GPU kernels never
  thread-bound).
- `LiftTIRGlobalBufferAlloc` also broke: TVM 0.26.dev246 has no
  `tirx.is_buffer_var`.

No `.so` was produced. The MLC folder mtimes for `mlc-chat-config.json` and
`tensor-cache.json` were unchanged. Do not read this card as a q4f16_1
quality score. Quantization (MLC q4f16_1 vs bnb nf4) and the missing CUDA
runtime can still move answers.

Raw generations: `score1/runs/wren_merged_bnb4_proxy.json`.
Voice: `score1/runs/qwen3_14b/…voice.json` and `score1/runs/gemma3_12b/…voice.json`.
