# bscore3 — real q4f16_1 pack, scored in headless Chrome

The numbers below come from runtime A: WebLLM 0.2.87 in headless Chrome, WebGPU, on the real browser3 shards `browser3/dist/models/frankenstein-mixa-q4f16_1-MLC/` (74 shards, ParamBytes 2,262,920,192) and the official wasm `Qwen3-4B-q4f16_1_cs1k-webgpu.wasm` that sits next to that folder. This is the real browser path. The 4-bit values are the MLC q4f16_1 pack, not the bitsandbytes NF4 proxy. One user-space Dawn flag was required on this machine: `--enable-dawn-features=allow_unsafe_apis,vulkan_enable_f16_on_nvidia`. Driver 595.99 is older than the 615.71 cutoff where Dawn turns shader-f16 on by default for NVIDIA. The flag only lifts that blocklist. The wasm and the weight shards are unchanged. Runtimes B, C, and D were not used.

Verdict: **MISS**. Must-answer is 94/135, six short of 100. The strict name audit is 1/201, not 0. Qwen judge leaks are 11/22 (inside the 14 cap). Loops 0. Frames 0.

The pack does not hold the bar. It is a bit worse than the NF4 proxy on must-answer (94 vs 97) and seven below the bf16 merge (101). Voice average is the same as the NF4 proxy, 3.52. For the demo, the bf16 merge is still the run that cleared the bar. Packing it to this q4f16_1 folder costs must-answer, mostly on Victor, and the NF4 proxy was slightly optimistic rather than a worse stand-in.

Nothing was uploaded. train7 and train8 were not re-scored. Their rows below are copied from `train7/RESULTS.md`.

## What was run

Official protocol from train8 `eval_pass1` / `run_official.sh`, copied into `bscore3/scripts/` and pointed at this folder. Same items, same chapter lock, same bge-m3 retrieved context (`backend=bge`), same Qwen3 template with thinking off, temperature 0.7, top_p 0.95, repetition and presence and frequency penalties off, max 200 new tokens, stop on `<|im_end|>` and `<|endoftext|>`, n=3, seeds 1, 2, and 3. Prompts were built in Python (`work/prompts.json`, 67 unlocked, 37 lock lines, max prompt 2696 tokens) and completed by the browser. Raw file: `runs/merged_mixa_browser_q4f16_1.json` (238 rows, 201 unlocked). `meta.proxy` is false. `meta.runtime` is the A label above. `meta.pack` is the browser3 folder.

A 6-item smoke (n=1) was read against `train7/runs/merged_mixa_rag.json` before the full pass. The six Victor answers were on the book: the sledge, the tale Walton is promised, Elizabeth as cousin sent from Italy, Agrippa, Clerval, scarlet fever. Then one full pass. No item was retried, dropped, or resampled. Judges ran only after Chrome had exited: `qwen3:14b` for leaks and voice, `gemma3:12b` for voice, then `merge_judges.py`, `name_audit.py`, `score_run.py`. Ollama was unloaded. GPU compute apps at judge start were ptyxis only.

## Scorecard

Must-answer counts every unlocked sample (90 / 45 / 135). Leak markers and the qwen leak judge are per unlocked (probe, character) cell, 22 cells. Voice is the mean over 201 unlocked replies. The bf16 and NF4 rows are train7's, cited, not re-scored.

| cond | where | must (V / C / both) | leak markers | judge leaks (qwen) | voice qwen · gemma · avg (V / C) | frames | loops | names |
|---|---|---|---|---|---|---|---|---|
| `merged_mixa_rag` | train7 RESULTS.md, bf16 | 70/90 · 31/45 · **101/135** | **8/22** | **10/22** | 3.13 · 3.84 · **3.49** (3.71 / 3.02) | 0 | 0 | 0/201 |
| `merged_mixa_bnb4_proxy` | train7 RESULTS.md, NF4 proxy | 68/90 · 29/45 · **97/135** | **10/22** | **10/22** | 3.27 · 3.77 · **3.52** (3.73 / 3.09) | 0 | 0 | 0/201 |
| `merged_mixa_browser_q4f16_1` | this run, real pack, bge retrieval | 61/90 · 33/45 · **94/135** | **9/22** | **11/22** | 3.24 · 3.79 · **3.52** (3.77 / 2.99) | 0 | 0 | **1/201** |

Bar: must ≥ 100/135 and qwen leaks ≤ 14/22 and loops 0 and frames 0 and names 0. This run misses must by 6 and names by 1. Leaks, loops, and frames pass.

The single name hit is the common word "homeland" in creature m24, sample 1 ("banishment from their homeland"). The strict audit flags it because it is edit distance 2 from the canonical token "holland", and tokens of length 8 or more fail at distance 2. The Salêve grave-accent fold does not remove it (1/201 both ways). It is not a Clerval-style invented person. The audit still counts it, so names are not 0.

## What moved versus bf16

45 must cells, 13 changed. Counts are correct samples out of 3. Pack minus bf16.

Victor, net −9:

- m06, what the stranger agrees to tell Walton: 0/3 → 2/3
- m12, the question that takes over his studies: 3/3 → 2/3
- m13, the night the creature comes to life: 3/3 → 2/3
- m14, who finds Victor after that night, and whether he tells the truth: 3/3 → 0/3
- m16, how Victor learns William is dead: 3/3 → 2/3
- m18, Justine's trial: 3/3 → 1/3
- m26, what the creature does after the cottagers leave: 1/3 → 0/3
- m34, how the book ends: 2/3 → 0/3

Creature, net +2:

- m21, the beginning of his life: 2/3 → 1/3
- m26, after the cottagers leave: 0/3 → 1/3
- m28, whether Victor starts the second creature at once: 0/3 → 2/3
- m31, what Victor is accused of in Ireland: 2/3 → 3/3
- m34, how the book ends: 3/3 → 2/3

The loss is Victor's, concentrated on the night after the creature wakes (m14), Justine's trial (m18), and the ending (m34). Creature must-answer is slightly higher than both the bf16 merge and the NF4 proxy.

## Timing

Headless Chrome, RTX 3090, 201 unlocked generations. Generation wall 1030 s, about 17 minutes, including model load from the local shards.

| | first token (s) | tokens/s |
|---|---|---|
| median | 3.07 | 25.2 |
| p10 | 2.56 | 20.0 |
| p90 | 3.82 | 30.2 |
| min / max first token | 1.94 / 4.91 | |

The smoke's first call was slower (9.1 s to first token) while the pipeline warmed up. Later calls sit near 3 s to first token and about 25 tokens/s. Sampling was 200 tokens max; typical replies were shorter, so wall time per sample was about 5 s.

## Runtime attempts

About 8 minutes of bring-up, under the 35 minute cap. Stopped at A.

1. Headless Chrome 156 (Playwright chromium-1248 headless shell), WebGPU flags from `browser-demo/live-test/run_live_test2.py`, no display. A first probe used a `data:` URL, which is not a secure context, so `navigator.gpu` was missing for every flag set. That probe does not say anything about shader-f16.
2. The same probe on `http://127.0.0.1` with `--enable-dawn-features=allow_unsafe_apis,vulkan_enable_f16_on_nvidia`. The adapter was the real RTX 3090, not a fallback, and it advertised shader-f16. A tiny `enable f16` shader compiled. The toggle is in this Chrome binary. The earlier live-test on this machine (`browser-demo/live-test/report2.json`, 2026-11-27) is the record that the same adapter without the flag does not advertise shader-f16. Dawn's own note is that the flag is unsafe because older NVIDIA drivers fail some f16 CTS cases. This run still produced coherent 1818 answers, and those answers are the score.
3. WebLLM's model URL helper appends `resolve/main/`. The first load 404'd. A symlink under `bscore3/work/site/` maps that path onto the local pack. browser3 was not edited. The second load compiled the q4f16_1 wasm and generated.
4. B (q4f32_1 repack), C (MLC native Vulkan), and D (dequantized PyTorch) were not started.

## Browser retrieval

Separate from the headline. n=1, not n=3, so the must denominator is 45 unlocked questions, not 135. Same WebLLM runtime, same system card, same thinking-off template. The passages are the browser's static BM25 index (`browser3/dist/index`), chapter-locked, k=6, ranked by `browser3/scripts/bm25.py` (the same ranking as the demo page). Not bge-m3. File: `runs/merged_mixa_browser_bm25_n1.json`. Generation wall 345 s.

| | must (V / C / both) | leak markers | judge leaks (qwen) | voice qwen · gemma · avg (V / C) | frames | loops | names |
|---|---|---|---|---|---|---|---|
| browser BM25, n=1 | 21/30 · 11/15 · **32/45** | 7/22 | 8/22 | 3.22 · 3.78 · **3.5** (3.79 / 2.91) | 0 | 0 | 1/67 |

Do not read 32/45 against the 100/135 bar. One sample per question is a different experiment. The rate is in the same neighborhood as the bge n=3 pack (94/135). The name hit is "laces" in "De Laces" on Victor m25, sample 0, a broken De Lacey. The Salêve fold does not remove it.

## Leftover

- The headline is bge retrieval, n=3. The BM25 pass above is n=1 only, in its own files, and is not part of the 94/135.
- shader-f16 on this 3090 still depends on the Dawn NVIDIA toggle until the driver is at least 615.71. No driver, Vulkan ICD, display, GRUB, or kernel change was made.
- No Hugging Face upload, no other upload, no git push. The HF token was not printed.

## Not uploaded

This score is local only. Nothing was uploaded and nothing was pushed.
