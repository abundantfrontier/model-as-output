# Frankenstein train2 results

The voice rewrite does not loop. It also does not beat the train1 RAG-only baseline.

The milestone was: beat `rag_base` on must-answer, then leak markers, then voice, with 0 frame breaks and 0 token loops, unlocked n=3. Must-answer is the first tie-break. Every train2 row is under 99/135, so the milestone is not met.

Use the train1 RAG-only model (`rag_base`): stock Qwen3-4B, character card, chapter-locked RAG, no LoRA. Do not ship the merged voice model.

1818 text. Reader position varies per question. Unlocked turns only (character already on the page). 37 lock-line rows are out of these rates. Must-answer is a gist regex. Leak-marker is the probe's `fail_if_mentions`. The leak judge and the voice judge are local Ollama `qwen3:14b`, temperature 0, seed 1.

## Scorecard (unlocked)

Train1 rows are copied from `../train1/RESULTS.md`. They were not regenerated, except the extra penalty check on the same baseline.

| cond | where | must (V / C / both) | leak markers | judge leaks | voice (V / C / both) | frames | loops |
|---|---|---|---|---|---|---|---|
| `rag_base` | train1, no repetition penalty | 66/90 · 33/45 · **99/135** | **8/22** | **14/22** | 3.10 · 2.45 · **2.89** | 0 | 0 |
| `rag_base_penalty` | train2 rerun, penalty on | 62/90 · 29/45 · 91/135 | 10/22 | 16/22 | 2.61 · 2.12 · 2.45 | 0 | 0 |
| `p1_facts_rag` | train1 facts LoRA + RAG | 59/90 · 22/45 · 81/135 | 11/22 | 21/22 | 3.90 · 2.68 · 3.50 | 0 | 0 |
| `p2_voice` | train1 two-pass, old voice LoRA | 1/90 · 0/45 · 1/135 | 0/22 | 0/22 | 1.78 · 1.64 · 1.73 | 0 | 110 |
| `p2_voice_guarded` | train1, guard fell back 200/201 | 59/90 · 22/45 · 81/135 | 11/22 | 21/22 | 3.90 · 2.68 · 3.50 | 0 | 0 |
| `merged_rag` | train1 facts+voice merged | 0/90 · 0/45 · 0/135 | 0/22 | 0/22 | 1.57 · 1.58 · 1.57 | 0 | 127 |
| `p2_rewrite` | train2 two-pass, new DPO rewrite LoRA | 39/90 · 18/45 · 57/135 | 7/22 | 11/22 | 3.32 · 2.74 · 3.13 | 0 | 0 |
| `p2_rewrite_guarded` | train2, fact-diff fallback 105/201 | 55/90 · 28/45 · 83/135 | 7/22 | 12/22 | 3.22 · 2.61 · 3.02 | 0 | 0 |
| `merged_rag` | train2 voice LoRA merged into stock Qwen3-4B + RAG | 46/90 · 20/45 · 66/135 | 10/22 | 17/22 | 3.61 · 2.77 · 3.34 | 0 | 0 |

`rag_base_penalty` is a regression check, not the official baseline. The published comparison is the train1 row (99/135, voice 2.89).

## What passed

The loop failure from train1 is gone. The early smoke test on 20 unlocked items (both characters) was 0/20 loops with the repetition penalty and 0/20 without it, on the CE adapter and again after light DPO. Full eval stayed at 0/201 loops and 0 frame breaks on the unguarded rewrite, the guarded rewrite, and the merged model.

The repetition penalty is a safety net, not the fix. The same 20 items were clean with the penalty turned off. A full `rag_base` rerun with the penalty still had 0 loops, and it scored worse than the train1 run that did not use it (must 91/135 vs 99/135, judge leaks 16/22 vs 14/22, voice 2.45 vs 2.89). Eval calls pass `eos_token_id` [151645, 151643], `repetition_penalty=1.15`, `no_repeat_ngram_size=3`, temperature 0.7, top_p 0.95, max_new_tokens 200, seed 1. The merged `generation_config.json` has the same eos ids, penalty, and n-gram block.

Leaks moved the right way on the two-pass rows. Unguarded judge leaks are 11/22 against 14/22 for `rag_base`. At sample level the rewrite cleared 6 answers the baseline judge had marked as leaks, and it added none. Markers are 7/22 against 8/22. Voice is a little higher (3.13 unguarded, 3.02 guarded, 3.34 merged) than 2.89.

The guard is doing real work, and it is no longer a total reject. Train1 threw away 200 of 201 unlocked rewrites. This run fell back on 105 of 201 and kept 96. Kept rewrites are why guarded must-answer rises from 57/135 to 83/135.

## What failed

Must-answer fell. The rewrite is a paraphrase, and the gist regex needs the book's words.

`m05` / Victor, pass-1 hit, rewrite miss: "We saw a sledge with a man in it on the ice" became "I saw a man on the sea ice, tied to his sled." "Sled" does not match `sledge`.

`m07` / Victor: "Elizabeth is my cousin, the daughter of my father's sister" became "Elizabeth is the niece of my mother." The cousin/Italy gist is gone.

The smoke set showed the same drift on names: "Henry Clerval" came out "Henry Cliversal", "Cornelius Agrippa" came out "Cornelis Agrippus", "Montanvert" came out "Montanvirt". Forty-nine unlocked must samples that pass-1 got right were missed after the rewrite. Fact-diff did not catch all of those wording swaps, so the guard could not put them back.

The merged browser candidate is the weakest accurate row of the three train2 runs: must 66/135, leak markers 10/22, judge leaks 17/22. Voice 3.34 does not offset that. It does not loop. It still should not ship.

Light DPO (120 steps, epoch loss 0.424, frozen reference, sum of token logprobs) did not add loops. It also did not fix the paraphrase problem. The CE-only adapter was smoked, not full-eval'd. On the 20-item smoke set its must hits were 10/20 against 9/20 after DPO, and it made the same "sled" swap. A full CE eval is unlikely to clear 99/135.

## Timings (PST, Wed Nov 25)

GPU lock taken 10:12 AM by `hold_gpu.sh` PID 3518483. The 10:24 session death left pass-1 at 346/610. Resumed 10:51 without taking a second lock.

| stage | time | wall |
|---|---|---|
| pass-1 resume (264 of 610) | 10:51–10:59 | 496 s |
| teacher `qwen3:14b` | 11:00–11:07 | 384 s (472 kept, 138 rejected) |
| rewrite CE, 551 steps, lr 2e-5 | 11:09–11:13 | 226 s, epoch CE 0.631, no early stop |
| loop smoke, CE | 11:14–11:17 | 0/20 with and without penalty |
| light DPO, 120 steps, lr 2e-6 | 11:18–11:20 | 91 s, epoch loss 0.424 |
| loop smoke, DPO | 11:20–11:23 | 0/20 with and without penalty |
| two-pass unguarded | 11:23–11:33 | 570 s |
| fact-diff | 11:34–11:37 | fallback 105/238 |
| judge unguarded + guarded | 11:38–11:41 | |
| CPU merge into `merged_voice/` | 11:38–11:41 | |
| merged RAG eval | 11:42–11:48 | 382 s |
| `rag_base` with penalty | 11:51–11:57 | 344 s |

Data: 551 rewrite pairs (471 teacher, 80 style-flatten of cleaned gold). Holdout check passed. Details in `data/HOW_MADE.md`.

## Milestone

Not met. Tie-break is must-answer, then leaks, then voice, and the row must have 0 loops and 0 frames.

`p2_rewrite_guarded` is the best train2 row: 83/135 must, 7/22 markers, judge 12/22, voice 3.02, 0 loops, 0 frames. It leaks less and sounds a bit more in character than `rag_base`, and it loses 16 must-answer samples. That is a loss.

## Recommended model

**`rag_base` (train1).** Stock Qwen3-4B with chapter-locked RAG. No voice LoRA.

Among the new rows, if a rewrite must be inspected, read `p2_rewrite_guarded`, not the merge. The merge raises voice and drops facts.

## Next step

Keep training a rewrite, not a Q&A voice. Reject a teacher rewrite that drops a proper name from pass-1 or swaps a gist word the must regex needs (`sledge`, `cousin`, `Agrippa`, `Clerval`, `Montanvert`, `yellow eye`). The current fact-diff lets those through, which is why the guard only recovered must-answer to 83/135. Do not stack this adapter on the facts LoRA, and do not init from train1 `lora_voice`. The repetition penalty can stay on eval calls. It is not the lever that removes loops, and on the baseline it costs must-answer and voice.
