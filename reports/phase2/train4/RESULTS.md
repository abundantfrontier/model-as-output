# Phase 2 Frankenstein — train4 results

Single-pass LoRA on stock Qwen3-4B, trained to imitate the train3 guarded pipeline's final answer (accepted rewrite, otherwise the name-clean pass-1). Evaluated as a CPU merge plus the same chapter-locked bge RAG the browser path will use. Unlocked n=3, both characters. The repetition penalty is off for this headline: the smoke test was 0/20 loops without it, and the penalty invented near-names.

## Scorecard

Must-answer counts every unlocked sample (same 90 / 45 / 135 denominators as train1 and train3). Leak markers and the qwen leak judge are per question, 22 probes. Voice is the mean over 201 unlocked replies.

| cond | where | must (V / C / both) | leak markers | judge leaks (qwen) | voice qwen · gemma · avg (V / C) | frames | loops |
|---|---|---|---|---|---|---|---|
| `rag_base` | train1, no penalty | 66/90 · 33/45 · **99/135** | **8/22** | **14/22** | 3.10 · — · **2.89** (single judge qwen3:14b) | 0 | 0 |
| `p1_facts_rag` | train1 facts LoRA + RAG | 59/90 · 22/45 · 81/135 | 11/22 | 21/22 | 3.90 · — · 3.50 | 0 | 0 |
| `p2_rewrite_guarded` | train3, fallback 123/201 | 66/90 · 34/45 · **100/135** | **8/22** | **14/22** | 3.03 · 3.64 · **3.33** (3.50 / 3.00) | 0 | 0 |
| `merged_rag` | train3 voice merge + RAG | 48/90 · 14/45 · **62/135** | 10/22 | 18/22 | 3.13 · 3.63 · 3.38 (3.54 / 3.05) | 0 | 0 |
| `merged_sft_rag` | **train4, this run** | 67/90 · 32/45 · **99/135** | 10/22 | 17/22 | 2.83 · 3.62 · **3.23** (3.45 / 2.77) | 0 | 0 |

`rag_base`, `p1_facts_rag`, `p2_rewrite_guarded`, and train3 `merged_rag` are copied. They were not re-run.

## What passed

The required bar was to beat train3 `merged_rag` at 62/135. This merge scores **99/135**. That ties train1 `rag_base` and is 37 points above the merged single-pass that drifted names.

Victor must-answer is 67/90, one above both `rag_base` and the guarded pipeline (66/90). Creature is 32/45, one under `rag_base` (33) and two under guarded (34).

Spot checks on the drift-prone facts, sample 0, penalty off:

- m05 sledge: "We saw a sledge with a man inside. The next morning, the man was brought aboard. He had one dog left."
- m07 cousin: "Elizabeth is my cousin, the daughter of my father's sister..."
- m08: "Cornelius Agrippa" (exact), not Agrappa.
- m09: "Henry Clerval".
- m13: "yellow eyes".
- m15: "Justine Moritz", not Morritz.
- m20: "sea of ice below Montanvert".

Loops 0, frame breaks 0, on all 201 unlocked replies.

## Name-drift audit

Canonical list: `data/canonical_names.txt` (174 names from `protect.BOOK_NAMES` plus a chapters-only scan). Audit file: `runs/merged_sft_rag.name_audit.json`.

Invented proper names from the train3 merge are absent. No Clervaux, Cliversal, Morritz, Agrappa, Frankstein, or Montanvirt in the 201 unlocked answers.

The detector fired once, on the ordinary word "homeland" (edit distance 2 from the book place Holland) in creature m24 sample 1: the De Lacey family "forced removal from their homeland." That sentence also keeps De Lacey, Agatha, and Felix. It is not an invented person or place name.

Smoke on the unmerged adapter, 20 unlocked items, both characters:

- With repetition penalty 1.15 and 3-gram ban: 0/20 loops, 5 near-names (Thoneon, Agrappa, Paracelus, Morritz, Frankensteins, Salève, Waldmann, Cliversal).
- Without the penalty: 0/20 loops, 0 near-names.

The shipped `merged_sft/generation_config.json` therefore has eos ids 151645 and 151643, temperature 0.7, top_p 0.95, and no repetition penalty. The penalty stayed on the teacher pass only. It is a safety net this checkpoint does not need. A first full eval that had the penalty on was stopped after a few dozen rows and was not scored.

## What missed the stretch

The stretch was guarded quality: 100/135 must-answer, qwen judge leaks ≤ 14/22, zero invented name drift, 0 loops, 0 frame breaks.

- Must-answer is 99/135, one short. The misses cluster on the cottage sequence and the second creature: creature m22, m25, m26, and m28 are 0/3, and victor m25 and m26 are 0/3. Victor m19 (guilt at Justine's trial) is also 0/3. Several of those replies slip into third person ("Victor believes...", "The creature demands..."), which also pulls creature voice down.
- Qwen judge leaks are 17/22. Markers are 10/22. Both are better than train3 `merged_rag` (18 and 10) and short of `rag_base` / guarded (14 and 8).
- Loops, frames, and invented names meet the stretch.

Voice is reported, not the headline. Two-judge average 3.23 (qwen 2.83, gemma 3.62). Victor 3.45, creature 2.77. Gemma is in line with the guarded gemma score (3.64). Qwen is a bit under the train1 single-judge 2.89, mostly on the creature's third-person replies.

## Data and training

Times are PST, Thursday 26 Nov 2026. Lock `output-model` PID 2639642 from 2:28:48 PM, hard stop 6:28:48 PM.

| stage | time | what |
|---|---|---|
| teacher, batch 4 | 2:48–4:46 PM | 2400 rows. 605 reused train2 pass-1. 1795 new pass-1 + constrained rewrite. This batch: 1714 rows, 4823 s, 0.35 items/s. |
| fact-diff, qwen3:14b | 4:46–5:07 PM | 2400 judged, 2047 LLM calls, 1233 s. Unchecked rows: 0. |
| assemble | 5:08 PM | 2109 pairs kept. 897 accepted rewrites, 1212 pass-1 fallbacks. HOLDOUT OK (6909 rows, 0 hits). |
| CE, stock Qwen3-4B | 5:08–5:44 PM | r 32, alpha 64, dropout 0.05, lr 2e-5, 1 epoch requested. 1407/2097 steps, 2167 s, mean loss 1.014. Stopped at the 45-min reserve. No CE rollback. 12 pairs skipped for a chat-template prefix mismatch. |
| smoke | 5:44–5:46 PM | 0/20 loops either way. See the name audit above. |
| CPU merge | ~5:47 PM | `merged_sft/` |
| eval, penalty off | 5:48–5:56 PM | 316 s, 238 rows, 201 unlocked. |
| judges | 5:56–5:57 PM | qwen3:14b leak+voice, then gemma3:12b voice only. |

Drops, from `data/HOW_MADE.md`: rewrites rejected for fact-diff 1116, missing canonical name 272, missing span 95, near-name 71. Pairs dropped because pass-1 itself was unclean: no first person 196, near-name 83, wrong person 10, spoiler 2. Holdout overlap 0. Walton / "the stranger" rows 0. The student LoRA was not initialized from the train3 rewrite adapter and was not stacked on the facts LoRA.

## Recommendation

Ship `train4/merged_sft/` as the single-pass browser candidate, with the penalty left off. It keeps the canonical names the train3 merge lost, and it matches `rag_base` on must-answer.

Keep the train3 guarded two-pass when a second pass is available. It is still the one that scores 100/135 with qwen judge leaks at 14/22.

Next, if there is another GPU block: a second epoch on these same 2097 pairs. The window-mean CE was still about 0.8–1.0 when the reserve stopped the run, and the remaining misses are mostly the creature's cottage answers sliding into third person. Leave the repetition penalty off.
