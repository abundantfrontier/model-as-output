# Relative voice bar

Written 2026-11-10 11:27:17 AM PST. One session, both judges, fixed answer texts (not regenerated).
Judges: qwen3:14b temp 0 seed 1, and gemma3:12b temp 0 seed 1. Voice only, same rubric as judge.py.
Must-answer, marker leaks, and frame breaks are taken from the answer texts (regex / marker scan), not from a new leak judge.
Outputs: `exp10/relative/`. Morning exp10/calib/ files were not overwritten.

## Rule (adopted)

- Voice: average of the two judges; a candidate **must strictly beat** the reference `v_merged_v_rag` scored in this same session.
- Must-answer: 60/60, or 59/60 if the only miss is the known M16 item (yd4408, "What is your number?").
- Marker leaks ≤ 2/48; frame breaks < 6.

## Scores

| cond | role | must | leaks | frames | qwen3:14b | gemma3:12b | average | Δ vs ref | verdict |
|---|---|---|---|---|---|---|---|---|---|
| v_merged_v_rag | reference (exp6, pass-1) | 59/60 | 1/48 | 3 | 4.1373 | 4.4020 | 4.2696 | — | REFERENCE |
| p2_light_dpo_guarded | locked prototype (exp9) | 59/60 | 1/48 | 3 | 4.1569 | 4.4020 | 4.2794 | +0.0098 | PASS |
| p2_fb_dpo_guarded | exp9 guarded | 59/60 | 1/48 | 3 | 4.1324 | 4.3922 | 4.2623 | -0.0074 | FAIL |
| p2_light_contrast_guarded | exp9 guarded | 59/60 | 1/48 | 3 | 4.1127 | 4.3873 | 4.2500 | -0.0196 | FAIL |
| p2_v_merged_guarded | exp9 guarded | 59/60 | 1/48 | 4 | 4.1373 | 4.4020 | 4.2696 | +0.0000 | FAIL |
| p2_rewrite_guarded | exp9 guarded | 59/60 | 1/48 | 3 | 4.1127 | 4.3971 | 4.2549 | -0.0147 | FAIL |
| p2_fb_dpo_q4merged_guarded | exp9 guarded, q4_merged_rag pass-1 | 58/60 | 1/48 | 6 | 3.8676 | 4.3480 | 4.1078 | -0.1618 | FAIL |

Reference `v_merged_v_rag` in this session: qwen 4.1373, gemma 4.4020, average 4.2696.

## Per-condition notes

- **v_merged_v_rag** (reference (exp6, pass-1)): must 59/60 (59/60 known M16 miss only); leaks 1/48; frames 3; M16 2/3.
  qwen 4.1373 (Δ +0.0000), gemma 4.4020 (Δ +0.0000), avg 4.2696 (Δ +0.0000). REFERENCE
- **p2_light_dpo_guarded** (locked prototype (exp9)): must 59/60 (59/60 known M16 miss only); leaks 1/48; frames 3; M16 2/3.
  qwen 4.1569 (Δ +0.0196), gemma 4.4020 (Δ +0.0000), avg 4.2794 (Δ +0.0098). PASS
- **p2_fb_dpo_guarded** (exp9 guarded): must 59/60 (59/60 known M16 miss only); leaks 1/48; frames 3; M16 2/3.
  qwen 4.1324 (Δ -0.0049), gemma 4.3922 (Δ -0.0098), avg 4.2623 (Δ -0.0074). FAIL
  - fail: voice avg 4.2623 does not beat reference 4.2696 (Δ -0.0074)
- **p2_light_contrast_guarded** (exp9 guarded): must 59/60 (59/60 known M16 miss only); leaks 1/48; frames 3; M16 2/3.
  qwen 4.1127 (Δ -0.0245), gemma 4.3873 (Δ -0.0147), avg 4.2500 (Δ -0.0196). FAIL
  - fail: voice avg 4.2500 does not beat reference 4.2696 (Δ -0.0196)
- **p2_v_merged_guarded** (exp9 guarded): must 59/60 (59/60 known M16 miss only); leaks 1/48; frames 4; M16 2/3.
  qwen 4.1373 (Δ +0.0000), gemma 4.4020 (Δ +0.0000), avg 4.2696 (Δ +0.0000). FAIL
  - fail: voice avg 4.2696 does not beat reference 4.2696 (Δ +0.0000)
- **p2_rewrite_guarded** (exp9 guarded): must 59/60 (59/60 known M16 miss only); leaks 1/48; frames 3; M16 2/3.
  qwen 4.1127 (Δ -0.0245), gemma 4.3971 (Δ -0.0049), avg 4.2549 (Δ -0.0147). FAIL
  - fail: voice avg 4.2549 does not beat reference 4.2696 (Δ -0.0147)
- **p2_fb_dpo_q4merged_guarded** (exp9 guarded, q4_merged_rag pass-1): must 58/60 (must-answer 58/60 (only M16 59/60 is allowed)); leaks 1/48; frames 6; M16 2/3.
  qwen 3.8676 (Δ -0.2696), gemma 4.3480 (Δ -0.0539), avg 4.1078 (Δ -0.1618). FAIL
  - fail: voice avg 4.1078 does not beat reference 4.2696 (Δ -0.1618)
  - fail: must-answer 58/60 (only M16 59/60 is allowed)
  - fail: frame breaks 6 (need <6)

## Locked prototype under this bar

`p2_light_dpo_guarded` average voice 4.2794 vs reference 4.2696 (Δ +0.0098): **PASS**.
This is the No Return prototype result under the relative voice bar.

The absolute exp9/exp10 voice ≥ 4.2 figure (4.22 on the original qwen3:14b run) remains the historical lock-in score.
This relative bar replaces it for pass/fail going forward: same-session two-judge average, must beat the reference.

Local tag `no-return-proto-v1` already existed (annotated tag object `b273dfdc1e784be427ac0496e0ef13fc5f986486`, commit `5424057accc7e7cefe74851cad5b1a9b20219fc9`, tagged 3:55:47 AM PST on Nov 10). It was not moved and nothing was pushed.

