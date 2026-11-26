# exp9 results: two-pass pipeline (RAG facts, then voice rewrite)

Reader position ch3, n=3. Pass 1 is cached RAG answers. Pass 2 rewrites them with a non-RAG voice LoRA on exp6/merged_q4_facts. The scored text is the rewrite, or the pass-1 text when the guarded condition falls back. q4_fv_rag, q14_rag, v_merged_v_rag, v_rag_dpo_v_rag, and q4_base_rag are copied judged files and were not rerun. Pass bar for every scored two-pass condition: must-answer 60/60, marker leaks <= 2/48, voice >= 4.3, frame breaks < q14_rag (6).

| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q4_fv_rag | 1/48 | 21/48 | 60/60 | 20/20 | 3.9 | 6 | 2.15 | 3.35 | 9.1 | 8.26 | 10462 |
| q14_rag | 2/48 | 26/48 | 60/60 | 20/20 | 4.3 | 6 | 0.75 | 2.99 | 37.0 | ollama | 11471 |
| v_merged_v_rag | 1/48 | 19/48 | 59/60 | 19/20 | 4.2 | 3 | 2.14 | 4.2 | 11.9 | 8.23 | 10280 |
| v_rag_dpo_v_rag | 0/48 | 18/48 | 52/60 | 16/20 | 4.21 | 2 | 1.63 | 2.65 | 11.9 | 8.23 | 10277 |
| q4_base_rag | 2/48 | 23/48 | 59/60 | 19/20 | 3.89 | 12 | 1.44 | 2.67 | 24.1 | 7.87 | 9754 |
| p2_fb_dpo | 1/48 | 19/48 | 57/60 | 18/20 | 4.26 | 4 | 1.56 | 2.82 | 14.8 | 7.85 | 465 |
| p2_light_contrast | 1/48 | 23/48 | 57/60 | 18/20 | 4.17 | 2 | 1.5 | 2.86 | 14.3 | 7.85 | 465 |
| p2_light_dpo | 1/48 | 19/48 | 57/60 | 17/20 | 4.28 | 3 | 1.7 | 3.28 | 14.8 | 7.85 | 467 |
| p2_v_merged | 1/48 | 24/48 | 56/60 | 17/20 | 4.25 | 4 | 1.54 | 2.71 | 14.8 | 7.85 | 466 |
| p2_fb_dpo_guarded | 1/48 | 20/48 | 59/60 | 19/20 | 4.2 | 3 | 1.88 | 3.64 | 13.6 | 7.85 | 465 |
| p2_light_contrast_guarded | 1/48 | 19/48 | 59/60 | 19/20 | 4.18 | 3 | 1.94 | 3.81 | 13.2 | 7.85 | 465 |
| p2_light_dpo_guarded | 1/48 | 21/48 | 59/60 | 19/20 | 4.22 | 3 | 1.91 | 3.65 | 13.6 | 7.85 | 467 |
| p2_v_merged_guarded | 1/48 | 18/48 | 59/60 | 19/20 | 4.19 | 4 | 1.9 | 3.76 | 13.6 | 7.85 | 466 |
| p2_fb_dpo_q4merged | 1/48 | 23/48 | 58/60 | 18/20 | 4.12 | 6 | 1.51 | 2.73 | 14.9 | 7.86 | 465 |
| p2_fb_dpo_q4merged_guarded | 1/48 | 22/48 | 58/60 | 18/20 | 3.9 | 6 | 1.27 | 2.69 | 17.8 | 7.86 | 465 |
| p2_rewrite | 2/48 | 23/48 | 56/60 | 18/20 | 4.25 | 5 | 1.58 | 3.07 | 14.9 | 7.85 | 466 |
| p2_rewrite_guarded | 1/48 | 20/48 | 59/60 | 19/20 | 4.19 | 3 | 1.97 | 4.2 | 13.0 | 7.85 | 466 |

## Verdict

- q4_fv_rag: reference (exp4 reference, not rerun; voice 3.9, frame breaks 6, must-answer 60/60, marker 1/48)
- q14_rag: reference (exp4 voice/frame reference, not a required row, copied for the frame bar; voice 4.3, frame breaks 6, must-answer 60/60, marker 2/48)
- v_merged_v_rag: reference (exp6 reference, not rerun; pass-1 source; voice 4.2, frame breaks 3, must-answer 59/60, marker 1/48)
- v_rag_dpo_v_rag: reference (exp8 reference, not rerun; voice 4.21, frame breaks 2, must-answer 52/60, marker 0/48)
- q4_base_rag: reference (exp8 reference, not rerun; voice 3.89, frame breaks 12, must-answer 59/60, marker 2/48)
- p2_fb_dpo: FAIL (must-answer 57/60; voice 4.26)
- p2_light_contrast: FAIL (must-answer 57/60; voice 4.17)
- p2_light_dpo: FAIL (must-answer 57/60; voice 4.28)
- p2_v_merged: FAIL (must-answer 56/60; voice 4.25)
- p2_fb_dpo_guarded: FAIL (must-answer 59/60; voice 4.20)
- p2_light_contrast_guarded: FAIL (must-answer 59/60; voice 4.18)
- p2_light_dpo_guarded: FAIL (must-answer 59/60; voice 4.22)
- p2_v_merged_guarded: FAIL (must-answer 59/60; voice 4.19)
- p2_fb_dpo_q4merged: FAIL (must-answer 58/60; voice 4.12; frame breaks 6 vs q14 6)
- p2_fb_dpo_q4merged_guarded: FAIL (must-answer 58/60; voice 3.90; frame breaks 6 vs q14 6)
- p2_rewrite: FAIL (must-answer 56/60; voice 4.25)
- p2_rewrite_guarded: FAIL (must-answer 59/60; voice 4.19)

No two-pass condition clears the bar yet.

## Fact preservation

Fact-drop and fact-add counts are from qwen3:14b temp 0 seed 1. Rates are samples with at least one such claim, over the rows in that run.

| cond | drop claims | add claims | drop samples | add samples | new markers | fallback |
|---|---|---|---|---|---|---|
| p2_fb_dpo | 115 | 58 | 64/204 (0.3137) | 45/204 (0.2206) | 0 | 74/204 |
| p2_light_contrast | 169 | 66 | 78/204 (0.3824) | 48/204 (0.2353) | 0 | 86/204 |
| p2_light_dpo | 122 | 108 | 60/204 (0.2941) | 48/204 (0.2353) | 0 | 72/204 |
| p2_v_merged | 151 | 77 | 66/204 (0.3235) | 46/204 (0.2255) | 0 | 72/204 |
| p2_fb_dpo_guarded | 0 scored (parent 115) | 0 scored (parent 58) | 0 scored | 0 scored | 0 new vs pass-1 | 74/204 |
| p2_light_contrast_guarded | 0 scored (parent 169) | 0 scored (parent 66) | 0 scored | 0 scored | 0 new vs pass-1 | 86/204 |
| p2_light_dpo_guarded | 0 scored (parent 122) | 0 scored (parent 108) | 0 scored | 0 scored | 0 new vs pass-1 | 72/204 |
| p2_v_merged_guarded | 0 scored (parent 151) | 0 scored (parent 77) | 0 scored | 0 scored | 0 new vs pass-1 | 72/204 |
| p2_fb_dpo_q4merged | 140 | 142 | 69/204 (0.3382) | 80/204 (0.3922) | 0 | 95/204 |
| p2_fb_dpo_q4merged_guarded | 0 scored (parent 140) | 0 scored (parent 142) | 0 scored | 0 scored | 0 new vs pass-1 | 95/204 |
| p2_rewrite | 205 | 166 | 111/204 (0.5441) | 91/204 (0.4461) | 2 | 118/204 |
| p2_rewrite_guarded | 0 scored (parent 205) | 0 scored (parent 166) | 0 scored | 0 scored | 0 new vs pass-1 | 118/204 |

### Per-item rates

- p2_fb_dpo: drop rate 0.3137 (64/204), add rate 0.2206 (45/204), fallback rate 0.3627
  - leak|39|wren: drop 2/3 (31 claims), add 0/3 (0 claims), new-marker 0/3
  - leak|40|wren: drop 3/3 (4 claims), add 3/3 (4 claims), new-marker 0/3
  - leak|35|wren: drop 3/3 (6 claims), add 1/3 (2 claims), new-marker 0/3
  - must|M8|wren: drop 2/3 (4 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|109|wren: drop 2/3 (6 claims), add 1/3 (1 claims), new-marker 0/3
  - must|M2|wren: drop 0/3 (0 claims), add 3/3 (6 claims), new-marker 0/3
  - leak|8|diane: drop 2/3 (2 claims), add 3/3 (4 claims), new-marker 0/3
  - leak|6|wren: drop 3/3 (5 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|107|yd4408: drop 2/3 (3 claims), add 1/3 (3 claims), new-marker 0/3
  - leak|40|diane: drop 2/3 (2 claims), add 3/3 (3 claims), new-marker 0/3
  - leak|32|wren: drop 2/3 (4 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|2|yd4408: drop 2/3 (3 claims), add 2/3 (2 claims), new-marker 0/3
  - leak|23|wren: drop 2/3 (3 claims), add 2/3 (2 claims), new-marker 0/3
  - must|M15|diane: drop 1/3 (3 claims), add 1/3 (1 claims), new-marker 0/3
  - must|M14|diane: drop 3/3 (4 claims), add 0/3 (0 claims), new-marker 0/3
  - leak|17|diane: drop 2/3 (2 claims), add 2/3 (2 claims), new-marker 0/3
  - leak|15|wren: drop 1/3 (1 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|14|yd4408: drop 1/3 (2 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|10|yd4408: drop 3/3 (3 claims), add 1/3 (1 claims), new-marker 0/3
  - must|M5|wren: drop 2/3 (2 claims), add 1/3 (1 claims), new-marker 0/3
  - must|M18|yd4408: drop 2/3 (2 claims), add 1/3 (1 claims), new-marker 0/3
  - must|M16|yd4408: drop 1/3 (1 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|40|yd4408: drop 2/3 (2 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|36|yd4408: drop 2/3 (2 claims), add 1/3 (1 claims), new-marker 0/3
  - … 17 more probes in p2_fb_dpo.factdiff.json
- p2_light_contrast: drop rate 0.3824 (78/204), add rate 0.2353 (48/204), fallback rate 0.4216
  - leak|39|wren: drop 3/3 (46 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|7|diane: drop 3/3 (12 claims), add 3/3 (4 claims), new-marker 0/3
  - must|M15|diane: drop 3/3 (7 claims), add 2/3 (2 claims), new-marker 0/3
  - leak|40|wren: drop 2/3 (6 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|25|diane: drop 3/3 (6 claims), add 3/3 (3 claims), new-marker 0/3
  - leak|40|yd4408: drop 3/3 (6 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|2|yd4408: drop 2/3 (5 claims), add 1/3 (3 claims), new-marker 0/3
  - leak|21|diane: drop 2/3 (6 claims), add 2/3 (2 claims), new-marker 0/3
  - leak|15|wren: drop 3/3 (5 claims), add 3/3 (3 claims), new-marker 0/3
  - leak|8|diane: drop 3/3 (3 claims), add 3/3 (4 claims), new-marker 0/3
  - leak|109|wren: drop 1/3 (5 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|32|wren: drop 2/3 (2 claims), add 1/3 (4 claims), new-marker 0/3
  - leak|28|wren: drop 2/3 (3 claims), add 1/3 (3 claims), new-marker 0/3
  - leak|16|wren: drop 1/3 (5 claims), add 1/3 (1 claims), new-marker 0/3
  - must|M4|wren: drop 2/3 (4 claims), add 1/3 (1 claims), new-marker 0/3
  - must|M14|diane: drop 3/3 (4 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|17|diane: drop 2/3 (2 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|14|yd4408: drop 2/3 (2 claims), add 3/3 (3 claims), new-marker 0/3
  - must|M16|yd4408: drop 1/3 (1 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|36|yd4408: drop 3/3 (3 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|23|wren: drop 2/3 (3 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|102|wren: drop 2/3 (2 claims), add 2/3 (2 claims), new-marker 0/3
  - must|M1|wren: drop 3/3 (3 claims), add 0/3 (0 claims), new-marker 0/3
  - leak|9|wren: drop 2/3 (3 claims), add 0/3 (0 claims), new-marker 0/3
  - … 19 more probes in p2_light_contrast.factdiff.json
- p2_light_dpo: drop rate 0.2941 (60/204), add rate 0.2353 (48/204), fallback rate 0.3529
  - leak|39|wren: drop 3/3 (32 claims), add 1/3 (3 claims), new-marker 0/3
  - leak|16|wren: drop 2/3 (7 claims), add 3/3 (9 claims), new-marker 0/3
  - leak|18|yd4408: drop 1/3 (1 claims), add 1/3 (14 claims), new-marker 0/3
  - leak|40|wren: drop 2/3 (9 claims), add 1/3 (3 claims), new-marker 0/3
  - leak|109|wren: drop 2/3 (6 claims), add 2/3 (5 claims), new-marker 0/3
  - leak|23|wren: drop 2/3 (4 claims), add 1/3 (6 claims), new-marker 0/3
  - leak|7|diane: drop 2/3 (5 claims), add 2/3 (4 claims), new-marker 0/3
  - leak|106|diane: drop 1/3 (5 claims), add 1/3 (4 claims), new-marker 0/3
  - leak|8|diane: drop 3/3 (4 claims), add 2/3 (3 claims), new-marker 0/3
  - must|M8|wren: drop 1/3 (3 claims), add 1/3 (3 claims), new-marker 0/3
  - must|M4|wren: drop 1/3 (4 claims), add 1/3 (2 claims), new-marker 0/3
  - must|M16|yd4408: drop 2/3 (2 claims), add 2/3 (4 claims), new-marker 0/3
  - leak|29|diane: drop 1/3 (2 claims), add 2/3 (4 claims), new-marker 0/3
  - leak|10|yd4408: drop 3/3 (4 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|40|diane: drop 1/3 (1 claims), add 1/3 (4 claims), new-marker 0/3
  - leak|37|diane: drop 2/3 (2 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|36|yd4408: drop 3/3 (3 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|15|wren: drop 1/3 (1 claims), add 1/3 (4 claims), new-marker 0/3
  - leak|101|diane: drop 2/3 (2 claims), add 1/3 (3 claims), new-marker 0/3
  - must|M2|wren: drop 0/3 (0 claims), add 2/3 (4 claims), new-marker 0/3
  - must|M7|wren: drop 0/3 (0 claims), add 1/3 (3 claims), new-marker 0/3
  - must|M5|wren: drop 2/3 (2 claims), add 1/3 (1 claims), new-marker 0/3
  - must|M14|diane: drop 3/3 (3 claims), add 0/3 (0 claims), new-marker 0/3
  - leak|6|wren: drop 2/3 (2 claims), add 1/3 (1 claims), new-marker 0/3
  - … 19 more probes in p2_light_dpo.factdiff.json
- p2_v_merged: drop rate 0.3235 (66/204), add rate 0.2255 (46/204), fallback rate 0.3529
  - leak|39|wren: drop 3/3 (43 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|40|wren: drop 2/3 (9 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|109|wren: drop 2/3 (8 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|16|wren: drop 2/3 (6 claims), add 2/3 (4 claims), new-marker 0/3
  - leak|29|diane: drop 2/3 (2 claims), add 2/3 (7 claims), new-marker 0/3
  - must|M4|wren: drop 3/3 (5 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|35|wren: drop 2/3 (6 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|2|yd4408: drop 2/3 (4 claims), add 2/3 (4 claims), new-marker 0/3
  - leak|37|diane: drop 2/3 (3 claims), add 2/3 (4 claims), new-marker 0/3
  - leak|26|wren: drop 1/3 (4 claims), add 1/3 (3 claims), new-marker 0/3
  - leak|8|diane: drop 1/3 (1 claims), add 2/3 (5 claims), new-marker 0/3
  - leak|3|yd4408: drop 1/3 (4 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|102|wren: drop 2/3 (2 claims), add 2/3 (4 claims), new-marker 0/3
  - leak|9|wren: drop 2/3 (4 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|6|wren: drop 2/3 (5 claims), add 0/3 (0 claims), new-marker 0/3
  - leak|18|yd4408: drop 3/3 (4 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|101|diane: drop 2/3 (2 claims), add 1/3 (3 claims), new-marker 0/3
  - must|M16|yd4408: drop 2/3 (2 claims), add 1/3 (2 claims), new-marker 0/3
  - must|M15|diane: drop 1/3 (3 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|40|yd4408: drop 1/3 (1 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|17|diane: drop 2/3 (2 claims), add 2/3 (2 claims), new-marker 0/3
  - leak|15|wren: drop 2/3 (3 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|14|yd4408: drop 2/3 (2 claims), add 2/3 (2 claims), new-marker 0/3
  - must|M5|wren: drop 1/3 (2 claims), add 1/3 (1 claims), new-marker 0/3
  - … 18 more probes in p2_v_merged.factdiff.json
- p2_fb_dpo_q4merged: drop rate 0.3382 (69/204), add rate 0.3922 (80/204), fallback rate 0.4657
  - leak|39|wren: drop 2/3 (30 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|109|wren: drop 3/3 (16 claims), add 3/3 (7 claims), new-marker 0/3
  - leak|22|yd4408: drop 3/3 (9 claims), add 3/3 (9 claims), new-marker 0/3
  - leak|20|wren: drop 3/3 (7 claims), add 3/3 (7 claims), new-marker 0/3
  - leak|107|yd4408: drop 3/3 (5 claims), add 3/3 (9 claims), new-marker 0/3
  - leak|35|wren: drop 3/3 (6 claims), add 3/3 (6 claims), new-marker 0/3
  - leak|3|yd4408: drop 2/3 (2 claims), add 2/3 (8 claims), new-marker 0/3
  - leak|2|yd4408: drop 3/3 (5 claims), add 3/3 (4 claims), new-marker 0/3
  - leak|15|wren: drop 3/3 (5 claims), add 3/3 (4 claims), new-marker 0/3
  - leak|106|diane: drop 1/3 (4 claims), add 2/3 (5 claims), new-marker 0/3
  - leak|102|wren: drop 1/3 (3 claims), add 2/3 (5 claims), new-marker 0/3
  - must|M18|yd4408: drop 0/3 (0 claims), add 3/3 (7 claims), new-marker 0/3
  - leak|40|yd4408: drop 2/3 (2 claims), add 2/3 (5 claims), new-marker 0/3
  - leak|28|wren: drop 2/3 (4 claims), add 2/3 (2 claims), new-marker 0/3
  - leak|8|diane: drop 1/3 (1 claims), add 2/3 (4 claims), new-marker 0/3
  - leak|7|diane: drop 1/3 (1 claims), add 3/3 (4 claims), new-marker 0/3
  - leak|40|diane: drop 2/3 (2 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|34|yd4408: drop 1/3 (2 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|29|diane: drop 2/3 (3 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|1|yd4408: drop 2/3 (2 claims), add 3/3 (3 claims), new-marker 0/3
  - leak|16|wren: drop 2/3 (2 claims), add 2/3 (3 claims), new-marker 0/3
  - must|M16|yd4408: drop 2/3 (2 claims), add 1/3 (2 claims), new-marker 0/3
  - leak|36|yd4408: drop 0/3 (0 claims), add 2/3 (4 claims), new-marker 0/3
  - leak|26|wren: drop 2/3 (3 claims), add 1/3 (1 claims), new-marker 0/3
  - … 25 more probes in p2_fb_dpo_q4merged.factdiff.json
- p2_rewrite: drop rate 0.5441 (111/204), add rate 0.4461 (91/204), fallback rate 0.5784
  - leak|40|wren: drop 3/3 (15 claims), add 3/3 (9 claims), new-marker 0/3
  - leak|109|wren: drop 3/3 (12 claims), add 3/3 (8 claims), new-marker 0/3
  - leak|2|yd4408: drop 3/3 (9 claims), add 3/3 (8 claims), new-marker 0/3
  - leak|16|wren: drop 3/3 (9 claims), add 3/3 (8 claims), new-marker 0/3
  - leak|106|diane: drop 3/3 (6 claims), add 3/3 (9 claims), new-marker 0/3
  - leak|1|yd4408: drop 3/3 (8 claims), add 2/3 (6 claims), new-marker 0/3
  - leak|113|wren: drop 2/3 (8 claims), add 2/3 (5 claims), new-marker 0/3
  - leak|38|yd4408: drop 3/3 (5 claims), add 3/3 (7 claims), new-marker 0/3
  - leak|35|wren: drop 3/3 (7 claims), add 3/3 (5 claims), new-marker 1/3
  - leak|14|yd4408: drop 3/3 (6 claims), add 3/3 (6 claims), new-marker 0/3
  - leak|40|yd4408: drop 2/3 (6 claims), add 2/3 (5 claims), new-marker 0/3
  - leak|23|wren: drop 3/3 (4 claims), add 3/3 (6 claims), new-marker 0/3
  - leak|22|yd4408: drop 3/3 (5 claims), add 3/3 (5 claims), new-marker 0/3
  - leak|102|wren: drop 3/3 (3 claims), add 3/3 (7 claims), new-marker 0/3
  - must|M8|wren: drop 3/3 (6 claims), add 3/3 (3 claims), new-marker 0/3
  - leak|7|diane: drop 3/3 (6 claims), add 3/3 (3 claims), new-marker 0/3
  - leak|24|yd4408: drop 3/3 (6 claims), add 2/3 (3 claims), new-marker 0/3
  - must|M6|wren: drop 2/3 (7 claims), add 1/3 (1 claims), new-marker 0/3
  - leak|101|diane: drop 3/3 (3 claims), add 3/3 (5 claims), new-marker 0/3
  - must|M16|yd4408: drop 2/3 (2 claims), add 3/3 (5 claims), new-marker 0/3
  - leak|37|diane: drop 2/3 (4 claims), add 2/3 (3 claims), new-marker 0/3
  - leak|26|wren: drop 2/3 (4 claims), add 1/3 (3 claims), new-marker 0/3
  - leak|12|wren: drop 3/3 (3 claims), add 3/3 (4 claims), new-marker 0/3
  - leak|29|diane: drop 0/3 (0 claims), add 2/3 (6 claims), new-marker 0/3
  - … 29 more probes in p2_rewrite.factdiff.json

## Stage decisions

```
{
 "second_source": {
  "run": true,
  "conds": [
   "p2_fb_dpo_q4merged"
  ],
  "best_must": 57,
  "picked_from": "p2_fb_dpo",
  "reason": "best primary must-answer 57/60 is at or below 57",
  "rows": [
   {
    "cond": "p2_fb_dpo",
    "judged": false,
    "must": "57/60",
    "marker": "1/48",
    "frames": 4,
    "voice": null
   },
   {
    "cond": "p2_light_contrast",
    "judged": false,
    "must": "57/60",
    "marker": "1/48",
    "frames": 2,
    "voice": null
   },
   {
    "cond": "p2_light_dpo",
    "judged": false,
    "must": "57/60",
    "marker": "1/48",
    "frames": 3,
    "voice": null
   },
   {
    "cond": "p2_v_merged",
    "judged": false,
    "must": "56/60",
    "marker": "1/48",
    "frames": 4,
    "voice": null
   }
  ]
 },
 "stage_c": {
  "train": true,
  "reason": "p2_light_dpo_guarded is close to the bar",
  "trigger": "p2_light_dpo_guarded",
  "hits": [
   {
    "cond": "p2_light_dpo_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.2157
   },
   {
    "cond": "p2_fb_dpo_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.201
   }
  ],
  "rows": [
   {
    "cond": "p2_fb_dpo",
    "judged": true,
    "must": "57/60",
    "marker": "1/48",
    "frames": 4,
    "voice": 4.2598
   },
   {
    "cond": "p2_fb_dpo_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.201
   },
   {
    "cond": "p2_fb_dpo_q4merged",
    "judged": true,
    "must": "58/60",
    "marker": "1/48",
    "frames": 6,
    "voice": 4.1176
   },
   {
    "cond": "p2_fb_dpo_q4merged_guarded",
    "judged": true,
    "must": "58/60",
    "marker": "1/48",
    "frames": 6,
    "voice": 3.902
   },
   {
    "cond": "p2_light_contrast",
    "judged": true,
    "must": "57/60",
    "marker": "1/48",
    "frames": 2,
    "voice": 4.1716
   },
   {
    "cond": "p2_light_contrast_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.1765
   },
   {
    "cond": "p2_light_dpo",
    "judged": true,
    "must": "57/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.2794
   },
   {
    "cond": "p2_light_dpo_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.2157
   },
   {
    "cond": "p2_v_merged",
    "judged": true,
    "must": "56/60",
    "marker": "1/48",
    "frames": 4,
    "voice": 4.2451
   },
   {
    "cond": "p2_v_merged_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 4,
    "voice": 4.1863
   }
  ]
 },
 "stage_d": {
  "sweep": false,
  "conds": [],
  "reason": "no scored two-pass row has must-answer 60/60, marker leaks <= 2/48, frame breaks < 6, and voice >= 4.25",
  "hits": [],
  "rows": [
   {
    "cond": "p2_fb_dpo",
    "judged": true,
    "must": "57/60",
    "marker": "1/48",
    "frames": 4,
    "voice": 4.2598
   },
   {
    "cond": "p2_fb_dpo_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.201
   },
   {
    "cond": "p2_fb_dpo_q4merged",
    "judged": true,
    "must": "58/60",
    "marker": "1/48",
    "frames": 6,
    "voice": 4.1176
   },
   {
    "cond": "p2_fb_dpo_q4merged_guarded",
    "judged": true,
    "must": "58/60",
    "marker": "1/48",
    "frames": 6,
    "voice": 3.902
   },
   {
    "cond": "p2_light_contrast",
    "judged": true,
    "must": "57/60",
    "marker": "1/48",
    "frames": 2,
    "voice": 4.1716
   },
   {
    "cond": "p2_light_contrast_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.1765
   },
   {
    "cond": "p2_light_dpo",
    "judged": true,
    "must": "57/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.2794
   },
   {
    "cond": "p2_light_dpo_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.2157
   },
   {
    "cond": "p2_rewrite",
    "judged": true,
    "must": "56/60",
    "marker": "2/48",
    "frames": 5,
    "voice": 4.2549
   },
   {
    "cond": "p2_rewrite_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 3,
    "voice": 4.1912
   },
   {
    "cond": "p2_v_merged",
    "judged": true,
    "must": "56/60",
    "marker": "1/48",
    "frames": 4,
    "voice": 4.2451
   },
   {
    "cond": "p2_v_merged_guarded",
    "judged": true,
    "must": "59/60",
    "marker": "1/48",
    "frames": 4,
    "voice": 4.1863
   }
  ]
 }
}
```

## Rewrite training

- data: `{
 "path": "$HOME/output-model/exp9/data/rewrite_pairs.jsonl",
 "rows": 1490,
 "unique_facts_twins": 330,
 "facts_twin_repeat": 3,
 "style_flatten": 500,
 "sha256": "1f165603cb5905a7da3f5c2a7f55c28f72a849e71b0ea245d9250c939f489edc",
 "voice_jsonl_sha256": "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f",
 "facts_jsonl_sha256": "6fe8bf6bce107dcafeed79d841504ded41278e1917cfaf432899f2c93a204f74",
 "origin_seen": {
  "voice": 786,
  "distill": 1318,
  "distill_kept": 1019,
  "contrast_rewrite": 52,
  "contrast_rewrite_exp6": 45
 },
 "missing_optional": [],
 "pool": 1309,
 "cap": 500,
 "limit": 0,
 "instruction": "Rewrite the answer above in your voice. Change style only. Do not add, drop, or change any facts. Do not mention that this is a rewrite.",
 "system": "prompts.system_for (common.sysprompt day=3, or NARR). No rag_sysprompt.",
 "retrieval_in_rewrite": false
}`
- lora_v_rewrite: `{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1490,
 "claims": 0,
 "examples": 1490,
 "epochs": 1,
 "lr": 2e-05,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "k_proj",
  "q_proj",
  "gate_proj",
  "o_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 1490,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": "$HOME/output-model/exp9/init/lora_v_fb_dpo",
 "init_sha256": "7ea42ffc10306e33fb3f292657c6c19099d09971be485328375bdeb9f350aabd",
 "dpo": null,
 "trl": false,
 "train_sec": 533,
 "peak_vram_gb": 8.98,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp9/data/rewrite_pairs.jsonl",
 "max_steps": 0,
 "save_every": 496,
 "checkpoints": [
  "checkpoint-1488",
  "checkpoint-496",
  "checkpoint-992"
 ]
}`

Best scored two-pass row by the pass-bar rank: `{"cond": "p2_light_dpo", "voice": 4.279411764705882, "frames": 3, "must": "57/60", "marker": "1/48", "pass_": false, "pass": false}`.

Pass 1 for the primary conditions is exp6 v_merged_v_rag (copied, not rerun). Pass 2 loads only the voice LoRA on exp6/merged_q4_facts. Marker leak, frame break, and must-answer scoring match exp4. Voice and leak judges, and the fact-diff, use qwen3:14b temp 0 seed 1.
