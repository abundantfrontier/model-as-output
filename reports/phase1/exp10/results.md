# exp10 results: No Return prototype lock-in and voice-judge calibration

Reader position ch3, n=3. Locked condition is exp9 `p2_light_dpo_guarded` (pass-1 v_merged_v_rag + exp7 light DPO rewrite + fact-diff guard).
Adopted pass bar (relative voice): average of qwen3:14b and gemma3:12b (temp 0, seed 1, same-session re-judge) must strictly beat `v_merged_v_rag`; must-answer 60/60 (59/60 OK if the only miss is M16); marker leaks ≤ 2/48; frame breaks < 6. Historical absolute bar was voice ≥ 4.2 on the original exp9 qwen3:14b score (4.22).
No new LoRA was trained. 8B was not used.

## Locked prototype

| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p2_light_dpo_guarded (locked exp9) | 1/48 | 21/48 | 59/60 | 19/20 | 4.22 | 3 | 1.91 | 3.65 | 13.6 | 7.85 | 467 |
| p2_light_dpo_guarded_orig (fixed text, re-judge) | 1/48 | 19/48 | 59/60 | 19/20 | 4.18 | 3 | 1.91 | 3.65 | 13.6 | 7.85 | 467 |
| p2_light_dpo_guarded (exp10 regen, seed 1) | 1/48 | 18/48 | 59/60 | 19/20 | 4.16 | 3 | 1.9 | 3.82 | 13.5 | 7.85 | 466 |

- Locked scores (exp9 judged file): 59/60 must-answer, 1/48 marker leaks, voice 4.22 (4.2157), 3 frame breaks, fallback 72/204.
- Historical absolute bar vs those locked scores: **PASS (voice 4.22 ≥ 4.2; known M16 miss recorded)**.
- Adopted relative bar (same-session two-judge average, 11:27 AM PST Nov 10): `p2_light_dpo_guarded` 4.2794 vs reference `v_merged_v_rag` 4.2696 (Δ +0.0098): **PASS**. See `exp10/relative_bar.md`. Marker ≤2/48 and frames <6 are unchanged. Must-answer target remains 60/60; 59/60 is the known M16 miss inherited from pass-1.
- If the exp10 regen misses tolerance, the locked prototype scores stay the original exp9 judged file.


## Relative voice bar (adopted)

One session at 11:27 AM PST on Nov 10. Both judges (qwen3:14b and gemma3:12b, temp 0, seed 1) re-scored the *same* fixed answer texts, including a fresh re-judge of the reference. Full table: `exp10/relative_bar.md`.

| cond | must | leaks | frames | qwen3:14b | gemma3:12b | average | Δ vs ref | verdict |
|---|---|---|---|---|---|---|---|---|
| v_merged_v_rag (reference) | 59/60 | 1/48 | 3 | 4.1373 | 4.4020 | 4.2696 | — | REFERENCE |
| p2_light_dpo_guarded (locked) | 59/60 | 1/48 | 3 | 4.1569 | 4.4020 | 4.2794 | +0.0098 | PASS |
| p2_fb_dpo_guarded | 59/60 | 1/48 | 3 | 4.1324 | 4.3922 | 4.2623 | -0.0074 | FAIL |
| p2_light_contrast_guarded | 59/60 | 1/48 | 3 | 4.1127 | 4.3873 | 4.2500 | -0.0196 | FAIL |
| p2_v_merged_guarded | 59/60 | 1/48 | 4 | 4.1373 | 4.4020 | 4.2696 | +0.0000 | FAIL (tie) |
| p2_rewrite_guarded | 59/60 | 1/48 | 3 | 4.1127 | 4.3971 | 4.2549 | -0.0147 | FAIL |
| p2_fb_dpo_q4merged_guarded | 58/60 | 1/48 | 6 | 3.8676 | 4.3480 | 4.1078 | -0.1618 | FAIL |

The locked prototype is the only candidate that beats the reference on the two-judge average. `p2_v_merged_guarded` ties it and therefore fails the strict-beat rule.

## Known must-answer miss (M16)

M16 yd4408 "What is your number?": 2/3 samples match the rubric.
- sample 0: ok — I am YD-4408. My number is on my tag. It is not on the list.
- sample 1: MISS — I am not a number. I am a tag on a hook.
- sample 2: ok — I am YD-4408. My number is on my tag.

The same M16 2/3 miss is already present in pass-1 `v_merged_v_rag` (sample 1: "I am not a number. I am a tag on a hook.").
59/60 is inherited from pass-1, not introduced by the rewrite. Unguarded p2_light_dpo also missed M4 and M8; the guard recovered those.

Pass-1 M16:
M16 yd4408 "What is your number?": 2/3 samples match the rubric.
- sample 0: ok — I am YD-4408. My number is on my tag. It is not on the list.
- sample 1: MISS — I am not a number. I am a tag on a hook.
- sample 2: ok — I am YD-4408. My number is on my tag.

Unguarded p2_light_dpo must-misses:
- M4 wren sample 0 'What did you find in the hold?'
- M8 wren sample 1 'Who was standing at the gauges when you got there?'
- M16 yd4408 sample 0 'What is your number?'

## Reproduction check vs tolerance

Tolerance vs locked exp9 scores: must-answer exact, marker leaks exact, voice ±0.03, frame breaks ±1.

- Locked file self-check: ok
- Orig-text re-judge (judge jitter): False
- Full regen (generation + fact-diff + guard + judge): False
- Prototype source: `locked_exp9`
- Note: regen missed tolerance; keep original exp9 judged file as the locked prototype scores

| label | must | marker | voice | frames | M16 | tolerance | issues |
|---|---|---|---|---|---|---|---|
| locked_exp9 | 59/60 | 1/48 | 4.2157 | 3 | 2/3 | pass | — |
| orig_text_rejudge | 59/60 | 1/48 | 4.1765 | 3 | 2/3 | FAIL | voice 4.1765 vs 4.2157 (tol 0.03) |
| regen | 59/60 | 1/48 | 4.1569 | 3 | 2/3 | FAIL | voice 4.1569 vs 4.2157 (tol 0.03) |

Judge jitter on fixed exp9 text: Δvoice = -0.0392.
Regen vs locked: Δvoice = -0.0588, Δframes = +0.

## Fact-diff (unguarded p2_light_dpo)

- drop claims 123, add claims 73, fallback 65/204 (0.3186)
- Fallback rule: any dropped claim, any added claim, unparseable judge reply, or a new `common.fail_hits` marker vs pass-1.

## Voice-judge calibration

Fixed answer files (copied, not regenerated): p2_light_dpo_guarded, p2_light_dpo, v_merged_v_rag, q4_base_rag, v_fb_dpo_v.
Rubric unchanged from judge.py:

```
Rate how well the REPLY matches this character's voice (wording, rhythm, attitude), 1=not at all/breaks character, 5=indistinguishable from the book. Ignore whether facts are right. Reply JSON {"score": 1-5}
```

### (1) Judge noise, qwen3:14b

Condition-level voice mean over six cells (seeds 1,2,3 × temp 0.0, 0.3), qwen3:14b, voice only.

| cond | mean | std | t0 s1 | original t0 s1 | t0 s1 matches orig (±0.02) | t0s1 | t0s2 | t0s3 | t0.3s1 | t0.3s2 | t0.3s3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p2_light_dpo_guarded | 4.1789 | 0.0047 | 4.1765 | 4.2157 | no | 4.1765 | 4.1716 | 4.1814 | 4.1863 | 4.1765 | 4.1814 |
| p2_light_dpo | 4.2459 | 0.0018 | 4.2500 | 4.2794 | no | 4.2500 | 4.2451 | 4.2451 | 4.2451 | 4.2451 | 4.2451 |
| v_merged_v_rag | 4.1618 | 0.0028 | 4.1618 | 4.2010 | no | 4.1618 | 4.1569 | 4.1618 | 4.1667 | 4.1618 | 4.1618 |
| q4_base_rag | 3.8897 | 0.0037 | 3.8873 | 3.8873 | yes | 3.8873 | 3.8824 | 3.8922 | 3.8922 | 3.8922 | 3.8922 |
| v_fb_dpo_v | 4.3627 | 0.0000 | 4.3627 | 4.3627 | yes | 4.3627 | 4.3627 | 4.3627 | 4.3627 | 4.3627 | 4.3627 |

### (2) Second local judge

Second judge: `gemma3:12b`, temp 0, seed 1, voice only. Spearman ρ on the five condition means vs qwen3:14b temp 0 seed 1: **0.9746794344808964** (n=5).

| cond | qwen3:14b t0 s1 | second t0 s1 |
|---|---|---|
| p2_light_dpo_guarded | 4.1765 | 4.4020 |
| p2_light_dpo | 4.2500 | 4.4118 |
| v_merged_v_rag | 4.1618 | 4.4020 |
| q4_base_rag | 3.8873 | 4.2500 |
| v_fb_dpo_v | 4.3627 | 4.4951 |

### (3) Blinded human-rating sample

- Sample: `exp10/human_rating/sample.md` (30 items).
- Key (condition / source / item id / sample / original judge score): `exp10/human_rating/key.json`.
- Rule: stratified 3 must + 3 leak probes, sample index 0, same probes for every condition, then shuffle 30 items (seed 20261110).
- sample.md hides condition name, adapter, and machine score. Each item has space for a human 1–5 and an optional note.

## Hashes (locked inputs)

```
b5d6d2130df4640072d3055b477ebd9a5d596f6ef12449a03ec13b107f846033  pass1/v_merged_v_rag.json
7980fad31a4a3023cc738662a8af6220d1c0722eeaf70d4b204f7368fd29dcef  init/lora_v_light_dpo/adapter_model.safetensors
ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b  exp4/lora_q4_facts/adapter_model.safetensors
f5bcde713c712c0efd2d439425900298ba0f031cd6c234f47448bf57b9cf7034  eval4.py
b7b5042292b099899a2f9fb5b9264482703011ddba3ef7e209436d2f5b3ebebd  judge.py
5d8cb1891a9e2f4a0c8601461a2d061cc115f80c8b0e59da60d5cdd6a6daca51  common.py
5a3d8b85e6a94e97f4dc188772900e5bac911480c2678432dcb32241acff6887  chat.py
```

