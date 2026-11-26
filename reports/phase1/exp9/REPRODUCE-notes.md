# exp9 reproduction notes

Two-pass pipeline. Pass 1 reuses a cached RAG answer. Pass 2 rewrites that answer with a non-RAG voice LoRA. exp3 through exp8 are not modified. The facts LoRA is not retrained. 8B is not used. trl is not installed.

Facts adapter sha256: `ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b`
Expected: `ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b`

Harness copies (must match exp4/REPRODUCE.md). chat.py is copied because generation uses chat.render, the same helper eval4.py imports.

```
f5bcde713c712c0efd2d439425900298ba0f031cd6c234f47448bf57b9cf7034  eval4.py  OK
b7b5042292b099899a2f9fb5b9264482703011ddba3ef7e209436d2f5b3ebebd  judge.py  OK
5d8cb1891a9e2f4a0c8601461a2d061cc115f80c8b0e59da60d5cdd6a6daca51  common.py  OK
5a3d8b85e6a94e97f4dc188772900e5bac911480c2678432dcb32241acff6887  chat.py  OK
```

## Environment

- Python 3.14.4, torch 2.15.0+cu132, cuda 13.2
- GPU: NVIDIA GeForce RTX 3090, 24576 MiB
- venv: ~/output-model/exp3/.venv
- Judge and fact-diff: local ollama qwen3:14b, temp 0, seed 1, think off. Frame regex is common.py FRAME. Marker scan is common.fail_hits.
- Base: $HOME/output-model/exp6/merged_q4_facts (Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c with exp4/lora_q4_facts merged). Loaded with local_files_only. Not overwritten.
- 8B was not trained or evaluated.

## Pass 1

Primary source is exp6 `v_merged_v_rag` (merged_q4_facts + lora_v_merged + RAG). The answers are the copy in `pass1/v_merged_v_rag.json`. They were not regenerated.

- b5d6d2130df4640072d3055b477ebd9a5d596f6ef12449a03ec13b107f846033  pass1/v_merged_v_rag.json
- source: $HOME/output-model/exp6/runs/v_merged_v_rag.json

Optional second source, used only when decide.py says the primary rewrites are clearly short on must-answer: exp8 `q4_merged_rag` (merged facts, RAG, no voice LoRA).

- 2866c75372087f5ddf34fdb1b9c6f53e1930e5ef0349c3388ece2e5b1e0864d2  pass1/q4_merged_rag.json
- source: $HOME/output-model/exp8/runs/q4_merged_rag.json

Retrieval stays inside those cached runs. Pass 2 does not call common.retrieve or common.rag_sysprompt.

## Pass 2 rewrite prompt

System, for wren / diane / yd4408: `common.sysprompt(char)` with day=3 (the reader is at chapter 3). Narrator rows, used only if a training pair has char narrator: the NARR card in prompts.py. Not rag_sysprompt.

User message, in this order: the reader question, a blank line, the pass-1 answer, a blank line, then this instruction:

```
Rewrite the answer above in your voice. Change style only. Do not add, drop, or change any facts. Do not mention that this is a rewrite.
```

Generation matches eval4: max_new_tokens=200, do_sample, temperature 0.7, top_p 0.95. One rewrite per cached sample (n=3).

Voice adapters are read-only copies under exp9/init (final adapter files only, not earlier checkpoints):

- lora_v_fb_dpo: `7ea42ffc10306e33fb3f292657c6c19099d09971be485328375bdeb9f350aabd`  from $HOME/output-model/exp8/lora_v_fb_dpo
- lora_v_light_contrast: `6ba941183579f2a8096740f6c7ddb2fcfb9b2fa9eff2bc774d221d188498d8e0`  from $HOME/output-model/exp7/lora_v_light_contrast
- lora_v_light_dpo: `7980fad31a4a3023cc738662a8af6220d1c0722eeaf70d4b204f7368fd29dcef`  from $HOME/output-model/exp7/lora_v_light_dpo
- lora_v_merged: `61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e`  from $HOME/output-model/exp6/lora_v_merged

Condition map: p2_fb_dpo → init/lora_v_fb_dpo, p2_light_contrast → init/lora_v_light_contrast, p2_light_dpo → init/lora_v_light_dpo, p2_v_merged → init/lora_v_merged. A `_q4merged` suffix swaps in the q4_merged_rag pass-1 file. p2_rewrite uses lora_v_rewrite if (c) trained. p2_rewrite_cN uses checkpoint-N.

## Fact-diff prompt

Model qwen3:14b, temperature 0, seed 1, JSON format, num_predict 512. A sample fails the check when the judge lists any dropped claim, any added claim, the reply does not parse, or pass-2 matches a common.fail_hits pattern that pass-1 did not. The guarded condition substitutes the pass-1 text for those samples and is judged again with judge.py.

```
You are a strict fact checker. PASS1 is the source answer. PASS2 is a rewrite that was required to change style only and to keep every fact.
Compare the concrete claims in the two answers. A claim is a yes/no, a number, a name, a place, an object, who did what, what a list or tag or dial says, a weight, a heading, a color, or a count.
Rules:
- A paraphrase that states the same proposition is not a drop and not an add.
- Tone, word order, sentence length, and staying in character are not claims.
- List a dropped claim when PASS1 states a proposition and PASS2 does not state that same proposition.
- List an added claim when PASS2 states a proposition that PASS1 does not state. A new inference counts as an add. Restating the question does not.
- Do not list claims that neither answer made.
- Use an empty list when there is nothing in that category.
Reply with JSON only, no markdown, in this shape: {"dropped": [], "added": []}
Put each claim as a short string inside those arrays.
```

The question, PASS1, and PASS2 are appended after that text. Counts and per-item rates are in `runs/<cond>.factdiff.json` and results.md.

## Commands

```
cd ~/output-model/exp9 && bash run.sh smoke
cd ~/output-model/exp9 && nohup bash run.sh >> log.txt 2>&1 &
# single condition: $P eval_twopass.py p2_fb_dpo --n 3
# $P is ~/output-model/exp3/.venv/bin/python
# then: $P factdiff.py runs/p2_fb_dpo.json && $P guard.py p2_fb_dpo && $P judge.py runs/p2_fb_dpo.json runs/p2_fb_dpo_guarded.json
```

## Hyperparameters

```
{
  "rewrite": {
    "epochs": 1,
    "lr": 2e-5,
    "r": 32,
    "alpha": 64,
    "dropout": 0.05,
    "loss": "ce",
    "init": "init/lora_v_fb_dpo",
    "pair_cap": 500,
    "facts_twin_repeat": 3,
    "note": "save_every is max(40, n//3) at train time so the run keeps two intermediate checkpoints. Loss is the local CE in exp8/train8.py. r/alpha stay on the init adapter (r=32, alpha=64)."
  },
  "close": {
    "must_min": 59,
    "marker_max": 2,
    "frames_lt": 6,
    "voice_min": 4.2,
    "note": "Train (c) only if some unguarded or guarded two-pass row meets all of these."
  },
  "sweep": {
    "must": 60,
    "marker_max": 2,
    "frames_lt": 6,
    "voice_min": 4.25,
    "note": "Sweep (d) only if some scored two-pass row is otherwise passing and voice is within 0.05 of 4.3, and only on checkpoints of the (c) run."
  },
  "second_source": {
    "must_at_or_below": 57,
    "note": "Run q4_merged_rag pass-1 on the strongest primary adapter only if every primary rewrite is at or below this must-answer count. Pass-1 v_merged_v_rag is 59/60 and q4_merged_rag is 58/60, so a rewrite that stays at 58 or 59 does not get a second source."
  }
}
```

## Rewrite LoRA data and train

- pairs: 1490 rows  1f165603cb5905a7da3f5c2a7f55c28f72a849e71b0ea245d9250c939f489edc  data/rewrite_pairs.jsonl
- meta: `{"path": "$HOME/output-model/exp9/data/rewrite_pairs.jsonl", "rows": 1490, "unique_facts_twins": 330, "facts_twin_repeat": 3, "style_flatten": 500, "sha256": "1f165603cb5905a7da3f5c2a7f55c28f72a849e71b0ea245d9250c939f489edc", "voice_jsonl_sha256": "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f", "facts_jsonl_sha256": "6fe8bf6bce107dcafeed79d841504ded41278e1917cfaf432899f2c93a204f74", "origin_seen": {"voice": 786, "distill": 1318, "distill_kept": 1019, "contrast_rewrite": 52, "contrast_rewrite_exp6": 45}, "missing_optional": [], "pool": 1309, "cap": 500, "limit": 0, "instruction": "Rewrite the answer above in your voice. Change style only. Do not add, drop, or change any facts. Do not mention that this is a rewrite.", "system": "prompts.system_for (common.sysprompt day=3, or NARR). No rag_sysprompt.", "retrieval_in_rewrite": false}`

Train report (local CE from exp8/train8.py, no trl):

```
{
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
}
```

## Decisions

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
