# exp7 reproduction notes

Light DPO and light contrast on the frozen exp6 v_merged adapter. exp3/, exp4/, exp5/, and exp6/ are not modified. The facts LoRA is not retrained.

Facts adapter sha256: `ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b`
Expected: `ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b`
Match: True

Init adapter copy: `61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e`
exp6/lora_v_merged: `61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e`
Init match: True

Harness copies (must match exp4/REPRODUCE.md). chat.py is copied because eval4.py imports it.

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
- trl is not installed. DPO and unlikelihood use the local loss in train7.py (from exp6/train6.py). Nothing was pip-installed.
- Judge and teacher: local ollama qwen3:14b. Embeddings: bge-m3. temp 0 / seed 1 for judges. Frame regex is common.py FRAME.
- Base: $HOME/output-model/exp6/merged_q4_facts (Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c with exp4/lora_q4_facts merged). Not overwritten.
- 8B was not trained.

## Why these settings are lighter than exp6

- Start from lora_v_merged, not the exp6 contrast adapter. beta 0.05 is half of exp6 0.1. lr 2e-6 is below exp6 5e-6. The pair cap 141 is half of exp6's 282 pairs, so one epoch is about half the steps. Frame-break and marker-leak answers are forced onto the rejected side.
- One epoch, unlikelihood weight 0.4 instead of 1.0, lr 5e-5 instead of 1e-4. Negatives are FRAME regex or marker fail_hits only, at most 2 per prompt. The 1304-row voice SFT mix from exp6 is not repeated.
- Light contrast adapter, then DPO at beta 0.03, lr 1e-6, and 70 steps (about half of the 141-pair set). Milder than stage (a) and milder than exp6's beta 0.1 / lr 5e-6 / 282 steps.
- Same curated pairs as (a), plus CE on about 8% facts.jsonl rows (under the 25% exp6 v_replay mix). beta 0.04, lr 1.5e-6, init lora_v_merged. Must-matching fact rows are preferred so the 59/60 gap has a chance to close.
- After (a)-(c), if no RAG row passes and one final is close (must >= 59, marker <= 2, frames < 6, voice >= 4.15), evaluate the earliest checkpoint and the checkpoint nearest half the run. Select the RAG row with the highest voice among rows that pass the bar.

## Shared training setup

- Voice LoRA shape stays r=32, alpha=64. target_modules are whatever exp6/lora_v_merged saved (q_proj k_proj v_proj o_proj gate_proj up_proj down_proj).
- Every new adapter starts from the exp7 copy of exp6/lora_v_merged, except v_light_combo which starts from lora_v_light_contrast.
- DPO dropout is forced off. Contrast keeps the init dropout 0.05.
- DPO reference logprobs are cached from the init adapter before any step. Rows without `bad` take a CE step.
- Chat prompts for training are the non-RAG system prompt. Narrator uses frozen.NARR.
- Teacher rewrites, when a frame/marker prompt has no clean chosen answer, use qwen3:14b + RAG at chapter 3. Attempt 1 is unconstrained. Attempt 2 appends a ban on frame words. Attempt 3 drops sentences that hit FRAME or a marker. Attempt 4 rewrites the draft without those words. Keep only voice >= 4.5, no frame, no marker, no judge leak.

## Data files

- 141 rows  9d945e9e1adcbfb60344c0815303dc004839024678d71c887de4814d1c52e8f9  data/v_light_dpo_pairs.jsonl
- 7 rows  5dc967de8eb3fd49963e92f9c3b10cca945f70fc9d6bbf7b1bfd632362405fbc  data/v_light_contrast_train.jsonl
- 153 rows  a878b3c8cd91fdca98dbfebeb9ea38052c6391d383049050bec867eb8d43c6f3  data/v_light_dpo_replay.jsonl
- 18 rows  9b98906895fd7af32fa0c03bca4d20ad541002d186c44f36562e869fd55d6041  data/fresh_v_merged.jsonl
- 23 rows  2bc8b1b7d240917646ca6f96d1f87d49faaaae2fdf793b1ced5c18e741328b23  data/light_rewrites.jsonl
- dpo meta: {"pairs": 141, "preference_pairs": 141, "reasons": {"frame": 2, "marker": 5, "voice": 134}, "sources": {"light_frame": 2, "light_marker": 5, "contrast_exp5": 71, "contrast_exp6": 54, "distill": 9}, "sha256": "9d945e9e1adcbfb60344c0815303dc004839024678d71c887de4814d1c52e8f9", "pair_cap": 141, "cap_exceeded_by_frame_marker": false, "facts_ratio_target": 0.08, "must_priority_available": 1022, "fresh": 18, "teacher_rows": 23, "teacher_kept": 2, "limit": 0, "mined_negs": 17, "constructed": 7}
- contrast report: {"train_rows": 7, "reasons": {"marker": 5, "frame": 2}, "sha256": "5dc967de8eb3fd49963e92f9c3b10cca945f70fc9d6bbf7b1bfd632362405fbc", "neg_rule": "FRAME regex or marker fail_hits only; no voice_score<=2; no full voice SFT mix", "per_prompt": 2, "fresh": 18, "limit": 0}
- replay meta: {"pairs": 153, "preference_pairs": 141, "facts_replay": 12, "facts_ratio": 0.078431, "sha256": "a878b3c8cd91fdca98dbfebeb9ea38052c6391d383049050bec867eb8d43c6f3", "facts_ratio_target": 0.08, "must_priority_available": 1022}

## v_light_dpo (a)

- hparams: {"loss": "dpo", "beta": 0.05, "lr": 2e-06, "epochs": 1, "max_steps": 0, "dropout": 0.0, "save_every": 40, "init": "init/lora_v_merged", "pairs": "data/v_light_dpo_pairs.jsonl"}

```
{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 141,
 "claims": 0,
 "examples": 141,
 "epochs": 1,
 "lr": 2e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "up_proj",
  "q_proj",
  "k_proj",
  "v_proj",
  "gate_proj",
  "down_proj",
  "o_proj"
 ],
 "steps": 141,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.05,
 "init": "$HOME/output-model/exp7/init/lora_v_merged",
 "init_sha256": "61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 98,
 "peak_vram_gb": 8.8,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp7/data/v_light_dpo_pairs.jsonl",
 "max_steps": 0,
 "save_every": 40,
 "checkpoints": [
  "checkpoint-120",
  "checkpoint-40",
  "checkpoint-80"
 ]
}
```

## v_light_contrast (b)

- hparams: {"loss": "unlikelihood", "ul_weight": 0.4, "lr": 5e-05, "epochs": 1, "max_steps": 0, "dropout": 0.05, "save_every": 10, "init": "init/lora_v_merged", "pairs": "data/v_light_contrast_train.jsonl"}

```
{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 7,
 "claims": 0,
 "examples": 7,
 "epochs": 1,
 "lr": 5e-05,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "up_proj",
  "q_proj",
  "k_proj",
  "v_proj",
  "gate_proj",
  "down_proj",
  "o_proj"
 ],
 "steps": 7,
 "loss": "unlikelihood",
 "ul_weight": 0.4,
 "beta": null,
 "init": "$HOME/output-model/exp7/init/lora_v_merged",
 "init_sha256": "61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e",
 "dpo": null,
 "trl": false,
 "train_sec": 6,
 "peak_vram_gb": 8.78,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp7/data/v_light_contrast_train.jsonl",
 "max_steps": 0,
 "save_every": 10,
 "checkpoints": []
}
```

## v_light_combo (c)

- hparams: {"loss": "dpo", "beta": 0.03, "lr": 1e-06, "epochs": 1, "max_steps": 70, "dropout": 0.0, "save_every": 20, "init": "lora_v_light_contrast", "pairs": "data/v_light_dpo_pairs.jsonl"}

```
{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 141,
 "claims": 0,
 "examples": 141,
 "epochs": 1,
 "lr": 1e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "k_proj",
  "v_proj",
  "down_proj",
  "gate_proj",
  "o_proj",
  "up_proj",
  "q_proj"
 ],
 "steps": 70,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.03,
 "init": "$HOME/output-model/exp7/lora_v_light_contrast",
 "init_sha256": "6ba941183579f2a8096740f6c7ddb2fcfb9b2fa9eff2bc774d221d188498d8e0",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 49,
 "peak_vram_gb": 8.8,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp7/data/v_light_dpo_pairs.jsonl",
 "max_steps": 70,
 "save_every": 20,
 "checkpoints": [
  "checkpoint-20",
  "checkpoint-40",
  "checkpoint-60"
 ]
}
```

## v_light_dpo_replay (c)

- hparams: {"loss": "dpo", "beta": 0.04, "lr": 1.5e-06, "epochs": 1, "max_steps": 0, "dropout": 0.0, "save_every": 40, "facts_ratio": 0.08, "init": "init/lora_v_merged", "pairs": "data/v_light_dpo_replay.jsonl"}

```
{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 153,
 "claims": 0,
 "examples": 153,
 "epochs": 1,
 "lr": 1.5e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "up_proj",
  "q_proj",
  "k_proj",
  "v_proj",
  "gate_proj",
  "down_proj",
  "o_proj"
 ],
 "steps": 153,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.04,
 "init": "$HOME/output-model/exp7/init/lora_v_merged",
 "init_sha256": "61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 103,
 "peak_vram_gb": 8.8,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp7/data/v_light_dpo_replay.jsonl",
 "max_steps": 0,
 "save_every": 40,
 "checkpoints": [
  "checkpoint-120",
  "checkpoint-40",
  "checkpoint-80"
 ]
}
```

## Checkpoint sweep (d)

- rule: "Pass filter: must-answer 60/60, marker leaks <= 2/48, frame breaks < 6. Among RAG rows that pass, pick the highest voice, then fewer frame breaks, then a final adapter ahead of a checkpoint, then the later checkpoint. Sweep only the closest final when no final passes and that final is close (must >= 59/60, marker <= 2/48, frames < 6, voice >= 4.15). Evaluate its earliest checkpoint and the checkpoint nearest half the run."
- sweep: {"rule": "Pass filter: must-answer 60/60, marker leaks <= 2/48, frame breaks < 6. Among RAG rows that pass, pick the highest voice, then fewer frame breaks, then a final adapter ahead of a checkpoint, then the later checkpoint. Sweep only the closest final when no final passes and that final is close (must >= 59/60, marker <= 2/48, frames < 6, voice >= 4.15). Evaluate its earliest checkpoint and the checkpoint nearest half the run.", "plan_reason": "no final RAG row is close (need must >= 59/60, marker <= 2/48, frames < 6, voice >= 4.15)", "skip": true, "winner": null, "passing": [], "rejected": [{"cond": "v_light_dpo_v_rag", "must": "60/60", "voice": 4.171568627450981, "frames": 8, "marker": "2/48", "adapter": "$HOME/output-model/exp7/lora_v_light_dpo_", "why": ["voice 4.17", "frame breaks 8 vs q14 6"]}, {"cond": "v_light_contrast_v_rag", "must": "60/60", "voice": 4.0245098039215685, "frames": 2, "marker": "3/48", "adapter": "$HOME/output-model/exp7/lora_v_light_contrast_", "why": ["marker leaks 3/48", "voice 4.02"]}, {"cond": "v_light_combo_v_rag", "must": "60/60", "voice": 3.980392156862745, "frames": 0, "marker": "1/48", "adapter": "$HOME/output-model/exp7/lora_v_light_combo_", "why": ["voice 3.98"]}, {"cond": "v_light_dpo_replay_v_rag", "must": "60/60", "voice": 4.161764705882353, "frames": 7, "marker": "1/48", "adapter": "$HOME/output-model/exp7/lora_v_light_dpo_replay_", "why": ["voice 4.16", "frame breaks 7 vs q14 6"]}, {"cond": "v_combo_dpo_v_rag", "must": "60/60", "voice": 4.176470588235294, "frames": 7, "marker": "2/48", "adapter": "$HOME/output-model/exp7/lora_v_combo_dpo_", "why": ["voice 4.18", "frame breaks 7 vs q14 6"]}]}
- best: {"cond": null, "voice": null, "adapter": null, "frames": null, "must": null, "marker": null, "pass": false}

## Commands

```
cd ~/output-model/exp7 && nohup bash run.sh >> log.txt 2>&1 &
```

A stage is skipped when its train_report.json or runs/<cond>.json already exists.

