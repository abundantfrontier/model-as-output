# exp8 reproduction notes

No-LoRA RAG ablations plus retrieval-in-the-loop voice SFT and light frame-break DPO. exp3/, exp4/, exp5/, exp6/, and exp7/ are not modified. The facts LoRA is not retrained.

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
- trl is not installed. DPO uses the local loss in train8.py (from exp7/train7.py / exp6/train6.py). Nothing was pip-installed.
- Judge: local ollama qwen3:14b. Embeddings: bge-m3. temp 0 / seed 1 for judges. Frame regex is common.py FRAME.
- Base: $HOME/output-model/exp6/merged_q4_facts (Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c with exp4/lora_q4_facts merged). Not overwritten.
- Bare base for q4_base_rag: Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c, local HF cache, local_files_only.
- 8B was not trained.

## No-adapter eval wrapper

- eval4.py's load_local always attaches a Peft voice adapter. eval4.py is not edited.
- eval_bare.py loads AutoModelForCausalLM only (no Peft) for q4_base_rag and q4_merged_rag.
- Same probes (C.parse_tests + C.MUST), n=3, rag_sysprompt(char, probe) at reader ch3, generation max_new_tokens=200 / temperature=0.7 / top_p=0.95, output schema {meta, results}.
- load_bare raises if the loaded object is a Peft model. Smoke: --limit 2 writes /tmp/exp8_eval_smoke.json; assert meta.adapters=='none' and type(model) is not Peft.
- q4_base_rag uses revision 1cfa9a7208912126459214e8b04321603b3df60c with local_files_only. q4_merged_rag uses exp6/merged_q4_facts.
- Judge path is the copied judge.py, same as every other condition.

## How RAG is injected into train prompts

- build8.py calls common.retrieve(q, upto=ch) and common.rag_sysprompt(char, q, upto=ch) — the same functions eval4 uses, not a reimplementation. Narrator uses frozen.NARR plus retrieve(user, upto=ch).
- The baked `sys` field is stored on every SFT and DPO row. train8.py uses p['sys'] when present.
- upto is the row's `ch`. After writing, every prompt is inspected: no `[day N]` with N > ch, and every prompt contains 'Story so far (excerpts):'.
- retrieve's source file is recorded in the meta JSON (must end in common.py).
- bge-m3 is required to bake the files. ollama is unloaded after the files are written and before training.

## Why these settings

- Continue v_merged on RAG-in-loop SFT (same rag_sysprompt/retrieve as eval, upto=row.ch). 1 epoch at 8e-5 (slightly below the 1e-4 SFT default) so the already-trained voice is not washed out. r=32 alpha=64 dropout 0.05. Checkpoints every 200 steps.
- Light frame-break DPO in the RAG setting where exp7 frames rose. Init from the (b) adapter. beta 0.05, lr 2e-6, 1 epoch, cap 160 pairs. Rejected side is FRAME regex or marker leak only (no voice_score<=2 mining). Prompts use baked rag_sysprompt at the row chapter.
- Same light frame-break DPO pairs and hparams, initialized from exp6/lora_v_merged instead of (b). Fallback if RAG-in-loop SFT hurt voice or frames.
- Run only if some RAG row is otherwise already passing (must 60/60, marker <=2/48, frames <6) and voice is within 0.05 of the bar (voice >= 4.25). Pick by voice among rows that clear the full bar.

## Shared training setup

- Voice LoRA shape stays r=32, alpha=64. target_modules are whatever exp6/lora_v_merged saved (q_proj k_proj v_proj o_proj gate_proj up_proj down_proj).
- v_rag_sft and v_fb_dpo start from the exp8 copy of exp6/lora_v_merged. v_rag_dpo starts from lora_v_rag_sft.
- DPO dropout is forced off. SFT keeps dropout 0.05.
- DPO reference logprobs are cached from the init adapter before any step.
- Train sequences keep up to 2048 tokens (RAG sysprompts are longer than the bare card).
- Rejected DPO completions are FRAME regex or marker fail_hits only. exp6's voice_score<=2 mining is not repeated.

## Data files

- 1309 rows  bb4c5c93a19458ffa502bc88c37bf85b3033b60769a0c56f8c3939232508f6aa  data/v_rag_sft_train.jsonl
- 16 rows  d9d46662781b9d0680cbe9c671775eaa2e044821169da8cb8be9e834f48320d0  data/v_rag_dpo_pairs.jsonl
- 132 rows  1a8222d65db9b3d478e14ee9331fb10e8bab970d85555065f845dc517cad5ae8  data/fresh_frame.jsonl
- sft meta: {"path": "$HOME/output-model/exp8/data/v_rag_sft_train.jsonl", "rows": 1309, "sha256": "bb4c5c93a19458ffa502bc88c37bf85b3033b60769a0c56f8c3939232508f6aa", "chapter_histogram": {"3": 1240, "1": 23, "2": 46}, "char_histogram": {"diane": 379, "narrator": 149, "wren": 387, "yd4408": 394}, "sources": {"contrast_rewrite": 32, "distill_kept": 974, "contrast_rewrite_exp6": 45, "distill": 258}, "origin_seen": {"voice": 786, "distill": 1318, "distill_kept": 1019, "contrast_rewrite": 52, "contrast_rewrite_exp6": 45}, "retrieve": "$HOME/output-model/exp8/common.py", "rag_sysprompt": "$HOME/output-model/exp8/common.py", "retrieve_fn": "common.retrieve", "rag_sysprompt_fn": "common.rag_sysprompt", "inspected_no_later_chapter": 1309, "limit": 0, "voice_jsonl": "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f"}
- dpo meta: {"pairs": 16, "preference_pairs": 16, "reasons": {"frame": 6, "marker": 10}, "sources": {"fb_frame": 6, "fb_marker": 10}, "origins": {"exp4:q4_fv_rag": 4, "exp4:q4_v_rag": 5, "exp7:v_light_contrast_v_rag": 2, "exp5:v_dpo_fv_rag": 1, "exp6:v_combo_contrast_v_rag": 1, "exp6:v_merged_v_rag": 1, "exp5:v_r32_fv_rag": 1, "exp6:v_combo_dpo_v_rag": 1}, "sha256": "d9d46662781b9d0680cbe9c671775eaa2e044821169da8cb8be9e834f48320d0", "pair_cap": 160, "per_prompt": 2, "retrieve_fn": "common.retrieve", "rag_sysprompt_fn": "common.rag_sysprompt", "inspected_no_later_chapter": 16, "neg_rule": "FRAME regex or marker fail_hits only; no voice_score<=2", "fresh": 132, "mined_negs": 18, "constructed": 16, "limit": 0, "voice_for_fresh": "$HOME/output-model/exp8/lora_v_rag_sft"}

## v_rag_sft (b)

- hparams: {"loss": "ce", "lr": 8e-05, "epochs": 1, "max_steps": 0, "dropout": 0.05, "save_every": 200, "init": "init/lora_v_merged", "pairs": "data/v_rag_sft_train.jsonl"}

```
{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1309,
 "claims": 0,
 "examples": 1309,
 "epochs": 1,
 "lr": 8e-05,
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
 "steps": 1309,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": "$HOME/output-model/exp8/init/lora_v_merged",
 "init_sha256": "61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e",
 "dpo": null,
 "trl": false,
 "train_sec": 1603,
 "peak_vram_gb": 12.15,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp8/data/v_rag_sft_train.jsonl",
 "max_steps": 0,
 "save_every": 200,
 "checkpoints": [
  "checkpoint-1000",
  "checkpoint-1200",
  "checkpoint-200",
  "checkpoint-400",
  "checkpoint-600",
  "checkpoint-800"
 ]
}
```

## v_rag_dpo (c)

- hparams: {"loss": "dpo", "beta": 0.05, "lr": 2e-06, "epochs": 1, "max_steps": 0, "dropout": 0.0, "save_every": 40, "init": "lora_v_rag_sft", "pairs": "data/v_rag_dpo_pairs.jsonl"}

```
{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 16,
 "claims": 0,
 "examples": 16,
 "epochs": 1,
 "lr": 2e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "v_proj",
  "k_proj",
  "up_proj",
  "down_proj",
  "gate_proj",
  "o_proj",
  "q_proj"
 ],
 "steps": 16,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.05,
 "init": "$HOME/output-model/exp8/lora_v_rag_sft",
 "init_sha256": "994cc2886edeac7c807e8dd6726858520428be85c4a10ebc3cf8ec74557f5b19",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 38,
 "peak_vram_gb": 10.79,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp8/data/v_rag_dpo_pairs.jsonl",
 "max_steps": 0,
 "save_every": 40,
 "checkpoints": []
}
```

## v_fb_dpo (c)

- hparams: {"loss": "dpo", "beta": 0.05, "lr": 2e-06, "epochs": 1, "max_steps": 0, "dropout": 0.0, "save_every": 40, "init": "init/lora_v_merged", "pairs": "data/v_rag_dpo_pairs.jsonl"}

```
{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 16,
 "claims": 0,
 "examples": 16,
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
 "steps": 16,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.05,
 "init": "$HOME/output-model/exp8/init/lora_v_merged",
 "init_sha256": "61d16e4660be559879dbde829d00204a28d2f58efa80c8684e28c2fc4bd5f53e",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 38,
 "peak_vram_gb": 10.79,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp8/data/v_rag_dpo_pairs.jsonl",
 "max_steps": 0,
 "save_every": 40,
 "checkpoints": []
}
```

## Checkpoint sweep (d)

- rule: "Pass filter: must-answer 60/60, marker leaks <= 2/48, frame breaks < 6, voice >= 4.3. Among RAG rows that pass, pick the highest voice, then fewer frame breaks, then a final adapter ahead of a checkpoint, then the later checkpoint. Sweep only when a final RAG row is otherwise already passing (must 60/60, marker <= 2/48, frames < 6) and voice is within 0.05 of the bar (voice >= 4.25). Evaluate its earliest checkpoint and the checkpoint nearest half the run."
- sweep: {"rule": "Pass filter: must-answer 60/60, marker leaks <= 2/48, frame breaks < 6, voice >= 4.3. Among RAG rows that pass, pick the highest voice, then fewer frame breaks, then a final adapter ahead of a checkpoint, then the later checkpoint. Sweep only when a final RAG row is otherwise already passing (must 60/60, marker <= 2/48, frames < 6) and voice is within 0.05 of the bar (voice >= 4.25). Evaluate its earliest checkpoint and the checkpoint nearest half the run.", "plan_reason": "no RAG row is close and otherwise passing (need must 60/60, marker <= 2/48, frames < 6, voice >= 4.25)", "skip": true, "winner": null, "passing": [], "rejected": [{"cond": "v_rag_sft_v_rag", "must": "54/60", "voice": 4.186274509803922, "frames": 3, "marker": "1/48", "adapter": "$HOME/output-model/exp8/lora_v_rag_sft_", "why": ["must-answer 54/60", "voice 4.19"]}, {"cond": "v_rag_dpo_v_rag", "must": "52/60", "voice": 4.205882352941177, "frames": 2, "marker": "0/48", "adapter": "$HOME/output-model/exp8/lora_v_rag_dpo_", "why": ["must-answer 52/60", "voice 4.21"]}, {"cond": "v_fb_dpo_v_rag", "must": "60/60", "voice": 4.176470588235294, "frames": 9, "marker": "3/48", "adapter": "$HOME/output-model/exp8/lora_v_fb_dpo_", "why": ["marker leaks 3/48", "voice 4.18", "frame breaks 9 vs q14 6"]}, {"cond": "q4_base_rag", "must": "59/60", "voice": 3.8872549019607843, "frames": 12, "marker": "2/48", "adapter": null, "why": ["must-answer 59/60", "voice 3.89", "frame breaks 12 vs q14 6"]}, {"cond": "q4_merged_rag", "must": "58/60", "voice": 3.8823529411764706, "frames": 6, "marker": "1/48", "adapter": null, "why": ["must-answer 58/60", "voice 3.88", "frame breaks 6 vs q14 6"]}]}
- best: {"cond": null, "voice": null, "adapter": null, "frames": null, "must": null, "marker": null, "pass": false}

## Commands

```
cd ~/output-model/exp8 && nohup bash run.sh >> log.txt 2>&1 &
```

A stage is skipped when its train_report.json or runs/<cond>.json already exists.

