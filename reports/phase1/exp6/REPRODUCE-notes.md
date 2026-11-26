# exp6 reproduction notes

Keep facts, raise voice. exp4/lora_q4_facts is loaded and never trained. exp3/, exp4/, and exp5/ are not modified.

Facts adapter sha256: `ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b`
Expected: `ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b`
Match: True

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
- trl is not installed. DPO uses the local loss in train6.py. Nothing was pip-installed.
- Judge and teacher: local ollama qwen3:14b. Embeddings: bge-m3. temp 0 / seed 1 for judges. Frame regex is common.py FRAME.
- Base: Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c

## Shared training setup

- Default voice LoRA: r=32, alpha=64, dropout 0.05, lr 1e-4, AdamW, 2 epochs, batch 1.
- target_modules: same as the facts adapter (q_proj k_proj v_proj o_proj gate_proj up_proj down_proj).
- Chat prompts for training are the non-RAG system prompt (narrator uses frozen.NARR).
- Teacher rewrites use qwen3:14b + RAG at chapter 3.
- Variant (a) stacks a new voice adapter on frozen exp4/lora_q4_facts.
- Variant (b) merges that facts adapter into a private copy under exp6/merged_q4_facts/, then trains a voice-only LoRA.
- Combo (c) takes the better of (a)/(b) by RAG must-answer, then voice, then fewer frame breaks.
- Optional (d) repeats the winning recipe on Qwen3-8B if the VRAM probe passes. No new download.

## v_replay (a)

- Voice pool: exp4 voice.jsonl overlaid with exp5 distill train / kept / contrast rewrites (same (char,user) prefers later source).
- Facts replay sampled from exp4/data/facts.jsonl at target 25% of the mixed file (inside 20-30%), seed 0.
- 786 rows  12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f  ../exp4/data/voice.jsonl
- 1882 rows  6fe8bf6bce107dcafeed79d841504ded41278e1917cfaf432899f2c93a204f74  ../exp4/data/facts.jsonl
- mix: {"path": "$HOME/output-model/exp6/data/v_replay_train.jsonl", "rows": 1739, "voice_rows": 1304, "facts_rows": 435, "facts_ratio": 0.250144, "facts_ratio_target": 0.25, "sha256": "6bd0acdd2454dbfdfad9bbcae5d4335d072376d135dbe43b13c745a9c985a24e", "sources": {"distill": 260, "distill_kept": 992, "facts_replay": 435, "contrast_rewrite": 52}, "limit": 0, "seed": 0, "voice_jsonl": "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f", "facts_jsonl": "6fe8bf6bce107dcafeed79d841504ded41278e1917cfaf432899f2c93a204f74"}
- 1739 rows  6bd0acdd2454dbfdfad9bbcae5d4335d072376d135dbe43b13c745a9c985a24e  data/v_replay_train.jsonl
- Train: train6.py --stage voice --r 32 --epochs 2 --lr 1e-4 --facts exp4/lora_q4_facts --pairs data/v_replay_train.jsonl
- Eval: v_replay_fv and v_replay_fv_rag

```
{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "facts_sha256": "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b",
 "q4": false,
 "pairs": 1739,
 "claims": 0,
 "examples": 1739,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "q_proj",
  "k_proj",
  "up_proj",
  "o_proj",
  "v_proj",
  "down_proj",
  "gate_proj"
 ],
 "steps": 3478,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 931,
 "peak_vram_gb": 11.19,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_replay_train.jsonl"
}
```

## v_merged (b)

- Merge: PeftModel.from_pretrained(Qwen3-4B, exp4/lora_q4_facts).merge_and_unload() -> exp6/merged_q4_facts/ (does not write the HF cache).
- merge: {"base": "Qwen/Qwen3-4B", "facts": "$HOME/output-model/exp4/lora_q4_facts", "facts_sha256": "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b", "out": "$HOME/output-model/exp6/merged_q4_facts", "q4": false, "dtype": "bfloat16", "n_params": 4022468096}
- Voice-only LoRA on the merged base. Same voice positives as (a), no facts mix.
- mix: {"path": "$HOME/output-model/exp6/data/v_merged_train.jsonl", "rows": 1304, "voice_rows": 1304, "facts_rows": 0, "facts_ratio": 0.0, "facts_ratio_target": 0.0, "sha256": "67b7bec5aa464a788aa09adafabc339ad3ddf8ac9dd5f6757e24deb594c2869e", "sources": {"distill_kept": 992, "distill": 260, "contrast_rewrite": 52}, "limit": 0, "seed": 0, "voice_jsonl": "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f", "facts_jsonl": "6fe8bf6bce107dcafeed79d841504ded41278e1917cfaf432899f2c93a204f74"}
- 1304 rows  67b7bec5aa464a788aa09adafabc339ad3ddf8ac9dd5f6757e24deb594c2869e  data/v_merged_train.jsonl
- Train: train6.py --stage voice --r 32 --epochs 2 --base exp6/merged_q4_facts --pairs data/v_merged_train.jsonl
- Eval: v_merged_v and v_merged_v_rag (adapters=v, no second facts adapter)

```
{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1304,
 "claims": 0,
 "examples": 1304,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 2608,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 544,
 "peak_vram_gb": 11.08,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_merged_train.jsonl"
}
```

## winner (a vs b)

- Rank: RAG must-answer, then voice, then fewer frame breaks.
- choice: {"winner": "v_merged", "mode": "v", "rag_cond": "v_merged_v_rag", "base": "$HOME/output-model/exp6/merged_q4_facts", "facts": null, "voice": "$HOME/output-model/exp6/lora_v_merged", "voice_dir": "lora_v_merged", "must": 59, "must_n": 60, "voice_score": 4.200980392156863, "frames": 3, "marker": 1, "q14_frames": 6, "ranking": [{"name": "v_replay", "mode": "fv", "rag": "v_replay_fv_rag", "must": "59/60", "voice": 4.107843137254902, "frames": 8, "marker": "0/48", "rank": [59, 4.107843137254902, -8], "passes": false}, {"name": "v_merged", "mode": "v", "rag": "v_merged_v_rag", "must": "59/60", "voice": 4.200980392156863, "frames": 3, "marker": "1/48", "rank": [59, 4.200980392156863, -3], "passes": false}], "reason": "RAG must-answer first, then voice, then fewer frame breaks. Winner v_merged must 59/60 voice 4.200980392156863 frames 3."}

## v_combo contrast

- Negatives: exp4 q4_fv_rag / q4_v_rag, all exp5 *_rag judged files, exp6 RAG judged files, plus fresh samples from the winning adapter.
- A negative hits common.py FRAME or has voice_score <= 2. At most 3 bad answers per prompt.
- Rewrites: qwen3:14b + RAG, keep when voice >= 4.5, no frame, no marker, no judge leak (one retry).
- Loss: token unlikelihood on the bad completion (weight 1.0) plus CE on the good completion. r=32, 2 epochs, lr 1e-4.
- If winner is (a), the contrast train file keeps ~25% facts replay so facts do not drift.
- contrast: {"mined": 350, "fresh": 777, "fresh_frame": 2, "fresh_break": 92, "negs": 186, "neg_origins": {"q4_v_rag": 30, "exp5:v_distill_fv_rag": 14, "q4_fv_rag": 29, "exp5:v_long_fv_rag": 8, "exp5:v_r32_fv_rag": 7, "fresh:v_merged": 81, "exp5:v_contrast_fv_rag": 10, "exp5:v_dpo_fv_rag": 2, "exp6:v_replay_fv_rag": 2, "exp6:v_merged_v_rag": 3}, "neg_reasons": {"frame+break": 11, "frame": 1, "break": 174}, "rewrite_groups": 125, "rewrites_kept": 45, "rewrites_rejected": 80, "pairs": 66, "train_rows": 1330, "facts_rows": 0, "facts_ratio": 0.0, "train_sha256": "e4d60546fb5a8eca1412244f2f148931ea035e3d864ae6c029274d9acc6c4d19", "break_rule": "frame regex or voice_score <= 2", "loss": "unlikelihood", "ul_weight": 1.0, "teacher": "qwen3:14b + RAG ch3", "fresh_model": "v_merged v", "winner": "v_merged", "limit": 0}
- 1330 rows  e4d60546fb5a8eca1412244f2f148931ea035e3d864ae6c029274d9acc6c4d19  data/v_contrast_train.jsonl

```
{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1330,
 "claims": 0,
 "examples": 1330,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 2660,
 "loss": "unlikelihood",
 "ul_weight": 1.0,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 586,
 "peak_vram_gb": 12.09,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_contrast_train.jsonl"
}
```

## v_combo DPO

- Starts from lora_v_combo_contrast. Skip below 16 preference pairs.
- Pairs: exp5 distill_pairs + exp5 contrast_pairs + exp6 contrast_pairs. If winner is (a), a 20% facts CE mix is appended (rows without bad).
- Loss: local DPO, beta 0.1, sum of completion-token logprobs, ref cached from init. Dropout off. lr 5e-6, 1 epoch, AdamW.
- choice: {"skip": false, "adapter": "$HOME/output-model/exp6/lora_v_combo_contrast", "name": "v_combo_contrast", "pairs": 282, "mode": "v", "winner": "v_merged", "reason": "DPO starts from the contrast adapter as specified"}
- pairs meta: {"pairs": 282, "preference_pairs": 282, "distill": 156, "contrast_exp5": 71, "contrast_exp6": 55, "facts_replay": 0, "sha256": "106fb2808efa259eba87f9bc4db6d246270b8f6f539674d3b303bf1c88cc8dc8", "facts_ratio": 0.0}
- 282 rows  106fb2808efa259eba87f9bc4db6d246270b8f6f539674d3b303bf1c88cc8dc8  data/v_dpo_pairs.jsonl

```
{
 "base": "$HOME/output-model/exp6/merged_q4_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 282,
 "claims": 0,
 "examples": 282,
 "epochs": 1,
 "lr": 5e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "gate_proj",
  "v_proj",
  "up_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "down_proj"
 ],
 "steps": 282,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.1,
 "init": "$HOME/output-model/exp6/lora_v_combo_contrast",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 190,
 "peak_vram_gb": 8.81,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_dpo_pairs.jsonl"
}
```

## optional 8B (d)

- Base: Qwen/Qwen3-8B from the local HF cache (exp3 m8b). No download.
- Probe: load bf16, require peak <= 18.5 GB and >= 4.5 GB free. Else try 4-bit. Else skip.
- Recipe: train lora_q8_facts on exp4 facts.jsonl + claims, then the winning (a) or (b) path, then the same contrast train file and DPO pairs.
- probe: {"ok": true, "q4": false, "peak_gb": 15.26, "free_after_gb": 7.9, "reason": "bf16 8B fits with headroom for LoRA train/eval"}
- skipped: no

### lora_q8_facts

```
{
 "base": "Qwen/Qwen3-8B",
 "stage": "facts",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1882,
 "claims": 39,
 "examples": 1950,
 "epochs": 2,
 "lr": 0.0001,
 "r": 16,
 "alpha": 32,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 3900,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 870,
 "peak_vram_gb": 22.48,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp4/data/facts.jsonl"
}
```

### lora_v8_merged

```
{
 "base": "$HOME/output-model/exp6/merged_q8_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1304,
 "claims": 0,
 "examples": 1304,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 2608,
 "loss": "ce",
 "ul_weight": null,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 587,
 "peak_vram_gb": 19.75,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_merged_train.jsonl"
}
```

### lora_v8_combo_contrast

```
{
 "base": "$HOME/output-model/exp6/merged_q8_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 1330,
 "claims": 0,
 "examples": 1330,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "alpha": 64,
 "dropout": 0.05,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 2660,
 "loss": "unlikelihood",
 "ul_weight": 1.0,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 628,
 "peak_vram_gb": 21.12,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_contrast_train.jsonl"
}
```

### lora_v8_combo_dpo

```
{
 "base": "$HOME/output-model/exp6/merged_q8_facts",
 "stage": "voice",
 "facts": null,
 "facts_sha256": null,
 "q4": false,
 "pairs": 282,
 "claims": 0,
 "examples": 282,
 "epochs": 1,
 "lr": 5e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "q_proj",
  "o_proj",
  "gate_proj",
  "k_proj",
  "v_proj",
  "up_proj",
  "down_proj"
 ],
 "steps": 282,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.1,
 "init": "$HOME/output-model/exp6/lora_v8_combo_contrast",
 "dpo": "sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows",
 "trl": false,
 "train_sec": 193,
 "peak_vram_gb": 16.94,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp6/data/v_dpo_pairs.jsonl"
}
```

## Eval

- exp6/eval_cond.py calls the copied eval4.py with --base / --facts / --voice / --adapters / --rag.
- n=3. Judge is the copied judge.py. A condition is skipped when runs/<cond>.json already exists.
- A train stage is skipped when lora_<name>/train_report.json exists.
- Reference runs q14_rag and q4_fv_rag are copied from exp4 and are not rerun.

## Commands

```
cd ~/output-model/exp6 && nohup bash run.sh >> log.txt 2>&1 &
```
