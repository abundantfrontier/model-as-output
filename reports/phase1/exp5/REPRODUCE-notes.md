# exp5 reproduction notes

Voice adapters only. exp4/lora_q4_facts is loaded and never trained. exp4/ and exp3/ are not modified.

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
- trl is not installed. v_dpo uses the local loss in train5.py. Nothing was pip-installed.
- Judge and teacher: local ollama qwen3:14b. Embeddings: bge-m3. temp 0 / seed 1 for judges. Frame regex is common.py FRAME.

## Shared training setup

- Base: Qwen/Qwen3-4B
- Facts adapter (frozen, active): exp4/lora_q4_facts
- Voice stage matches exp4/train4.py: second LoRA, facts requires_grad false, both adapters active, AdamW, batch 1, lr 1e-4 except DPO.
- Rank 32 uses lora_alpha 64 (2*r) and dropout 0.05, same target modules as the facts adapter.
- Chat prompts for training are the non-RAG system prompt (narrator uses exp4/gen.py NARR). Teacher answers are generated with RAG.
- v_long and v_r32 and v_distill call exp4/train4.py. v_contrast and v_dpo call exp5/train5.py.

## v_long

- Data: exp4/data/voice.jsonl
- 786 rows  12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f  ../exp4/data/voice.jsonl
- epochs 4, r 32, alpha 64, lr 1e-4, dropout 0.05
- Output: lora_v_long
- Eval: v_long_fv and v_long_fv_rag, n=3

```
{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "q4": false,
 "pairs": 786,
 "claims": 0,
 "examples": 990,
 "epochs": 4,
 "lr": 0.0001,
 "r": 32,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 3960,
 "train_sec": 1044,
 "peak_vram_gb": 10.77,
 "prefix_mismatch_skipped": 0,
 "limit": 0
}
```

## v_r32

- Same data and rank as v_long, epochs 2. This separates rank from the extra epochs.
- Output: lora_v_r32
- Ran before distill because it is cheap next to teacher generation and the ablation is part of variant a.

```
{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "q4": false,
 "pairs": 786,
 "claims": 0,
 "examples": 990,
 "epochs": 2,
 "lr": 0.0001,
 "r": 32,
 "targets": [
  "down_proj",
  "gate_proj",
  "k_proj",
  "o_proj",
  "q_proj",
  "up_proj",
  "v_proj"
 ],
 "steps": 1980,
 "train_sec": 516,
 "peak_vram_gb": 10.76,
 "prefix_mismatch_skipped": 0,
 "limit": 0
}
```

## v_distill

- Teacher: qwen3:14b, RAG at chapter 3, same rag_sysprompt as the q14_rag condition, num_predict 200.
- First sample temp 0.7 seed 10000+i. If it fails the filter, one retry temp 0.3 seed 20000+i.
- Narrator is not in q14_rag. Those prompts use gen.py's narrator line plus the same chapter-locked retrieval.
- Keep when voice score >= 4.5 (float, so an integer 4 is rejected), no FRAME hit, no marker (fail_hits), no judge leak.
- Judge prompts match exp4/judge.py. temp 0, seed 1, json format, think off.
- Mix: a kept answer replaces the exp4 voice row for that prompt. A rejected voice prompt keeps the original voice.jsonl answer. Kept reader prompts are added.
- Train: exp4/train4.py --stage voice --epochs 2 --r 32 on data/v_distill_train.jsonl
- One teacher answer per unique (char, user). voice.jsonl has 786 rows and 711 unique prompts; a repeated prompt shares that answer. The train mix still walks every voice.jsonl row.
- Reader cap: first 1000 rows of exp3/data/reader_q.jsonl, deduped against those prompts.
- voice.jsonl 786 rows  12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f  ../exp4/data/voice.jsonl
- reader_q.jsonl 1011 rows  2a1d7d5e208cdf7f252e933c1e1a9d3b728d5900ff2dbf5b0e2011b01b1b9541  ../exp3/data/reader_q.jsonl

```
{
 "voice_prompts": 711,
 "unique_prompts": 1544,
 "attempts": 2227,
 "kept": 1019,
 "rejected": 525,
 "reasons": {
  "voice": 333,
  "judge_leak": 189,
  "frame": 3
 },
 "score_hist": {
  "5.0": 1471,
  "2.0": 300,
  "4.0": 77,
  "3.0": 351,
  "1.0": 17
 },
 "dpo_pairs": 158,
 "voice_rows_added": 299,
 "voice_rows_replaced": 487,
 "reader_rows_added": 575,
 "train_rows": 1318,
 "train_sha256": "9d0c38a22d072a4ffc565fcd1ce0ade37af79315e199e0f3bf478db2f93fe8e6",
 "voice_sha256": "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f",
 "threshold": 4.5,
 "judge": "qwen3:14b temp 0 seed 1",
 "teacher": "qwen3:14b + rag_sysprompt ch3, temp 0.7 then 0.3, num_predict 200",
 "reader_prompts_in_manifest": 833
}
```

- train file 1318 rows  9d0c38a22d072a4ffc565fcd1ce0ade37af79315e199e0f3bf478db2f93fe8e6  data/v_distill_train.jsonl

## v_contrast

- Negatives: exp4 runs q4_fv_rag and q4_v_rag, plus exp5 *_fv_rag judged files that exist when the stage starts, plus fresh samples.
- A negative hits common.py FRAME or has voice_score <= 2. Score 1 is the judge's break-character bin. Score 2 is included so the set is not only a handful of rows. At most 3 bad answers per prompt.
- Fresh samples: Qwen3-4B + frozen facts + exp4 lora_q4_voice, RAG system prompt, temperature 0.7, top_p 0.95, max_new_tokens 200. Prompts are voice.jsonl plus the leak and must probes.
- Retrieval is computed before the 4B model is loaded. ollama is unloaded before that load.
- Each negative prompt is rewritten by qwen3:14b + RAG and kept only if it passes the same filter as v_distill (one retry).
- Train loss: token unlikelihood on the bad completion (weight 1.0) plus the usual CE on the good completion. Rows without a bad answer are CE only, including exp4 voice.jsonl prompts that were not rewritten.
- A rewritten prompt replaces the voice.jsonl row so the two targets do not fight. epochs 2, r 32, alpha 64, lr 1e-4.
- Output: lora_v_contrast. Train file data/v_contrast_train.jsonl.

```
{
 "mined": 224,
 "fresh": 777,
 "fresh_frame": 3,
 "fresh_break": 99,
 "negs": 184,
 "neg_origins": {
  "q4_v_rag": 31,
  "exp5:v_distill_fv_rag": 17,
  "q4_fv_rag": 30,
  "exp5:v_long_fv_rag": 10,
  "exp5:v_r32_fv_rag": 8,
  "fresh": 88
 },
 "neg_reasons": {
  "frame+break": 12,
  "break": 172
 },
 "rewrite_groups": 127,
 "rewrites_kept": 52,
 "rewrites_rejected": 75,
 "pairs": 71,
 "train_rows": 822,
 "train_sha256": "bfa434635d5cf00dd35086a05eafa79643c1dc0a02513f7fdaf0d4972872c06a",
 "break_rule": "frame regex or voice_score <= 2",
 "loss": "unlikelihood",
 "ul_weight": 1.0,
 "teacher": "qwen3:14b + RAG ch3",
 "fresh_model": "Qwen3-4B + exp4 lora_q4_facts + exp4 lora_q4_voice",
 "limit": 0
}
```

```
{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "facts_sha256": "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b",
 "q4": false,
 "pairs": 822,
 "claims": 0,
 "examples": 1008,
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
 "steps": 2016,
 "loss": "unlikelihood",
 "ul_weight": 1.0,
 "beta": null,
 "init": null,
 "dpo": null,
 "trl": false,
 "train_sec": 575,
 "peak_vram_gb": 12.2,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp5/data/v_contrast_train.jsonl"
}
```

## v_dpo

- Optional last stage. Starts from the best of v_long, v_r32, v_distill, v_contrast on the RAG condition.
- Ranking: pass bar first, then higher voice, fewer frame breaks, more must-answer hits, fewer marker leaks.
- Pairs: chosen/rejected from data/distill_pairs.jsonl and data/contrast_pairs.jsonl. Skip the stage below 16 pairs.
- Loss: local DPO, beta 0.1, on the sum of completion-token logprobs. Reference logprobs are cached from the init adapter before any update. Dropout is off. lr 5e-6, 1 epoch, AdamW.
- trl was not installed, so it was not used. No package was installed into the shared venv.

```
{
 "skip": false,
 "adapter": "$HOME/output-model/exp5/lora_v_distill",
 "name": "v_distill",
 "pairs": 227,
 "ranking": [
  {
   "name": "v_distill",
   "rank": [
    0,
    4.009803921568627,
    -8,
    56,
    -1
   ],
   "voice": 4.009803921568627,
   "frames": 8,
   "must": "56/60",
   "marker": "1/48",
   "passes": false
  },
  {
   "name": "v_contrast",
   "rank": [
    0,
    4,
    -2,
    57,
    0
   ],
   "voice": 4,
   "frames": 2,
   "must": "57/60",
   "marker": "0/48",
   "passes": false
  },
  {
   "name": "v_long",
   "rank": [
    0,
    3.950980392156863,
    -7,
    59,
    -1
   ],
   "voice": 3.950980392156863,
   "frames": 7,
   "must": "59/60",
   "marker": "1/48",
   "passes": false
  },
  {
   "name": "v_r32",
   "rank": [
    0,
    3.9362745098039214,
    -2,
    59,
    -2
   ],
   "voice": 3.9362745098039214,
   "frames": 2,
   "must": "59/60",
   "marker": "2/48",
   "passes": false
  }
 ],
 "q14_frames": 6,
 "voice": 4.009803921568627,
 "frames": 8,
 "reason": "best RAG voice adapter is v_distill"
}
```

- pairs meta `{"pairs": 227, "distill": 156, "contrast": 71}`
- 227 rows  7cab776b3a2731a1d223fa29af22db24b1853c975366df2fedb8e0931d299f4e  data/v_dpo_pairs.jsonl

```
{
 "base": "Qwen/Qwen3-4B",
 "stage": "voice",
 "facts": "$HOME/output-model/exp4/lora_q4_facts",
 "facts_sha256": "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b",
 "q4": false,
 "pairs": 227,
 "claims": 0,
 "examples": 227,
 "epochs": 1,
 "lr": 5e-06,
 "r": 32,
 "alpha": 64,
 "dropout": 0.0,
 "targets": [
  "q_proj",
  "k_proj",
  "up_proj",
  "o_proj",
  "v_proj",
  "down_proj",
  "gate_proj"
 ],
 "steps": 227,
 "loss": "dpo",
 "ul_weight": null,
 "beta": 0.1,
 "init": "$HOME/output-model/exp5/lora_v_distill",
 "dpo": "sum token logprob, beta on the sum, ref cached from init",
 "trl": false,
 "train_sec": 197,
 "peak_vram_gb": 8.92,
 "prefix_mismatch_skipped": 0,
 "limit": 0,
 "pairs_file": "$HOME/output-model/exp5/data/v_dpo_pairs.jsonl"
}
```

## Eval

- exp5/eval_voice.py calls the copied eval4.py with --base, --facts, --voice, --adapters fv, and --rag when the condition ends in _rag.
- n=3. Judge is the copied judge.py. A condition is skipped when runs/<cond>.json or the judged file already exists.
- A train stage is skipped when lora_<name>/train_report.json exists.
