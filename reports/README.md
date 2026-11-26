# Reports

One folder per experiment, copied from the working tree after each run.

- `phase1/exp1` … `exp10`: the *No Return* experiments, in order. exp1–exp3 explore the chapter lock, chapter LoRAs, model sizes and retrieval (exp2 has only a summary). exp4 trains the facts adapter. exp5–exp8 work on voice (distillation, contrast, DPO) and a no-LoRA retrieval ablation. exp9 builds the two-pass rewrite with a fact guard. exp10 locks the prototype. Only exp4, exp6, exp7, exp9 and exp10 have their code in `src/phase1/`, because those are the steps behind the locked model. The other reports are kept for the record.
- `phase1/exp10/locked/p2_light_dpo_guarded.judged.json`: every judged answer from the locked model.
- `browser-demo/results.md`: what the browser build ships and the closest recorded scores.
- `phase2/train1`: the first *Frankenstein* run. `phase2/dev1`: the CPU diagnosis of the voice loops. `phase2/train2`: the bounded rewrite retrain (no loops, but under the baseline on story questions).

Each `REPRODUCE.md` / `REPRODUCE-notes.md` lists the exact commands, hashes and settings for that run. Paths that start with `$HOME/output-model/` refer to the original working tree.
