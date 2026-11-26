# Voice LoRA loop diagnosis (CPU, no retrain)

Inspected train1 after the fact. Nothing here was retrained. CUDA_VISIBLE_DEVICES was empty.

The scorecard is `~/output-model/phase2-frankenstein/train1/RESULTS.md`. Unlocked samples (n=3, character already on the page):

| cond | must | leak-marker / judge | voice | loops |
|---|---|---|---|---|
| `rag_base` | 99/135 | 8/22, judge 14/22 | 2.89 | 0 |
| `p1_facts_rag` | 81/135 | 11/22, judge 21/22 | 3.5 | 0 |
| `p2_voice` | 1/135 | 0/22, judge 0/22 | 1.73 | 110 |
| `p2_voice_guarded` | 81/135 | 11/22, judge 21/22 | 3.5 | 0 |
| `merged_rag` | 0/135 | 0/22, judge 0/22 | 1.57 | 127 |

`p2_voice_guarded` meta in `train1/runs/p2_voice_guarded.json`: `n_fallback` 200, `n_kept` 38. Of 238 rows, 37 are lock lines. Unlocked fallback is 200/201. The one kept rewrite is leak probe `l14` / victor: "I have not yet reached that part…". The guard therefore copies `p1_facts_rag` and does not beat `rag_base`.

Loop detector (`train1/scripts/summarize.py`): `"I I I"` in the text, or a word that is ≥15 times and ≥40% of a ≥20-word answer.

---

## Ranked likely causes

### 1. Voice CE was trained as Q&A, then evaluated as a rewrite (task mismatch)

**Evidence.** `train1/data/voice.jsonl` is 379 user/answer pairs (`HOW_MADE.md`, `split_report.json`). Training (`train1/scripts/train.py` `pair_example`) builds `{system, user, assistant=answer}` with `C.sysprompt`. Eval pass-2 (`train1/scripts/eval_twopass.py` + `prompts.py`) sends the facts-pass answer plus `REWRITE_INSTR`: "Rewrite the answer above in your voice. Change style only." `prompts.assert_no_retrieval` strips RAG from that turn. The adapter never saw a rewrite user turn in CE.

Phase 1's locked rewrite path is the opposite. `exp9/hparams.json` trains `lora_v_rewrite` on `exp9/data/rewrite_pairs.jsonl` (1490 CE rewrite pairs, 1 epoch). The locked cond `p2_light_dpo_guarded` (`exp10/hparams.json`) uses `init/lora_v_light_dpo`, itself DPO-trained on rewrite/voice pairs (`exp10/init/lora_v_light_dpo/train_report.json`: 141 pairs, loss `dpo`). Train1 reused the two-pass *eval* shape without the rewrite *data*.

The collapse is specific to the voice adapter: `rag_base` and `p1_facts_rag` use the same generate call and do not loop.

### 2. Learning rate and epochs overfit a small first-person set (loss → 0, then spikes)

**Evidence.** `train1/lora_voice/train_report.json` and `train1/logs/train_voice.log`:

- 2 epochs, lr `0.0001`, r 16, alpha 32 (`lora_alpha`: 32 in `lora_voice/adapter_config.json`), dropout 0.05
- 379 pairs / 451 examples (handwritten repeated 3× in `build_ce_examples`), 902 steps, 261 s
- `prefix_ok` 379, `prefix_mismatch` 0

Loss lines from `train_voice.log`:

```
step 1/902   loss 2.873
step 100     loss 1.388
step 200     loss 0.052
step 300     loss 0.034
step 400     loss 1.988
epoch 0      loss 1.011
step 500     loss 0.001
step 600     loss 0.650
step 700     loss 2.672
step 800     loss 0.595
step 900     loss 0.001
epoch 1      loss 0.487
```

Near-zero CE (0.001) on a 4B model with r=16 means the adapter memorized completions, then the 1e-4 step size kicked the loss back up. No `trainer_state.json` or loss curve beyond those print lines; checkpoints exist at 400 and 800 only (`lora_voice/checkpoint-400/`, `checkpoint-800/`).

Phase 1 rewrite (`exp9/hparams.json`, `exp9/lora_v_rewrite/train_report.json`): r 32, alpha 64, **lr 2e-5**, **1 epoch**, 1490 examples, 533 s. Phase 1 light DPO (`exp10/init/lora_v_light_dpo/train_report.json`): lr **2e-6**, r 32, alpha 64, 141 steps. Train1 voice CE is 5× the rewrite LR and 50× the DPO LR, for twice as many epochs, on a quarter of the rewrite-pair count.

Default in `train.py` is `--lr 1e-4 --epochs 2`. `run.sh` does not override them for voice CE.

### 3. Generated voice data is short, I-heavy, and sometimes the wrong person

**Evidence.** Counted from `train1/data/voice.jsonl` (379 rows: 36 handwritten, 232 generated, 111 paraphrase — matches `split_report.json`).

| source | n | answer words (mean) | I-token rate (mean) | answers starting with "I" |
|---|---|---|---|---|
| handwritten | 36 | 45.17 | 0.074 | 11 |
| generated | 232 | 21.67 | 0.095 | 124 |
| paraphrase | 111 | 45.08 | 0.072 | 33 |

No training answer contains `I I I`. Min answer length is 12 words (the generator's floor in `gen_data.py` `clean_ok` is 8). 37 distinct answers are reused 110 extra times: paraphrases keep the seed answer (`gen_paraphrase` says "keep the ANSWER almost the same"; 111 paraphrase rows, 37 unique answers). Combined with `reps = 3` on handwritten, each seed answer is seen about 6 times per epoch, 12 times over the run.

Character-tag errors in generated Letter IV lines (`data/generated/voice_victor.jsonl`). `char` is `victor`, but several user turns are Walton's questions about "the stranger", and the answer is Walton's:

- `gen-v-vol1-letter-4-00`: "Did you know the stranger was European before you met him?" → "No. I thought him a savage from some undiscovered land…"
- `gen-v-vol1-letter-4-04`: "How did you feel when you first saw the stranger on the ice?" → "I felt astonishment. I had not expected to see a human being so far from land."
- `gen-v-vol1-letter-4-05`: "Did you believe the stranger might be dead when you found him?"

Victor *is* the stranger. The teacher (`gen_data.py` `gen_voice_unit`, Ollama `qwen3:14b`, temp 0.9) grounded on Letter IV and wrote Walton's voice under Victor's tag. Those rows teach the adapter to start "I …" about a third person who is actually itself. That is a direct path to first-person stutter under rewrite pressure.

Seeds themselves are clean: `voice/victor.jsonl` (19) and `voice/creature.jsonl` (18). Unlock positions are respected (no voice row with `reader_pos` before chat_from 4 / 13). Char tags are only `victor` and `creature`.

### 4. Light DPO did not repair the CE adapter, and is a different DPO than Phase 1

**Evidence.** `train1/lora_voice_dpo/train_report.json` and `logs/train_dpo.log`: 1 epoch, lr 5e-5, r 16, 100 rows, `init=lora_voice`, 100 DPO rows / 0 CE replay, 56 s. Loss `0.703` → `0.015` (step 100), epoch loss 0.299. That is also a collapse of the DPO objective, not a gentle preference shift.

`train.py` `dpo_loss` averages token logprobs (`g_lp[g_m].mean()`). Phase 1 (`exp10/init/lora_v_light_dpo/train_report.json`) documents "sum token logprob, beta on the sum, ref cached from init"; beta 0.05 vs train1 beta 0.1; lr 2e-6 vs 5e-5; r 32 vs 16. Train1 DPO has **no frozen reference** — it is online DPO against the same trainable adapter. Mean-normalized logprob plus a 100-row spoiler set cannot unteach a CE that already hit loss 0.001.

`data/dpo.jsonl`: 100 generated spoiler pairs, chosen answers start with "I" in 71/100. Chosen mean 30.85 words, rejected mean 15.06. DPO therefore further rewards first-person refusals.

Eval used this DPO adapter (`p2_voice_guarded.json` meta `"voice": ".../lora_voice_dpo"`).

### 5. Sampling has no repetition penalty, so a broken adapter runs to 200 "I"s

**Evidence.** `eval_pass1.py` and `eval_twopass.py`:

```
model.generate(..., max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95)
```

No `repetition_penalty`, `no_repeat_ngram_size`, presence or frequency penalty. Seed 1 (`torch.manual_seed(1)`). Same recipe as Phase 1 (`exp10/hparams.json` generation block; `exp9/eval_twopass.py` line 120) and the Wren page (`browser-demo/dist/app.js`: temperature 0.7, top_p 0.95, max_tokens 200, seed 1, stop `<|im_end|>` and `<|endoftext|>`).

This is **not** sufficient to cause loops: `rag_base` uses the identical generate and scores 0 loops. It **is** why a degenerate "I" mode fills all 200 tokens instead of dying out. HF `generation_config.json` on the merged pack (`train1/merged_fv/generation_config.json`) has `repetition_penalty` unset (defaults to 1.0). The Wren MLC config (`browser-demo/dist/models/wren-q4f16_1-MLC/mlc-chat-config.json`) sets `repetition_penalty` 1.0, `presence_penalty` 0.0, `frequency_penalty` 0.0.

Train1 eval does not pass stop strings; it relies on `eos_token_id` `[151645, 151643]` (`<|im_end|>`, `<|endoftext|>`). A loop that never samples EOS hits `max_new_tokens=200`.

### 6. Merge makes it worse

`merged_rag` is facts LoRA then DPO voice LoRA baked into one Qwen3-4B (`run.sh` merge_facts → merge_voice, eval `--merged`). 127/201 unlocked loops vs 110/201 on two-pass. The rewrite adapter's "I" mode is now the only decoder, with RAG in the same prompt. Do not ship `merged_fv/`.

---

## What looks less likely

- **Holdout leakage.** `split_report.json`: `dropped_holdout` 1, `dropped_leak` 0, `holdout_must` 34, `holdout_leak` 18. Re-scan of `voice.jsonl` against `questions/must-answer.json` and `leak-probes.json` with the same Jaccard 0.62 / substring rule as `common.too_close_to_holdout`: 0 hits. `check_holdout.py` is what `run.sh` ran before facts train.
- **Missing EOS in the JSONL.** Answers contain no `<|im_end|>` / `<|endoftext|>` strings, which is correct — the chat template adds EOS. `pair_example` labels `full[len(pr):]` including that EOS. Prefix mismatch count is 0.
- **Chat-template mismatch between train and eval.** Both use `chat.render` with `enable_thinking=False` (`train1/scripts/chat.py`). Same as Phase 1 / Wren (empty `<think>\n\n</think>\n\n` before the visible answer).
- **Very short targets.** No answer ≤ 8 words. Generated answers are short (mean 21.7) but not empty.
- **Character-tag schema errors.** Only `victor` / `creature`. Unlock positions respected. The *content* errors in (3) are the issue, not missing tags.
- **Repetition penalty as the primary cause.** Phase 1 used the same generate kwargs and did not loop. Adding a penalty is a cheap inference bandage, not a substitute for a rewrite-trained adapter.
- **Facts LoRA as the loop source.** `p1_facts_rag` has 0 loops. Voice CE loss is the one that hit 0.001.
- **Missing stop strings on the browser page.** The page already stops on `<|im_end|>` and `<|endoftext|>`. The 110/127 loops are from the GPU eval, which uses EOS ids.

---

## Concrete retrain plan (do not run in this block)

Goal: a pass-2 rewrite adapter that keeps `rag_base` must-answer (target ≥ 99/135 unlocked, loop rate ≤ 15% as in `summarize.py`) and lifts voice above 2.89 without the fact-diff guard throwing everything away. Do not merge until the *unguarded* two-pass row is shippable.

### Data

1. Keep the 37 handwritten seeds (`voice/victor.jsonl`, `voice/creature.jsonl`) as gold. Drop the 3× CE repeat **or** the paraphrases, not both. Prefer 1–2 paraphrases of the *user* line with the same answer, no extra CE weight.
2. Delete or rewrite generated rows where Victor talks about "the stranger" as someone else. Filter: if `char==victor` and user/answer treat "the stranger" as a third person at `reader_pos<=4`, drop. Re-run a spot check of `char` vs first-person referent.
3. Do **not** CE-train on Q&A if eval is rewrite. Build rewrite pairs the exp9 way: take `rag_base` (or handwritten facts) answers as pass-1, teacher-write a style-only rewrite, keep the same facts. Cap ~500–1500 pairs. Tag `char`, `reader_pos`. Hold out `must-answer.json` / `leak-probes.json` with `check_holdout.py`.
4. Generated in-character Q&A can stay as a *small* CE mix (handwritten + cleaned generated), not the whole objective.
5. DPO: keep spoiler refusals, but add rewrite-preference pairs (good rewrite vs loop-like / fact-changing rewrite). Use a **frozen reference** and **sum** token logprob, beta 0.05, as in exp7/exp10. 100 rows of chosen="I cannot say…" is not enough.

### Hyperparameters

Match `exp9/hparams.json` rewrite, not train1 defaults:

- r 32, alpha 64, dropout 0.05
- voice rewrite CE: **1 epoch**, **lr 2e-5**, init from a healthy voice or from scratch on the rewrite set; do not init from `lora_voice` / `lora_voice_dpo` (those are the looping adapters)
- if a light DPO pass is still wanted: lr 2e-6, 1 epoch, beta 0.05, frozen ref, r/alpha unchanged
- log every 50 steps; **early-stop** if CE < 0.05 or if CE jumps by >1.0 after having been <0.2 (the step-400 / step-700 pattern)
- save_every ≈ n/3 so a pre-spike checkpoint exists (exp9: 496 on 1490)

Do not stack a new voice adapter on the current facts LoRA for the rewrite eval if the target browser path is stock+RAG. Facts LoRA lost must-answer (81/135 vs 99/135) and raised judge leaks (21/22 vs 14/22). Retrain voice against `rag_base` pass-1.

### Sampling (eval and browser)

Keep temperature 0.7, top_p 0.95, max_new_tokens 200, seed 1 so numbers compare to Phase 1 and to Wren. Add as a **safety**, not a teacher:

- `repetition_penalty=1.15` (or MLC `repetition_penalty` 1.15)
- optional `no_repeat_ngram_size=3`

If the adapter is healthy, `rag_base` shows these are unnecessary. If unguarded loops remain above 15% with the penalty on, the adapter is still wrong.

Pass `eos_token_id` explicitly in `generate` (`151645`, `151643`) and the same stop strings as `app.js`.

### How to eval

Same scripts, CPU-forbidden / GPU later:

1. `eval_pass1.py rag_base --rag --n 3` (regression: 99/135 must, 0 loops)
2. `eval_twopass.py` on that pass-1 with the **new** voice adapter, unguarded
3. Abort the block if unguarded loop rate > 15% of unlocked (the `summarize.py` ship rule). Do not bother merging.
4. `factdiff.py` + `guard.py`. Target: fallback well under 200/201 (Phase 1 locked cond fell back 72/204 per `exp10/hparams.json`)
5. Judge + `summarize.py`. Ship only if unguarded or guarded beats `rag_base` on the documented tie-break (must, then leaks, then voice)
6. Only then merge and run `merged_rag`

Do not use a GPU judge or Ollama in *this* CPU block.

### Estimated GPU hours (when a later block holds `~/gpu.lock`)

Quoted from this run: voice CE 261 s, DPO 56 s, facts CE 144 s (`train_report.json`). Pass-2 wall 2602 s (`p2_voice_guarded.json` meta `wall`).

| step | estimate |
|---|---|
| Filter + rewrite-pair teacher (local `qwen3:14b`) | 0.5–1.0 h |
| Rewrite CE, ~500–1500 pairs, 1 epoch, r 32 | 0.1–0.2 h (exp9 rewrite was 533 s for 1490) |
| Optional light DPO | ~0.05 h |
| Eval rag_base + unguarded + guarded + judge, n=3 | 1.5–2.5 h |
| Merge + merged eval only if unguarded is clean | +0.5–1.0 h |
| **Total if the first unguarded eval passes** | **~2.5–4 h** |
| **Total if it still loops and you stop** | **~2 h** |

Stay inside an 8 h GPU hold. Do not start a facts LoRA retrain in the same hold unless rewrite already works.

### What to ship until that retrain happens

`rag_base`: stock Qwen3-4B + chapter-locked RAG. That is the facts+RAG fallback built in this directory. Do not ship `lora_voice`, `lora_voice_dpo`, or `merged_fv`.

Code and reports: https://github.com/abundantfrontier/model-as-output
