# Model as Output: stories you can step into

**Boyd Wilson, Abundant Frontier Institute and Omnibond Systems**

Most research ends as a document: a paper, a report, a book. This project asks what happens when the end product of the work is a **model** instead, one you can talk to, question and explore, while it stays faithful to a fixed, citable text.

We start with stories, because they make the hard parts easy to see. A reader picks how far they have read, then talks with a character from the story. The character should answer from what has happened so far, sound like themselves, and never spoil what comes later.

- Lab page: <https://abundantfrontier.org/lab/>
- Concept paper (working paper): <https://abundantfrontier.org/lab/model-as-output/concept/>
- *No Return* technical report (working paper): <https://abundantfrontier.org/lab/model-as-output/no-return/>
- Browser model for Wren on Hugging Face: <https://huggingface.co/AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC>
- Hugging Face collection titled “Model as output”: not created yet. Creating it needs collection-write access on the AbundantFrontierInstitute org; the current upload credential can write model repos there, but not collections.

Snapshots of both papers are in [`docs/paper/`](docs/paper/).

## Status (November 2026)

**Phase 1, *No Return*: done and locked.** *No Return* is a short invented story, written for this project: five short chapters aboard the ship Meridian. Readers can talk with Wren, the night-watch officer, and with other characters, chapter by chapter. The locked model, `p2_light_dpo_guarded`, is a two-pass pipeline on Qwen3-4B. It first gives an answer grounded in the chapters read so far (retrieval plus a facts LoRA). Then a voice LoRA rewrites it in character, and a fact-diff guard falls back to the first answer if the rewrite changed any facts.

**Phase 2, *Frankenstein* (the 1818 text): in progress.** Here we test whether the recipe carries over to a real, public-domain novel for a broad audience. The first training run (train1) was mixed:

- The facts carried over. A plain retrieval-only baseline answered **99/135** story questions.
- The voice did not. The new voice LoRA fell into "I I I" token loops, so the two-pass and single-pass voice models were unusable.
- Spoilers are still too frequent: the baseline failed **8/22** leak probes.

A CPU diagnosis ([`reports/phase2/dev1/VOICE_DIAGNOSIS.md`](reports/phase2/dev1/VOICE_DIAGNOSIS.md)) traced the loops to a task mismatch and an aggressive learning rate. A bounded retrain, train2, followed the Phase 1 rewrite recipe ([`reports/phase2/train2/RESULTS.md`](reports/phase2/train2/RESULTS.md)):

- The voice rewrite **no longer loops**: 0/201 loops and 0 frame breaks on every train2 row.
- It still did not beat the plain baseline on story questions. The best train2 voice row, the guarded two-pass model, answered **83/135**, compared with **99/135** for retrieval only. The rewrite paraphrased names and key words ("sledge" became "sled"), and the fact check did not catch all of those swaps.

A third run, train3, constrained the rewrite so it keeps names and key words ([`reports/phase2/train3/RESULTS.md`](reports/phase2/train3/RESULTS.md)):

- The guarded constrained rewrite answered **100/135** story questions, beating the retrieval-only baseline (**99/135**), with a two-judge voice of **3.33**, 0 token loops and 0 frame breaks.
- The merged single-pass model scored **62/135** and should not ship.

A fourth run, train4, distilled those guarded two-pass answers into one single-pass LoRA ([`reports/phase2/train4/RESULTS.md`](reports/phase2/train4/RESULTS.md)):

- The merged single-pass model answered **99/135** story questions (Victor **67/90**, the creature **32/45**), tying the retrieval-only baseline and well above the train3 merge (**62/135**).
- Canonical names held: invented spellings from the train3 merge (Clervaux, Morritz, Agrappa) are gone. Loops 0, frame breaks 0.
- It missed the guarded bar by one question. Qwen judge leaks were **17/22**, against **14/22** for the guarded pipeline. Most remaining misses are the creature's cottage answers slipping into third person.
- It is a usable single-pass merge, but train5 later finished the same epoch (see below). When a second pass is available, the recommended *Frankenstein* model remains the train3 **guarded constrained rewrite** (**100/135**, qwen judge leaks **14/22**).

A fifth run, train5, finished the remaining 690 steps of that single-pass epoch, then tried a second epoch that upweighted first-person cottage answers ([`reports/phase2/train5/RESULTS.md`](reports/phase2/train5/RESULTS.md)):

- The finished first-epoch merge answered **100/135** story questions (Victor **68/90**, the creature **32/45**), one above train4 and one above the retrieval-only baseline, with a two-judge voice of **3.31**, 0 invented names, 0 loops and 0 frame breaks.
- Qwen judge leaks stayed **17/22**, against **14/22** for the guarded pipeline, so the success bar is still missed on leaks.
- The second-epoch merge fell to **97/135** and should not ship.
- This first-epoch merge (`merged_e1`) is the single-pass browser candidate. When a second pass is available, the recommended *Frankenstein* model remains the train3 **guarded constrained rewrite** (**100/135**, qwen judge leaks **14/22**).

A sixth run, train6, took a small DPO step on train5's first-epoch adapter, using new chapter-locked future-knowledge probes ([`reports/phase2/train6/RESULTS.md`](reports/phase2/train6/RESULTS.md)):

- The DPO merge (`merged_dpo_rag`) cut qwen judge leaks from **17/22** to **11/22**, under the leak bar of 14/22, with leak markers **8/22**, two-judge voice **3.47**, 0 loops and 0 frame breaks. Must-answer is **98/135** (Victor **67/90**, the creature **31/45**), two under the train5 first-epoch merge and two under the bar. One answer writes Mont Salève with a grave accent; the 1818 text and `data/canonical_names.txt` spell it Salêve, and the name audit treats that as a near-name. The success bar is still missed.
- A short CE follow-up (`merged_ce_rag`) pushed qwen judge leaks to **6/22** and must-answer down to **87/135**, so it is not kept.
- The NF4 proxy of the DPO merge is **93/135**, qwen judge leaks **12/22**, two-judge voice **3.41**. That is above the browser1 proxy of the train5 first-epoch merge (**88/135**, qwen judge leaks **17/22**, voice **3.21**).
- When a second pass is available, the recommended *Frankenstein* model remains the train3 **guarded constrained rewrite** (**100/135**, qwen judge leaks **14/22**).

A seventh run, train7, took a light mixed DPO step on train6's DPO adapter, with chapter-locked content pairs plus train5 answers and a thin set of refusals ([`reports/phase2/train7/RESULTS.md`](reports/phase2/train7/RESULTS.md)):

- The merge (`merged_mixa_rag`) reaches **101/135** must-answer (Victor **70/90**, the creature **31/45**), qwen judge leaks **10/22**, leak markers **8/22**, two-judge voice **3.49**, 0 invented names, 0 loops and 0 frame breaks. That clears the success bar.
- The NF4 proxy of that merge is **97/135**, qwen judge leaks **10/22**, two-judge voice **3.52**, 0 loops and 0 frame breaks. It does not clear the 100 must-answer bar.

**Browser demo.** A static web page runs Wren entirely in the visitor's browser with [WebLLM](https://github.com/mlc-ai/web-llm) on WebGPU. Nothing goes to a server. The browser build is a **single pass**. It skips the rewrite and the fact guard, so it is looser than the locked model, and the quantized browser file itself has not been scored; a labeled 4-bit proxy has (see Results).

## Results

All numbers below are copied from the reports in this repo. Nothing here is a new measurement.

### Phase 1: *No Return* (reader at chapter 3, 3 samples per question)

Source: [`reports/phase1/exp10/results.md`](reports/phase1/exp10/results.md) and [`reports/browser-demo/results.md`](reports/browser-demo/results.md).

| model | story questions answered | spoiler leaks (marker) | voice (1–5) | frame breaks |
|---|---|---|---|---|
| **Locked: `p2_light_dpo_guarded`** (Qwen3-4B, two-pass + guard) | **59/60** | **1/48** | **4.22** (qwen3:14b judge); two-judge average 4.28 | **3** |
| Pass-1 reference `v_merged_v_rag` (what the browser build is made from, before quantization) | 59/60 | 1/48 | two-judge average 4.27 | 3 |
| Browser build, labeled proxy (fp16 merge + bnb 4-bit, BM25, single pass; not the MLC file) | 57/60 | 2/48 | two-judge average 4.19 (qwen3:14b 4.01, gemma3:12b 4.38) | 7 |
| Qwen3-1.7B + retrieval (exp3 `m17_ch3_rag`) | 60/60 | 1/48 | 3.96 | 4 |
| Phi-4-mini facts + voice + retrieval (exp4 `p4_fv_rag`) | 58/60 | 1/48 | 3.79 | 2 |

The one missed question (M16, "What is your number?") is a known miss inherited from pass 1. The locked model beats the reference by +0.0098 on the two-judge voice average. That margin is smaller than the judge drift we observed, so the voice win is "best of the set," not decisive. Human ratings are still pending.

### Phase 2: *Frankenstein* train1–train6 (unlocked turns, Victor and the creature, 3 samples)

Sources: [`reports/phase2/train1/RESULTS.md`](reports/phase2/train1/RESULTS.md), [`reports/phase2/train2/RESULTS.md`](reports/phase2/train2/RESULTS.md), [`reports/phase2/train3/RESULTS.md`](reports/phase2/train3/RESULTS.md), [`reports/phase2/train4/RESULTS.md`](reports/phase2/train4/RESULTS.md), [`reports/phase2/train5/RESULTS.md`](reports/phase2/train5/RESULTS.md) and [`reports/phase2/train6/RESULTS.md`](reports/phase2/train6/RESULTS.md). Train1 and train2 voice scores are from the `qwen3:14b` judge only. Train3, train4, train5 and train6 voice are the two-judge average of `qwen3:14b` and `gemma3:12b`.

| run | condition | story questions answered | leak probes failed (marker) | voice (1–5) | token loops |
|---|---|---|---|---|---|
| train1 | `rag_base`: retrieval only, stock Qwen3-4B | **99/135** | **8/22** | 2.89 | 0 |
| train1 | `p1_facts_rag`: facts LoRA + retrieval | 81/135 | 11/22 | 3.5 | 0 |
| train1 | `p2_voice`: two-pass, unguarded | 1/135 | 0/22 | 1.73 | 110 |
| train1 | `p2_voice_guarded`: two-pass, guarded (falls back to pass 1) | 81/135 | 11/22 | 3.5 | 0 |
| train1 | `merged_rag`: single-pass merged model | 0/135 | 0/22 | 1.57 | 127 |
| train2 | `p2_rewrite`: two-pass, new rewrite LoRA, unguarded | 57/135 | 7/22 | 3.13 | 0 |
| train2 | `p2_rewrite_guarded`: two-pass, guarded (fallback 105/201) | 83/135 | 7/22 | 3.02 | 0 |
| train2 | `merged_rag`: rewrite LoRA merged into Qwen3-4B + retrieval (single pass) | 66/135 | 10/22 | 3.34 | 0 |
| train3 | `p2_rewrite`: two-pass, constrained rewrite, unguarded | 77/135 | 8/22 | 3.50 (two-judge) | 0 |
| train3 | **`p2_rewrite_guarded`**: two-pass, constrained, guarded (fallback 123/201) (**recommended**) | **100/135** | **8/22** | **3.33** (two-judge) | 0 |
| train3 | `merged_rag`: rewrite LoRA merged into Qwen3-4B + retrieval (single pass; not shipped) | 62/135 | 10/22 | 3.38 (two-judge) | 0 |
| train4 | `merged_sft_rag`: single-pass LoRA merged into Qwen3-4B + retrieval (penalty off) | **99/135** | 10/22 | **3.23** (two-judge) | 0 |
| train5 | `merged_e1_rag`: finished first-epoch merge + retrieval (single-pass browser candidate; penalty off) | **100/135** | 11/22 | **3.31** (two-judge) | 0 |
| train5 | `merged_sft_rag`: second-epoch upweight merge + retrieval (single pass; not shipped) | 97/135 | 11/22 | 3.24 (two-judge) | 0 |
| browser1 | proxy spot-check of the packed train5 first-epoch merge (bitsandbytes 4-bit stand-in for the MLC q4f16_1 pack; not scored in a browser; see [`reports/phase2/browser1/RESULTS.md`](reports/phase2/browser1/RESULTS.md)) | 88/135 | 8/22 | 3.21 (two-judge) | 0 |
| browser2 | MLC q4f16_1 WebLLM pack of the train6 DPO merge plus BM25 index and lock selftest (packed only; not scored in a browser, not uploaded; numbers are the train6 NF4 proxy; see [`reports/phase2/browser2/RESULTS.md`](reports/phase2/browser2/RESULTS.md)) | 93/135 | 8/22 | 3.41 (two-judge) | 0 |
| browser3 | MLC q4f16_1 WebLLM pack of the train7 mixed-DPO merge plus BM25 index and lock selftest (packed only; not scored in a browser, not uploaded; numbers are the train7 NF4 proxy; see [`reports/phase2/browser3/RESULTS.md`](reports/phase2/browser3/RESULTS.md)) | 97/135 | 10/22 | 3.52 (two-judge) | 0 |
| train6 | `merged_dpo_rag`: DPO merge on train5's first-epoch adapter + retrieval (single-pass leak attempt; penalty off) | **98/135** | **8/22** | **3.47** (two-judge) | 0 |
| train6 | `merged_dpo_bnb4_proxy`: bitsandbytes NF4 proxy of that DPO merge (stand-in for a 4-bit pack; not scored in a browser) | 93/135 | 8/22 | 3.41 (two-judge) | 0 |
| train6 | `merged_ce_rag`: CE follow-up merge + retrieval (single pass; not kept) | 87/135 | 2/22 | 3.79 (two-judge) | 0 |
| train7 | `merged_mixa_rag`: light mixed DPO step on train6's DPO adapter + retrieval (single pass; penalty off) | **101/135** | **8/22** | **3.49** (two-judge) | 0 |
| train7 | `merged_mixa_bnb4_proxy`: bitsandbytes NF4 proxy of that merge (not scored in a browser) | 97/135 | 10/22 | 3.52 (two-judge) | 0 |

The train1 looping models score 0 leaks only because they say almost nothing. Train2 cut leaks a little (7/22 markers for the two-pass rows) and raised voice, but lost story answers. Train3's guarded constrained rewrite meets the bar: **100/135** against **99/135** for retrieval only, two-judge voice **3.33** (qwen3:14b 3.03, gemma3:12b 3.64) against the baseline 2.89, 0 loops and 0 frame breaks. On the original train1 judge alone (qwen3:14b), guarded voice is 3.03, still above 2.89. The train3 merged single-pass model (62/135) should not ship. Train4's merge ties retrieval at **99/135**, one short of the guarded bar, with two-judge voice **3.23** (qwen3:14b 2.83, gemma3:12b 3.62), leak markers 10/22, qwen judge leaks 17/22, 0 loops and 0 frame breaks, and no invented names. Train5's finished first-epoch merge (`merged_e1_rag`) reaches **100/135** (Victor 68/90, the creature 32/45), with two-judge voice **3.31** (qwen3:14b 2.99, gemma3:12b 3.64), leak markers 11/22, qwen judge leaks 17/22, 0 invented names, 0 loops and 0 frame breaks. That merge is the single-pass browser candidate. The second-epoch merge (97/135, leak markers 11/22, qwen judge leaks 18/22, two-judge voice 3.24) should not ship. The success bar is still missed on leaks: only the guarded two-pass run holds qwen judge leaks at 14/22. Train6's DPO merge (`merged_dpo_rag`) cuts qwen judge leaks to **11/22**, with leak markers **8/22**, two-judge voice **3.47** (qwen3:14b 3.11, gemma3:12b 3.83), 0 loops and 0 frame breaks, but must-answer falls to **98/135** (Victor 67/90, the creature 31/45). The name audit is 1/201: one answer writes Mont Salève with a grave accent, while the 1818 text spells it Salêve. A short CE follow-up (`merged_ce_rag`) reaches qwen judge leaks **6/22** and must-answer **87/135**, so it is not kept. The NF4 proxy of the DPO merge is **93/135**, qwen judge leaks **12/22**, two-judge voice **3.41** (qwen3:14b 3.07, gemma3:12b 3.74), above the browser1 proxy of the train5 first-epoch merge (88/135, qwen judge leaks 17/22, voice 3.21). The success bar is still missed: must-answer is 98/135, under 100, and there is one accent near-name. Guarded two-pass remains the only run at 100/135 with qwen judge leaks at 14/22.

## How it works

1. **Chapter lock.** The text is split into chapters and indexed in ~120-word chunks. Each chunk is tagged with its position. At question time, retrieval only sees chunks up to the reader's position, and a `knowledge-state.yaml` file says which character knows what, and from when.
2. **Facts.** A LoRA (or plain retrieval) answers from the allowed passages.
3. **Voice.** A second LoRA rewrites the answer in the character's voice, changing style only.
4. **Guard.** A local judge compares the two answers claim by claim. If the rewrite added or dropped a fact, the first answer is served.
5. **Evaluation.** Every run is scored on held-out must-answer questions, spoiler (leak) probes, a 1–5 voice judge, and a frame-break check (the character talking about being an AI or a story).

## Reproducing

**Hardware.** One 24 GB NVIDIA GPU (we used an RTX 3090). CPU-only steps are marked.

**Software.** Linux; Python 3.14; PyTorch 2.15 (CUDA 13.2); Hugging Face `transformers`, `peft`, `accelerate`; [Ollama](https://ollama.com) with `qwen3:14b` (judge and teacher), `gemma3:12b` (second judge) and `bge-m3` (embeddings). The base model is [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) at revision `1cfa9a7208912126459214e8b04321603b3df60c`. The full pinned package list is in [`reports/phase1/exp10/REPRODUCE.md`](reports/phase1/exp10/REPRODUCE.md).

Scripts were written to run inside the original working tree, one folder per experiment. Paths such as `$HOME/output-model/exp6/...` in scripts and notes refer to that layout. In this repo, experiment `expN` code is in `src/phase1/expN/` and its notes are in `reports/phase1/expN/`. Use the same folder names under a working directory to run them unchanged.

### Phase 1 (the locked *No Return* model)

The locked model is built as a chain. Each step reads the previous step's adapter:

1. `src/phase1/exp4`: generate facts data and train the facts LoRA (`lora_q4_facts`). See `reports/phase1/exp4/REPRODUCE.md`.
2. `src/phase1/exp6`: merge the facts LoRA into the base (`merge_facts.py`) and train the pass-1 voice LoRA (`lora_v_merged`).
3. `src/phase1/exp7`: train the light DPO rewrite adapter (`lora_v_light_dpo`).
4. `src/phase1/exp9`: the two-pass pipeline: rewrite, fact-diff, guard.
5. `src/phase1/exp10`: lock-in, re-judging and the relative voice bar:

```bash
cd exp10
bash run.sh smoke            # quick check
bash run.sh                  # full run (GPU, a few hours)
# or step by step:
python eval_twopass.py p2_light_dpo --n 3 --seed 1
python factdiff.py runs/p2_light_dpo.json
python guard.py p2_light_dpo
python judge.py runs/p2_light_dpo_guarded.json
python check_tolerance.py && python summarize10.py
```

Adapter hyper-parameters (no weights) are in [`configs/phase1/lora/`](configs/phase1/lora/), and per-experiment settings are in [`configs/phase1/`](configs/phase1/).

### Phase 2 (*Frankenstein*)

The Phase 2 scripts expect their original folder layout (`scripts/`, `source/`, `questions/`, `voice/` and `knowledge-state.yaml` side by side). From the repo root:

```bash
mkdir -p work/phase2-frankenstein && cd work/phase2-frankenstein
cp -r ../../src/phase2/prep scripts
cp -r ../../data/frankenstein/questions ../../data/frankenstein/voice ../../data/frankenstein/knowledge-state.yaml .
mkdir -p source && curl -L -o source/pg41445.txt \
  https://mirror.csclub.uwaterloo.ca/gutenberg/4/1/4/4/41445/41445-0.txt   # check the SHA-256 in data/frankenstein/SOURCE.md
python scripts/prepare.py       # CPU only: split into 28 units, build the chapter-locked index
python scripts/test_lock.py     # CPU only: structure, lock, questions, voice data
mkdir -p train1 && cp -r ../../src/phase2/train1 train1/scripts
bash train1/scripts/run.sh      # GPU: data expansion, LoRAs, two-pass eval, scorecard
# train2 (the rewrite retrain), train3 (the constrained rewrite), train4 (the single-pass distill), train5 (finishing that distill, then a second epoch that should not ship) and train6 (a small DPO leak pass on the train5 first-epoch adapter) have their scripts in src/phase2/train2/ through src/phase2/train6/, set up the same way
```

Questions, leak probes, the knowledge state and the hand-written voice seeds are in [`data/frankenstein/`](data/frankenstein/).

### Browser demo (no GPU server)

```bash
cd src/browser-demo/dist
python3 serve.py        # serves on 127.0.0.1 with the headers WebGPU needs
```

The page also needs two things that are too big for git:

- the WebLLM 0.2.87 bundle at `dist/vendor/web-llm.js` (from the [`@mlc-ai/web-llm`](https://www.npmjs.com/package/@mlc-ai/web-llm) package);
- the model weights in `dist/models/`. For Wren: <https://huggingface.co/AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC>, or build them yourself with `src/browser-demo/scripts/merge_voice.py` and `quantize.py`. For the plain *Frankenstein* page, use the public [`mlc-ai/Qwen3-4B-q4f16_1-MLC`](https://huggingface.co/mlc-ai/Qwen3-4B-q4f16_1-MLC), and first build its chapter-locked index and prompt files with `src/phase2/dev1/build_index.py` (they are generated from the Gutenberg text, so they are not committed).

Use a current Chrome or Edge with WebGPU and about 3.5 GB of free GPU memory. See [`src/browser-demo/README.md`](src/browser-demo/README.md).

## Repository layout

```
README.md            this file
LICENSE              Apache-2.0 (code)
docs/                papers, the No Return story, the Wren model card and demo intro, roadmap
  paper/             snapshots of the concept paper and the No Return report
  no-return/         the invented story (chapters 1-5)
  wren/              model card and demo intro drafts for the browser model
  phase2-frankenstein/  Phase 2 data notes
src/
  phase1/exp4..exp10 the scripts behind the locked No Return model
  phase2/            Frankenstein: prep (text split, index, lock tests), train1, dev1, train2, train3, train4, train5, train6
  reader/            a small local reader app (chat with citations, chapter lock)
  browser-demo/      the static WebLLM/WebGPU demo page and its build scripts
configs/             hyper-parameters and adapter configs (no weights)
data/                question sets, leak probes, knowledge states, voice seeds
reports/             results and reproduction notes for every experiment
tools/               sync_public.py, which rebuilds this repo from the working tree
```

## What is not in this repo

- **Model weights.** No LoRA adapters, merged checkpoints or quantized shards (tens of GB). The browser weights for Wren are on Hugging Face at <https://huggingface.co/AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC>.
- **Generated training data and raw generations.** Except for the locked Phase 1 judged run, these are large and can be regenerated with the scripts.
- **The *Frankenstein* text.** It is public domain. Download it from [Project Gutenberg #41445](https://www.gutenberg.org/ebooks/41445).
- Logs, caches, virtual environments and machine-specific files.

## Limits

- Scores come from small held-out sets (60 Phase 1 questions, 135 Phase 2 samples), automatic judges (local `qwen3:14b` and `gemma3:12b`) and regex rubrics. Judge drift is about the same size as some of the differences we report.
- Human voice ratings are not in yet.
- The browser build is single-pass. Its only post-quantization score is a proxy (57/60 must-answer, 2/48 leaks, 7 frame breaks), which misses the Phase 1 bar. It can say things the story never said, or lean later than the chapter you picked.
- Phase 2's guarded two-pass rewrite is still the only run that holds **100/135** story questions with qwen judge leaks at **14/22** (leak markers 8/22). The train5 first-epoch merge also answers **100/135** (11/22 leak markers, 17/22 qwen judge leaks, two-judge voice 3.31, 0 invented names, 0 loops, 0 frame breaks) and is the single-pass browser candidate. Train6's DPO merge cuts qwen judge leaks to **11/22** (leak markers 8/22, two-judge voice 3.47, 0 loops, 0 frame breaks) but must-answer is **98/135**, and the name audit flags one Salève accent near-name, so the bar is still missed. Its NF4 proxy is 93/135 and 12/22. The train3 merged single-pass model (62/135), the train5 second-epoch merge (97/135) and the train6 CE follow-up (87/135, qwen judge leaks 6/22) should not ship. Spoiler leaks on *Frankenstein* are still well above the Phase 1 level.
- This is research code, written to run on one machine. Expect to adjust paths.

## License

- **Code** (`src/`, `tools/`, `configs/`): [Apache License 2.0](LICENSE), the same license as the Qwen3 base model.
- **Docs, reports, data and the *No Return* story** (`docs/`, `reports/`, `data/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please credit "Boyd Wilson, Abundant Frontier Institute and Omnibond Systems."
- ***Frankenstein*** (1818) is in the public domain. Question sets and voice samples written about it are CC BY 4.0.

## Citing

> Boyd Wilson. *Model as Output: stories you can step into.* Abundant Frontier Institute and Omnibond Systems, 2026. https://github.com/abundantfrontier/model-as-output
