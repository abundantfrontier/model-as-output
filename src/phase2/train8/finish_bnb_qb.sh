#!/bin/bash
# Finish the killed merged_qb NF4 proxy. Generation and the qwen judge are already on disk.
# Do not regenerate: temperature 0.7 would draw a new sample.
set -u
cd $HOME/output-model/phase2-frankenstein/train8
PY=$HOME/output-model/exp3/.venv/bin/python
COND=merged_qb_bnb4_proxy
echo "===== gemma judge resume $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/judge.py \
  "runs/${COND}.json" \
  --model=gemma3:12b \
  --voice-key=voice_gemma \
  --voice-only \
  --out-suffix=.gemma.voice.json
echo "gemma_exit=$?"
ollama stop gemma3:12b || true
echo "===== merge judges $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/merge_judges.py \
  "runs/${COND}.qwen.judged.json" \
  "runs/${COND}.gemma.voice.json" \
  --out "runs/${COND}.judged.json"
"$PY" -u scripts/name_audit.py \
  "runs/${COND}.json" \
  "runs/${COND}.name_audit.json"
"$PY" -u scripts/score_run.py "runs/${COND}.judged.json"
"$PY" -u scripts/compare_train7.py "runs/${COND}.judged.json"
echo "===== bnb done $(TZ=America/Los_Angeles date) $COND ====="
