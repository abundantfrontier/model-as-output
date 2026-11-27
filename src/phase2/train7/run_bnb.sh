#!/bin/bash
# 4-bit proxy of one train7 merge. ptyxis is allowed on the idle gate.
set -u
cd $HOME/output-model/phase2-frankenstein/train7
PY=$HOME/output-model/exp3/.venv/bin/python
NAME="${1:-mixa}"
MERGED="$HOME/output-model/phase2-frankenstein/train7/merged_${NAME}"
COND="merged_${NAME}_bnb4_proxy"
if [ ! -f "$MERGED/merge_report.json" ]; then
  echo "FAILED: missing $MERGED/merge_report.json"
  exit 1
fi
# shellcheck source=gpu_ok.sh
source $HOME/output-model/phase2-frankenstein/train7/scripts/gpu_ok.sh
gpu_ok || exit 1
echo "===== bnb start $(TZ=America/Los_Angeles date) $COND ====="
"$PY" -u scripts/eval_bnb_proxy.py \
  --merged "$MERGED" \
  --cond "$COND" \
  --out "$HOME/output-model/phase2-frankenstein/train7/runs/${COND}.json"
echo "===== qwen judge $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/judge.py \
  "runs/${COND}.json" \
  --model=qwen3:14b \
  --voice-key=voice_qwen \
  --out-suffix=.qwen.judged.json
ollama stop qwen3:14b || true
echo "===== gemma judge $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/judge.py \
  "runs/${COND}.json" \
  --model=gemma3:12b \
  --voice-key=voice_gemma \
  --voice-only \
  --out-suffix=.gemma.voice.json
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
echo "===== bnb done $(TZ=America/Los_Angeles date) $COND ====="
