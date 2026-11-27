#!/bin/bash
# Judges and name audit after the no-penalty merged eval. Does not release the lock.
set -euo pipefail
cd $HOME/output-model/phase2-frankenstein/train4
PY=$HOME/output-model/exp3/.venv/bin/python
LOG=logs/after_eval.log
exec >>"$LOG" 2>&1
echo "===== after_eval $(TZ=America/Los_Angeles date) ====="
set -x
"$PY" -u scripts/judge.py \
  runs/merged_sft_rag.json \
  --model=qwen3:14b \
  --voice-key=voice_qwen \
  --out-suffix=.qwen.judged.json
ollama stop qwen3:14b || true
"$PY" -u scripts/judge.py \
  runs/merged_sft_rag.json \
  --model=gemma3:12b \
  --voice-key=voice_gemma \
  --voice-only \
  --out-suffix=.gemma.voice.json
ollama stop gemma3:12b || true
"$PY" -u scripts/merge_judges.py \
  runs/merged_sft_rag.qwen.judged.json \
  runs/merged_sft_rag.gemma.voice.json \
  --out runs/merged_sft_rag.judged.json
"$PY" -u scripts/name_audit.py \
  runs/merged_sft_rag.json \
  runs/merged_sft_rag.name_audit.json
"$PY" -u scripts/score_run.py runs/merged_sft_rag.judged.json
echo "===== after_eval done $(TZ=America/Los_Angeles date) ====="
