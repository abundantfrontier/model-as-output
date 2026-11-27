#!/bin/bash
# 4-bit proxy of merged_dpo, the better of the two train6 merges.
set -u
cd $HOME/output-model/phase2-frankenstein/train6
PY=$HOME/output-model/exp3/.venv/bin/python
echo "===== bnb start $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/eval_bnb_proxy.py \
  --merged $HOME/output-model/phase2-frankenstein/train6/merged_dpo \
  --cond merged_dpo_bnb4_proxy \
  --out $HOME/output-model/phase2-frankenstein/train6/runs/merged_dpo_bnb4_proxy.json
echo "===== qwen judge $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/judge.py \
  runs/merged_dpo_bnb4_proxy.json \
  --model=qwen3:14b \
  --voice-key=voice_qwen \
  --out-suffix=.qwen.judged.json
ollama stop qwen3:14b || true
echo "===== gemma judge $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/judge.py \
  runs/merged_dpo_bnb4_proxy.json \
  --model=gemma3:12b \
  --voice-key=voice_gemma \
  --voice-only \
  --out-suffix=.gemma.voice.json
ollama stop gemma3:12b || true
echo "===== merge judges $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/merge_judges.py \
  runs/merged_dpo_bnb4_proxy.qwen.judged.json \
  runs/merged_dpo_bnb4_proxy.gemma.voice.json \
  --out runs/merged_dpo_bnb4_proxy.judged.json
"$PY" -u scripts/name_audit.py \
  runs/merged_dpo_bnb4_proxy.json \
  runs/merged_dpo_bnb4_proxy.name_audit.json
"$PY" -u scripts/score_run.py runs/merged_dpo_bnb4_proxy.judged.json
echo "===== bnb done $(TZ=America/Los_Angeles date) ====="
