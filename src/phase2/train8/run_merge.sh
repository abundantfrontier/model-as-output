#!/bin/bash
# CPU bf16 merge. Refuses to run if CUDA is visible. Do not overlap with a GPU load.
set -u
cd $HOME/output-model/phase2-frankenstein/train8
PY=$HOME/output-model/exp3/.venv/bin/python
NAME="${1:-qa}"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "===== merge start $(TZ=America/Los_Angeles date) $NAME ====="
exec "$PY" -u scripts/merge.py \
  --base Qwen/Qwen3-4B \
  --adapter "$HOME/output-model/phase2-frankenstein/train8/lora_${NAME}" \
  --out "$HOME/output-model/phase2-frankenstein/train8/merged_${NAME}" \
  --name default
