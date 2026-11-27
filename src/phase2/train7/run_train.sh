#!/bin/bash
# Light mixed DPO/CE from train6 lora_dpo. One epoch, lr 4e-6.
set -u
cd $HOME/output-model/phase2-frankenstein/train7
PY=$HOME/output-model/exp3/.venv/bin/python
PAIRS="${1:-data/mixed_a.jsonl}"
OUT="${2:-lora_mixa}"
LR="${3:-4e-6}"
INIT="${4:-$HOME/output-model/phase2-frankenstein/train6/lora_dpo}"
# shellcheck source=gpu_ok.sh
source "$(dirname "$0")/gpu_ok.sh"
gpu_ok || exit 1
if ! grep -q "HOLDOUT OK" logs/holdout_before_train.txt 2>/dev/null; then
  echo "FAILED: missing HOLDOUT OK in logs/holdout_before_train.txt"
  exit 1
fi
echo "===== train start $(TZ=America/Los_Angeles date) $OUT lr $LR init $INIT ====="
exec "$PY" -u scripts/train.py \
  --stage voice \
  --loss dpo \
  --init "$INIT" \
  --pairs "$PAIRS" \
  --out "$OUT" \
  --lr "$LR" \
  --beta 0.1 \
  --epochs 1 \
  --r 32 \
  --alpha 64 \
  --dropout 0 \
  --save-every 80 \
  --log-every 10 \
  --reserve-min 50
