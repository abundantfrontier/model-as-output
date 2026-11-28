#!/bin/bash
# Light QLoRA DPO on train7 lora_mixa. Base weights in NF4, adapter in bf16.
# One epoch, lr 4e-6, beta 0.1. Not the train6 CE follow-up.
set -u
cd $HOME/output-model/phase2-frankenstein/train8
PY=$HOME/output-model/exp3/.venv/bin/python
PAIRS="${1:-data/mixed_qa.jsonl}"
OUT="${2:-lora_qa}"
LR="${3:-4e-6}"
INIT="${4:-$HOME/output-model/phase2-frankenstein/train7/lora_mixa}"
# shellcheck source=gpu_ok.sh
source "$(dirname "$0")/gpu_ok.sh"
gpu_ok || exit 1
if ! grep -q "HOLDOUT OK" logs/holdout_before_train.txt 2>/dev/null; then
  echo "FAILED: missing HOLDOUT OK in logs/holdout_before_train.txt"
  exit 1
fi
case "$INIT" in
  *"/train6/lora_ce"*|*"/train6/merged_ce"*|*"/train7/lora_mixb"*|*"/train5/lora_sft"*)
    echo "FAILED: banned init $INIT"
    exit 1
    ;;
esac
echo "===== train start $(TZ=America/Los_Angeles date) $OUT lr $LR init $INIT nf4 ====="
exec "$PY" -u scripts/train.py \
  --stage voice \
  --loss dpo \
  --nf4 \
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
