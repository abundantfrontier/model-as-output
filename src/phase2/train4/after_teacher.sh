#!/bin/bash
# Stages after teacher generation. GPU lock must still be held.
# Stops early on loop-smoke abort. Does not release the lock.
set -euo pipefail
cd $HOME/output-model/phase2-frankenstein/train4
PY=$HOME/output-model/exp3/.venv/bin/python
LOG=logs/after_teacher.log
exec >>"$LOG" 2>&1
echo "===== after_teacher $(TZ=America/Los_Angeles date) ====="
set -x
"$PY" -u scripts/factdiff_train.py --reserve-min 70
"$PY" -u scripts/assemble_sft.py
"$PY" -u scripts/check_holdout.py
"$PY" -u scripts/train.py \
  --stage voice \
  --out lora_sft \
  --pairs data/sft_pairs.jsonl \
  --epochs 1 \
  --lr 2e-5 \
  --r 32 \
  --alpha 64 \
  --dropout 0.05 \
  --loss ce \
  --log-every 25 \
  --reserve-min 45
"$PY" -u scripts/smoke_single.py --adapter $HOME/output-model/phase2-frankenstein/train4/lora_sft
if [ -f smoke/ABORT_LOOPS ]; then
  echo "ABORT_LOOPS: skip merge and full eval"
  exit 2
fi
"$PY" -u scripts/merge.py \
  --base Qwen/Qwen3-4B \
  --adapter $HOME/output-model/phase2-frankenstein/train4/lora_sft \
  --out $HOME/output-model/phase2-frankenstein/train4/merged_sft \
  --name voice
"$PY" -u scripts/eval_pass1.py merged_sft_rag \
  --n 3 \
  --merged $HOME/output-model/phase2-frankenstein/train4/merged_sft \
  --rag \
  --chars victor,creature \
  --backend bge \
  --out $HOME/output-model/phase2-frankenstein/train4/runs/merged_sft_rag.json
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
echo "===== after_teacher done $(TZ=America/Los_Angeles date) ====="
