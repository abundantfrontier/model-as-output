#!/bin/bash
# Finish epoch 1 from the train4 adapter, run epoch 2, smoke, CPU-merge, eval, judges.
# Does not release the GPU lock. Repetition penalty stays off.
set -euo pipefail
cd $HOME/output-model/phase2-frankenstein/train5
PY=$HOME/output-model/exp3/.venv/bin/python
LOG=logs/pipeline.log
exec >>"$LOG" 2>&1
echo "===== pipeline $(TZ=America/Los_Angeles date) ====="
set -x

"$PY" -u scripts/check_holdout.py

"$PY" - << 'PY'
import os, random, hashlib, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "scripts")
from transformers import AutoTokenizer
import train
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
pairs = train.C.load_jsonl("data/sft_pairs.jsonl")
ex = train.build_ce_examples(tok, pairs, 0)
print("precheck examples", len(ex), "mismatch", getattr(train.pair_example, "mismatch", 0), flush=True)
if len(ex) != 2097:
    raise SystemExit("precheck expected 2097 examples")
random.seed(0)
random.shuffle(ex)
fp = hashlib.sha256()
for item in ex:
    fp.update(",".join(str(t) for t in item[0][:6]).encode())
    fp.update(b"|")
print("precheck epoch1 order", fp.hexdigest()[:16], "remain", len(ex) - 1407, flush=True)
PY

"$PY" -u scripts/train.py \
  --stage voice \
  --out lora_sft \
  --init $HOME/output-model/phase2-frankenstein/train4/lora_sft \
  --pairs data/sft_pairs.jsonl \
  --epoch2-pairs data/sft_pairs_epoch2.jsonl \
  --resume-steps 1407 \
  --epochs 1 \
  --lr 2e-5 \
  --r 32 \
  --alpha 64 \
  --dropout 0.05 \
  --loss ce \
  --log-every 25 \
  --save-every 699 \
  --reserve-min 45

"$PY" -u scripts/smoke_single.py \
  --adapter $HOME/output-model/phase2-frankenstein/train5/lora_sft
if [ -f smoke/ABORT_LOOPS ]; then
  echo "ABORT_LOOPS: skip merge and full eval"
  exit 2
fi

"$PY" -u scripts/merge.py \
  --base Qwen/Qwen3-4B \
  --adapter $HOME/output-model/phase2-frankenstein/train5/lora_sft \
  --out $HOME/output-model/phase2-frankenstein/train5/merged_sft \
  --name voice

"$PY" -u scripts/eval_pass1.py merged_sft_rag \
  --n 3 \
  --merged $HOME/output-model/phase2-frankenstein/train5/merged_sft \
  --rag \
  --chars victor,creature \
  --backend bge \
  --out $HOME/output-model/phase2-frankenstein/train5/runs/merged_sft_rag.json

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
echo "===== pipeline done $(TZ=America/Los_Angeles date) ====="
