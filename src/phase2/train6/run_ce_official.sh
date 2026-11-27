#!/bin/bash
# n=3 eval of the CE recovery merge. Same harness as merged_dpo_rag.
set -u
cd $HOME/output-model/phase2-frankenstein/train6
PY=$HOME/output-model/exp3/.venv/bin/python
MERGED=$HOME/output-model/phase2-frankenstein/train6/merged_ce
if [ ! -f "$MERGED/merge_report.json" ]; then
  echo "FAILED: missing $MERGED/merge_report.json"
  exit 1
fi
mapfile -t apps < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
for row in "${apps[@]}"; do
  [ -z "$row" ] && continue
  case "$row" in
    *ollama*|*llama-server*) ;;
    *) echo "FAILED: unexpected GPU compute app: $row"; exit 1 ;;
  esac
done
echo "===== eval start $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/eval_pass1.py merged_ce_rag \
  --n 3 \
  --merged "$MERGED" \
  --rag \
  --chars victor,creature \
  --backend bge \
  --out $HOME/output-model/phase2-frankenstein/train6/runs/merged_ce_rag.json
echo "===== qwen judge $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/judge.py \
  runs/merged_ce_rag.json \
  --model=qwen3:14b \
  --voice-key=voice_qwen \
  --out-suffix=.qwen.judged.json
ollama stop qwen3:14b || true
echo "===== gemma judge $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/judge.py \
  runs/merged_ce_rag.json \
  --model=gemma3:12b \
  --voice-key=voice_gemma \
  --voice-only \
  --out-suffix=.gemma.voice.json
ollama stop gemma3:12b || true
echo "===== merge judges $(TZ=America/Los_Angeles date) ====="
"$PY" -u scripts/merge_judges.py \
  runs/merged_ce_rag.qwen.judged.json \
  runs/merged_ce_rag.gemma.voice.json \
  --out runs/merged_ce_rag.judged.json
"$PY" -u scripts/name_audit.py \
  runs/merged_ce_rag.json \
  runs/merged_ce_rag.name_audit.json
"$PY" -u scripts/score_run.py runs/merged_ce_rag.judged.json
echo "===== official done $(TZ=America/Los_Angeles date) ====="
