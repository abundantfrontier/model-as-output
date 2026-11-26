#!/bin/bash
# Tiny-slice checks. Writes only under exp6/smoke/ plus the real merge (idempotent).
# Does not write lora_v_* train reports used by run.sh.
set -euo pipefail
cd "$(dirname "$0")"
P="$HOME/output-model/exp3/.venv/bin/python"
export PYTHONUNBUFFERED=1
mkdir -p smoke/data smoke/contrast
echo "$(date) exp6 smoke start"
$P -c 'import frozen; print("facts", frozen.check())'
$P -c 'import voicefilt; voicefilt.self_test()'
$P mix_replay.py --out-dir smoke/data --limit 20 --facts-ratio 0.25
test -f smoke/data/v_replay_train.jsonl
test -f smoke/data/v_merged_train.jsonl
$P -c 'import json; d=json.load(open("smoke/data/v_replay_mix.json")); assert 0.2 <= d["facts_ratio"] <= 0.3, d; print("mix", d["rows"], d["facts_ratio"], d["sha256"])'
unload() {
  if ! command -v ollama >/dev/null 2>&1; then return 0; fi
  ollama ps 2>/dev/null | awk 'NR>1 {print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" >/dev/null 2>&1 || true
  done
}
unload
sleep 1
rm -rf smoke/lora_v_replay
$P train6.py --stage voice --out "$PWD/smoke/lora_v_replay" \
  --facts "$HOME/output-model/exp4/lora_q4_facts" \
  --pairs "$PWD/smoke/data/v_replay_train.jsonl" \
  --epochs 1 --r 32 --lr 1e-4 --loss ce --limit 20 > smoke/train_replay.log 2>&1
test -f smoke/lora_v_replay/train_report.json
echo "$(date) smoke train replay ok"
if [ ! -f merged_q4_facts/merge_report.json ]; then
  unload
  $P merge_facts.py --out "$PWD/merged_q4_facts" > smoke/merge.log 2>&1
  echo "$(date) smoke merge ok"
else
  echo "$(date) smoke merge skip (already present)"
fi
unload
rm -rf smoke/lora_v_merged
$P train6.py --stage voice --out "$PWD/smoke/lora_v_merged" \
  --base "$PWD/merged_q4_facts" --facts "" \
  --pairs "$PWD/smoke/data/v_merged_train.jsonl" \
  --epochs 1 --r 32 --lr 1e-4 --loss ce --limit 20 > smoke/train_merged.log 2>&1
test -f smoke/lora_v_merged/train_report.json
echo "$(date) smoke train merged ok"
$P eval4.py smoke_fv --n 1 --limit 2 --base Qwen/Qwen3-4B \
  --facts "$HOME/output-model/exp4/lora_q4_facts" \
  --voice "$PWD/smoke/lora_v_replay" --adapters fv > smoke/eval.log 2>&1
echo "$(date) smoke eval ok"
$P judge_sanity.py --limit 4 --gold-n 1 --out-dir smoke --md smoke/judge_sanity.md > smoke/judge_sanity.log 2>&1
test -f smoke/judge_sanity.md
echo "$(date) smoke judge_sanity ok"
cat > smoke/winner.json <<EOF
{"winner":"exp4_fv","mode":"fv","base":"Qwen/Qwen3-4B","facts":"$HOME/output-model/exp4/lora_q4_facts","voice":"$HOME/output-model/exp4/lora_q4_voice"}
EOF
$P build_contrast.py fresh --out-dir smoke/contrast --limit 3 --winner "$PWD/smoke/winner.json" > smoke/contrast_fresh.log 2>&1
$P build_contrast.py finish --out-dir smoke/contrast --limit 3 --winner "$PWD/smoke/winner.json" > smoke/contrast_finish.log 2>&1
test -f smoke/contrast/v_contrast_train.jsonl
echo "$(date) smoke contrast ok"
unload
rm -rf smoke/lora_ul
$P train6.py --stage voice --out "$PWD/smoke/lora_ul" \
  --facts "$HOME/output-model/exp4/lora_q4_facts" \
  --pairs "$PWD/smoke/contrast/v_contrast_train.jsonl" \
  --epochs 1 --r 32 --lr 1e-4 --loss unlikelihood --ul-weight 1.0 --limit 8 > smoke/train_ul.log 2>&1
test -f smoke/lora_ul/train_report.json
echo "$(date) smoke unlikelihood ok"
# tiny DPO if contrast produced a pair
$P -c '
import json,os
rows=[json.loads(l) for l in open("smoke/contrast/v_contrast_train.jsonl") if l.strip() and json.loads(l).get("bad")]
open("smoke/dpo.jsonl","w").writelines(json.dumps(r)+"\n" for r in rows[:4])
print("dpo rows", len(rows[:4]))
'
if [ "$(wc -l < smoke/dpo.jsonl)" -ge 1 ]; then
  unload
  rm -rf smoke/lora_dpo
  $P train6.py --stage voice --out "$PWD/smoke/lora_dpo" \
    --facts "$HOME/output-model/exp4/lora_q4_facts" \
    --init "$PWD/smoke/lora_ul" \
    --pairs "$PWD/smoke/dpo.jsonl" \
    --epochs 1 --lr 5e-6 --beta 0.1 --loss dpo --r 32 --limit 4 > smoke/train_dpo.log 2>&1
  test -f smoke/lora_dpo/train_report.json
  echo "$(date) smoke dpo ok"
else
  echo "$(date) smoke dpo skip (no pairs)"
fi
$P -c 'import frozen; frozen.check()'
echo "$(date) SMOKE OK"
