#!/bin/bash
# Tiny-slice checks. Writes under exp7/smoke/ only (plus /tmp for limited eval).
# Does not write lora_v_light_*/train_report.json, so run.sh will not skip real trains.
set -euo pipefail
cd "$(dirname "$0")"
P="$HOME/output-model/exp3/.venv/bin/python"
export PYTHONUNBUFFERED=1
rm -rf smoke
mkdir -p smoke/data
echo "$(date) exp7 smoke start"
$P -c 'import frozen; print("facts", frozen.check())'
$P -c 'import voicefilt; voicefilt.self_test(); print("voicefilt ok")'
$P build7.py self-test
$P commit7.py --self-test
$P -c 'import eval_cond; assert eval_cond.parse_core("v_light_dpo_c40_v_rag")==("v_light_dpo_c40", True); assert eval_cond.parse_core("v_light_dpo_replay_v")==("v_light_dpo_replay", False); print("parse ok")'
$P build7.py prepare --out-dir smoke/data --pair-cap 141 --facts-ratio 0.08 --per-prompt 2 --skip-fresh --skip-teacher --limit 20
$P - << 'PY'
import json, os
dpo=[json.loads(l) for l in open("smoke/data/v_light_dpo_pairs.jsonl") if l.strip()]
con=[json.loads(l) for l in open("smoke/data/v_light_contrast_train.jsonl") if l.strip()]
rep=[json.loads(l) for l in open("smoke/data/v_light_dpo_replay.jsonl") if l.strip()]
meta=json.load(open("smoke/data/v_light_replay_meta.json"))
assert dpo, "no dpo pairs"
assert con, "no contrast rows"
assert all(r["reason"] in ("frame","marker") for r in con), set(r["reason"] for r in con)
import re, sys
sys.path.insert(0,".")
import common as C
for r in dpo + con:
    assert not re.search(C.FRAME, r["answer"], re.I), r["answer"][:80]
    assert not C.fail_hits(r["answer"]), r["answer"][:80]
    if r.get("bad"):
        # frame/marker pairs must carry the hit on the rejected side; voice pairs may be clean
        if r["reason"] in ("frame","marker"):
            bad_hit = bool(re.search(C.FRAME, r["bad"], re.I) or C.fail_hits(r["bad"]))
            assert bad_hit, r["bad"][:80]
facts=[r for r in rep if not r.get("bad")]
assert facts, "replay added no facts"
assert meta["facts_ratio"] < 0.25, meta
assert meta["facts_ratio"] > 0, meta
print("data", len(dpo), "contrast", len(con), "replay", len(rep), "facts_ratio", meta["facts_ratio"])
PY
echo "$(date) cpu prepare ok"
unload() {
  if ! command -v ollama >/dev/null 2>&1; then return 0; fi
  ollama ps 2>/dev/null | awk 'NR>1 {print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" >/dev/null 2>&1 || true
  done
}
# Exercise fresh samples and one teacher rewrite, then train on the CPU slice
# (the GPU slice is smaller and may not include a bad completion).
unload
$P build7.py prepare --out-dir smoke/live --pair-cap 20 --facts-ratio 0.08 --per-prompt 2 \
  --fresh-per 2 --fresh-limit 1 --teacher-limit 1 --limit 8 --force > smoke/live_prepare.log 2>&1
test -f smoke/live/fresh_v_merged.jsonl
echo "$(date) fresh+teacher ok"
unload
rm -rf smoke/lora_dpo
$P train7.py --stage voice --out "$PWD/smoke/lora_dpo" \
  --base "$HOME/output-model/exp6/merged_q4_facts" --facts "" \
  --init "$PWD/init/lora_v_merged" \
  --pairs "$PWD/smoke/data/v_light_dpo_pairs.jsonl" \
  --epochs 1 --lr 2e-6 --beta 0.05 --loss dpo --r 32 --dropout 0 \
  --limit 8 --max-steps 4 --save-every 2 > smoke/train_dpo.log 2>&1
test -f smoke/lora_dpo/train_report.json
$P -c 'import json,os; d=json.load(open("smoke/lora_dpo/train_report.json")); assert d["loss"]=="dpo" and 2<=d["steps"]<=4 and d["beta"]==0.05, d; assert d["steps"]<=2 or os.path.isdir("smoke/lora_dpo/checkpoint-2"); print("dpo report", d["steps"], d["peak_vram_gb"])'
echo "$(date) smoke dpo ok"
unload
rm -rf smoke/lora_ul
$P train7.py --stage voice --out "$PWD/smoke/lora_ul" \
  --base "$HOME/output-model/exp6/merged_q4_facts" --facts "" \
  --init "$PWD/init/lora_v_merged" \
  --pairs "$PWD/smoke/data/v_light_contrast_train.jsonl" \
  --epochs 1 --lr 5e-5 --loss unlikelihood --ul-weight 0.4 --r 32 --dropout 0.05 \
  --limit 4 --max-steps 3 --save-every 2 > smoke/train_ul.log 2>&1
test -f smoke/lora_ul/train_report.json
$P -c 'import json; d=json.load(open("smoke/lora_ul/train_report.json")); assert d["loss"]=="unlikelihood" and d["ul_weight"]==0.4 and d["steps"]<=3, d; print("ul report", d["steps"])'
echo "$(date) smoke contrast ok"
unload
rm -rf smoke/lora_combo
$P train7.py --stage voice --out "$PWD/smoke/lora_combo" \
  --base "$HOME/output-model/exp6/merged_q4_facts" --facts "" \
  --init "$PWD/smoke/lora_ul" \
  --pairs "$PWD/smoke/data/v_light_dpo_pairs.jsonl" \
  --epochs 1 --lr 1e-6 --beta 0.03 --loss dpo --r 32 --dropout 0 \
  --limit 4 --max-steps 2 --save-every 2 > smoke/train_combo.log 2>&1
test -f smoke/lora_combo/train_report.json
echo "$(date) smoke combo ok"
unload
$P eval4.py smoke_v --n 1 --limit 2 --base "$HOME/output-model/exp6/merged_q4_facts" \
  --voice "$PWD/smoke/lora_dpo" --adapters v > smoke/eval.log 2>&1
$P -c 'import json; d=json.load(open("/tmp/exp4_eval_smoke.json")); assert d["meta"]["cond"]=="smoke_v" and len(d["results"])==2, d["meta"]; print("eval", len(d["results"]))'
# parser sees a real checkpoint directory, then remove it so run.sh starts clean
mkdir -p lora_v_light_dpo/checkpoint-40
cp smoke/lora_dpo/adapter_config.json lora_v_light_dpo/checkpoint-40/
$P -c 'import eval_cond; v=eval_cond.voice_for("v_light_dpo_c40"); assert v.endswith("lora_v_light_dpo/checkpoint-40"), v; print("voice_for", v)'
rm -rf lora_v_light_dpo
echo "$(date) exp7 smoke ok"
