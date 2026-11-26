#!/bin/bash
# exp7 driver. Resumable: skip a stage when its report or runs/<cond>.json exists.
# Does not modify exp3/exp4/exp5/exp6, does not push, does not pip-install.
cd "$(dirname "$0")"
P="$HOME/output-model/exp3/.venv/bin/python"
export PYTHONUNBUFFERED=1
mkdir -p runs data smoke init
echo $$ > run.pid
exec 9>run.lock
if ! flock -n 9; then
  echo "$(date) another exp7 run holds run.lock"
  exit 1
fi
stage=start
done_ok=0
flag() { if [ -f "$1" ]; then echo "- $2: done"; else echo "- $2: pending"; fi; }
write_status() {
  local cur="$1" st="$2" nxt="$3"
  echo "$(date)	$cur	$st	$nxt" >> status.tsv
  {
    echo "# exp7 build status"
    echo
    echo "Updated: $(date)"
    echo
    echo "- Current stage: $cur"
    echo "- State: $st"
    echo "- Next: $nxt"
    echo "- PID: $(cat run.pid 2>/dev/null)"
    echo
    echo "## Checklist"
    echo
    flag init/lora_v_merged/adapter_model.safetensors "init copy of lora_v_merged"
    flag data/v_light_dpo_meta.json "light dpo pairs"
    flag lora_v_light_dpo/train_report.json "v_light_dpo train"
    flag runs/v_light_dpo_v.judged.json "v_light_dpo judge v"
    flag runs/v_light_dpo_v_rag.judged.json "v_light_dpo judge v_rag"
    flag data/v_light_contrast_report.json "light contrast data"
    flag lora_v_light_contrast/train_report.json "v_light_contrast train"
    flag runs/v_light_contrast_v.judged.json "v_light_contrast judge v"
    flag runs/v_light_contrast_v_rag.judged.json "v_light_contrast judge v_rag"
    flag lora_v_light_combo/train_report.json "v_light_combo train"
    flag runs/v_light_combo_v.judged.json "v_light_combo judge v"
    flag runs/v_light_combo_v_rag.judged.json "v_light_combo judge v_rag"
    flag data/v_light_replay_meta.json "facts-replay mix"
    flag lora_v_light_dpo_replay/train_report.json "v_light_dpo_replay train"
    flag runs/v_light_dpo_replay_v.judged.json "v_light_dpo_replay judge v"
    flag runs/v_light_dpo_replay_v_rag.judged.json "v_light_dpo_replay judge v_rag"
    flag data/sweep.json "checkpoint sweep"
    flag results.md "results.md"
    flag REPRODUCE-notes.md "REPRODUCE-notes.md"
    echo
    echo "## History"
    echo
    echo '```'
    cat status.tsv 2>/dev/null
    echo '```'
  } > build-status.md
}
fail() {
  echo "$(date) FAIL $stage exit $1"
  write_status "$stage" "failed" "re-run bash run.sh"
  exit "$1"
}
trap 'if [ "$done_ok" != 1 ]; then write_status "$stage" "failed" "re-run bash run.sh"; fi' EXIT
unload() {
  if ! command -v ollama >/dev/null 2>&1; then return 0; fi
  ollama ps 2>/dev/null | awk 'NR>1 {print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" >/dev/null 2>&1 || true
  done
  return 0
}
facts_ok() {
  $P -c 'import frozen; frozen.check()'
  rc=$?
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
}
hj() {
  $P -c 'import json,sys
d=json.load(open("hparams.json"))
for k in sys.argv[1:]:
    d=d[k]
print(d)' "$@"
}
refresh() {
  stage=summarize
  $P summarize7.py > summarize.log 2>&1
  rc=$?
  echo "$(date) summarize exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  $P notes7.py > notes.log 2>&1
  rc=$?
  echo "$(date) notes exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
}
ensure_inputs() {
  local f src
  for f in eval4.py judge.py common.py chat.py; do
    src="$HOME/output-model/exp4/$f"
    if [ ! -f "$f" ] || ! cmp -s "$src" "$f"; then
      cp -a "$src" "$f"
    fi
  done
  mkdir -p data/chapters runs init
  ln -sfn $HOME/output-model/exp3/data/chapters/ch1.md data/chapters/ch1.md
  ln -sfn $HOME/output-model/exp3/data/chapters/ch2.md data/chapters/ch2.md
  ln -sfn $HOME/output-model/exp3/data/chapters/ch3.md data/chapters/ch3.md
  ln -sfn $HOME/output-model/exp3/data/chapters/ch4.md data/chapters/ch4.md
  ln -sfn $HOME/output-model/exp3/data/chapters/ch5.md data/chapters/ch5.md
  ln -sfn $HOME/output-model/exp3/data/knowledge-state.yaml data/knowledge-state.yaml
  ln -sfn $HOME/output-model/exp3/data/leak-tests.md data/leak-tests.md
  if [ ! -f runs/q4_fv_rag.judged.json ]; then cp -a "$HOME/output-model/exp4/runs/q4_fv_rag.judged.json" runs/; fi
  if [ ! -f runs/q14_rag.judged.json ]; then cp -a "$HOME/output-model/exp4/runs/q14_rag.judged.json" runs/; fi
  if [ ! -f runs/v_merged_v_rag.judged.json ]; then cp -a "$HOME/output-model/exp6/runs/v_merged_v_rag.judged.json" runs/; fi
  if [ ! -f runs/v_combo_dpo_v_rag.judged.json ]; then cp -a "$HOME/output-model/exp6/runs/v_combo_dpo_v_rag.judged.json" runs/; fi
  src="$HOME/output-model/exp6/lora_v_merged/adapter_model.safetensors"
  if [ ! -f init/lora_v_merged/adapter_model.safetensors ] || ! cmp -s "$src" init/lora_v_merged/adapter_model.safetensors; then
    rm -rf init/lora_v_merged
    mkdir -p init/lora_v_merged
    cp -a "$HOME/output-model/exp6/lora_v_merged/." init/lora_v_merged/
  fi
}
eval_one() {
  local cond="$1"
  if [ -f "runs/${cond}.json" ]; then
    echo "$(date) eval $cond skip"
    return 0
  fi
  unload
  sleep 2
  stage="eval $cond"
  write_status "$stage" "running" "judge $cond"
  $P eval_cond.py "$cond" --n 3 > "runs/${cond}.log" 2>&1
  rc=$?
  echo "$(date) eval $cond exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  $P -c 'import json,sys; d=json.load(open(sys.argv[1])); n=len(d["results"]); assert n==204, n' "runs/${cond}.json"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$(date) eval $cond bad result count"
    rm -f "runs/${cond}.json"
    fail "$rc"
  fi
  facts_ok
}
judge_some() {
  local files=()
  local c
  for c in "$@"; do
    if [ -f "runs/$c.json" ] && [ ! -f "runs/$c.judged.json" ]; then
      files+=("runs/$c.json")
    fi
  done
  if [ "${#files[@]}" -eq 0 ]; then
    echo "$(date) judge $* skip"
    return 0
  fi
  unload
  sleep 2
  stage="judge $*"
  write_status "$stage" "running" "summarize"
  $P judge.py "${files[@]}" >> runs/judge.log 2>&1
  rc=$?
  echo "$(date) judge $* exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
}
train7() {
  local name="$1"; shift
  local out="lora_${name}"
  if [ -f "$out/train_report.json" ]; then
    echo "$(date) train $out skip"
    return 0
  fi
  unload
  sleep 2
  stage="train $name"
  write_status "$stage" "running" "eval ${name}"
  rm -rf "$out"
  facts_ok
  $P train7.py --out "$PWD/$out" "$@" > "train_${out}.log" 2>&1
  rc=$?
  echo "$(date) train $out exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  if [ ! -f "$out/train_report.json" ]; then fail 3; fi
  facts_ok
}

echo "$(date) exp7 start"
ensure_inputs
facts_ok
write_status "start" "running" "prepare"
echo "$(date) hparams dpo beta $(hj v_light_dpo beta) lr $(hj v_light_dpo lr) cap $(hj pair_cap)"
echo "$(date) hparams contrast ul $(hj v_light_contrast ul_weight) lr $(hj v_light_contrast lr)"
echo "$(date) hparams combo beta $(hj v_light_combo beta) lr $(hj v_light_combo lr) steps $(hj v_light_combo max_steps)"
echo "$(date) hparams replay beta $(hj v_light_dpo_replay beta) lr $(hj v_light_dpo_replay lr) facts $(hj facts_ratio)"
refresh

if [ ! -f data/v_light_dpo_meta.json ] || [ ! -f data/v_light_contrast_report.json ] || [ ! -f data/v_light_replay_meta.json ]; then
  stage=prepare
  write_status "$stage" "running" "train v_light_dpo"
  $P build7.py prepare --out-dir data \
    --pair-cap "$(hj pair_cap)" \
    --facts-ratio "$(hj facts_ratio)" \
    --per-prompt "$(hj per_prompt_negs)" \
    --fresh-per "$(hj fresh_per_prompt)" > prepare.log 2>&1
  rc=$?
  echo "$(date) prepare exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
else
  echo "$(date) prepare skip"
fi
facts_ok

BASE="$(hj base)"
INIT="$PWD/init/lora_v_merged"
train7 v_light_dpo --stage voice --base "$BASE" --facts "" \
  --init "$INIT" --pairs "$PWD/data/v_light_dpo_pairs.jsonl" \
  --epochs "$(hj v_light_dpo epochs)" --lr "$(hj v_light_dpo lr)" \
  --beta "$(hj v_light_dpo beta)" --loss dpo --r 32 \
  --dropout "$(hj v_light_dpo dropout)" \
  --max-steps "$(hj v_light_dpo max_steps)" \
  --save-every "$(hj v_light_dpo save_every)"
eval_one v_light_dpo_v
eval_one v_light_dpo_v_rag
judge_some v_light_dpo_v v_light_dpo_v_rag
refresh

train7 v_light_contrast --stage voice --base "$BASE" --facts "" \
  --init "$INIT" --pairs "$PWD/data/v_light_contrast_train.jsonl" \
  --epochs "$(hj v_light_contrast epochs)" --lr "$(hj v_light_contrast lr)" \
  --loss unlikelihood --ul-weight "$(hj v_light_contrast ul_weight)" --r 32 \
  --dropout "$(hj v_light_contrast dropout)" \
  --max-steps "$(hj v_light_contrast max_steps)" \
  --save-every "$(hj v_light_contrast save_every)"
eval_one v_light_contrast_v
eval_one v_light_contrast_v_rag
judge_some v_light_contrast_v v_light_contrast_v_rag
refresh

train7 v_light_combo --stage voice --base "$BASE" --facts "" \
  --init "$PWD/lora_v_light_contrast" --pairs "$PWD/data/v_light_dpo_pairs.jsonl" \
  --epochs "$(hj v_light_combo epochs)" --lr "$(hj v_light_combo lr)" \
  --beta "$(hj v_light_combo beta)" --loss dpo --r 32 \
  --dropout "$(hj v_light_combo dropout)" \
  --max-steps "$(hj v_light_combo max_steps)" \
  --save-every "$(hj v_light_combo save_every)"
eval_one v_light_combo_v
eval_one v_light_combo_v_rag
judge_some v_light_combo_v v_light_combo_v_rag
refresh

train7 v_light_dpo_replay --stage voice --base "$BASE" --facts "" \
  --init "$INIT" --pairs "$PWD/data/v_light_dpo_replay.jsonl" \
  --epochs "$(hj v_light_dpo_replay epochs)" --lr "$(hj v_light_dpo_replay lr)" \
  --beta "$(hj v_light_dpo_replay beta)" --loss dpo --r 32 \
  --dropout "$(hj v_light_dpo_replay dropout)" \
  --max-steps "$(hj v_light_dpo_replay max_steps)" \
  --save-every "$(hj v_light_dpo_replay save_every)"
eval_one v_light_dpo_replay_v
eval_one v_light_dpo_replay_v_rag
judge_some v_light_dpo_replay_v v_light_dpo_replay_v_rag
refresh

stage=sweep
write_status "$stage" "running" "checkpoint eval or skip"
$P sweep7.py plan > sweep_plan.log 2>&1
rc=$?
echo "$(date) sweep plan exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
mapfile -t SWEEP_CONDS < <($P -c 'import json; d=json.load(open("data/sweep_plan.json")); print("\n".join(e["cond"] for e in d["evals"]))')
if [ "${#SWEEP_CONDS[@]}" -eq 0 ]; then
  echo "$(date) sweep eval skip ($($P -c 'import json; print(json.load(open("data/sweep_plan.json")).get("reason"))'))"
else
  for c in "${SWEEP_CONDS[@]}"; do
    [ -n "$c" ] || continue
    eval_one "$c"
  done
  judge_some "${SWEEP_CONDS[@]}"
fi
$P sweep7.py select > sweep_select.log 2>&1
rc=$?
echo "$(date) sweep select exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
refresh

stage=reproduce
$P reproduce7.py > reproduce.log 2>&1
rc=$?
echo "$(date) reproduce exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
stage=commit
$P commit7.py commit > commit.log 2>&1
rc=$?
echo "$(date) commit exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
facts_ok
write_status "done" "done" "none"
done_ok=1
echo "$(date) ALL DONE"
