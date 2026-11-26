#!/bin/bash
# exp6 driver. Resumable: a stage is skipped when its report or runs/<cond>.json exists.
# Facts adapter exp4/lora_q4_facts is loaded and never trained.
cd "$(dirname "$0")"
P="$HOME/output-model/exp3/.venv/bin/python"
export PYTHONUNBUFFERED=1
mkdir -p runs data smoke
echo $$ > run.pid
exec 9>run.lock
if ! flock -n 9; then
  echo "$(date) another exp6 run holds run.lock"
  exit 1
fi
stage=start
done_ok=0
flag() { if [ -f "$1" ]; then echo "- $2: done"; else echo "- $2: pending"; fi; }
write_status() {
  local cur="$1" st="$2" nxt="$3"
  echo "$(date)	$cur	$st	$nxt" >> status.tsv
  {
    echo "# exp6 build status"
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
    flag judge_sanity.md "judge sanity"
    flag data/v_replay_mix.json "v_replay mix"
    flag lora_v_replay/train_report.json "v_replay train"
    flag runs/v_replay_fv.judged.json "v_replay judge fv"
    flag runs/v_replay_fv_rag.judged.json "v_replay judge fv_rag"
    flag merged_q4_facts/merge_report.json "merge facts into base"
    flag lora_v_merged/train_report.json "v_merged train"
    flag runs/v_merged_v.judged.json "v_merged judge v"
    flag runs/v_merged_v_rag.judged.json "v_merged judge v_rag"
    flag data/winner.json "pick winner a vs b"
    flag data/contrast_report.json "combo contrast data"
    flag lora_v_combo_contrast/train_report.json "combo contrast train"
    flag data/dpo_choice.json "dpo choice"
    flag lora_v_combo_dpo/train_report.json "combo dpo train"
    flag data/dpo_skipped.json "dpo skipped"
    flag data/8b_probe.json "8B probe"
    flag data/8b_skipped.json "8B skipped"
    flag lora_q8_facts/train_report.json "8B facts train"
    flag lora_v8_replay/train_report.json "8B replay train"
    flag lora_v8_merged/train_report.json "8B merged train"
    flag lora_v8_combo_contrast/train_report.json "8B contrast train"
    flag lora_v8_combo_dpo/train_report.json "8B dpo train"
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
refresh() {
  stage=summarize
  $P summarize6.py > summarize.log 2>&1
  rc=$?
  echo "$(date) summarize exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  $P notes6.py > notes.log 2>&1
  rc=$?
  echo "$(date) notes exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
}
eval_one() {
  local cond="$1"
  local soft="${2:-}"
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
  if [ "$rc" -ne 0 ]; then
    if [ "$soft" = "soft" ]; then return "$rc"; fi
    fail "$rc"
  fi
  $P -c 'import json,sys; d=json.load(open(sys.argv[1])); n=len(d["results"]); assert n==204, n' "runs/${cond}.json"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$(date) eval $cond bad result count"
    rm -f "runs/${cond}.json"
    if [ "$soft" = "soft" ]; then return "$rc"; fi
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
train6() {
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
  $P train6.py --out "$PWD/$out" "$@" > "train_${out}.log" 2>&1
  rc=$?
  echo "$(date) train $out exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  if [ ! -f "$out/train_report.json" ]; then fail 3; fi
  facts_ok
}
try_train6() {
  # like train6 but on failure write skip and return 1 instead of aborting the whole run (8B path)
  local name="$1"; shift
  local out="lora_${name}"
  if [ -f "$out/train_report.json" ]; then
    echo "$(date) train $out skip"
    return 0
  fi
  unload
  sleep 2
  stage="train $name"
  write_status "$stage" "running" "eval or skip 8B"
  rm -rf "$out"
  $P train6.py --out "$PWD/$out" "$@" > "train_${out}.log" 2>&1
  rc=$?
  echo "$(date) train $out exit $rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$out/train_report.json" ]; then
    return 1
  fi
  return 0
}

echo "$(date) exp6 start"
facts_ok
write_status "start" "running" "judge sanity"
refresh

# --- judge sanity ---
if [ ! -f judge_sanity.md ]; then
  unload
  sleep 1
  stage="judge sanity"
  write_status "$stage" "running" "mix replay"
  $P judge_sanity.py > judge_sanity.log 2>&1
  rc=$?
  echo "$(date) judge sanity exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
else
  echo "$(date) judge sanity skip"
fi
refresh

# --- data mix ---
if [ ! -f data/v_replay_mix.json ] || [ ! -f data/v_merged_mix.json ]; then
  stage="mix replay"
  write_status "$stage" "running" "train v_replay"
  $P mix_replay.py --out-dir data > mix_replay.log 2>&1
  rc=$?
  echo "$(date) mix replay exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
else
  echo "$(date) mix replay skip"
fi

# --- (a) v_replay ---
train6 v_replay --stage voice --facts "$HOME/output-model/exp4/lora_q4_facts" \
  --pairs "$PWD/data/v_replay_train.jsonl" --epochs 2 --r 32 --lr 1e-4 --loss ce
eval_one v_replay_fv
eval_one v_replay_fv_rag
judge_some v_replay_fv v_replay_fv_rag
refresh

# --- (b) merge then v_merged ---
if [ ! -f merged_q4_facts/merge_report.json ]; then
  unload
  sleep 2
  stage="merge facts"
  write_status "$stage" "running" "train v_merged"
  $P merge_facts.py --out "$PWD/merged_q4_facts" > merge_facts.log 2>&1
  rc=$?
  echo "$(date) merge facts exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
else
  echo "$(date) merge facts skip"
fi
train6 v_merged --stage voice --base "$PWD/merged_q4_facts" --facts "" \
  --pairs "$PWD/data/v_merged_train.jsonl" --epochs 2 --r 32 --lr 1e-4 --loss ce
eval_one v_merged_v
eval_one v_merged_v_rag
judge_some v_merged_v v_merged_v_rag
refresh

# --- pick winner ---
if [ ! -f data/winner.json ]; then
  stage="pick winner"
  write_status "$stage" "running" "combo contrast"
  $P pick_winner.py > pick_winner.log 2>&1
  rc=$?
  echo "$(date) pick winner exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
else
  echo "$(date) pick winner skip"
fi
refresh

MODE=$($P -c 'import json; print(json.load(open("data/winner.json"))["mode"])')
WINNER=$($P -c 'import json; print(json.load(open("data/winner.json"))["winner"])')
echo "$(date) winner=$WINNER mode=$MODE"

# --- (c) contrast ---
if [ ! -f data/contrast_report.json ]; then
  if [ -f data/contrast_fresh.done ]; then
    echo "$(date) contrast fresh skip"
  else
    unload
    sleep 2
    stage="combo contrast fresh"
    write_status "$stage" "running" "combo contrast finish"
    $P build_contrast.py fresh --out-dir data > contrast_fresh.log 2>&1
    rc=$?
    echo "$(date) contrast fresh exit $rc"
    if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  fi
  stage="combo contrast finish"
  write_status "$stage" "running" "train combo contrast"
  $P build_contrast.py finish --out-dir data > contrast_finish.log 2>&1
  rc=$?
  echo "$(date) contrast finish exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
else
  echo "$(date) contrast skip"
fi

if [ "$MODE" = "fv" ]; then
  train6 v_combo_contrast --stage voice --facts "$HOME/output-model/exp4/lora_q4_facts" \
    --pairs "$PWD/data/v_contrast_train.jsonl" --epochs 2 --r 32 --lr 1e-4 --loss unlikelihood --ul-weight 1.0
else
  train6 v_combo_contrast --stage voice --base "$PWD/merged_q4_facts" --facts "" \
    --pairs "$PWD/data/v_contrast_train.jsonl" --epochs 2 --r 32 --lr 1e-4 --loss unlikelihood --ul-weight 1.0
fi
eval_one "v_combo_contrast_${MODE}"
eval_one "v_combo_contrast_${MODE}_rag"
judge_some "v_combo_contrast_${MODE}" "v_combo_contrast_${MODE}_rag"
refresh

# --- (c) DPO ---
if [ ! -f lora_v_combo_dpo/train_report.json ] && [ ! -f data/dpo_skipped.json ]; then
  stage="dpo pairs"
  write_status "$stage" "running" "dpo train or skip"
  $P make_dpo_pairs.py > dpo_pairs.log 2>&1
  rc=$?
  echo "$(date) dpo pairs exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  $P pick_dpo.py > pick_dpo.log 2>&1
  rc=$?
  echo "$(date) pick dpo exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  skip=$($P -c 'import json; print("yes" if json.load(open("data/dpo_choice.json")).get("skip") else "no")')
  echo "$(date) dpo skip=$skip"
  if [ "$skip" = "no" ]; then
    init=$($P -c 'import json; print(json.load(open("data/dpo_choice.json"))["adapter"])')
    if [ "$MODE" = "fv" ]; then
      train6 v_combo_dpo --stage voice --facts "$HOME/output-model/exp4/lora_q4_facts" \
        --init "$init" --pairs "$PWD/data/v_dpo_pairs.jsonl" --epochs 1 --lr 5e-6 --beta 0.1 --loss dpo --r 32
    else
      train6 v_combo_dpo --stage voice --base "$PWD/merged_q4_facts" --facts "" \
        --init "$init" --pairs "$PWD/data/v_dpo_pairs.jsonl" --epochs 1 --lr 5e-6 --beta 0.1 --loss dpo --r 32
    fi
    eval_one "v_combo_dpo_${MODE}"
    eval_one "v_combo_dpo_${MODE}_rag"
    judge_some "v_combo_dpo_${MODE}" "v_combo_dpo_${MODE}_rag"
  fi
else
  echo "$(date) dpo skip (report or skipped file exists)"
fi
refresh

# --- (d) optional 8B ---
if [ ! -f data/8b_probe.json ]; then
  unload
  sleep 2
  stage="8B probe"
  write_status "$stage" "running" "8B train or skip"
  $P probe_8b.py > probe_8b.log 2>&1
  rc=$?
  echo "$(date) 8B probe exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
else
  echo "$(date) 8B probe skip"
fi

eight_ok=$($P -c 'import json,os; p="data/8b_probe.json"; d=json.load(open(p)) if os.path.exists(p) else {"ok":False}; print("yes" if d.get("ok") else "no")')
eight_q4=$($P -c 'import json,os; p="data/8b_probe.json"; d=json.load(open(p)) if os.path.exists(p) else {}; print("yes" if d.get("q4") else "no")')
echo "$(date) 8B ok=$eight_ok q4=$eight_q4"

skip8() {
  local why="$1"
  echo "$(date) 8B skip: $why"
  $P -c 'import json,sys; json.dump({"ok":False,"reason":sys.argv[1]}, open("data/8b_skipped.json","w"), indent=1)' "$why"
}

if [ "$eight_ok" != "yes" ]; then
  skip8 "probe said no"
elif [ "$eight_q4" = "yes" ] && [ "$MODE" = "v" ]; then
  skip8 "4-bit 8B cannot merge facts into base; winner was (b)"
else
  Q4FLAG=""
  if [ "$eight_q4" = "yes" ]; then Q4FLAG="--q4"; fi
  if ! try_train6 q8_facts --stage facts --base Qwen/Qwen3-8B $Q4FLAG \
      --pairs "$HOME/output-model/exp4/data/facts.jsonl" \
      --claims "$HOME/output-model/exp4/data/claims.jsonl" \
      --epochs 2 --r 16 --lr 1e-4 --loss ce; then
    skip8 "q8_facts train failed (see train_lora_q8_facts.log)"
  else
    eight_fail=0
    if [ "$WINNER" = "v_replay" ]; then
      if ! try_train6 v8_replay --stage voice --base Qwen/Qwen3-8B --facts "$PWD/lora_q8_facts" $Q4FLAG \
          --pairs "$PWD/data/v_replay_train.jsonl" --epochs 2 --r 32 --lr 1e-4 --loss ce; then
        skip8 "v8_replay train failed"; eight_fail=1
      else
        eval_one v8_replay_fv soft || eight_fail=1
        eval_one v8_replay_fv_rag soft || eight_fail=1
        judge_some v8_replay_fv v8_replay_fv_rag
      fi
    else
      if [ ! -f merged_q8_facts/merge_report.json ]; then
        unload
        $P merge_facts.py --base Qwen/Qwen3-8B --facts "$PWD/lora_q8_facts" --out "$PWD/merged_q8_facts" > merge_facts8.log 2>&1
        rc=$?
        echo "$(date) merge 8B facts exit $rc"
        if [ "$rc" -ne 0 ]; then skip8 "8B merge failed"; eight_fail=1; fi
      fi
      if [ "$eight_fail" -eq 0 ]; then
        if ! try_train6 v8_merged --stage voice --base "$PWD/merged_q8_facts" --facts "" $Q4FLAG \
            --pairs "$PWD/data/v_merged_train.jsonl" --epochs 2 --r 32 --lr 1e-4 --loss ce; then
          skip8 "v8_merged train failed"; eight_fail=1
        else
          eval_one v8_merged_v soft || eight_fail=1
          eval_one v8_merged_v_rag soft || eight_fail=1
          judge_some v8_merged_v v8_merged_v_rag
        fi
      fi
    fi
    if [ "$eight_fail" -eq 0 ] && [ -f data/v_contrast_train.jsonl ]; then
      if [ "$MODE" = "fv" ]; then
        if ! try_train6 v8_combo_contrast --stage voice --base Qwen/Qwen3-8B --facts "$PWD/lora_q8_facts" $Q4FLAG \
            --pairs "$PWD/data/v_contrast_train.jsonl" --epochs 2 --r 32 --lr 1e-4 --loss unlikelihood --ul-weight 1.0; then
          skip8 "v8_combo_contrast train failed"; eight_fail=1
        fi
      else
        if ! try_train6 v8_combo_contrast --stage voice --base "$PWD/merged_q8_facts" --facts "" $Q4FLAG \
            --pairs "$PWD/data/v_contrast_train.jsonl" --epochs 2 --r 32 --lr 1e-4 --loss unlikelihood --ul-weight 1.0; then
          skip8 "v8_combo_contrast train failed"; eight_fail=1
        fi
      fi
      if [ "$eight_fail" -eq 0 ]; then
        eval_one "v8_combo_contrast_${MODE}" soft || eight_fail=1
        eval_one "v8_combo_contrast_${MODE}_rag" soft || eight_fail=1
        judge_some "v8_combo_contrast_${MODE}" "v8_combo_contrast_${MODE}_rag"
      fi
    fi
    if [ "$eight_fail" -eq 0 ] && [ -f data/v_dpo_pairs.jsonl ] && [ -f lora_v8_combo_contrast/train_report.json ]; then
      if [ "$MODE" = "fv" ]; then
        if ! try_train6 v8_combo_dpo --stage voice --base Qwen/Qwen3-8B --facts "$PWD/lora_q8_facts" $Q4FLAG \
            --init "$PWD/lora_v8_combo_contrast" --pairs "$PWD/data/v_dpo_pairs.jsonl" \
            --epochs 1 --lr 5e-6 --beta 0.1 --loss dpo --r 32; then
          skip8 "v8_combo_dpo train failed"; eight_fail=1
        fi
      else
        if ! try_train6 v8_combo_dpo --stage voice --base "$PWD/merged_q8_facts" --facts "" $Q4FLAG \
            --init "$PWD/lora_v8_combo_contrast" --pairs "$PWD/data/v_dpo_pairs.jsonl" \
            --epochs 1 --lr 5e-6 --beta 0.1 --loss dpo --r 32; then
          skip8 "v8_combo_dpo train failed"; eight_fail=1
        fi
      fi
      if [ "$eight_fail" -eq 0 ]; then
        eval_one "v8_combo_dpo_${MODE}" soft || eight_fail=1
        eval_one "v8_combo_dpo_${MODE}_rag" soft || eight_fail=1
        judge_some "v8_combo_dpo_${MODE}" "v8_combo_dpo_${MODE}_rag"
      fi
    fi
  fi
fi
refresh
facts_ok
write_status "done" "done" "none"
done_ok=1
echo "$(date) ALL DONE"
