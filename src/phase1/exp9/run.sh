#!/bin/bash
# exp9 driver. Resumable: skip a stage when its run json, judged json, or train report exists.
# Does not modify exp3–exp8, does not push, does not pip-install, does not reboot.
cd "$(dirname "$0")"
P="$HOME/output-model/exp3/.venv/bin/python"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p runs data smoke init pass1
stage=start
done_ok=0

flag() { if [ -f "$1" ]; then echo "- $2: done"; else echo "- $2: pending"; fi; }
write_status() {
  local cur="$1" st="$2" nxt="$3"
  echo "$(date)	$cur	$st	$nxt" >> status.tsv
  {
    echo "# exp9 build status"
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
    flag init/lora_v_fb_dpo/adapter_model.safetensors "init lora_v_fb_dpo"
    flag init/lora_v_light_contrast/adapter_model.safetensors "init lora_v_light_contrast"
    flag init/lora_v_light_dpo/adapter_model.safetensors "init lora_v_light_dpo"
    flag init/lora_v_merged/adapter_model.safetensors "init lora_v_merged"
    flag pass1/v_merged_v_rag.json "pass1 v_merged_v_rag"
    flag smoke/ok.json "smoke"
    flag runs/p2_fb_dpo.judged.json "p2_fb_dpo judge"
    flag runs/p2_light_contrast.judged.json "p2_light_contrast judge"
    flag runs/p2_light_dpo.judged.json "p2_light_dpo judge"
    flag runs/p2_v_merged.judged.json "p2_v_merged judge"
    flag runs/p2_fb_dpo.factdiff.json "p2_fb_dpo fact-diff"
    flag runs/p2_light_contrast.factdiff.json "p2_light_contrast fact-diff"
    flag runs/p2_light_dpo.factdiff.json "p2_light_dpo fact-diff"
    flag runs/p2_v_merged.factdiff.json "p2_v_merged fact-diff"
    flag runs/p2_fb_dpo_guarded.judged.json "p2_fb_dpo guarded judge"
    flag runs/p2_light_contrast_guarded.judged.json "p2_light_contrast guarded judge"
    flag runs/p2_light_dpo_guarded.judged.json "p2_light_dpo guarded judge"
    flag runs/p2_v_merged_guarded.judged.json "p2_v_merged guarded judge"
    flag data/decide.json "stage decisions"
    flag data/rewrite_meta.json "rewrite pairs (only if c trains)"
    flag lora_v_rewrite/train_report.json "rewrite LoRA (only if c trains)"
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
  echo "$(date) STAGE_FAIL $stage exit $1"
  write_status "$stage" "failed" "re-run bash run.sh"
  exit "$1"
}
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
copy_adapter() {
  local src="$1"
  local name="$2"
  local dest="init/$name"
  if [ ! -f "$src/adapter_model.safetensors" ]; then
    echo "missing adapter $src"
    return 1
  fi
  if [ -f "$dest/adapter_model.safetensors" ] && cmp -s "$src/adapter_model.safetensors" "$dest/adapter_model.safetensors"; then
    return 0
  fi
  rm -rf "$dest"
  mkdir -p "$dest"
  cp -a "$src/adapter_config.json" "$src/adapter_model.safetensors" "$dest/"
  [ -f "$src/README.md" ] && cp -a "$src/README.md" "$dest/" || true
  [ -f "$src/train_report.json" ] && cp -a "$src/train_report.json" "$dest/" || true
}
ensure_inputs() {
  local f src
  for f in eval4.py judge.py common.py chat.py; do
    src="$HOME/output-model/exp4/$f"
    if [ ! -f "$f" ] || ! cmp -s "$src" "$f"; then
      cp -a "$src" "$f"
    fi
  done
  mkdir -p data/chapters runs init pass1 smoke
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
  if [ ! -f runs/v_rag_dpo_v_rag.judged.json ]; then cp -a "$HOME/output-model/exp8/runs/v_rag_dpo_v_rag.judged.json" runs/; fi
  if [ ! -f runs/q4_base_rag.judged.json ]; then cp -a "$HOME/output-model/exp8/runs/q4_base_rag.judged.json" runs/; fi
  if [ ! -f pass1/v_merged_v_rag.json ] || ! cmp -s "$HOME/output-model/exp6/runs/v_merged_v_rag.json" pass1/v_merged_v_rag.json; then
    cp -a "$HOME/output-model/exp6/runs/v_merged_v_rag.json" pass1/v_merged_v_rag.json
  fi
  if [ ! -f pass1/q4_merged_rag.json ] || ! cmp -s "$HOME/output-model/exp8/runs/q4_merged_rag.json" pass1/q4_merged_rag.json; then
    cp -a "$HOME/output-model/exp8/runs/q4_merged_rag.json" pass1/q4_merged_rag.json
  fi
  copy_adapter "$HOME/output-model/exp8/lora_v_fb_dpo" lora_v_fb_dpo || return 1
  copy_adapter "$HOME/output-model/exp7/lora_v_light_contrast" lora_v_light_contrast || return 1
  copy_adapter "$HOME/output-model/exp7/lora_v_light_dpo" lora_v_light_dpo || return 1
  copy_adapter "$HOME/output-model/exp6/lora_v_merged" lora_v_merged || return 1
}
refresh() {
  stage=summarize
  $P summarize9.py > summarize.log 2>&1
  rc=$?
  echo "$(date) summarize exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  $P notes9.py > notes.log 2>&1
  rc=$?
  echo "$(date) notes exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
}
count_ok() {
  $P -c 'import json,sys; d=json.load(open(sys.argv[1])); n=len(d["results"]); assert n==204, n; assert d["meta"].get("retrieval_in_rewrite") is False; assert all(r["answer"]==r["pass2"] for r in d["results"])' "$1"
}
eval_one() {
  local cond="$1"
  if [ -f "runs/${cond}.json" ]; then
    if count_ok "runs/${cond}.json"; then
      echo "$(date) eval $cond skip"
      return 0
    fi
    echo "$(date) eval $cond bad result; removing"
    rm -f "runs/${cond}.json" "runs/${cond}.judged.json"
  fi
  unload
  sleep 5
  stage="eval $cond"
  write_status "$stage" "running" "judge $cond"
  $P eval_twopass.py "$cond" --n 3 > "runs/${cond}.log" 2>&1
  rc=$?
  echo "$(date) eval $cond exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  if ! count_ok "runs/${cond}.json"; then
    echo "$(date) eval $cond bad result count"
    rm -f "runs/${cond}.json"
    fail 2
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
  stage="judge $*"
  write_status "$stage" "running" "fact-diff or next"
  echo "$(date) judge start ${files[*]}"
  (
    while true; do
      sleep 60
      echo "$(date) judge heartbeat ${files[*]}"
    done
  ) &
  local hb=$!
  $P judge.py "${files[@]}" >> runs/judge.log 2>&1
  rc=$?
  kill "$hb" 2>/dev/null || true
  wait "$hb" 2>/dev/null || true
  echo "$(date) judge $* exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  for c in "$@"; do
    if [ -f "runs/$c.json" ] && [ ! -f "runs/$c.judged.json" ]; then
      echo "$(date) judge missing runs/$c.judged.json"
      fail 2
    fi
  done
}
fact_one() {
  local cond="$1"
  stage="factdiff $cond"
  write_status "$stage" "running" "guard $cond"
  echo "$(date) factdiff start $cond"
  $P factdiff.py "runs/${cond}.json"
  rc=$?
  echo "$(date) factdiff $cond exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
}
guard_one() {
  local cond="$1"
  stage="guard $cond"
  if [ -f "runs/${cond}_guarded.json" ]; then
    echo "$(date) guard $cond skip"
    return 0
  fi
  $P guard.py "$cond"
  rc=$?
  echo "$(date) guard $cond exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
}
smoke_hash() {
  $P -c 'import hashlib, pathlib
h=hashlib.sha256()
for n in ["eval_twopass.py","factdiff.py","guard.py","prompts.py","train9.py","build_rewrite.py"]:
    h.update(pathlib.Path(n).read_bytes())
print(h.hexdigest())'
}
smoke_current() {
  [ -f smoke/ok.json ] || return 1
  local want have
  want="$(smoke_hash)"
  have="$($P -c 'import json; print(json.load(open("smoke/ok.json")).get("scripts",""))')"
  [ "$want" = "$have" ]
}
do_smoke() {
  stage=smoke
  echo "$(date) smoke start"
  $P factdiff.py --self-test || return $?
  $P guard.py --self-test || return $?
  $P commit9.py --self-test || return $?
  $P eval_twopass.py p2_fb_dpo --limit 20 --include-must 1 --dry-prompt > smoke/dry.log 2>&1
  rc=$?
  echo "$(date) dry-prompt exit $rc"
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  if ! grep -q DRY_OK smoke/dry.log; then
    echo "dry-prompt missing DRY_OK"
    return 2
  fi
  if ! grep -q "RETRIEVAL False" smoke/dry.log; then
    echo "dry-prompt still shows retrieval"
    return 2
  fi
  local need=1
  if [ -f smoke/p2_fb_dpo.json ]; then
    need="$($P -c 'import json,hashlib,pathlib
want=hashlib.sha256(pathlib.Path("eval_twopass.py").read_bytes()).hexdigest()
d=json.load(open("smoke/p2_fb_dpo.json"))
ok=d["meta"].get("eval_sha")==want and len(d["results"])==63 and any(r["kind"]=="must" for r in d["results"]) and d["meta"].get("retrieval_in_rewrite") is False
print("0" if ok else "1")')"
  fi
  if [ "$need" != "0" ]; then
    unload
    sleep 5
    $P eval_twopass.py p2_fb_dpo --limit 20 --include-must 1 --n 3 > smoke/eval.log 2>&1
    rc=$?
    echo "$(date) smoke eval exit $rc"
    if [ "$rc" -ne 0 ]; then return "$rc"; fi
  else
    echo "$(date) smoke eval skip"
  fi
  $P -c 'import json
d=json.load(open("smoke/p2_fb_dpo.json"))
assert len(d["results"])==63, len(d["results"])
assert any(r["kind"]=="must" for r in d["results"])
assert all(r["answer"]==r["pass2"] and isinstance(r["pass1"], str) for r in d["results"])
assert d["meta"]["rag"] is False and d["meta"]["retrieval_in_rewrite"] is False
print("smoke schema ok", len(d["results"]))'
  rc=$?
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  $P factdiff.py smoke/p2_fb_dpo.json > smoke/factdiff.log 2>&1
  rc=$?
  echo "$(date) smoke factdiff exit $rc"
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  $P -c 'import json
d=json.load(open("smoke/p2_fb_dpo.json"))
rows=[r for r in d["results"] if r["kind"]=="leak"][:1] + [r for r in d["results"] if r["kind"]=="must"][:1]
json.dump({"meta": d["meta"], "results": rows}, open("smoke/judge_slice.json","w"), indent=1)
print("slice", len(rows))'
  if [ ! -f smoke/judge_slice.judged.json ]; then
    $P judge.py smoke/judge_slice.json > smoke/judge.log 2>&1
    rc=$?
    echo "$(date) smoke judge exit $rc"
    if [ "$rc" -ne 0 ]; then return "$rc"; fi
  fi
  $P -c 'import json; d=json.load(open("smoke/judge_slice.judged.json")); assert all("voice_score" in r for r in d["results"]); print("smoke judge ok", [r["voice_score"] for r in d["results"]])'
  rc=$?
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  $P guard.py --run smoke/p2_fb_dpo.json --out smoke/p2_fb_dpo_guarded.json --force > smoke/guard.log 2>&1
  rc=$?
  echo "$(date) smoke guard exit $rc"
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  $P build_rewrite.py --limit 8 --out smoke/rewrite_pairs.jsonl --meta smoke/rewrite_meta.json --force > smoke/build.log 2>&1
  rc=$?
  echo "$(date) smoke build exit $rc"
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  if [ ! -f smoke/lora_v_rewrite/train_report.json ]; then
    unload
    sleep 5
    $P train9.py --pairs smoke/rewrite_pairs.jsonl --out "$PWD/smoke/lora_v_rewrite" --init "$PWD/init/lora_v_fb_dpo" \
      --epochs 1 --lr 2e-5 --r 32 --dropout 0.05 --max-steps 2 --save-every 0 --limit 8 > smoke/train.log 2>&1
    rc=$?
    echo "$(date) smoke train exit $rc"
    if [ "$rc" -ne 0 ]; then return "$rc"; fi
  else
    echo "$(date) smoke train skip"
  fi
  $P -c 'import json; d=json.load(open("smoke/lora_v_rewrite/train_report.json")); assert d["steps"]>=1, d; assert d["loss"]=="ce"; print("smoke train ok", d["steps"], d["loss"])'
  rc=$?
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  $P -c 'import json,hashlib,pathlib,datetime
h=hashlib.sha256()
for n in ["eval_twopass.py","factdiff.py","guard.py","prompts.py","train9.py","build_rewrite.py"]:
    h.update(pathlib.Path(n).read_bytes())
json.dump({"scripts": h.hexdigest(), "limit": 20, "include_must": 1, "rows": 63, "when": datetime.datetime.now().isoformat(timespec="seconds")}, open("smoke/ok.json","w"), indent=1)
print("smoke ok")'
}

cmd="${1:-run}"
if [ "$cmd" = "ensure" ]; then
  ensure_inputs || exit 1
  $P -c 'import frozen; frozen.check()'
  exit $?
fi
if [ "$cmd" = "smoke" ]; then
  ensure_inputs || exit 1
  $P -c 'import frozen; frozen.check()' || exit 1
  if smoke_current; then
    echo "$(date) smoke already current"
    exit 0
  fi
  do_smoke
  exit $?
fi
if [ "$cmd" != "run" ]; then
  echo "usage: bash run.sh [ensure|smoke|run]"
  exit 2
fi

echo $$ > run.pid
exec 9>run.lock
if ! flock -n 9; then
  echo "$(date) another exp9 run holds run.lock"
  exit 1
fi
trap 'if [ "$done_ok" != 1 ]; then write_status "$stage" "failed" "re-run bash run.sh"; fi' EXIT

echo "$(date) exp9 start"
ensure_inputs || fail 1
facts_ok
write_status "start" "running" "smoke"
if smoke_current; then
  echo "$(date) smoke skip (smoke/ok.json matches these scripts)"
else
  do_smoke
  rc=$?
  echo "$(date) smoke exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
fi
facts_ok
refresh

# (a) pass 2 on the four voice adapters, primary pass-1.
PRIMARY=(p2_fb_dpo p2_light_contrast p2_light_dpo p2_v_merged)
for c in "${PRIMARY[@]}"; do
  eval_one "$c"
done
stage="decide-second"
$P decide.py second
rc=$?
echo "$(date) decide second exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
mapfile -t EXTRA < <($P -c 'import json; print("\n".join(json.load(open("data/decide.json"))["second_source"]["conds"]))')
if [ "${#EXTRA[@]}" -eq 0 ]; then
  echo "$(date) second pass1 skip ($($P -c 'import json; print(json.load(open("data/decide.json"))["second_source"]["reason"])'))"
else
  for c in "${EXTRA[@]}"; do
    [ -n "$c" ] || continue
    eval_one "$c"
  done
fi
judge_some "${PRIMARY[@]}" "${EXTRA[@]}"
refresh

# (b) fact-diff and guarded fallback for every unguarded two-pass run so far.
mapfile -t UNGUARDED < <($P -c '
import glob, os
out=[]
for p in sorted(glob.glob("runs/p2_*.json")):
    b=os.path.basename(p)
    if b.endswith(".judged.json") or ".factdiff" in b or b.endswith("_guarded.json"):
        continue
    out.append(b[:-5])
print("\n".join(out))
')
if [ "${#UNGUARDED[@]}" -eq 0 ]; then
  echo "$(date) no unguarded two-pass runs to fact-diff"
  fail 2
fi
for c in "${UNGUARDED[@]}"; do
  [ -n "$c" ] || continue
  fact_one "$c"
  guard_one "$c"
done
GUARDED=()
for c in "${UNGUARDED[@]}"; do
  [ -n "$c" ] || continue
  GUARDED+=("${c}_guarded")
done
judge_some "${GUARDED[@]}"
refresh

# (c) rewrite LoRA only if a scored row is close to the bar.
stage="decide-train"
$P decide.py train
rc=$?
echo "$(date) decide train exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
TRAIN="$($P -c 'import json; print("yes" if json.load(open("data/decide.json"))["stage_c"]["train"] else "no")')"
echo "$(date) stage c train=$TRAIN ($($P -c 'import json; print(json.load(open("data/decide.json"))["stage_c"]["reason"])'))"
if [ -f lora_v_rewrite/train_report.json ]; then
  echo "$(date) train lora_v_rewrite skip"
elif [ "$TRAIN" != "yes" ]; then
  echo "$(date) train lora_v_rewrite skipped: not close"
  write_status "stage-c" "skipped" "sweep check"
else
  stage="rewrite-data"
  write_status "$stage" "running" "train lora_v_rewrite"
  $P build_rewrite.py > data/build_rewrite.log 2>&1
  rc=$?
  echo "$(date) rewrite data exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  SAVE="$($P -c 'import json; n=json.load(open("data/rewrite_meta.json"))["rows"]; print(max(40, n//3))')"
  unload
  sleep 5
  stage="train-rewrite"
  write_status "$stage" "running" "eval p2_rewrite"
  rm -rf lora_v_rewrite
  (
    while true; do
      sleep 60
      echo "$(date) train heartbeat lora_v_rewrite"
    done
  ) &
  hb=$!
  $P train9.py --pairs "$PWD/data/rewrite_pairs.jsonl" --out "$PWD/lora_v_rewrite" --init "$PWD/init/lora_v_fb_dpo" \
    --epochs 1 --lr 2e-5 --r 32 --dropout 0.05 --max-steps 0 --save-every "$SAVE" > train_lora_v_rewrite.log 2>&1
  rc=$?
  kill "$hb" 2>/dev/null || true
  wait "$hb" 2>/dev/null || true
  echo "$(date) train lora_v_rewrite exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  if [ ! -f lora_v_rewrite/train_report.json ]; then fail 3; fi
  facts_ok
fi
if [ -f lora_v_rewrite/train_report.json ]; then
  eval_one p2_rewrite
  judge_some p2_rewrite
  fact_one p2_rewrite
  guard_one p2_rewrite
  judge_some p2_rewrite_guarded
  refresh
fi

# (d) checkpoint sweep only if a row is otherwise passing and within 0.05 voice.
stage=sweep
write_status "$stage" "running" "checkpoint eval or skip"
$P decide.py sweep
rc=$?
echo "$(date) decide sweep exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
mapfile -t SWEEP < <($P -c 'import json; print("\n".join(json.load(open("data/decide.json"))["stage_d"]["conds"]))')
if [ "${#SWEEP[@]}" -eq 0 ]; then
  echo "$(date) sweep skip ($($P -c 'import json; print(json.load(open("data/decide.json"))["stage_d"]["reason"])'))"
else
  for c in "${SWEEP[@]}"; do
    [ -n "$c" ] || continue
    eval_one "$c"
  done
  judge_some "${SWEEP[@]}"
  for c in "${SWEEP[@]}"; do
    [ -n "$c" ] || continue
    fact_one "$c"
    guard_one "$c"
  done
  SG=()
  for c in "${SWEEP[@]}"; do
    [ -n "$c" ] || continue
    SG+=("${c}_guarded")
  done
  judge_some "${SG[@]}"
fi
refresh

stage=reproduce
$P reproduce9.py > reproduce.log 2>&1
rc=$?
echo "$(date) reproduce exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
stage=commit
$P commit9.py commit > commit.log 2>&1
rc=$?
echo "$(date) commit exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
facts_ok
write_status "done" "done" "none"
done_ok=1
echo "$(date) ALL DONE"
