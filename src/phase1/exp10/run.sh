#!/bin/bash
# exp10 driver. Resumable: skip a stage when its output already exists.
# Does not modify exp3–exp9, does not push, does not pip-install, does not reboot.
cd "$(dirname "$0")"
P="$HOME/output-model/exp3/.venv/bin/python"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p runs data/chapters smoke init/lora_v_light_dpo pass1 calib/src calib/qwen3_14b calib/gemma3_12b human_rating locked
stage=start
done_ok=0

flag() { if [ -f "$1" ]; then echo "- $2: done"; else echo "- $2: pending"; fi; }
write_status() {
  local cur="$1" st="$2" nxt="$3"
  echo "$(date)	$cur	$st	$nxt" >> status.tsv
  {
    echo "# exp10 build status"
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
    flag init/lora_v_light_dpo/adapter_model.safetensors "init lora_v_light_dpo"
    flag pass1/v_merged_v_rag.json "pass1 v_merged_v_rag"
    flag locked/p2_light_dpo_guarded.judged.json "locked exp9 judged copy"
    flag smoke/ok.json "smoke"
    flag runs/p2_light_dpo.json "regen pass-2 answers"
    flag runs/p2_light_dpo.factdiff.json "regen fact-diff"
    flag runs/p2_light_dpo_guarded.json "regen guarded"
    flag runs/p2_light_dpo_guarded.judged.json "regen judge"
    flag runs/p2_light_dpo_guarded_orig.judged.json "orig-text re-judge"
    flag runs/tolerance.json "tolerance check"
    flag human_rating/sample.md "human-rating sample"
    flag calib/report.json "calib report"
    flag results.md "results.md"
    flag REPRODUCE.md "REPRODUCE.md"
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
  $P frozen.py
  rc=$?
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
}
copy_adapter() {
  local src="$1"
  local dest="init/lora_v_light_dpo"
  if [ ! -f "$src/adapter_model.safetensors" ]; then
    echo "missing adapter $src"
    return 1
  fi
  if [ -f "$dest/adapter_model.safetensors" ] && cmp -s "$src/adapter_model.safetensors" "$dest/adapter_model.safetensors"; then
    return 0
  fi
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
  mkdir -p data/chapters runs init/lora_v_light_dpo pass1 smoke calib/src locked human_rating
  ln -sfn $HOME/output-model/exp3/data/chapters/ch1.md data/chapters/ch1.md
  ln -sfn $HOME/output-model/exp3/data/chapters/ch2.md data/chapters/ch2.md
  ln -sfn $HOME/output-model/exp3/data/chapters/ch3.md data/chapters/ch3.md
  ln -sfn $HOME/output-model/exp3/data/chapters/ch4.md data/chapters/ch4.md
  ln -sfn $HOME/output-model/exp3/data/chapters/ch5.md data/chapters/ch5.md
  ln -sfn $HOME/output-model/exp3/data/knowledge-state.yaml data/knowledge-state.yaml
  ln -sfn $HOME/output-model/exp3/data/leak-tests.md data/leak-tests.md
  if [ ! -f pass1/v_merged_v_rag.json ] || ! cmp -s "$HOME/output-model/exp9/pass1/v_merged_v_rag.json" pass1/v_merged_v_rag.json; then
    cp -a "$HOME/output-model/exp9/pass1/v_merged_v_rag.json" pass1/v_merged_v_rag.json
  fi
  copy_adapter "$HOME/output-model/exp7/lora_v_light_dpo" || return 1
  if [ ! -f locked/p2_light_dpo_guarded.judged.json ]; then
    cp -a "$HOME/output-model/exp9/runs/p2_light_dpo.json" locked/
    cp -a "$HOME/output-model/exp9/runs/p2_light_dpo.factdiff.json" locked/
    cp -a "$HOME/output-model/exp9/runs/p2_light_dpo_guarded.json" locked/
    cp -a "$HOME/output-model/exp9/runs/p2_light_dpo_guarded.judged.json" locked/
    cp -a "$HOME/output-model/exp9/pass1/v_merged_v_rag.json" locked/v_merged_v_rag.json
  fi
  if [ ! -f runs/p2_light_dpo_guarded_orig.json ]; then
    cp -a "$HOME/output-model/exp9/runs/p2_light_dpo_guarded.json" runs/p2_light_dpo_guarded_orig.json
  fi
  local pair
  for pair in \
    "p2_light_dpo_guarded:$HOME/output-model/exp9/runs/p2_light_dpo_guarded.json" \
    "p2_light_dpo:$HOME/output-model/exp9/runs/p2_light_dpo.json" \
    "v_merged_v_rag:$HOME/output-model/exp9/pass1/v_merged_v_rag.json" \
    "q4_base_rag:$HOME/output-model/exp8/runs/q4_base_rag.json" \
    "v_fb_dpo_v:$HOME/output-model/exp8/runs/v_fb_dpo_v.json"
  do
    local name="${pair%%:*}"
    local src="${pair#*:}"
    if [ ! -f "calib/src/${name}.json" ]; then
      cp -a "$src" "calib/src/${name}.json"
    fi
  done
  [ -f calib/src/p2_light_dpo_guarded.judged.json ] || cp -a "$HOME/output-model/exp9/runs/p2_light_dpo_guarded.judged.json" calib/src/
  [ -f calib/src/p2_light_dpo.judged.json ] || cp -a "$HOME/output-model/exp9/runs/p2_light_dpo.judged.json" calib/src/
  [ -f calib/src/v_merged_v_rag.judged.json ] || cp -a "$HOME/output-model/exp6/runs/v_merged_v_rag.judged.json" calib/src/
  [ -f calib/src/q4_base_rag.judged.json ] || cp -a "$HOME/output-model/exp8/runs/q4_base_rag.judged.json" calib/src/
  [ -f calib/src/v_fb_dpo_v.judged.json ] || cp -a "$HOME/output-model/exp8/runs/v_fb_dpo_v.judged.json" calib/src/
}
refresh_docs() {
  stage=summarize
  $P summarize10.py
  rc=$?
  echo "$(date) summarize exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  $P reproduce10.py
  rc=$?
  echo "$(date) reproduce exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  $P notes10.py
  rc=$?
  echo "$(date) notes exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
}
count_ok() {
  $P -c 'import json,sys,hashlib,pathlib
d=json.load(open(sys.argv[1])); n=len(d["results"]); want=hashlib.sha256(pathlib.Path("eval_twopass.py").read_bytes()).hexdigest()
assert n==204, n
assert d["meta"].get("retrieval_in_rewrite") is False
assert d["meta"].get("generation_seed")==1
assert d["meta"].get("eval_sha")==want, (d["meta"].get("eval_sha"), want)
assert all(r["answer"]==r["pass2"] for r in d["results"])' "$1"
}
eval_regen() {
  if [ -f runs/p2_light_dpo.json ]; then
    if count_ok runs/p2_light_dpo.json; then
      echo "$(date) eval p2_light_dpo skip"
      return 0
    fi
    echo "$(date) eval p2_light_dpo bad result; removing"
    rm -f runs/p2_light_dpo.json runs/p2_light_dpo.judged.json runs/p2_light_dpo.factdiff.json runs/p2_light_dpo_guarded.json runs/p2_light_dpo_guarded.judged.json
  fi
  unload
  sleep 5
  stage="eval p2_light_dpo"
  write_status "$stage" "running" "factdiff p2_light_dpo"
  $P eval_twopass.py p2_light_dpo --n 3 --seed 1 > runs/p2_light_dpo.log 2>&1
  rc=$?
  echo "$(date) eval p2_light_dpo exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  if ! count_ok runs/p2_light_dpo.json; then
    echo "$(date) eval p2_light_dpo bad result count"
    rm -f runs/p2_light_dpo.json
    fail 2
  fi
  facts_ok
}
judge_files() {
  local files=()
  local f
  for f in "$@"; do
    if [ -f "$f" ] && [ ! -f "${f%.json}.judged.json" ]; then
      files+=("$f")
    fi
  done
  if [ "${#files[@]}" -eq 0 ]; then
    echo "$(date) judge $* skip"
    return 0
  fi
  stage="judge ${files[*]}"
  write_status "$stage" "running" "tolerance or next"
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
  echo "$(date) judge exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
  for f in "${files[@]}"; do
    if [ ! -f "${f%.json}.judged.json" ]; then
      echo "$(date) judge missing ${f%.json}.judged.json"
      fail 2
    fi
  done
}
smoke_hash() {
  $P -c 'import hashlib, pathlib
h=hashlib.sha256()
for n in ["eval_twopass.py","factdiff.py","guard.py","prompts.py","judge_wrap.py","frozen.py"]:
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
  $P frozen.py || return $?
  $P factdiff.py --self-test || return $?
  $P guard.py --self-test || return $?
  $P commit10.py --self-test || return $?
  $P judge_wrap.py --self-test || return $?
  $P calib.py self-test || return $?
  $P human_sample.py --self-test || return $?
  $P eval_twopass.py p2_light_dpo --limit 20 --include-must 1 --seed 1 --dry-prompt > smoke/dry.log 2>&1
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
  if ! grep -q "SEED 1" smoke/dry.log; then
    echo "dry-prompt missing SEED 1"
    return 2
  fi
  local need=1
  if [ -f smoke/p2_light_dpo.json ]; then
    need="$($P -c 'import json,hashlib,pathlib
want=hashlib.sha256(pathlib.Path("eval_twopass.py").read_bytes()).hexdigest()
d=json.load(open("smoke/p2_light_dpo.json"))
ok=d["meta"].get("eval_sha")==want and d["meta"].get("generation_seed")==1 and len(d["results"])==63 and any(r["kind"]=="must" for r in d["results"]) and d["meta"].get("retrieval_in_rewrite") is False
print("0" if ok else "1")')"
  fi
  if [ "$need" != "0" ]; then
    unload
    sleep 5
    $P eval_twopass.py p2_light_dpo --limit 20 --include-must 1 --n 3 --seed 1 > smoke/eval.log 2>&1
    rc=$?
    echo "$(date) smoke eval exit $rc"
    if [ "$rc" -ne 0 ]; then return "$rc"; fi
  else
    echo "$(date) smoke eval skip"
  fi
  $P -c 'import json
d=json.load(open("smoke/p2_light_dpo.json"))
assert len(d["results"])==63, len(d["results"])
assert any(r["kind"]=="must" for r in d["results"])
assert all(r["answer"]==r["pass2"] and isinstance(r["pass1"], str) for r in d["results"])
assert d["meta"]["rag"] is False and d["meta"]["retrieval_in_rewrite"] is False
assert d["meta"]["generation_seed"]==1
print("smoke schema ok", len(d["results"]))'
  rc=$?
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  $P factdiff.py smoke/p2_light_dpo.json > smoke/factdiff.log 2>&1
  rc=$?
  echo "$(date) smoke factdiff exit $rc"
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  $P -c 'import json
d=json.load(open("smoke/p2_light_dpo.json"))
rows=[r for r in d["results"] if r["kind"]=="leak"][:1] + [r for r in d["results"] if r["kind"]=="must"][:1]
json.dump({"meta": d["meta"], "results": rows}, open("smoke/judge_slice.json","w"), indent=1)
print("slice", len(rows))'
  if [ ! -f smoke/judge_slice.voice.json ]; then
    $P judge_wrap.py smoke/judge_slice.json --out smoke/judge_slice.voice.json --voice-only --model qwen3:14b --temp 0 --seed 1 > smoke/judge.log 2>&1
    rc=$?
    echo "$(date) smoke judge_wrap exit $rc"
    if [ "$rc" -ne 0 ]; then return "$rc"; fi
  fi
  $P -c 'import json; d=json.load(open("smoke/judge_slice.voice.json")); assert all("voice_score" in r for r in d["results"]); print("smoke judge ok", [r["voice_score"] for r in d["results"]])'
  rc=$?
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  $P guard.py --run smoke/p2_light_dpo.json --out smoke/p2_light_dpo_guarded.json --force > smoke/guard.log 2>&1
  rc=$?
  echo "$(date) smoke guard exit $rc"
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  $P -c 'import json,hashlib,pathlib,datetime
h=hashlib.sha256()
for n in ["eval_twopass.py","factdiff.py","guard.py","prompts.py","judge_wrap.py","frozen.py"]:
    h.update(pathlib.Path(n).read_bytes())
json.dump({"scripts": h.hexdigest(), "limit": 20, "include_must": 1, "rows": 63, "seed": 1, "when": datetime.datetime.now().isoformat(timespec="seconds")}, open("smoke/ok.json","w"), indent=1)
print("smoke ok")'
}

cmd="${1:-run}"
if [ "$cmd" = "ensure" ]; then
  ensure_inputs || exit 1
  $P frozen.py
  exit $?
fi
if [ "$cmd" = "smoke" ]; then
  ensure_inputs || exit 1
  $P frozen.py || exit 1
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
  echo "$(date) another exp10 run holds run.lock"
  exit 1
fi
trap 'if [ "$done_ok" != 1 ]; then write_status "$stage" "failed" "re-run bash run.sh"; fi' EXIT

echo "$(date) exp10 start"
ensure_inputs || fail 1
facts_ok
write_status "start" "running" "docs then smoke"
$P reproduce10.py
rc=$?
echo "$(date) reproduce draft exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
$P notes10.py
rc=$?
echo "$(date) notes draft exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
$P summarize10.py
rc=$?
echo "$(date) summarize draft exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
if [ ! -f human_rating/sample.md ] || [ ! -f human_rating/key.json ]; then
  stage="human-sample"
  write_status "$stage" "running" "smoke"
  $P human_sample.py
  rc=$?
  echo "$(date) human_sample exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
else
  echo "$(date) human_sample skip"
fi
if smoke_current; then
  echo "$(date) smoke skip (smoke/ok.json matches these scripts)"
else
  do_smoke
  rc=$?
  echo "$(date) smoke exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
fi
facts_ok

# A. reproduction run
eval_regen
stage="factdiff p2_light_dpo"
write_status "$stage" "running" "guard p2_light_dpo"
echo "$(date) factdiff start p2_light_dpo"
$P factdiff.py runs/p2_light_dpo.json
rc=$?
echo "$(date) factdiff p2_light_dpo exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
stage="guard p2_light_dpo"
if [ -f runs/p2_light_dpo_guarded.json ]; then
  echo "$(date) guard p2_light_dpo skip"
else
  $P guard.py p2_light_dpo
  rc=$?
  echo "$(date) guard p2_light_dpo exit $rc"
  if [ "$rc" -ne 0 ]; then fail "$rc"; fi
fi
judge_files runs/p2_light_dpo_guarded.json runs/p2_light_dpo_guarded_orig.json
stage=tolerance
write_status "$stage" "running" "commit+tag"
$P check_tolerance.py
rc=$?
echo "$(date) tolerance exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
refresh_docs

stage="commit-lock"
write_status "$stage" "running" "calib noise"
$P commit10.py lock > commit.lock.log 2>&1
rc=$?
echo "$(date) commit lock exit $rc"
cat commit.lock.log
if [ "$rc" -ne 0 ]; then fail "$rc"; fi

# B. judge calibration
stage="calib-noise"
write_status "$stage" "running" "calib second"
echo "$(date) calib noise start"
(
  while true; do
    sleep 60
    echo "$(date) calib noise heartbeat"
  done
) &
hb=$!
$P calib.py noise
rc=$?
kill "$hb" 2>/dev/null || true
wait "$hb" 2>/dev/null || true
echo "$(date) calib noise exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi

stage="calib-second"
write_status "$stage" "running" "summarize"
unload
sleep 2
echo "$(date) calib second start"
$P calib.py second
rc=$?
echo "$(date) calib second exit $rc"
if [ "$rc" -ne 0 ]; then fail "$rc"; fi

refresh_docs
stage="commit-followup"
$P commit10.py followup > commit.followup.log 2>&1
rc=$?
echo "$(date) commit followup exit $rc"
cat commit.followup.log
if [ "$rc" -ne 0 ]; then fail "$rc"; fi
facts_ok
write_status "done" "done" "none"
done_ok=1
echo "$(date) ALL DONE"
