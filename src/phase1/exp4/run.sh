#!/bin/bash
# exp4 driver (resumable: each stage skipped if its output exists).
# Capture rc before $(date): command substitution resets $?.
cd ~/output-model/exp4
P=~/output-model/exp3/.venv/bin/python
export PYTHONUNBUFFERED=1
mkdir -p runs data
echo $$ > run.pid
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
unload() {
  if ! command -v ollama >/dev/null 2>&1; then return 0; fi
  ollama ps 2>/dev/null | awk 'NR>1 {print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" >/dev/null 2>&1 || true
  done
  return 0
}
echo "$(date) exp4 start"

if [ ! -f data/clean.jsonl ] || [ ! -f data/clean_report.json ]; then
  $P clean.py > clean.log 2>&1
  rc=$?
  echo "$(date) clean exit $rc"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
else
  echo "$(date) clean skip"
fi

if [ "$(lines data/contrast.jsonl)" -lt 300 ]; then
  $P contrast.py --n 400 > contrast.log 2>&1
  rc=$?
  echo "$(date) contrast exit $rc rows $(lines data/contrast.jsonl)"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
else
  echo "$(date) contrast skip"
fi

$P split.py > split.log 2>&1
rc=$?
echo "$(date) split exit $rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

if [ ! -f kd_report.json ]; then
  $P kd.py > kd.log 2>&1
  rc=$?
  echo "$(date) kd exit $rc"
fi

unload
echo "$(date) ollama unloaded"

$P -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-4B")' > hfdl_qwen.log 2>&1
rc=$?
echo "$(date) qwen download exit $rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

train_stage() {
  local base="$1" stage="$2" out="$3"
  shift 3
  if [ -f "$out/train_report.json" ]; then
    echo "$(date) train $out skip"
    return 0
  fi
  rm -rf "$out"
  $P train4.py --base "$base" --stage "$stage" --out "$out" "$@" > "train_${out}.log" 2>&1
  rc=$?
  echo "$(date) train $out exit $rc"
  return "$rc"
}

train_stage Qwen/Qwen3-4B facts lora_q4_facts
rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
train_stage Qwen/Qwen3-4B voice lora_q4_voice --facts lora_q4_facts
rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

if [ ! -f lora_p4_voice/train_report.json ] && [ ! -f lora_s3_voice/train_report.json ]; then
  $P -c 'from huggingface_hub import snapshot_download; snapshot_download("microsoft/Phi-4-mini-instruct")' > hfdl_phi.log 2>&1
  rc=$?
  echo "$(date) phi download exit $rc"
  if [ "$rc" -ne 0 ]; then
    echo "phi download failed" > phi_download_error.txt
  fi
  if [ "$rc" -eq 0 ]; then
    train_stage microsoft/Phi-4-mini-instruct facts lora_p4_facts
    rc=$?
    if [ "$rc" -eq 0 ]; then
      train_stage microsoft/Phi-4-mini-instruct voice lora_p4_voice --facts lora_p4_facts
      rc=$?
    fi
  fi
  if [ "$rc" -ne 0 ]; then
    echo "$(date) phi failed; fallback HuggingFaceTB/SmolLM3-3B"
    {
      echo "Phi-4-mini-instruct failed to load or train. Fallback: HuggingFaceTB/SmolLM3-3B."
      echo
      echo "download:"
      cat phi_download_error.txt 2>/dev/null
      echo
      echo "facts log tail:"
      tail -n 40 train_lora_p4_facts.log 2>/dev/null
      echo
      echo "voice log tail:"
      tail -n 40 train_lora_p4_voice.log 2>/dev/null
    } > base_fallback.txt
    rm -rf lora_p4_facts lora_p4_voice
    $P -c 'from huggingface_hub import snapshot_download; snapshot_download("HuggingFaceTB/SmolLM3-3B")' > hfdl_smol.log 2>&1
    rc=$?
    echo "$(date) smol download exit $rc"
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
    train_stage HuggingFaceTB/SmolLM3-3B facts lora_s3_facts
    rc=$?
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
    train_stage HuggingFaceTB/SmolLM3-3B voice lora_s3_voice --facts lora_s3_facts
    rc=$?
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
  fi
else
  echo "$(date) second base skip"
fi

unload
echo "$(date) ollama unloaded before eval"

tags="q4"
if [ -f lora_p4_voice/train_report.json ]; then tags="q4 p4"; fi
if [ -f lora_s3_voice/train_report.json ]; then tags="q4 s3"; fi
for s in $tags; do
  for c in ${s}_fv ${s}_fv_rag ${s}_v_rag; do
    if [ ! -f "runs/$c.json" ]; then
      $P eval4.py "$c" > "runs/$c.log" 2>&1
      rc=$?
      echo "$(date) eval $c exit $rc"
    else
      echo "$(date) eval $c skip"
    fi
  done
done

if [ ! -f runs/q14_rag.json ]; then
  cp -a ../exp3/runs/q14_rag.json runs/q14_rag.json
  rc=$?
  echo "$(date) copy q14_rag.json exit $rc"
fi
if [ ! -f runs/q14_rag.judged.json ]; then
  cp -a ../exp3/runs/q14_rag.judged.json runs/q14_rag.judged.json
  rc=$?
  echo "$(date) copy q14_rag.judged.json exit $rc"
fi

J=""
for c in q4_fv q4_fv_rag q4_v_rag p4_fv p4_fv_rag p4_v_rag s3_fv s3_fv_rag s3_v_rag q14_rag; do
  [ -f "runs/$c.json" ] && J="$J runs/$c.json"
done
$P judge.py $J > runs/judge.log 2>&1
rc=$?
echo "$(date) judged exit $rc"

$P summarize4.py > summarize.log 2>&1
rc=$?
echo "$(date) results.md written exit $rc"
systemctl --user is-active noreturn.service 2>/dev/null | sed "s/^/$(date) noreturn.service: /"
echo "$(date) ALL DONE"
