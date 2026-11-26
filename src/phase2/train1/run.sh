#!/bin/bash
# Resumable train1 driver. Skip a stage when its output exists.
set -u
cd "$(dirname "$0")/.."
P="${P:-$HOME/output-model/exp3/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs runs data/generated data/handwritten index lora_facts lora_voice
echo $$ > run.pid
log() { echo "$(TZ=America/Chicago date '+%a %b %d %I:%M:%S %p %Z')  $*" | tee -a logs/run.log; }
status() {
  {
    echo "# Phase 2 Frankenstein — train1 build status"
    echo
    echo "Updated: $(TZ=America/Chicago date)"
    echo
    echo "- Current stage: $1"
    echo "- State: $2"
    echo "- Next: $3"
    echo "- GPU lock: HOLDING (holder PID $(cat .gpu-hold.pid 2>/dev/null || echo '?'))"
    echo "- Time cap: Sat Nov 21 5:30 PM CST"
    echo
    echo "## Checklist"
    echo
    grep -E '^- \[' build-status.md | head -20
  } > /tmp/train1-status.md
  # keep the checklist lines from this file if present; rewrite simply:
  cat > build-status.md <<EOF
# Phase 2 Frankenstein — train1 build status

Updated: $(TZ=America/Chicago date)

- Current stage: $1
- State: $2
- Next: $3
- GPU lock: HOLDING (holder PID $(cat .gpu-hold.pid 2>/dev/null || echo unknown))
- Time cap: Sat Nov 21 5:30 PM CST

## Checklist

- [done] Read Phase 2 README, knowledge-state, questions, voice, and Phase 1 recipe
- [done] Took ~/gpu.lock (live holder PID); updated ~/GPU_SCHEDULE.md
- [$([ -f index/bge.npy ] && echo done || echo running)] Embed 391 chunks with bge-m3; TF-IDF kept as fallback
- [$([ -f logs/test_bge_lock.ok ] && echo done || echo pending)] Anchor checks on the bge index
- [$([ -f data/voice.jsonl ] && echo done || echo pending)] Expand voice/facts data (eval questions held out)
- [$([ -f lora_facts/adapter_model.safetensors ] && echo done || echo pending)] Train facts LoRA on Qwen3-4B
- [$([ -f lora_voice/adapter_model.safetensors ] && echo done || echo pending)] Train one character-tagged voice LoRA
- [$([ -f lora_voice_dpo/adapter_model.safetensors ] && echo done || echo pending)] Light DPO against spoilers/frame breaks
- [$([ -f runs/p2_voice_guarded.judged.json ] && echo done || echo pending)] Eval two-pass guarded
- [$([ -f runs/merged_rag.judged.json ] && echo done || echo pending)] Eval single-pass merged
- [$([ -f runs/rag_base.judged.json ] && echo done || echo pending)] Eval RAG-only base baseline
- [$([ -f RESULTS.md ] && echo done || echo pending)] RESULTS.md, local commit, release lock
EOF
}
need() { [ ! -e "$1" ]; }

log "start pid $$"
status embed running "bge lock test"

if need index/bge.npy; then
  log "embed"
  $P scripts/embed.py >> logs/embed.log 2>&1 || { log "FAIL embed"; exit 1; }
fi
status bge-lock running "data"
if need logs/test_bge_lock.ok; then
  log "test_bge_lock"
  $P scripts/test_bge_lock.py > logs/test_bge_lock.log 2>&1 && touch logs/test_bge_lock.ok || { log "FAIL bge lock"; exit 1; }
fi

status data running "train facts"
if need data/voice.jsonl; then
  log "gen_data"
  $P scripts/gen_data.py >> logs/gen_data.log 2>&1 || { log "FAIL gen_data"; exit 1; }
fi

status train-facts running "train voice"
if need lora_facts/adapter_model.safetensors; then
  log "holdout check"
  $P scripts/check_holdout.py >> logs/holdout.log 2>&1 || { log "FAIL holdout"; exit 1; }
  log "smoke train facts"
  $P scripts/train.py --stage facts --out /tmp/train1_smoke_facts --pairs data/facts.jsonl --claims data/claims.jsonl --epochs 1 --r 16 --limit 4 \
    >> logs/smoke_facts.log 2>&1 || { log "FAIL smoke facts"; exit 1; }
  log "train facts"
  $P scripts/train.py --stage facts --out lora_facts --pairs data/facts.jsonl --claims data/claims.jsonl --epochs 2 --r 16 --save-every 400 \
    >> logs/train_facts.log 2>&1 || { log "FAIL train facts"; exit 1; }
fi
status train-voice running "dpo or eval"
if need lora_voice/adapter_model.safetensors; then
  log "train voice"
  $P scripts/train.py --stage voice --out lora_voice --facts lora_facts --pairs data/voice.jsonl --epochs 2 --r 16 --save-every 400 \
    >> logs/train_voice.log 2>&1 || { log "FAIL train voice"; exit 1; }
fi

VOICE=lora_voice
if [ -f data/dpo.jsonl ] && need lora_voice_dpo/adapter_model.safetensors; then
  # skip DPO if the deadline is inside the eval reserve (train.py also checks)
  log "train light DPO"
  if $P scripts/train.py --stage voice --out lora_voice_dpo --facts lora_facts --init lora_voice \
      --pairs data/dpo.jsonl --loss dpo --epochs 1 --r 16 --lr 5e-5 \
      >> logs/train_dpo.log 2>&1; then
    VOICE=lora_voice_dpo
  else
    log "DPO failed or skipped; using lora_voice"
  fi
fi
# A finished DPO adapter from an earlier session must be the rewrite (Phase 1 p2_light_dpo).
# The train block above is skipped when the file already exists, which previously left VOICE=lora_voice.
if [ -f lora_voice_dpo/adapter_model.safetensors ]; then
  VOICE=lora_voice_dpo
fi
echo "$VOICE" > logs/voice_adapter.txt
[ -f logs/voice_adapter.txt ] && VOICE=$(cat logs/voice_adapter.txt)

status eval-base running "pass1"
if need runs/rag_base.json; then
  log "eval rag_base"
  $P scripts/eval_pass1.py rag_base --rag --n 3 --backend bge --out runs/rag_base.json \
    >> logs/eval_rag_base.log 2>&1 || { log "FAIL rag_base"; exit 1; }
fi
status eval-pass1 running "pass2"
if need runs/p1_facts_rag.json; then
  log "eval p1_facts_rag"
  $P scripts/eval_pass1.py p1_facts_rag --rag --n 3 --backend bge \
    --facts "$PWD/lora_facts" --out runs/p1_facts_rag.json \
    >> logs/eval_p1.log 2>&1 || { log "FAIL p1"; exit 1; }
fi
status eval-pass2 running "factdiff"
if need runs/p2_voice.json; then
  log "eval p2_voice"
  $P scripts/eval_twopass.py p2_voice --pass1 runs/p1_facts_rag.json \
    --facts "$PWD/lora_facts" --voice "$PWD/$VOICE" \
    --out runs/p2_voice.json \
    >> logs/eval_p2.log 2>&1 || { log "FAIL p2"; exit 1; }
fi
status factdiff running "guard"
if need runs/p2_voice.factdiff.json; then
  log "factdiff"
  $P scripts/factdiff.py runs/p2_voice.json >> logs/factdiff.log 2>&1 || { log "FAIL factdiff"; exit 1; }
fi
if need runs/p2_voice_guarded.json; then
  log "guard"
  $P scripts/guard.py runs/p2_voice.json runs/p2_voice.factdiff.json --out runs/p2_voice_guarded.json \
    >> logs/guard.log 2>&1 || { log "FAIL guard"; exit 1; }
fi

status merge running "merged eval"
if need merged_facts/config.json; then
  log "merge facts (CPU)"
  $P scripts/merge.py --base Qwen/Qwen3-4B --adapter "$PWD/lora_facts" --out merged_facts --name facts \
    >> logs/merge_facts.log 2>&1 || { log "FAIL merge facts"; exit 1; }
fi
if need merged_fv/config.json; then
  log "merge voice (CPU)"
  $P scripts/merge.py --base "$PWD/merged_facts" --adapter "$PWD/$VOICE" --out merged_fv --name voice \
    >> logs/merge_voice.log 2>&1 || { log "FAIL merge voice"; exit 1; }
fi
status eval-merged running "judge"
if need runs/merged_rag.json; then
  log "eval merged_rag"
  $P scripts/eval_pass1.py merged_rag --rag --n 3 --backend bge \
    --merged "$PWD/merged_fv" --out runs/merged_rag.json \
    >> logs/eval_merged.log 2>&1 || { log "FAIL merged"; exit 1; }
fi

status judge running "summarize"
for f in runs/rag_base.json runs/p1_facts_rag.json runs/p2_voice.json runs/p2_voice_guarded.json runs/merged_rag.json; do
  [ -f "$f" ] || continue
  j="${f%.json}.judged.json"
  if [ ! -f "$j" ]; then
    log "judge $f"
    $P scripts/judge.py "$f" >> logs/judge.log 2>&1 || { log "FAIL judge $f"; exit 1; }
  fi
done
status summarize running "commit"
$P scripts/summarize.py >> logs/summarize.log 2>&1 || { log "FAIL summarize"; exit 1; }
log "ALL STAGES DONE"
status done done none
echo ALL_STAGES_DONE >> logs/run.log
