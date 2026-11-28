#!/bin/bash
# n=1 NF4 smoke on an unmerged adapter. Not an official score.
# Same NF4 settings as the proxy. ptyxis is allowed. ollama is allowed (bge).
set -u
cd $HOME/output-model/phase2-frankenstein/train8
PY=$HOME/output-model/exp3/.venv/bin/python
NAME="${1:-qa}"
ADAPTER="$HOME/output-model/phase2-frankenstein/train8/lora_${NAME}"
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "FAILED: missing $ADAPTER/adapter_config.json"
  exit 1
fi
# shellcheck source=gpu_ok.sh
source $HOME/output-model/phase2-frankenstein/train8/scripts/gpu_ok.sh
gpu_ok || exit 1
echo "===== nf4 smoke start $(TZ=America/Los_Angeles date) lora_${NAME} ====="
"$PY" -u scripts/eval_pass1.py "smoke_${NAME}_nf4" \
  --n 1 \
  --nf4 \
  --base Qwen/Qwen3-4B \
  --voice "$ADAPTER" \
  --rag \
  --chars victor,creature \
  --backend bge \
  --out "$HOME/output-model/phase2-frankenstein/train8/runs/smoke_${NAME}_nf4.json"
"$PY" - << PY
import json
from pathlib import Path
p = Path("runs/smoke_${NAME}_nf4.json")
doc = json.loads(p.read_text())
rows = [r for r in doc["results"] if not r.get("locked")]
must = [r for r in rows if r["kind"]=="must"]
ok = sum(1 for r in must if r.get("correct"))
print(f"SMOKE NF4 must {ok}/{len(must)}")
focus = {"m12","m14","m16","m21","m22","m25","m26","m28","m29","m34"}
for r in rows:
    if r["kind"]=="must" and r["id"] in focus:
        print(r["id"], r["char"], "ok" if r.get("correct") else "MISS", (r.get("answer") or "")[:180].replace("\\n"," "))
leaks = {}
for r in rows:
    if r["kind"]!="leak":
        continue
    leaks.setdefault((r["id"], r["char"]), []).append(r)
marked = sum(1 for rs in leaks.values() if any(x.get("fail") for x in rs))
print(f"SMOKE NF4 leak markers {marked}/{len(leaks)}")
PY
echo "===== nf4 smoke done $(TZ=America/Los_Angeles date) ====="
