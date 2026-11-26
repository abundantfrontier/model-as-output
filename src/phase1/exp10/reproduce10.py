"""Write exp10/REPRODUCE.md in the style of exp4/REPRODUCE.md."""
import os, json, subprocess, datetime
import frozen
import prompts

H = frozen.H
PY = os.path.join(frozen.ROOT, "exp3", ".venv", "bin", "python")


def sh(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()


def script_shas():
    names = [
        "run.sh", "hparams.json", "frozen.py", "prompts.py", "scores.py",
        "eval_twopass.py", "factdiff.py", "guard.py", "judge_wrap.py",
        "eval4.py", "judge.py", "common.py", "chat.py",
        "summarize10.py", "notes10.py", "reproduce10.py", "commit10.py",
        "check_tolerance.py", "calib.py", "human_sample.py", "BRIEF.md",
    ]
    lines = []
    for name in names:
        path = os.path.join(H, name)
        if os.path.isfile(path):
            lines.append(f"{frozen.sha256(path)}  {name}")
    return lines


def main():
    os.chdir(H)
    sha_lines = script_shas()
    try:
        py = sh([PY, "-c", "import torch,sys; print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda)"]).splitlines()
    except Exception as e:
        py = [str(e)]
    try:
        gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    except Exception as e:
        gpu = str(e)
    try:
        kernel = sh(["uname", "-r"])
    except Exception as e:
        kernel = str(e)
    try:
        oll = sh(["ollama", "list"])
    except Exception as e:
        oll = str(e)
    try:
        freeze = sh([PY, "-m", "pip", "freeze"])
    except Exception as e:
        freeze = str(e)
    hp = open(os.path.join(H, "hparams.json")).read().rstrip()
    tol_path = os.path.join(H, "runs", "tolerance.json")
    tol = json.load(open(tol_path)) if os.path.isfile(tol_path) else None
    regen_note = "Reproduction run not scored yet."
    if tol:
        regen_note = (
            f"regen_tolerance_ok={tol.get('regen_tolerance_ok')}; "
            f"orig_rejudge_tolerance_ok={tol.get('orig_rejudge_tolerance_ok')}; "
            f"proto_source={tol.get('proto_source')}; {tol.get('proto_note')}"
        )
    when = datetime.datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y").strip()
    fact_head = prompts.FACT_DIFF_HEAD.rstrip()
    rewrite = prompts.REWRITE_INSTR
    text = f"""# exp10 reproduction (No Return prototype: p2_light_dpo_guarded)

Generated {when}. Locked prototype p2_light_dpo_guarded: must-answer 59/60, marker leaks 1/48, voice 4.22 (4.2157), frame breaks 3. Reader at ch3, n=3. Fallback 72/204. No new LoRA. 8B not used.

Revised pass bar for this prototype: voice ≥ 4.2; marker leaks ≤ 2/48 and frame breaks < 6 unchanged; must-answer target remains 60/60, with 59/60 recorded as the known miss (M16 / yd4408 / "What is your number?", 2/3, inherited from pass-1).

## Commands in order
```
cd ~/output-model/exp10 && bash run.sh smoke
cd ~/output-model/exp10 && nohup bash run.sh >> log.txt 2>&1 &
# P=~/output-model/exp3/.venv/bin/python
# regenerate pass-2 from the cached pass-1 file, then fact-diff, guard, judge:
#   ollama stop  # unload before local GPU generate
#   $P eval_twopass.py p2_light_dpo --n 3 --seed 1
#   $P factdiff.py runs/p2_light_dpo.json
#   $P guard.py p2_light_dpo
#   $P judge.py runs/p2_light_dpo_guarded.json
# re-judge original exp9 answers (fixed text, new output path):
#   $P judge.py runs/p2_light_dpo_guarded_orig.json
#   $P check_tolerance.py && $P summarize10.py
```

Pass-2 generation pins `torch.manual_seed(1)` and `torch.cuda.manual_seed_all(1)` once before the rewrite loop. Sampling stays `do_sample=True`, temperature 0.7, top_p 0.95, max_new_tokens=200 (same as eval4). The original exp9 pass-2 generate call did not set a torch seed.

If the full regen misses tolerance (must exact, marker exact, voice ±0.03, frames ±1), keep the original exp9 judged file as the locked prototype scores. Current check: {regen_note}

## Pass-1 model
- Condition: exp6 `v_merged_v_rag` = exp6/merged_q4_facts (Qwen3-4B revision `{frozen.BASE_REV}` with exp4/lora_q4_facts merged) + exp6/lora_v_merged + RAG.
- Cached answers: `exp10/pass1/v_merged_v_rag.json` (copy of exp9/pass1/v_merged_v_rag.json, from exp6/runs/v_merged_v_rag.json).
- sha256: `{frozen.PASS1_SHA}`
- Retriever: `common.retrieve` / `common.rag_sysprompt` (bge-m3, 120-word chunks, chapters 1..upto only, top k=6). Reader at ch3 so upto=3. Pass-1 is not regenerated.

## Pass-2 adapter
- exp7 light DPO, copied as `exp10/init/lora_v_light_dpo` (same bytes as exp9/init/lora_v_light_dpo and exp7/lora_v_light_dpo).
- sha256 of adapter_model.safetensors: `{frozen.P2_ADAPTER_SHA}`
- Base: `{frozen.MERGED}` (not overwritten).
- No retrieval in the rewrite prompt. `common.retrieve` / `common.rag_sysprompt` are not called.

## Rewrite prompt
System, for wren / diane / yd4408: `common.sysprompt(char, day=3)`. Narrator rows would use the NARR card in prompts.py. Not rag_sysprompt.

User message, in this order: the reader question, a blank line, the pass-1 answer, a blank line, then:

```
{rewrite}
```

## Fact-diff, guard, judge
- Fact-diff and voice/leak judge: local ollama qwen3:14b, temp 0, seed 1, think off, JSON format.
- Frame regex: `common.py` FRAME. Marker scan: `common.fail_hits`.
- A sample fails and falls back to pass-1 text when the judge lists any dropped claim, any added claim, the reply does not parse, or pass-2 has a fail_hits marker that pass-1 did not. Locked fallback rate 72/204.

Fact-diff head:

```
{fact_head}
```

## Known must-answer miss
Item M16, character yd4408, probe "What is your number?". Rubric: `4408|forty-four`. 2/3 samples match; sample 1 fails ("I am not a number. I am a tag on a hook."). The same M16 2/3 miss is already present in pass-1 v_merged_v_rag, so 59/60 is inherited from pass-1. Unguarded p2_light_dpo also missed M4 and M8; the guard recovered those.

## Scripts (sha256)
```
{chr(10).join(sha_lines)}
```

Harness copies must match exp4/REPRODUCE.md:
```
{frozen.HARNESS['eval4.py']}  eval4.py
{frozen.HARNESS['judge.py']}  judge.py
{frozen.HARNESS['common.py']}  common.py
{frozen.HARNESS['chat.py']}  chat.py
```

Do not edit judge.py / common.py / eval4.py. Thin wrappers (eval_twopass.py, judge_wrap.py) change output path, seed, temperature, or judge model and write only under exp10/.

## Hyperparameters
```
{hp}
```

## Data and weights (sha256)
```
{frozen.PASS1_SHA}  pass1/v_merged_v_rag.json
{frozen.P2_ADAPTER_SHA}  init/lora_v_light_dpo/adapter_model.safetensors
{frozen.FACTS_SHA}  exp4/lora_q4_facts/adapter_model.safetensors
```

Facts adapter is read-only. Merged base is exp6/merged_q4_facts (Qwen/Qwen3-4B revision {frozen.BASE_REV} with that facts adapter merged).

## RAG (common.py) — pass-1 only
- embedder ollama bge-m3, 120-word chunks over chapters 1..upto (upto=3, the reader's chapter lock), top k=6 by dot product, put back in chapter order
- Pass-2 does not retrieve

## Judge
- qwen3:14b through ollama, temp 0.0, seed 1, JSON format, think off. Prompts are in judge.py (sha above): a leak judge against C.SECRET and a 1-5 voice judge using the character card plus 8 real lines from the book. Frame-break regex: common.py FRAME.

## Environment
```
Python {py[0] if py else ''}
torch {py[1] if len(py)>1 else ''} cuda {py[2] if len(py)>2 else ''}
{gpu}
{kernel}
venv: ~/output-model/exp3/.venv
```

### ollama models
```
{oll}
```

### pip freeze
```
{freeze}
```

## Locked prototype scores
- cond: p2_light_dpo_guarded
- must-answer: 59/60
- marker leaks: 1/48
- voice: 4.22 (4.2157)
- frame breaks: 3
- source: exp9/runs/p2_light_dpo_guarded.judged.json (copied to exp10/locked/)
- local tag: `no-return-proto-v1` (annotated, not pushed)
"""
    path = os.path.join(H, "REPRODUCE.md")
    with open(path, "w") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
