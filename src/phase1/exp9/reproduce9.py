"""Write exp9/REPRODUCE.md when a two-pass row clears the pass bar.

Called at the end of run.sh. Exits 0 without the file when nothing passes.
"""
import os, json, subprocess
import frozen

H = frozen.H
PY = os.path.join(frozen.ROOT, "exp3", ".venv", "bin", "python")

def sh(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()

def main():
    os.chdir(H)
    best_path = os.path.join(H, "data", "best.json")
    if not os.path.isfile(best_path):
        print("no best.json; not writing REPRODUCE.md")
        return
    best = json.load(open(best_path))
    if not best.get("pass"):
        print("no two-pass pass; not writing REPRODUCE.md")
        if os.path.isfile(os.path.join(H, "REPRODUCE.md")):
            print("leaving existing REPRODUCE.md in place")
        return
    scripts = [
        "run.sh", "hparams.json", "frozen.py", "prompts.py", "scores.py", "eval_twopass.py",
        "factdiff.py", "guard.py", "build_rewrite.py", "train9.py", "decide.py",
        "summarize9.py", "notes9.py", "reproduce9.py", "commit9.py",
        "eval4.py", "judge.py", "common.py", "chat.py",
    ]
    sha_lines = []
    for name in scripts:
        path = os.path.join(H, name)
        if os.path.isfile(path):
            sha_lines.append(f"{frozen.sha256(path)}  {name}")
    py = sh([PY, "-c", "import torch,sys; print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda)"]).splitlines()
    gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    kernel = sh(["uname", "-r"])
    try:
        oll = sh(["ollama", "list"])
    except Exception as e:
        oll = str(e)
    freeze = sh([PY, "-m", "pip", "freeze"])
    hp = open(os.path.join(H, "hparams.json")).read().rstrip()
    report = ""
    rp = os.path.join(H, "lora_v_rewrite", "train_report.json")
    if os.path.isfile(rp):
        report = open(rp).read().rstrip()
    meta = ""
    mp = os.path.join(H, "data", "rewrite_meta.json")
    if os.path.isfile(mp):
        meta = open(mp).read().rstrip()
    import prompts
    adapter_lines = []
    for name in frozen.ADAPTER_SRC:
        path = os.path.join(H, "init", name, "adapter_model.safetensors")
        if os.path.isfile(path):
            adapter_lines.append(f"{frozen.sha256(path)}  init/{name}/adapter_model.safetensors")
    rw = os.path.join(H, "lora_v_rewrite", "adapter_model.safetensors")
    if os.path.isfile(rw):
        adapter_lines.append(f"{frozen.sha256(rw)}  lora_v_rewrite/adapter_model.safetensors")
    data_lines = []
    for path in [
        os.path.join(H, "pass1", "v_merged_v_rag.json"),
        os.path.join(H, "pass1", "q4_merged_rag.json"),
        os.path.join(H, "data", "rewrite_pairs.jsonl"),
    ]:
        if os.path.isfile(path):
            data_lines.append(f"{frozen.sha256(path)}  {os.path.relpath(path, H)}")
    cond = best.get("cond")
    text = f"""# exp9 reproduction (best: {cond})

Best two-pass result {cond}: must-answer {best.get('must')}, marker leaks {best.get('marker')}, voice {None if best.get('voice') is None else round(best['voice'], 2)}, frame breaks {best.get('frames')}. Reader at ch3, n=3. The scored text is the pass-2 rewrite, or the pass-1 answer where the guarded check fell back.

## Commands in order
```
cd ~/output-model/exp9 && bash run.sh smoke
cd ~/output-model/exp9 && nohup bash run.sh >> log.txt 2>&1 &
# run.sh: verify harness → smoke (if smoke/ok.json is stale) → pass-2 eval of the four voice adapters on cached v_merged_v_rag answers → judge.py → factdiff.py → guarded fallback → optional q4_merged pass-1 → optional rewrite LoRA → optional checkpoint sweep → summarize9.py → notes9.py → this file → commit9.py
# P=~/output-model/exp3/.venv/bin/python
```

## Scripts (sha256)
```
{chr(10).join(sha_lines)}
```

## Two-pass prompt
System: common.sysprompt(char) with day=3. Not rag_sysprompt. No retrieved passages.

User: the reader question, a blank line, the pass-1 answer, a blank line, then:

```
{prompts.REWRITE_INSTR}
```

Generation: max_new_tokens=200, temperature 0.7, top_p 0.95, do_sample. Same probes and n=3 as eval4. Pass 1 is the cached file pass1/v_merged_v_rag.json (exp6 v_merged_v_rag), not a new retrieval.

## Fact-diff
qwen3:14b, temp 0, seed 1. Prompt head is prompts.FACT_DIFF_HEAD. Fallback when any claim is dropped, any claim is added, the judge reply does not parse, or pass-2 has a new common.fail_hits marker. The guarded condition scores pass-1 text on those rows.

## Hyperparameters
(c) trains only if a two-pass row is close (must-answer >= 59/60, marker leaks <= 2/48, frame breaks < 6, voice >= 4.2). Loss is the local CE in exp8/train8.py. Init is exp9/init/lora_v_fb_dpo. r=32, alpha=64, dropout 0.05, 1 epoch, lr 2e-5.

```
{hp}
```

```
{report or "no train_report.json"}
```

```
{meta or "no rewrite_meta.json"}
```

## Data (sha256)
```
{chr(10).join(data_lines)}
```

## Base model
- Merged facts base: {frozen.MERGED}
- Qwen/Qwen3-4B revision {frozen.BASE_REV}
- Facts adapter sha256 {frozen.FACTS_SHA} (exp4/lora_q4_facts, not retrained)

## RAG
Pass 1 retrieval was done by the exp6 / exp8 harness (common.retrieve, bge-m3, chapters 1..3, k=6). Pass 2 does not retrieve.

## Judge
qwen3:14b through ollama, temp 0.0, seed 1, JSON format, think off. judge.py is the exp4 file (sha above). Frame-break regex: common.py FRAME.

## Environment
```
Python {py[0]}
torch {py[1]} cuda {py[2]}
{gpu}
{kernel}
```

### ollama models
```
{oll}
```

### pip freeze
```
{freeze}
```

## Adapters (sha256)
```
{chr(10).join(adapter_lines)}
```
"""
    open(os.path.join(H, "REPRODUCE.md"), "w").write(text)
    print("wrote REPRODUCE.md for", cond)

if __name__ == "__main__":
    main()
