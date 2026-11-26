"""Write exp7/REPRODUCE.md in the exp4 style when a RAG condition clears the bar."""
import os, json, subprocess, glob
import frozen

H = os.path.dirname(os.path.abspath(__file__))

def sha(path):
    return frozen.sha256(path) if os.path.isfile(path) else "missing"

def rows(path):
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())

def main():
    os.chdir(H)
    best_path = os.path.join(H, "data", "best.json")
    if not os.path.isfile(best_path):
        print("no best.json")
        return
    best = json.load(open(best_path))
    if not best.get("pass"):
        print("no RAG pass; REPRODUCE.md not written")
        return
    hp = json.load(open(os.path.join(H, "hparams.json")))
    py = subprocess.check_output([os.path.join(frozen.ROOT, "exp3", ".venv", "bin", "python"), "-c",
        "import sys,torch; print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda)"], text=True).strip().splitlines()
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    kern = subprocess.check_output(["uname", "-r"], text=True).strip()
    try:
        oll = subprocess.check_output(["ollama", "list"], text=True).strip()
    except Exception as e:
        oll = str(e)
    pip = subprocess.check_output([os.path.join(frozen.ROOT, "exp3", ".venv", "bin", "pip"), "freeze"], text=True).strip()
    scripts = [
        "build7.py", "train7.py", "eval_cond.py", "eval4.py", "judge.py", "common.py", "chat.py",
        "frozen.py", "sweep7.py", "summarize7.py", "notes7.py", "reproduce7.py", "commit7.py",
        "voicefilt.py", "gpuutil.py", "scores.py", "run.sh", "smoke.sh", "hparams.json", "BRIEF.md",
    ]
    data_files = [
        "data/v_light_dpo_pairs.jsonl", "data/v_light_contrast_train.jsonl", "data/v_light_dpo_replay.jsonl",
        "data/fresh_v_merged.jsonl", "data/light_rewrites.jsonl",
    ]
    lines = [
        f"# exp7 reproduction (best: {best.get('cond')})",
        "",
        f"Best RAG result {best.get('cond')}: must-answer {best.get('must')}, marker leaks {best.get('marker')}, "
        f"voice {best.get('voice')}, frame breaks {best.get('frames')}. Reader at ch3, n=3.",
        f"Adapter: `{best.get('adapter')}`.",
        "",
        "## Commands in order",
        "```",
        "cd ~/output-model/exp7 && nohup bash run.sh >> log.txt 2>&1 &",
        "# run.sh: verify frozen inputs, build7.py prepare, train7.py v_light_dpo, v_light_contrast,",
        "# v_light_combo, v_light_dpo_replay, eval_cond.py + judge.py, sweep7.py, summarize7.py, notes7.py",
        "```",
        "",
        "## Scripts (sha256)",
        "```",
    ]
    for name in scripts:
        lines.append(f"{sha(os.path.join(H, name))}  {name}")
    lines += ["```", "", "## Hyperparameters", "```", json.dumps(hp, indent=1), "```", ""]
    lines += ["## Data (rows, sha256)", "```"]
    for rel in data_files:
        path = os.path.join(H, rel)
        if os.path.isfile(path):
            lines.append(f"{rows(path)} rows  {sha(path)}  {rel}")
    lines += ["```", ""]
    for name in ("v_light_dpo_meta.json", "v_light_contrast_report.json", "v_light_replay_meta.json"):
        path = os.path.join(H, "data", name)
        if os.path.isfile(path):
            lines.append(f"- {name}: `{open(path).read().strip()}`")
    lines += [
        "",
        "## Base model",
        f"- Qwen/Qwen3-4B revision {frozen.BASE_REV}",
        f"- Merged facts base (not overwritten): {frozen.MERGED}",
        f"- Facts adapter sha256 {frozen.FACTS_SHA}",
        f"- Init voice adapter copy of exp6/lora_v_merged, sha256 {sha(os.path.join(frozen.INIT, 'adapter_model.safetensors'))}",
        "",
        "## RAG (common.py)",
        "- embedder ollama bge-m3, 120-word chunks over chapters 1..upto (upto=3), top k=6 by dot product, put back in chapter order",
        "",
        "## Judge",
        "- qwen3:14b through ollama, temp 0.0, seed 1, JSON format, think off. Prompts are in judge.py. "
        "Frame-break regex: common.py FRAME. Reference judged files were copied, not rerun.",
        "",
        "## Environment",
        "```",
        f"Python {py[0]}",
        f"torch {py[1]} cuda {py[2]}",
        gpu,
        kern,
        "```",
        "",
        "### ollama models",
        "```",
        oll,
        "```",
        "",
        "### pip freeze",
        "```",
        pip,
        "```",
        "",
        "## Adapters (sha256 of adapter_model.safetensors)",
        "```",
    ]
    for path in sorted(glob.glob(os.path.join(H, "lora_*", "adapter_model.safetensors"))):
        lines.append(f"{sha(path)}  {os.path.relpath(path, H)}")
    for path in sorted(glob.glob(os.path.join(H, "lora_*", "checkpoint-*", "adapter_model.safetensors"))):
        lines.append(f"{sha(path)}  {os.path.relpath(path, H)}")
    lines += [
        "```",
        "",
        "Weights are gitignored. This file records their hashes.",
        "",
    ]
    open(os.path.join(H, "REPRODUCE.md"), "w").write("\n".join(lines))
    print("wrote REPRODUCE.md", best.get("cond"))

if __name__ == "__main__":
    main()
