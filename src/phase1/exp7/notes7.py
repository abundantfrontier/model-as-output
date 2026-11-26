"""Hyperparameters and data hashes for exp7. Rewritten from artifacts so a resume stays accurate."""
import json, os, subprocess
import frozen

H = os.path.dirname(os.path.abspath(__file__))

def load(path):
    path = os.path.join(H, path) if not os.path.isabs(path) else path
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def sha_line(path):
    if not os.path.exists(path):
        return f"(missing)  {path}"
    n = ""
    if path.endswith(".jsonl"):
        with open(path) as f:
            n = str(sum(1 for line in f if line.strip())) + " rows  "
    return f"{n}{frozen.sha256(path)}  {os.path.relpath(path, H)}"

def block(title, lines):
    return [f"## {title}", ""] + lines + [""]

def report(name):
    doc = load(os.path.join(H, f"lora_{name}", "train_report.json"))
    if not doc:
        return ["- not trained yet."]
    return ["```", json.dumps(doc, indent=1), "```"]

def main():
    os.chdir(H)
    facts_sha = frozen.sha256(os.path.join(frozen.FACTS, "adapter_model.safetensors"))
    init_sha = frozen.sha256(os.path.join(frozen.INIT, "adapter_model.safetensors"))
    src_sha = frozen.sha256(os.path.join(frozen.INIT_SRC, "adapter_model.safetensors"))
    hp = load("hparams.json") or {}
    lines = [
        "# exp7 reproduction notes",
        "",
        "Light DPO and light contrast on the frozen exp6 v_merged adapter. "
        "exp3/, exp4/, exp5/, and exp6/ are not modified. The facts LoRA is not retrained.",
        "",
        f"Facts adapter sha256: `{facts_sha}`",
        f"Expected: `{frozen.FACTS_SHA}`",
        f"Match: {facts_sha == frozen.FACTS_SHA}",
        "",
        f"Init adapter copy: `{init_sha}`",
        f"exp6/lora_v_merged: `{src_sha}`",
        f"Init match: {init_sha == src_sha}",
        "",
        "Harness copies (must match exp4/REPRODUCE.md). chat.py is copied because eval4.py imports it.",
        "",
        "```",
    ]
    for name, want in frozen.HARNESS.items():
        got = frozen.sha256(os.path.join(H, name))
        lines.append(f"{got}  {name}  {'OK' if got == want else 'MISMATCH'}")
    lines += ["```", ""]
    py = subprocess.check_output([os.path.join(frozen.ROOT, "exp3", ".venv", "bin", "python"), "-c",
        "import torch,sys; print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda)"], text=True).strip().splitlines()
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    except Exception as e:
        gpu = str(e)
    lines += block("Environment", [
        f"- Python {py[0]}, torch {py[1]}, cuda {py[2]}",
        f"- GPU: {gpu}",
        "- venv: ~/output-model/exp3/.venv",
        "- trl is not installed. DPO and unlikelihood use the local loss in train7.py (from exp6/train6.py). Nothing was pip-installed.",
        "- Judge and teacher: local ollama qwen3:14b. Embeddings: bge-m3. temp 0 / seed 1 for judges. Frame regex is common.py FRAME.",
        f"- Base: {frozen.MERGED} (Qwen/Qwen3-4B revision {frozen.BASE_REV} with exp4/lora_q4_facts merged). Not overwritten.",
        "- 8B was not trained.",
    ])
    lines += block("Why these settings are lighter than exp6", [
        "- " + v for v in (hp.get("why") or {}).values()
    ])
    lines += block("Shared training setup", [
        "- Voice LoRA shape stays r=32, alpha=64. target_modules are whatever exp6/lora_v_merged saved (q_proj k_proj v_proj o_proj gate_proj up_proj down_proj).",
        "- Every new adapter starts from the exp7 copy of exp6/lora_v_merged, except v_light_combo which starts from lora_v_light_contrast.",
        "- DPO dropout is forced off. Contrast keeps the init dropout 0.05.",
        "- DPO reference logprobs are cached from the init adapter before any step. Rows without `bad` take a CE step.",
        "- Chat prompts for training are the non-RAG system prompt. Narrator uses frozen.NARR.",
        "- Teacher rewrites, when a frame/marker prompt has no clean chosen answer, use qwen3:14b + RAG at chapter 3. Attempt 1 is unconstrained. Attempt 2 appends a ban on frame words. Attempt 3 drops sentences that hit FRAME or a marker. Attempt 4 rewrites the draft without those words. Keep only voice >= 4.5, no frame, no marker, no judge leak.",
    ])
    lines += block("Data files", [
        f"- {sha_line(os.path.join(H, 'data', 'v_light_dpo_pairs.jsonl'))}",
        f"- {sha_line(os.path.join(H, 'data', 'v_light_contrast_train.jsonl'))}",
        f"- {sha_line(os.path.join(H, 'data', 'v_light_dpo_replay.jsonl'))}",
        f"- {sha_line(os.path.join(H, 'data', 'fresh_v_merged.jsonl'))}",
        f"- {sha_line(os.path.join(H, 'data', 'light_rewrites.jsonl'))}",
        "- dpo meta: " + json.dumps(load("data/v_light_dpo_meta.json") or "not built yet"),
        "- contrast report: " + json.dumps(load("data/v_light_contrast_report.json") or "not built yet"),
        "- replay meta: " + json.dumps(load("data/v_light_replay_meta.json") or "not built yet"),
    ])
    for title, name in (
        ("v_light_dpo (a)", "v_light_dpo"),
        ("v_light_contrast (b)", "v_light_contrast"),
        ("v_light_combo (c)", "v_light_combo"),
        ("v_light_dpo_replay (c)", "v_light_dpo_replay"),
    ):
        lines += block(title, ["- hparams: " + json.dumps(hp.get(name) or {}), ""] + report(name))
    lines += block("Checkpoint sweep (d)", [
        "- rule: " + json.dumps((load("data/sweep.json") or {}).get("rule") or (hp.get("why") or {}).get("sweep")),
        "- sweep: " + json.dumps(load("data/sweep.json") or "not run yet"),
        "- best: " + json.dumps(load("data/best.json") or "not chosen yet"),
    ])
    lines += block("Commands", [
        "```",
        "cd ~/output-model/exp7 && nohup bash run.sh >> log.txt 2>&1 &",
        "```",
        "",
        "A stage is skipped when its train_report.json or runs/<cond>.json already exists.",
    ])
    open("REPRODUCE-notes.md", "w").write("\n".join(lines) + "\n")
    print("wrote REPRODUCE-notes.md")

if __name__ == "__main__":
    main()
