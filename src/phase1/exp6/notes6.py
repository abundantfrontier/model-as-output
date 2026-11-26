"""Hyperparameters and data hashes for exp6. Rewritten from artifacts so a resume stays accurate."""
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
    n = 0
    if path.endswith(".jsonl"):
        with open(path) as f:
            n = sum(1 for line in f if line.strip())
        return f"{n} rows  {frozen.sha256(path)}  {os.path.relpath(path, H)}"
    return f"{frozen.sha256(path)}  {os.path.relpath(path, H)}"

def block(title, lines):
    return [f"## {title}", ""] + lines + [""]

def main():
    os.chdir(H)
    facts_sha = frozen.sha256(os.path.join(frozen.FACTS, "adapter_model.safetensors"))
    lines = [
        "# exp6 reproduction notes",
        "",
        "Keep facts, raise voice. exp4/lora_q4_facts is loaded and never trained. exp3/, exp4/, and exp5/ are not modified.",
        "",
        f"Facts adapter sha256: `{facts_sha}`",
        f"Expected: `{frozen.FACTS_SHA}`",
        f"Match: {facts_sha == frozen.FACTS_SHA}",
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
    gpu = ""
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    except Exception as e:
        gpu = str(e)
    lines += block("Environment", [
        f"- Python {py[0]}, torch {py[1]}, cuda {py[2]}",
        f"- GPU: {gpu}",
        "- venv: ~/output-model/exp3/.venv",
        "- trl is not installed. DPO uses the local loss in train6.py. Nothing was pip-installed.",
        "- Judge and teacher: local ollama qwen3:14b. Embeddings: bge-m3. temp 0 / seed 1 for judges. Frame regex is common.py FRAME.",
        f"- Base: {frozen.BASE} revision {frozen.BASE_REV}",
    ])
    lines += block("Shared training setup", [
        "- Default voice LoRA: r=32, alpha=64, dropout 0.05, lr 1e-4, AdamW, 2 epochs, batch 1.",
        "- target_modules: same as the facts adapter (q_proj k_proj v_proj o_proj gate_proj up_proj down_proj).",
        "- Chat prompts for training are the non-RAG system prompt (narrator uses frozen.NARR).",
        "- Teacher rewrites use qwen3:14b + RAG at chapter 3.",
        "- Variant (a) stacks a new voice adapter on frozen exp4/lora_q4_facts.",
        "- Variant (b) merges that facts adapter into a private copy under exp6/merged_q4_facts/, then trains a voice-only LoRA.",
        "- Combo (c) takes the better of (a)/(b) by RAG must-answer, then voice, then fewer frame breaks.",
        "- Optional (d) repeats the winning recipe on Qwen3-8B if the VRAM probe passes. No new download.",
    ])
    mix_a = load("data/v_replay_mix.json")
    mix_b = load("data/v_merged_mix.json")
    lines += block("v_replay (a)", [
        "- Voice pool: exp4 voice.jsonl overlaid with exp5 distill train / kept / contrast rewrites (same (char,user) prefers later source).",
        "- Facts replay sampled from exp4/data/facts.jsonl at target 25% of the mixed file (inside 20-30%), seed 0.",
        f"- {sha_line(frozen.VOICE_JSONL)}",
        f"- {sha_line(frozen.FACTS_JSONL)}",
        "- mix: " + (json.dumps(mix_a) if mix_a else "not built yet"),
        f"- {sha_line(os.path.join(H, 'data', 'v_replay_train.jsonl'))}",
        "- Train: train6.py --stage voice --r 32 --epochs 2 --lr 1e-4 --facts exp4/lora_q4_facts --pairs data/v_replay_train.jsonl",
        "- Eval: v_replay_fv and v_replay_fv_rag",
    ])
    rep = load("lora_v_replay/train_report.json")
    if rep:
        lines += ["```", json.dumps(rep, indent=1), "```", ""]
    merge = load("merged_q4_facts/merge_report.json")
    lines += block("v_merged (b)", [
        "- Merge: PeftModel.from_pretrained(Qwen3-4B, exp4/lora_q4_facts).merge_and_unload() -> exp6/merged_q4_facts/ (does not write the HF cache).",
        "- merge: " + (json.dumps(merge) if merge else "not done yet"),
        "- Voice-only LoRA on the merged base. Same voice positives as (a), no facts mix.",
        "- mix: " + (json.dumps(mix_b) if mix_b else "not built yet"),
        f"- {sha_line(os.path.join(H, 'data', 'v_merged_train.jsonl'))}",
        "- Train: train6.py --stage voice --r 32 --epochs 2 --base exp6/merged_q4_facts --pairs data/v_merged_train.jsonl",
        "- Eval: v_merged_v and v_merged_v_rag (adapters=v, no second facts adapter)",
    ])
    rep = load("lora_v_merged/train_report.json")
    if rep:
        lines += ["```", json.dumps(rep, indent=1), "```", ""]
    win = load("data/winner.json")
    lines += block("winner (a vs b)", [
        "- Rank: RAG must-answer, then voice, then fewer frame breaks.",
        "- choice: " + (json.dumps(win) if win else "not picked yet"),
    ])
    con = load("data/contrast_report.json")
    lines += block("v_combo contrast", [
        "- Negatives: exp4 q4_fv_rag / q4_v_rag, all exp5 *_rag judged files, exp6 RAG judged files, plus fresh samples from the winning adapter.",
        "- A negative hits common.py FRAME or has voice_score <= 2. At most 3 bad answers per prompt.",
        "- Rewrites: qwen3:14b + RAG, keep when voice >= 4.5, no frame, no marker, no judge leak (one retry).",
        "- Loss: token unlikelihood on the bad completion (weight 1.0) plus CE on the good completion. r=32, 2 epochs, lr 1e-4.",
        "- If winner is (a), the contrast train file keeps ~25% facts replay so facts do not drift.",
        "- contrast: " + (json.dumps(con) if con else "not built yet"),
        f"- {sha_line(os.path.join(H, 'data', 'v_contrast_train.jsonl'))}",
    ])
    rep = load("lora_v_combo_contrast/train_report.json")
    if rep:
        lines += ["```", json.dumps(rep, indent=1), "```", ""]
    choice = load("data/dpo_choice.json")
    meta = load("data/dpo_pairs_meta.json")
    lines += block("v_combo DPO", [
        "- Starts from lora_v_combo_contrast. Skip below 16 preference pairs.",
        "- Pairs: exp5 distill_pairs + exp5 contrast_pairs + exp6 contrast_pairs. If winner is (a), a 20% facts CE mix is appended (rows without bad).",
        "- Loss: local DPO, beta 0.1, sum of completion-token logprobs, ref cached from init. Dropout off. lr 5e-6, 1 epoch, AdamW.",
        "- choice: " + (json.dumps(choice) if choice else "not built yet"),
        "- pairs meta: " + (json.dumps(meta) if meta else "not built yet"),
        f"- {sha_line(os.path.join(H, 'data', 'v_dpo_pairs.jsonl'))}",
    ])
    rep = load("lora_v_combo_dpo/train_report.json")
    if rep:
        lines += ["```", json.dumps(rep, indent=1), "```", ""]
    p8 = load("data/8b_probe.json")
    skip8 = load("data/8b_skipped.json")
    lines += block("optional 8B (d)", [
        f"- Base: {frozen.BASE8} from the local HF cache (exp3 m8b). No download.",
        "- Probe: load bf16, require peak <= 18.5 GB and >= 4.5 GB free. Else try 4-bit. Else skip.",
        "- Recipe: train lora_q8_facts on exp4 facts.jsonl + claims, then the winning (a) or (b) path, then the same contrast train file and DPO pairs.",
        "- probe: " + (json.dumps(p8) if p8 else "not run yet"),
        "- skipped: " + (json.dumps(skip8) if skip8 else "no"),
    ])
    for name in ("q8_facts", "v8_replay", "v8_merged", "v8_combo_contrast", "v8_combo_dpo"):
        rep = load(f"lora_{name}/train_report.json")
        if rep:
            lines += [f"### lora_{name}", "", "```", json.dumps(rep, indent=1), "```", ""]
    lines += block("Eval", [
        "- exp6/eval_cond.py calls the copied eval4.py with --base / --facts / --voice / --adapters / --rag.",
        "- n=3. Judge is the copied judge.py. A condition is skipped when runs/<cond>.json already exists.",
        "- A train stage is skipped when lora_<name>/train_report.json exists.",
        "- Reference runs q14_rag and q4_fv_rag are copied from exp4 and are not rerun.",
    ])
    lines += block("Commands", [
        "```",
        "cd ~/output-model/exp6 && nohup bash run.sh >> log.txt 2>&1 &",
        "```",
    ])
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    open(os.path.join(H, "REPRODUCE-notes.md"), "w").write(text)
    print("wrote REPRODUCE-notes.md", flush=True)

if __name__ == "__main__":
    main()
