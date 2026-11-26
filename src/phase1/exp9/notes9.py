"""Reproduction notes for exp9. Rewritten from code and artifacts so a resume stays accurate."""
import json, os, subprocess
import frozen
import prompts

H = frozen.H

def load(path):
    path = path if os.path.isabs(path) else os.path.join(H, path)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)

def sha_line(path):
    if not os.path.isfile(path):
        return f"(missing)  {path}"
    n = ""
    if path.endswith(".jsonl"):
        with open(path) as f:
            n = str(sum(1 for line in f if line.strip())) + " rows  "
    rel = os.path.relpath(path, H)
    return f"{n}{frozen.sha256(path)}  {rel}"

def main():
    os.chdir(H)
    py = subprocess.check_output(
        [os.path.join(frozen.ROOT, "exp3", ".venv", "bin", "python"), "-c",
         "import torch,sys; print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda)"],
        text=True,
    ).strip().splitlines()
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    except Exception as e:
        gpu = str(e)
    lines = [
        "# exp9 reproduction notes",
        "",
        "Two-pass pipeline. Pass 1 reuses a cached RAG answer. Pass 2 rewrites that answer with a non-RAG voice LoRA. "
        "exp3 through exp8 are not modified. The facts LoRA is not retrained. 8B is not used. trl is not installed.",
        "",
        f"Facts adapter sha256: `{frozen.sha256(os.path.join(frozen.FACTS, 'adapter_model.safetensors'))}`",
        f"Expected: `{frozen.FACTS_SHA}`",
        "",
        "Harness copies (must match exp4/REPRODUCE.md). chat.py is copied because generation uses chat.render, the same helper eval4.py imports.",
        "",
        "```",
    ]
    for name, want in frozen.HARNESS.items():
        got = frozen.sha256(os.path.join(H, name))
        lines.append(f"{got}  {name}  {'OK' if got == want else 'MISMATCH'}")
    lines += ["```", ""]
    lines += [
        "## Environment",
        "",
        f"- Python {py[0]}, torch {py[1]}, cuda {py[2]}",
        f"- GPU: {gpu}",
        "- venv: ~/output-model/exp3/.venv",
        "- Judge and fact-diff: local ollama qwen3:14b, temp 0, seed 1, think off. Frame regex is common.py FRAME. Marker scan is common.fail_hits.",
        f"- Base: {frozen.MERGED} (Qwen/Qwen3-4B revision {frozen.BASE_REV} with exp4/lora_q4_facts merged). Loaded with local_files_only. Not overwritten.",
        "- 8B was not trained or evaluated.",
        "",
        "## Pass 1",
        "",
        "Primary source is exp6 `v_merged_v_rag` (merged_q4_facts + lora_v_merged + RAG). "
        "The answers are the copy in `pass1/v_merged_v_rag.json`. They were not regenerated.",
        "",
        f"- {sha_line(os.path.join(H, 'pass1', 'v_merged_v_rag.json'))}",
        f"- source: {frozen.PASS1_SRC['v_merged_v_rag']}",
        "",
        "Optional second source, used only when decide.py says the primary rewrites are clearly short on must-answer: "
        "exp8 `q4_merged_rag` (merged facts, RAG, no voice LoRA).",
        "",
        f"- {sha_line(os.path.join(H, 'pass1', 'q4_merged_rag.json'))}",
        f"- source: {frozen.PASS1_SRC['q4_merged_rag']}",
        "",
        "Retrieval stays inside those cached runs. Pass 2 does not call common.retrieve or common.rag_sysprompt.",
        "",
        "## Pass 2 rewrite prompt",
        "",
        "System, for wren / diane / yd4408: `common.sysprompt(char)` with day=3 (the reader is at chapter 3). "
        "Narrator rows, used only if a training pair has char narrator: the NARR card in prompts.py. Not rag_sysprompt.",
        "",
        "User message, in this order: the reader question, a blank line, the pass-1 answer, a blank line, then this instruction:",
        "",
        "```",
        prompts.REWRITE_INSTR,
        "```",
        "",
        "Generation matches eval4: max_new_tokens=200, do_sample, temperature 0.7, top_p 0.95. One rewrite per cached sample (n=3).",
        "",
        "Voice adapters are read-only copies under exp9/init (final adapter files only, not earlier checkpoints):",
        "",
    ]
    for name, src in frozen.ADAPTER_SRC.items():
        dest = os.path.join(H, "init", name, "adapter_model.safetensors")
        lines.append(f"- {name}: `{frozen.sha256(dest) if os.path.isfile(dest) else 'missing'}`  from {src}")
    lines += [
        "",
        "Condition map: p2_fb_dpo → init/lora_v_fb_dpo, p2_light_contrast → init/lora_v_light_contrast, "
        "p2_light_dpo → init/lora_v_light_dpo, p2_v_merged → init/lora_v_merged. "
        "A `_q4merged` suffix swaps in the q4_merged_rag pass-1 file. "
        "p2_rewrite uses lora_v_rewrite if (c) trained. p2_rewrite_cN uses checkpoint-N.",
        "",
        "## Fact-diff prompt",
        "",
        "Model qwen3:14b, temperature 0, seed 1, JSON format, num_predict 512. "
        "A sample fails the check when the judge lists any dropped claim, any added claim, the reply does not parse, "
        "or pass-2 matches a common.fail_hits pattern that pass-1 did not. "
        "The guarded condition substitutes the pass-1 text for those samples and is judged again with judge.py.",
        "",
        "```",
        prompts.FACT_DIFF_HEAD.rstrip(),
        "```",
        "",
        "The question, PASS1, and PASS2 are appended after that text. Counts and per-item rates are in `runs/<cond>.factdiff.json` and results.md.",
        "",
        "## Commands",
        "",
        "```",
        "cd ~/output-model/exp9 && bash run.sh smoke",
        "cd ~/output-model/exp9 && nohup bash run.sh >> log.txt 2>&1 &",
        "# single condition: $P eval_twopass.py p2_fb_dpo --n 3",
        "# $P is ~/output-model/exp3/.venv/bin/python",
        "# then: $P factdiff.py runs/p2_fb_dpo.json && $P guard.py p2_fb_dpo && $P judge.py runs/p2_fb_dpo.json runs/p2_fb_dpo_guarded.json",
        "```",
        "",
        "## Hyperparameters",
        "",
        "```",
        open(os.path.join(H, "hparams.json")).read().rstrip(),
        "```",
        "",
    ]
    meta = load("data/rewrite_meta.json")
    report = load("lora_v_rewrite/train_report.json")
    lines.append("## Rewrite LoRA data and train")
    lines.append("")
    if meta:
        lines.append(f"- pairs: {sha_line(meta['path'])}")
        lines.append(f"- meta: `{json.dumps(meta)}`")
    else:
        lines.append("- pairs: not built. (c) trains only when a two-pass row is close to the bar.")
    lines.append("")
    if report:
        lines.append("Train report (local CE from exp8/train8.py, no trl):")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(report, indent=1))
        lines.append("```")
    else:
        lines.append("Not trained, or the train report is not written yet. See data/decide.json for the skip reason when the run has reached that stage.")
    lines += ["", "## Decisions", ""]
    dec = load("data/decide.json")
    if dec:
        lines.append("```")
        lines.append(json.dumps(dec, indent=1))
        lines.append("```")
    else:
        lines.append("- not written yet")
    lines.append("")
    text = "\n".join(lines)
    open(os.path.join(H, "REPRODUCE-notes.md"), "w").write(text)
    print("wrote REPRODUCE-notes.md", len(text), "chars")

if __name__ == "__main__":
    main()
