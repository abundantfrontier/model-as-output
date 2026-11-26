"""Write exp7/results.md. Reference rows are copied judged files, not reruns."""
import json, os, glob
import scores
import frozen

H = os.path.dirname(os.path.abspath(__file__))
REFS = ["q4_fv_rag", "q14_rag", "v_merged_v_rag", "v_combo_dpo_v_rag"]
OURS = [
    "v_light_dpo_v", "v_light_dpo_v_rag",
    "v_light_contrast_v", "v_light_contrast_v_rag",
    "v_light_combo_v", "v_light_combo_v_rag",
    "v_light_dpo_replay_v", "v_light_dpo_replay_v_rag",
]
REF_NOTE = {
    "q4_fv_rag": "exp4 reference, not rerun",
    "q14_rag": "exp4 voice/frame reference, not rerun",
    "v_merged_v_rag": "exp6 reference, not rerun",
    "v_combo_dpo_v_rag": "exp6 reference, not rerun",
}

def find(cond):
    for suf in (".judged.json", ".json"):
        p = os.path.join(H, "runs", cond + suf)
        if os.path.exists(p):
            return p
    return None

def check_known():
    q4 = scores.stats(scores.load_json(os.path.join(H, "runs", "q4_fv_rag.judged.json")))
    q14 = scores.stats(scores.load_json(os.path.join(H, "runs", "q14_rag.judged.json")))
    vm = scores.stats(scores.load_json(os.path.join(H, "runs", "v_merged_v_rag.judged.json")))
    vc = scores.stats(scores.load_json(os.path.join(H, "runs", "v_combo_dpo_v_rag.judged.json")))
    assert (q4["marker"], q4["must_ok"], q4["frames"]) == (1, 60, 6), q4
    assert round(q4["voice"], 2) == 3.9
    assert (q14["marker"], q14["must_ok"], q14["frames"]) == (2, 60, 6), q14
    assert round(q14["voice"], 2) == 4.3
    assert (vm["marker"], vm["must_ok"], vm["frames"]) == (1, 59, 3), vm
    assert round(vm["voice"], 2) == 4.2
    assert (vc["marker"], vc["must_ok"], vc["frames"]) == (2, 60, 7), vc
    assert round(vc["voice"], 2) == 4.18

def discovered():
    extra = []
    for path in sorted(glob.glob(os.path.join(H, "runs", "*_c*_v*.judged.json"))):
        cond = os.path.basename(path).replace(".judged.json", "")
        if cond not in OURS and cond not in extra:
            extra.append(cond)
    return extra

def load_report(path):
    path = os.path.join(H, path) if not os.path.isabs(path) else path
    if not os.path.exists(path):
        return None
    return json.load(open(path))

def main():
    os.chdir(H)
    check_known()
    q14_frames = 6
    order = REFS + OURS + discovered()
    packed = []
    for cond in order:
        path = find(cond)
        if not path:
            continue
        d = scores.load_json(path)
        s = scores.stats(d)
        if cond == "q14_rag":
            q14_frames = s["frames"]
        packed.append((cond, [cond] + s["cells"], s))
    lines = [
        "# exp7 results: light DPO / contrast on v_merged",
        "",
        "Reader position ch3, n=3. RAG stays the chapter lock. "
        "q4_fv_rag, q14_rag, v_merged_v_rag, and v_combo_dpo_v_rag are copied judged files and were not rerun. "
        f"Pass bar for every exp7 RAG condition: must-answer 60/60, marker leaks <= 2/48, voice >= 4.3, frame breaks < q14_rag ({q14_frames}). "
        "A variant clears the bar when its `_v_rag` row passes. Non-RAG rows are reported with the same thresholds.",
        "",
        "| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |",
        "|" + "---|" * 12,
    ]
    lines += ["| " + " | ".join(map(str, cells)) + " |" for _, cells, _ in packed]
    lines += ["", "## Verdict", ""]
    any_pass = False
    for cond, cells, s in packed:
        if cond in REF_NOTE:
            lines.append(f"- {cond}: reference ({REF_NOTE[cond]}; voice {s['voice'] if s['voice'] is None else round(s['voice'], 2)}, "
                         f"frame breaks {s['frames']}, must-answer {s['must_ok']}/{s['must_n']}, marker {s['marker']}/{s['leak_n']})")
            continue
        checks = scores.failures(s, q14_frames)
        rag = cond.endswith("_rag")
        bar = "RAG pass bar" if rag else "pass bar (non-RAG)"
        if checks:
            lines.append(f"- {cond}: FAIL vs {bar} ({'; '.join(checks)})")
        else:
            if rag:
                any_pass = True
            lines.append(f"- {cond}: PASS")
    if any_pass:
        lines += ["", "At least one RAG condition clears the bar."]
    else:
        lines += ["", "No RAG condition clears the bar yet."]
    lines += ["", "## Data counts and mix ratios", ""]
    for name, path in (
        ("v_light_dpo", "data/v_light_dpo_meta.json"),
        ("v_light_contrast", "data/v_light_contrast_report.json"),
        ("v_light_dpo_replay", "data/v_light_replay_meta.json"),
        ("sweep", "data/sweep.json"),
        ("best", "data/best.json"),
    ):
        doc = load_report(path)
        lines.append(f"- {name}: " + (json.dumps(doc) if doc else "not built yet."))
    lines += ["", "## Training", ""]
    for name in ("v_light_dpo", "v_light_contrast", "v_light_combo", "v_light_dpo_replay"):
        p = os.path.join(H, f"lora_{name}", "train_report.json")
        if os.path.exists(p):
            lines.append(f"- lora_{name}: `{open(p).read().strip()}`")
        else:
            lines.append(f"- lora_{name}: not trained yet.")
    lines += [
        "",
        "Init adapter is a copy of exp6/lora_v_merged. Base is exp6/merged_q4_facts (Qwen3-4B with exp4/lora_q4_facts merged). "
        "Facts LoRA is not retrained. Marker leak, frame break, and must-answer scoring match exp4. "
        "Voice and leak judge = qwen3:14b, temp 0, seed 1.",
        "",
    ]
    text = "\n".join(lines)
    open("results.md", "w").write(text)
    print(text)

if __name__ == "__main__":
    main()
