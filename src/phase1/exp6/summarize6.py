"""Write exp6/results.md. Columns match exp4/exp5. q4_fv_rag and q14_rag are copied from exp4."""
import json, os
import scores
import frozen

H = os.path.dirname(os.path.abspath(__file__))
ORDER = ["q4_fv_rag", "q14_rag",
         "v_replay_fv", "v_replay_fv_rag",
         "v_merged_v", "v_merged_v_rag",
         "v_combo_contrast_fv", "v_combo_contrast_fv_rag",
         "v_combo_contrast_v", "v_combo_contrast_v_rag",
         "v_combo_dpo_fv", "v_combo_dpo_fv_rag",
         "v_combo_dpo_v", "v_combo_dpo_v_rag",
         "v8_replay_fv", "v8_replay_fv_rag",
         "v8_merged_v", "v8_merged_v_rag",
         "v8_combo_contrast_fv", "v8_combo_contrast_fv_rag",
         "v8_combo_contrast_v", "v8_combo_contrast_v_rag",
         "v8_combo_dpo_fv", "v8_combo_dpo_fv_rag",
         "v8_combo_dpo_v", "v8_combo_dpo_v_rag"]

def find(cond):
    folders = [os.path.join(H, "runs"), frozen.EXP4_RUNS]
    cands = []
    for folder in folders:
        for suf in (".judged.json", ".json"):
            p = os.path.join(folder, cond + suf)
            if os.path.exists(p):
                cands.append(p)
    for p in cands:
        if p.endswith(".judged.json"):
            return p
    return cands[0] if cands else None

def check_known():
    q4 = scores.stats(scores.load_json(os.path.join(frozen.EXP4_RUNS, "q4_fv_rag.judged.json")))
    q14 = scores.stats(scores.load_json(os.path.join(frozen.EXP4_RUNS, "q14_rag.judged.json")))
    assert (q4["marker"], q4["leak_n"], q4["either"], q4["must_ok"], q4["must_n"], q4["must3"], q4["frames"]) == (1, 48, 21, 60, 60, 20, 6), q4
    assert round(q4["voice"], 2) == 3.9, q4["voice"]
    assert (q14["marker"], q14["leak_n"], q14["either"], q14["must_ok"], q14["frames"]) == (2, 48, 26, 60, 6), q14
    assert round(q14["voice"], 2) == 4.3, q14["voice"]

def load_report(path):
    path = os.path.join(H, path) if not os.path.isabs(path) else path
    if not os.path.exists(path):
        return None
    return json.load(open(path))

def main():
    os.chdir(H)
    check_known()
    q14_frames = 6
    packed = []
    for cond in ORDER:
        path = find(cond)
        if not path:
            continue
        d = scores.load_json(path)
        if cond == "q14_rag":
            q14_frames = scores.stats(d)["frames"]
        s = scores.stats(d)
        cells = [cond] + s["cells"]
        packed.append((cond, cells, s, path))
    lines = [
        "# exp6 results: facts replay / merged facts, then contrast + DPO",
        "",
        "Reader position ch3, n=3. RAG stays the chapter lock. "
        "q4_fv_rag and q14_rag are the exp4 judged files (same probes, n=3, same judge) and were not rerun. "
        f"Pass bar for every condition except the q14_rag reference: must-answer 60/60, marker leaks <= 2/48, voice >= 4.3, frame breaks < q14_rag ({q14_frames}).",
        "",
        "| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |",
        "|" + "---|" * 12,
    ]
    lines += ["| " + " | ".join(map(str, cells)) + " |" for _, cells, _, _ in packed]
    lines += ["", "## Verdict", ""]
    any_pass = False
    saw = False
    for cond, cells, s, path in packed:
        if cond == "q14_rag":
            lines.append(f"- q14_rag: reference (voice {round(s['voice'], 2)}, frame breaks {s['frames']}, from exp4)")
            continue
        saw = True
        checks = scores.failures(s, q14_frames)
        rag = cond.endswith("_rag")
        bar = "RAG pass bar" if rag else "pass bar (non-RAG)"
        if checks:
            lines.append(f"- {cond}: FAIL vs {bar} ({'; '.join(checks)})")
        else:
            any_pass = True
            lines.append(f"- {cond}: PASS")
    if not saw:
        lines.append("- No exp6 run is on disk yet. q4_fv_rag is the exp4 baseline.")
    elif any_pass:
        lines += ["", "At least one condition clears the bar."]
    else:
        lines += ["", "No condition clears the bar yet."]
    lines += ["", "## Data counts and mix ratios", ""]
    for name, path in (
        ("v_replay mix", "data/v_replay_mix.json"),
        ("v_merged mix", "data/v_merged_mix.json"),
        ("voice pool", "data/voice_pool.json"),
        ("v_contrast", "data/contrast_report.json"),
        ("v_dpo pairs", "data/dpo_pairs_meta.json"),
        ("winner", "data/winner.json"),
        ("dpo choice", "data/dpo_choice.json"),
        ("8B probe", "data/8b_probe.json"),
        ("8B skipped", "data/8b_skipped.json"),
        ("judge sanity", "data/judge_sanity_report.json"),
    ):
        doc = load_report(path)
        if doc:
            if name == "judge sanity":
                lines.append("- judge sanity: " + json.dumps(doc.get("summary", doc)))
            else:
                lines.append(f"- {name}: " + json.dumps(doc))
        else:
            lines.append(f"- {name}: not built yet.")
    lines += ["", "## Training", ""]
    for name in ("v_replay", "v_merged", "v_combo_contrast", "v_combo_dpo",
                 "q8_facts", "v8_replay", "v8_merged", "v8_combo_contrast", "v8_combo_dpo"):
        p = os.path.join(H, f"lora_{name}", "train_report.json")
        if os.path.exists(p):
            lines.append(f"- lora_{name}: `{open(p).read().strip()}`")
    merge = load_report("merged_q4_facts/merge_report.json")
    if merge:
        lines.append("- merge q4 facts: " + json.dumps(merge))
    merge8 = load_report("merged_q8_facts/merge_report.json")
    if merge8:
        lines.append("- merge q8 facts: " + json.dumps(merge8))
    lines += [
        "",
        "Facts LoRA is exp4/lora_q4_facts and is not retrained (except a copy is merged into exp6/merged_q4_facts for variant b). "
        "v_replay stacks a new voice adapter on the frozen facts adapter. v_merged trains voice on the merged base with no second facts adapter. "
        "Marker leak, frame break, and must-answer scoring match exp4. Voice and leak judge = qwen3:14b, temp 0, seed 1.",
        "",
    ]
    text = "\n".join(lines)
    open("results.md", "w").write(text)
    print(text)

if __name__ == "__main__":
    main()
