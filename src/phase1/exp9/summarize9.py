"""Write exp9/results.md. Reference rows are copied judged files, not reruns."""
import json, os, glob
import scores
import frozen

H = frozen.H
REFS = ["q4_fv_rag", "q14_rag", "v_merged_v_rag", "v_rag_dpo_v_rag", "q4_base_rag"]
PRIMARY = ["p2_fb_dpo", "p2_light_contrast", "p2_light_dpo", "p2_v_merged"]
REF_NOTE = {
    "q4_fv_rag": "exp4 reference, not rerun",
    "q14_rag": "exp4 voice/frame reference, not a required row, copied for the frame bar",
    "v_merged_v_rag": "exp6 reference, not rerun; pass-1 source",
    "v_rag_dpo_v_rag": "exp8 reference, not rerun",
    "q4_base_rag": "exp8 reference, not rerun",
}
REF_EXPECT = {
    "q4_fv_rag": (1, 60, 6, 3.9),
    "q14_rag": (2, 60, 6, 4.3),
    "v_merged_v_rag": (1, 59, 3, 4.2),
    "v_rag_dpo_v_rag": (0, 52, 2, 4.21),
    "q4_base_rag": (2, 59, 12, 3.89),
}

def find(cond):
    for suf in (".judged.json", ".json"):
        p = os.path.join(H, "runs", cond + suf)
        if os.path.isfile(p):
            return p
    return None

def check_known():
    for cond, (marker, must, frames, voice) in REF_EXPECT.items():
        path = os.path.join(H, "runs", cond + ".judged.json")
        s = scores.stats(scores.load_json(path))
        got = (s["marker"], s["must_ok"], s["frames"], round(s["voice"], 2))
        if got != (marker, must, frames, voice):
            raise SystemExit(f"reference {cond} drifted: {got} != {(marker, must, frames, voice)}")

def discovered():
    extra = []
    for path in sorted(glob.glob(os.path.join(H, "runs", "p2_*.json"))):
        base = os.path.basename(path)
        if base.endswith(".judged.json") or ".factdiff" in base or base.endswith(".bad.json"):
            continue
        cond = base[:-len(".json")]
        if cond in PRIMARY or cond in extra:
            continue
        if cond.endswith("_guarded") and cond[: -len("_guarded")] in PRIMARY:
            continue
        extra.append(cond)
    return extra

def fact_line(cond):
    path = os.path.join(H, "runs", cond + ".factdiff.json")
    parent = cond[:-len("_guarded")] if cond.endswith("_guarded") else cond
    if cond.endswith("_guarded"):
        gpath = os.path.join(H, "runs", cond + ".json")
        if not os.path.isfile(gpath):
            return None
        meta = json.load(open(gpath)).get("meta") or {}
        parent_fd = os.path.join(H, "runs", parent + ".factdiff.json")
        parent_sum = json.load(open(parent_fd))["summary"] if os.path.isfile(parent_fd) else None
        return dict(
            cond=cond,
            kind="guarded",
            fallback=meta.get("n_fallback"),
            n=meta.get("n_fallback", 0) + meta.get("n_kept", 0) if meta.get("n_kept") is not None else None,
            kept=meta.get("n_kept"),
            parent_drop_claims=None if not parent_sum else parent_sum["drop_claims"],
            parent_add_claims=None if not parent_sum else parent_sum["add_claims"],
            parent_drop_rate=None if not parent_sum else parent_sum["drop_sample_rate"],
            parent_add_rate=None if not parent_sum else parent_sum["add_sample_rate"],
            scored_drop_claims=0,
            scored_add_claims=0,
            note="scored text drops/adds are 0 by construction: failing rewrites fall back to pass-1",
        )
    if not os.path.isfile(path):
        return None
    summary = json.load(open(path))["summary"]
    return dict(cond=cond, kind="rewrite", **{k: summary[k] for k in (
        "n", "drop_claims", "add_claims", "drop_samples", "add_samples",
        "drop_sample_rate", "add_sample_rate", "new_marker_samples", "parse_fail", "fallback_samples", "fallback_rate",
    )})

def per_item_lines(cond, limit=24):
    path = os.path.join(H, "runs", cond + ".factdiff.json")
    if not os.path.isfile(path):
        return []
    per = json.load(open(path))["summary"]["per_item"]
    rows = []
    for key, slot in per.items():
        if slot["drop_samples"] or slot["add_samples"] or slot["new_marker_samples"]:
            rows.append((slot["drop_claims"] + slot["add_claims"], key, slot))
    rows.sort(reverse=True)
    lines = []
    for _, key, slot in rows[:limit]:
        lines.append(
            f"  - {key}: drop {slot['drop_samples']}/{slot['n']} ({slot['drop_claims']} claims), "
            f"add {slot['add_samples']}/{slot['n']} ({slot['add_claims']} claims), "
            f"new-marker {slot['new_marker_samples']}/{slot['n']}"
        )
    if len(rows) > limit:
        lines.append(f"  - … {len(rows) - limit} more probes in {os.path.basename(path)}")
    if not lines:
        lines.append("  - no probe had a drop, an add, or a new marker")
    return lines

def main():
    os.chdir(H)
    check_known()
    guarded = [c + "_guarded" for c in PRIMARY]
    order = []
    for cond in REFS + PRIMARY + guarded + discovered():
        if cond not in order:
            order.append(cond)
    packed = []
    for cond in order:
        path = find(cond)
        if not path:
            continue
        s = scores.stats(scores.load_json(path))
        packed.append((cond, [cond] + s["cells"], s))
    q14_frames = 6
    lines = [
        "# exp9 results: two-pass pipeline (RAG facts, then voice rewrite)",
        "",
        "Reader position ch3, n=3. Pass 1 is cached RAG answers. Pass 2 rewrites them with a non-RAG voice LoRA on exp6/merged_q4_facts. "
        "The scored text is the rewrite, or the pass-1 text when the guarded condition falls back. "
        "q4_fv_rag, q14_rag, v_merged_v_rag, v_rag_dpo_v_rag, and q4_base_rag are copied judged files and were not rerun. "
        f"Pass bar for every scored two-pass condition: must-answer 60/60, marker leaks <= 2/48, voice >= 4.3, frame breaks < q14_rag ({q14_frames}).",
        "",
        "| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |",
        "|" + "---|" * 12,
    ]
    lines += ["| " + " | ".join(map(str, cells)) + " |" for _, cells, _ in packed]
    lines += ["", "## Verdict", ""]
    any_pass = False
    best = None
    for cond, cells, s in packed:
        if cond in REF_NOTE:
            voice = "-" if s["voice"] is None else round(s["voice"], 2)
            lines.append(
                f"- {cond}: reference ({REF_NOTE[cond]}; voice {voice}, frame breaks {s['frames']}, "
                f"must-answer {s['must_ok']}/{s['must_n']}, marker {s['marker']}/{s['leak_n']})"
            )
            continue
        if not s["judged"]:
            lines.append(f"- {cond}: pending judge (must-answer {s['must_ok']}/{s['must_n']}, frame breaks {s['frames']}, marker {s['marker']}/{s['leak_n']})")
            continue
        checks = scores.failures(s, q14_frames)
        if checks:
            lines.append(f"- {cond}: FAIL ({'; '.join(checks)})")
        else:
            any_pass = True
            lines.append(f"- {cond}: PASS")
        if best is None or scores.rank_key(s, q14_frames) > scores.rank_key(best[1], q14_frames):
            best = (cond, s)
    planned = PRIMARY + guarded
    have = {c for c, _, _ in packed}
    for cond in planned:
        if cond not in have:
            lines.append(f"- {cond}: pending")
    if any_pass:
        lines += ["", "At least one two-pass condition clears the bar."]
    else:
        lines += ["", "No two-pass condition clears the bar yet."]
    lines += ["", "## Fact preservation", ""]
    lines.append("Fact-drop and fact-add counts are from qwen3:14b temp 0 seed 1. Rates are samples with at least one such claim, over the rows in that run.")
    lines.append("")
    lines.append("| cond | drop claims | add claims | drop samples | add samples | new markers | fallback |")
    lines.append("|---|---|---|---|---|---|---|")
    fact_docs = []
    for cond, _, _ in packed:
        if cond in REFS:
            continue
        info = fact_line(cond)
        if not info:
            continue
        fact_docs.append(info)
        if info["kind"] == "rewrite":
            lines.append(
                f"| {cond} | {info['drop_claims']} | {info['add_claims']} | "
                f"{info['drop_samples']}/{info['n']} ({info['drop_sample_rate']}) | "
                f"{info['add_samples']}/{info['n']} ({info['add_sample_rate']}) | "
                f"{info['new_marker_samples']} | {info['fallback_samples']}/{info['n']} |"
            )
        else:
            lines.append(
                f"| {cond} | 0 scored (parent {info['parent_drop_claims']}) | 0 scored (parent {info['parent_add_claims']}) | "
                f"0 scored | 0 scored | 0 new vs pass-1 | {info['fallback']}/{info['n']} |"
            )
    if not fact_docs:
        lines.append("| (none yet) | | | | | | |")
    lines += ["", "### Per-item rates", ""]
    shown = False
    for cond, _, _ in packed:
        if cond in REFS or cond.endswith("_guarded"):
            continue
        path = os.path.join(H, "runs", cond + ".factdiff.json")
        if not os.path.isfile(path):
            continue
        shown = True
        summary = json.load(open(path))["summary"]
        lines.append(
            f"- {cond}: drop rate {summary['drop_sample_rate']} ({summary['drop_samples']}/{summary['n']}), "
            f"add rate {summary['add_sample_rate']} ({summary['add_samples']}/{summary['n']}), "
            f"fallback rate {summary['fallback_rate']}"
        )
        lines.extend(per_item_lines(cond))
    if not shown:
        lines.append("- no fact-diff yet")
    lines += ["", "## Stage decisions", ""]
    dec = os.path.join(H, "data", "decide.json")
    if os.path.isfile(dec):
        lines.append("```")
        lines.append(open(dec).read().rstrip())
        lines.append("```")
    else:
        lines.append("- decide.json not written yet")
    lines += ["", "## Rewrite training", ""]
    report = os.path.join(H, "lora_v_rewrite", "train_report.json")
    meta = os.path.join(H, "data", "rewrite_meta.json")
    if os.path.isfile(meta):
        lines.append(f"- data: `{open(meta).read().strip()}`")
    else:
        lines.append("- data: not built (training has not been requested)")
    if os.path.isfile(report):
        lines.append(f"- lora_v_rewrite: `{open(report).read().strip()}`")
    else:
        lines.append("- lora_v_rewrite: not trained")
    best_doc = dict(cond=None, voice=None, frames=None, must=None, marker=None, pass_=False, pass_flag=False)
    if best is not None:
        cond, s = best
        best_doc = dict(
            cond=cond,
            voice=s["voice"],
            frames=s["frames"],
            must=f"{s['must_ok']}/{s['must_n']}",
            marker=f"{s['marker']}/{s['leak_n']}",
            pass_=not scores.failures(s, q14_frames),
        )
        best_doc["pass"] = best_doc["pass_"]
    # `pass` is reserved; write the key the commit script reads.
    out_best = dict(best_doc)
    out_best["pass"] = bool(out_best.get("pass_"))
    frozen.dump_json(os.path.join(H, "data", "best.json"), out_best)
    lines += [
        "",
        f"Best scored two-pass row by the pass-bar rank: `{json.dumps(out_best)}`.",
        "",
        "Pass 1 for the primary conditions is exp6 v_merged_v_rag (copied, not rerun). "
        "Pass 2 loads only the voice LoRA on exp6/merged_q4_facts. "
        "Marker leak, frame break, and must-answer scoring match exp4. "
        "Voice and leak judges, and the fact-diff, use qwen3:14b temp 0 seed 1.",
        "",
    ]
    text = "\n".join(lines)
    open(os.path.join(H, "results.md"), "w").write(text)
    print(text)

if __name__ == "__main__":
    main()
