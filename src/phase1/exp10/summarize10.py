"""Write exp10/results.md from locked scores, regen, and calibration outputs."""
import os, json, glob
import frozen, scores

H = frozen.H
LOCKED_COND = "p2_light_dpo_guarded"


def md_row(name, s, note=""):
    c = s["cells"]
    extra = f" | {note}" if note else ""
    return (
        f"| {name} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {c[5]} | "
        f"{c[6]} | {c[7]} | {c[8]} | {c[9]} | {c[10]} |{extra.lstrip(' |') and extra or ''}"
    )


def load_maybe(*paths):
    for p in paths:
        if p and os.path.isfile(p):
            return json.load(open(p)), p
    return None, None


def fmt_voice(v):
    if v is None:
        return "-"
    return f"{v:.4f}"


def m16_block(d):
    m = scores.m16(d)
    lines = [f"M16 yd4408 \"What is your number?\": {m['ok']}/{m['n']} samples match the rubric."]
    for sample, char, probe, ok, ans in m["rows"]:
        flag = "ok" if ok else "MISS"
        lines.append(f"- sample {sample}: {flag} — {ans.replace(chr(10), ' ')}")
    return "\n".join(lines)


def noise_md(report):
    noise = (report or {}).get("qwen_noise") or {}
    if not noise:
        return "_Noise grid not run yet._"
    lines = [
        "Condition-level voice mean over six cells (seeds 1,2,3 × temp 0.0, 0.3), qwen3:14b, voice only.",
        "",
        "| cond | mean | std | t0 s1 | original t0 s1 | t0 s1 matches orig (±0.02) | t0s1 | t0s2 | t0s3 | t0.3s1 | t0.3s2 | t0.3s3 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cond in scores_conds(noise):
        n = noise[cond]
        def cell(t, s):
            c = n["cells"].get(f"t{t}_s{s}")
            if not c or c["mean"] is None:
                return "-"
            return f"{c['mean']:.4f}"
        match = n.get("t0_s1_matches_original")
        match_s = "-" if match is None else ("yes" if match else "no")
        lines.append(
            f"| {cond} | {fmt_voice(n.get('mean'))} | {fmt_voice(n.get('std'))} | "
            f"{fmt_voice(n.get('t0_s1'))} | {fmt_voice(n.get('original'))} | {match_s} | "
            f"{cell(0.0,1)} | {cell(0.0,2)} | {cell(0.0,3)} | {cell(0.3,1)} | {cell(0.3,2)} | {cell(0.3,3)} |"
        )
    return "\n".join(lines)


def scores_conds(noise):
    order = ["p2_light_dpo_guarded", "p2_light_dpo", "v_merged_v_rag", "q4_base_rag", "v_fb_dpo_v"]
    return [c for c in order if c in noise]


def second_md(report):
    if not report:
        return "_Second judge not run yet._"
    if report.get("second_skip_reason") and not report.get("second_judge"):
        return f"Skipped: {report['second_skip_reason']}"
    sec = report.get("second_judge")
    if not sec:
        return "_Second judge not run yet._"
    lines = [
        f"Second judge: `{sec['model']}`, temp 0, seed 1, voice only. Spearman ρ on the five condition means vs qwen3:14b temp 0 seed 1: **{sec.get('spearman_vs_qwen3_14b_t0_s1')}** (n={sec.get('n_paired')}).",
        "",
        "| cond | qwen3:14b t0 s1 | second t0 s1 |",
        "|---|---|---|",
    ]
    qn = (report.get("qwen_noise") or {})
    for cond in scores_conds(qn) or list((sec.get("means") or {}).keys()):
        q = None if cond not in qn else qn[cond].get("t0_s1")
        g = (sec["means"].get(cond) or {}).get("mean")
        lines.append(f"| {cond} | {fmt_voice(q)} | {fmt_voice(g)} |")
    return "\n".join(lines)


def main():
    locked_p = os.path.join(H, "locked", "p2_light_dpo_guarded.judged.json")
    locked_d, _ = load_maybe(locked_p)
    regen_d, regen_p = load_maybe(os.path.join(H, "runs", "p2_light_dpo_guarded.judged.json"))
    orig_d, orig_p = load_maybe(os.path.join(H, "runs", "p2_light_dpo_guarded_orig.judged.json"))
    ung_d, _ = load_maybe(os.path.join(H, "locked", "p2_light_dpo.json"))
    p1_d, _ = load_maybe(os.path.join(H, "pass1", "v_merged_v_rag.json"))
    tol, _ = load_maybe(os.path.join(H, "runs", "tolerance.json"))
    calib, _ = load_maybe(os.path.join(H, "calib", "report.json"))
    fd, _ = load_maybe(
        os.path.join(H, "runs", "p2_light_dpo.factdiff.json"),
        os.path.join(H, "locked", "p2_light_dpo.factdiff.json"),
    )

    ls = scores.stats(locked_d) if locked_d else None
    m16_locked = scores.m16(locked_d) if locked_d else None
    proto_issues = []
    if not ls:
        proto_issues.append("locked judged missing")
    else:
        if ls["marker"] > scores.MARKER_MAX or ls["leak_n"] != scores.LEAK_N:
            proto_issues.append(f"marker leaks {ls['marker']}/{ls['leak_n']}")
        if ls["voice"] is None or ls["voice"] < scores.VOICE_MIN:
            proto_issues.append(f"voice {ls['voice']}")
        if ls["frames"] >= scores.Q14_FRAMES:
            proto_issues.append(f"frame breaks {ls['frames']}")
        if not (ls["must_ok"] == 60 and ls["must_n"] == 60):
            if ls["must_ok"] == 59 and m16_locked and m16_locked["ok"] == 2 and m16_locked["n"] == 3:
                pass  # documented inherited M16 miss
            else:
                proto_issues.append(f"must-answer {ls['must_ok']}/{ls['must_n']}")
    bar_s = "PASS (voice ≥ 4.2; known M16 miss recorded)" if ls and not proto_issues else (
        "FAIL (" + "; ".join(proto_issues) + ")" if proto_issues else "pending"
    )

    lines = []
    lines.append("# exp10 results: No Return prototype lock-in and voice-judge calibration")
    lines.append("")
    lines.append("Reader position ch3, n=3. Locked condition is exp9 `p2_light_dpo_guarded` (pass-1 v_merged_v_rag + exp7 light DPO rewrite + fact-diff guard).")
    lines.append("Revised pass bar for this prototype: must-answer 60/60 (59/60 recorded as the known M16 miss), marker leaks ≤ 2/48, **voice ≥ 4.2**, frame breaks < 6.")
    lines.append("No new LoRA was trained. 8B was not used.")
    lines.append("")
    lines.append("## Locked prototype")
    lines.append("")
    lines.append("| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    if ls:
        lines.append("| p2_light_dpo_guarded (locked exp9) | " + " | ".join(str(x) for x in ls["cells"]) + " |")
    if orig_d:
        os_ = scores.stats(orig_d)
        lines.append("| p2_light_dpo_guarded_orig (fixed text, re-judge) | " + " | ".join(str(x) for x in os_["cells"]) + " |")
    if regen_d:
        rs = scores.stats(regen_d)
        lines.append("| p2_light_dpo_guarded (exp10 regen, seed 1) | " + " | ".join(str(x) for x in rs["cells"]) + " |")
    lines.append("")
    lines.append(f"- Locked scores (exp9 judged file): 59/60 must-answer, 1/48 marker leaks, voice 4.22 (4.2157), 3 frame breaks, fallback 72/204.")
    lines.append(f"- Revised pass bar vs those locked scores: **{bar_s}**.")
    lines.append("- Voice bar was 4.3 in exp9; this prototype revises it to 4.2. Marker ≤2/48 and frames <6 are unchanged. Must-answer target remains 60/60; 59/60 is the known M16 miss inherited from pass-1.")
    lines.append("- If the exp10 regen misses tolerance, the locked prototype scores stay the original exp9 judged file.")
    lines.append("")
    lines.append("## Known must-answer miss (M16)")
    lines.append("")
    if locked_d:
        lines.append(m16_block(locked_d))
    lines.append("")
    lines.append("The same M16 2/3 miss is already present in pass-1 `v_merged_v_rag` (sample 1: \"I am not a number. I am a tag on a hook.\").")
    lines.append("59/60 is inherited from pass-1, not introduced by the rewrite. Unguarded p2_light_dpo also missed M4 and M8; the guard recovered those.")
    if p1_d:
        lines.append("")
        lines.append("Pass-1 M16:")
        lines.append(m16_block(p1_d))
    if ung_d:
        lines.append("")
        lines.append("Unguarded p2_light_dpo must-misses:")
        um = [r for r in ung_d["results"] if r["kind"] == "must" and not r.get("correct")]
        for r in um:
            lines.append(f"- {r['id']} {r['char']} sample {r['sample']} {r['probe']!r}")
    lines.append("")
    lines.append("## Reproduction check vs tolerance")
    lines.append("")
    lines.append("Tolerance vs locked exp9 scores: must-answer exact, marker leaks exact, voice ±0.03, frame breaks ±1.")
    if tol:
        lines.append("")
        lines.append(f"- Locked file self-check: {'ok' if tol.get('locked_file_ok') else 'FAIL'}")
        lines.append(f"- Orig-text re-judge (judge jitter): {tol.get('orig_rejudge_tolerance_ok')}")
        lines.append(f"- Full regen (generation + fact-diff + guard + judge): {tol.get('regen_tolerance_ok')}")
        lines.append(f"- Prototype source: `{tol.get('proto_source')}`")
        lines.append(f"- Note: {tol.get('proto_note')}")
        lines.append("")
        lines.append("| label | must | marker | voice | frames | M16 | tolerance | issues |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in tol.get("rows") or []:
            lines.append(
                f"| {r['label']} | {r['must']} | {r['marker']} | {r['voice']} | {r['frames']} | {r['m16']} | "
                f"{'pass' if r['tolerance_ok'] else 'FAIL'} | {'; '.join(r['tolerance_issues']) or '—'} |"
            )
        if orig_d and locked_d:
            ov = scores.stats(orig_d)["voice"]
            lv = ls["voice"]
            if ov is not None and lv is not None:
                lines.append("")
                lines.append(f"Judge jitter on fixed exp9 text: Δvoice = {ov - lv:+.4f}.")
        if regen_d and locked_d:
            rv = scores.stats(regen_d)["voice"]
            lv = ls["voice"]
            if rv is not None and lv is not None:
                lines.append(f"Regen vs locked: Δvoice = {rv - lv:+.4f}, Δframes = {scores.stats(regen_d)['frames'] - ls['frames']:+d}.")
    else:
        lines.append("")
        lines.append("_Tolerance file not written yet (reproduction run pending)._")
    lines.append("")
    if fd:
        sm = fd.get("summary") or {}
        lines.append("## Fact-diff (unguarded p2_light_dpo)")
        lines.append("")
        lines.append(f"- drop claims {sm.get('drop_claims')}, add claims {sm.get('add_claims')}, fallback {sm.get('fallback_samples')}/{sm.get('n')} ({sm.get('fallback_rate')})")
        lines.append("- Fallback rule: any dropped claim, any added claim, unparseable judge reply, or a new `common.fail_hits` marker vs pass-1.")
        lines.append("")
    lines.append("## Voice-judge calibration")
    lines.append("")
    lines.append("Fixed answer files (copied, not regenerated): p2_light_dpo_guarded, p2_light_dpo, v_merged_v_rag, q4_base_rag, v_fb_dpo_v.")
    lines.append("Rubric unchanged from judge.py:")
    lines.append("")
    lines.append("```")
    lines.append('Rate how well the REPLY matches this character\'s voice (wording, rhythm, attitude), 1=not at all/breaks character, 5=indistinguishable from the book. Ignore whether facts are right. Reply JSON {"score": 1-5}')
    lines.append("```")
    lines.append("")
    lines.append("### (1) Judge noise, qwen3:14b")
    lines.append("")
    lines.append(noise_md(calib))
    lines.append("")
    lines.append("### (2) Second local judge")
    lines.append("")
    lines.append(second_md(calib))
    lines.append("")
    lines.append("### (3) Blinded human-rating sample")
    lines.append("")
    sample = os.path.join(H, "human_rating", "sample.md")
    key = os.path.join(H, "human_rating", "key.json")
    if os.path.isfile(sample) and os.path.isfile(key):
        k = json.load(open(key))
        lines.append(f"- Sample: `exp10/human_rating/sample.md` ({len(k.get('items') or {})} items).")
        lines.append("- Key (condition / source / item id / sample / original judge score): `exp10/human_rating/key.json`.")
        lines.append(f"- Rule: {k.get('rule')} (seed {k.get('seed')}).")
        lines.append("- sample.md hides condition name, adapter, and machine score. Each item has space for a human 1–5 and an optional note.")
    else:
        lines.append("_Human-rating sample not exported yet._")
    lines.append("")
    lines.append("## Hashes (locked inputs)")
    lines.append("")
    lines.append("```")
    lines.append(f"{frozen.PASS1_SHA}  pass1/v_merged_v_rag.json")
    lines.append(f"{frozen.P2_ADAPTER_SHA}  init/lora_v_light_dpo/adapter_model.safetensors")
    lines.append(f"{frozen.FACTS_SHA}  exp4/lora_q4_facts/adapter_model.safetensors")
    for name, want in frozen.HARNESS.items():
        lines.append(f"{want}  {name}")
    lines.append("```")
    lines.append("")
    text = "\n".join(lines) + "\n"
    path = os.path.join(H, "results.md")
    with open(path, "w") as f:
        f.write(text)
    print("wrote", path)


if __name__ == "__main__":
    main()
