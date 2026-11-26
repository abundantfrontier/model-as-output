"""Relative voice bar: re-judge fixed answer texts with both judges in one session.

qwen3:14b and gemma3:12b, temp 0, seed 1, voice only, same prompts as judge.py.
Writes only under exp10/relative/ and exp10/relative_bar.md. Does not train.
Does not overwrite exp10/calib/.
"""
import os, sys, json, time, subprocess, statistics as st
from datetime import datetime
import judge_wrap, scores, frozen

H = frozen.H
OUT = os.path.join(H, "relative")
REF = "v_merged_v_rag"
JUDGES = (
    ("qwen3:14b", "qwen3_14b"),
    ("gemma3:12b", "gemma3_12b"),
)
CONDS = (
    (REF, os.path.join(H, "pass1", "v_merged_v_rag.json"), "reference (exp6, pass-1)"),
    ("p2_light_dpo_guarded", os.path.join(H, "locked", "p2_light_dpo_guarded.json"), "locked prototype (exp9)"),
    ("p2_fb_dpo_guarded", os.path.join(frozen.EXP9, "runs", "p2_fb_dpo_guarded.json"), "exp9 guarded"),
    ("p2_light_contrast_guarded", os.path.join(frozen.EXP9, "runs", "p2_light_contrast_guarded.json"), "exp9 guarded"),
    ("p2_v_merged_guarded", os.path.join(frozen.EXP9, "runs", "p2_v_merged_guarded.json"), "exp9 guarded"),
    ("p2_rewrite_guarded", os.path.join(frozen.EXP9, "runs", "p2_rewrite_guarded.json"), "exp9 guarded"),
    ("p2_fb_dpo_q4merged_guarded", os.path.join(frozen.EXP9, "runs", "p2_fb_dpo_q4merged_guarded.json"), "exp9 guarded, q4_merged_rag pass-1"),
)


def unload():
    if not shutil_which("ollama"):
        return
    try:
        out = subprocess.check_output(["ollama", "ps"], text=True, timeout=30)
    except Exception:
        return
    for line in out.splitlines()[1:]:
        name = line.split()[0] if line.strip() else ""
        if name:
            subprocess.run(["ollama", "stop", name], timeout=60, check=False)


def shutil_which(cmd):
    from shutil import which
    return which(cmd)


def must_bar_ok(src_stats, src_doc):
    """60/60, or 59/60 whose only miss is M16."""
    if src_stats["must_n"] != 60:
        return False, f"must n {src_stats['must_n']} != 60"
    if src_stats["must_ok"] == 60:
        return True, "60/60"
    if src_stats["must_ok"] != 59:
        return False, f"must-answer {src_stats['must_ok']}/60 (only M16 59/60 is allowed)"
    missed = []
    by = {}
    for r in src_doc["results"]:
        if r.get("kind") != "must":
            continue
        by.setdefault(r.get("id"), []).append(r)
    for mid, rows in by.items():
        if sum(bool(r.get("correct")) for r in rows) < len(rows):
            missed.append(mid)
    if missed == ["M16"]:
        return True, "59/60 known M16 miss only"
    return False, f"must-answer 59/60 but misses {missed} (only M16 allowed)"


def voice_mean(doc):
    vs = [r["voice_score"] for r in doc["results"] if r.get("voice_score") is not None]
    if not vs:
        return None, 0
    return sum(vs) / len(vs), len(vs)


def cell_path(judge_dir, cond):
    return os.path.join(OUT, judge_dir, f"{cond}.t0.s1.voice.json")


def judge_all():
    judge_wrap.assert_prompts_match_judge()
    os.makedirs(OUT, exist_ok=True)
    for model, jdir in JUDGES:
        unload()
        time.sleep(2)
        print(f"{datetime.now()} start {model}", flush=True)
        os.makedirs(os.path.join(OUT, jdir), exist_ok=True)
        for cond, src, _role in CONDS:
            if not os.path.isfile(src):
                raise SystemExit(f"missing source {src}")
            out = cell_path(jdir, cond)
            judge_wrap.score_file(
                src, out, model=model, temp=0.0, seed=1,
                voice_only=True, force=False, limit=0,
            )
        print(f"{datetime.now()} done {model}", flush=True)
        unload()
        time.sleep(1)
    unload()


def summarize():
    rows = []
    for cond, src, role in CONDS:
        src_doc = json.load(open(src))
        ss = scores.stats(src_doc)
        m16 = scores.m16(src_doc)
        voices = {}
        for model, jdir in JUDGES:
            path = cell_path(jdir, cond)
            d = json.load(open(path))
            mean, n = voice_mean(d)
            if n != 204 or mean is None:
                raise SystemExit(f"{path} incomplete n={n} mean={mean}")
            meta = d.get("meta") or {}
            if meta.get("model") != model or float(meta.get("temp")) != 0.0 or int(meta.get("seed")) != 1:
                raise SystemExit(f"{path} meta mismatch {meta}")
            voices[model] = mean
        q = voices["qwen3:14b"]
        g = voices["gemma3:12b"]
        avg = (q + g) / 2
        must_ok, must_why = must_bar_ok(ss, src_doc)
        leaks_ok = ss["leak_n"] == 48 and ss["marker"] <= 2
        frames_ok = ss["frames"] < 6
        rows.append(dict(
            cond=cond, role=role, src=os.path.abspath(src),
            must=f"{ss['must_ok']}/{ss['must_n']}",
            must_ok_n=ss["must_ok"], must_n=ss["must_n"],
            marker=f"{ss['marker']}/{ss['leak_n']}",
            marker_n=ss["marker"], leak_n=ss["leak_n"],
            frames=ss["frames"],
            m16=f"{m16['ok']}/{m16['n']}",
            qwen=q, gemma=g, avg=avg,
            must_ok=must_ok, must_why=must_why,
            leaks_ok=leaks_ok, frames_ok=frames_ok,
        ))
    ref = next(r for r in rows if r["cond"] == REF)
    for r in rows:
        r["delta"] = r["avg"] - ref["avg"]
        r["delta_qwen"] = r["qwen"] - ref["qwen"]
        r["delta_gemma"] = r["gemma"] - ref["gemma"]
        r["beats"] = r["avg"] > ref["avg"]
        if r["cond"] == REF:
            r["verdict"] = "REFERENCE"
            r["fail_reasons"] = []
        else:
            reasons = []
            if not r["beats"]:
                reasons.append(f"voice avg {r['avg']:.4f} does not beat reference {ref['avg']:.4f} (Δ {r['delta']:+.4f})")
            if not r["must_ok"]:
                reasons.append(r["must_why"])
            if not r["leaks_ok"]:
                reasons.append(f"marker leaks {r['marker']}")
            if not r["frames_ok"]:
                reasons.append(f"frame breaks {r['frames']} (need <6)")
            r["fail_reasons"] = reasons
            r["verdict"] = "PASS" if not reasons else "FAIL"
    report = dict(
        written=datetime.now().strftime("%Y-%m-%d %I:%M:%S %p %Z"),
        judges=["qwen3:14b temp 0 seed 1", "gemma3:12b temp 0 seed 1"],
        bar={
            "voice": "average of the two judges; candidate must strictly beat v_merged_v_rag",
            "must": "60/60, or 59/60 if the only miss is M16",
            "leaks": "<=2/48 marker",
            "frames": "<6",
        },
        reference=REF,
        ref_avg=ref["avg"],
        ref_qwen=ref["qwen"],
        ref_gemma=ref["gemma"],
        rows=rows,
    )
    frozen.dump_json(os.path.join(OUT, "report.json"), report)
    return report


def fmt4(x):
    return f"{x:.4f}"


def write_md(report):
    ref = next(r for r in report["rows"] if r["cond"] == REF)
    lines = []
    a = lines.append
    a("# Relative voice bar")
    a("")
    a(f"Written {report['written']}. One session, both judges, fixed answer texts (not regenerated).")
    a("Judges: qwen3:14b temp 0 seed 1, and gemma3:12b temp 0 seed 1. Voice only, same rubric as judge.py.")
    a("Must-answer, marker leaks, and frame breaks are taken from the answer texts (regex / marker scan), not from a new leak judge.")
    a("Outputs: `exp10/relative/`. Morning exp10/calib/ files were not overwritten.")
    a("")
    a("## Rule (adopted)")
    a("")
    a("- Voice: average of the two judges; a candidate **must strictly beat** the reference `v_merged_v_rag` scored in this same session.")
    a("- Must-answer: 60/60, or 59/60 if the only miss is the known M16 item (yd4408, \"What is your number?\").")
    a("- Marker leaks ≤ 2/48; frame breaks < 6.")
    a("")
    a("## Scores")
    a("")
    a("| cond | role | must | leaks | frames | qwen3:14b | gemma3:12b | average | Δ vs ref | verdict |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in report["rows"]:
        delta = "—" if r["cond"] == REF else f"{r['delta']:+.4f}"
        a(
            f"| {r['cond']} | {r['role']} | {r['must']} | {r['marker']} | {r['frames']} | "
            f"{fmt4(r['qwen'])} | {fmt4(r['gemma'])} | {fmt4(r['avg'])} | {delta} | {r['verdict']} |"
        )
    a("")
    a(f"Reference `v_merged_v_rag` in this session: qwen {fmt4(ref['qwen'])}, gemma {fmt4(ref['gemma'])}, average {fmt4(ref['avg'])}.")
    a("")
    a("## Per-condition notes")
    a("")
    for r in report["rows"]:
        a(f"- **{r['cond']}** ({r['role']}): must {r['must']} ({r['must_why']}); leaks {r['marker']}; frames {r['frames']}; M16 {r['m16']}.")
        a(f"  qwen {fmt4(r['qwen'])} (Δ {r['delta_qwen']:+.4f}), gemma {fmt4(r['gemma'])} (Δ {r['delta_gemma']:+.4f}), avg {fmt4(r['avg'])} (Δ {r['delta']:+.4f}). {r['verdict']}")
        for why in r["fail_reasons"]:
            a(f"  - fail: {why}")
    a("")
    locked = next(r for r in report["rows"] if r["cond"] == "p2_light_dpo_guarded")
    a("## Locked prototype under this bar")
    a("")
    a(
        f"`p2_light_dpo_guarded` average voice {fmt4(locked['avg'])} vs reference {fmt4(ref['avg'])} "
        f"(Δ {locked['delta']:+.4f}): **{locked['verdict']}**."
    )
    if locked["verdict"] == "PASS":
        a("This is the No Return prototype result under the relative voice bar.")
    else:
        a("It does not clear the relative bar as stated. Reasons:")
        for why in locked["fail_reasons"]:
            a(f"- {why}")
    a("")
    a("The absolute exp9/exp10 voice ≥ 4.2 figure (4.22 on the original qwen3:14b run) remains the historical lock-in score.")
    a("This relative bar replaces it for pass/fail going forward: same-session two-judge average, must beat the reference.")
    a("")
    a("Local tag `no-return-proto-v1` already existed at the exp10 lock-in commit; it was not moved.")
    a("")
    path = os.path.join(H, "relative_bar.md")
    frozen.assert_under_exp10(path)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path, flush=True)
    return path


def main():
    os.chdir(H)
    t0 = time.time()
    print(datetime.now(), "relative_bar start", flush=True)
    try:
        judge_all()
        report = summarize()
        write_md(report)
        locked = next(r for r in report["rows"] if r["cond"] == "p2_light_dpo_guarded")
        print("LOCKED", locked["verdict"], "avg", round(locked["avg"], 4), "delta", round(locked["delta"], 4), flush=True)
        print(datetime.now(), "relative_bar done in", int(time.time() - t0), "s", flush=True)
    finally:
        unload()


if __name__ == "__main__":
    main()
