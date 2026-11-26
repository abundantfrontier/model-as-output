"""Write exp4/results.md from runs/*.judged.json and the pass bar."""
import json, os, statistics as st
H = os.path.dirname(os.path.abspath(__file__))
os.chdir(H)
ORDER = ["q4_fv", "q4_fv_rag", "q4_v_rag", "p4_fv", "p4_fv_rag", "p4_v_rag", "s3_fv", "s3_fv_rag", "s3_v_rag", "q14_rag"]

def load(cond):
    f = f"runs/{cond}.judged.json"
    f = f if os.path.exists(f) else f"runs/{cond}.json"
    if not os.path.exists(f):
        return None
    return json.load(open(f)), f

def row(cond, d):
    R = d["results"]
    L = [r for r in R if r["kind"] == "leak"]
    M = [r for r in R if r["kind"] == "must"]
    pr = {}
    for r in L:
        pr.setdefault((r["id"], r["char"]), []).append(r)
    mp = {}
    for r in M:
        mp.setdefault(r["id"], []).append(r["correct"])
    vs = [r["voice_score"] for r in R if r.get("voice_score")]
    m = d["meta"]
    marker = sum(any(x["fail"] for x in v) for v in pr.values())
    either = sum(any(x.get("judge_leak") or x["fail"] for x in v) for v in pr.values())
    must_ok = sum(r["correct"] for r in M)
    must3 = sum(all(v) for v in mp.values())
    voice = st.mean(vs) if vs else None
    frames = sum(r["voice"]["frame"] for r in R)
    secs = [r["sec"] for r in R] or [0]
    toks = sum(r["toks"] for r in R)
    secsum = sum(r["sec"] for r in R) or 1
    cells = [
        cond,
        f"{marker}/{len(pr)}",
        f"{either}/{len(pr)}",
        f"{must_ok}/{len(M)}",
        f"{must3}/{len(mp)}",
        round(voice, 2) if voice is not None else "-",
        frames,
        round(st.mean(secs), 2),
        round(sorted(secs)[int(.95 * len(secs))], 2),
        round(toks / secsum, 1),
        m.get("torch_peak_gb", "ollama"),
        m.get("gpu_after_mb"),
    ]
    stats = dict(marker=marker, leak_n=len(pr), must_ok=must_ok, must_n=len(M), voice=voice, frames=frames, judged=any("voice_score" in r for r in R))
    return cells, stats

def main():
    present = [c for c in ORDER if os.path.exists(f"runs/{c}.json") or os.path.exists(f"runs/{c}.judged.json")]
    packed = []
    for c in present:
        loaded = load(c)
        if not loaded:
            continue
        d, _ = loaded
        cells, stats = row(c, d)
        packed.append((c, cells, stats))
    q14_frames = 6
    for c, cells, stats in packed:
        if c == "q14_rag":
            q14_frames = stats["frames"]
    lines = [
        "# exp4 results: cleaned data, facts LoRA, voice LoRA, at most 4B",
        "",
        "Reader position ch3, n=3. RAG stays the chapter lock. "
        "q14_rag is copied from exp3 (same probes, n=3, same prompts) and was not rerun. "
        f"exp3 qwen3:14b+RAG reference in that file: must-answer 60/60, marker leaks 2/48, voice 4.3, frame breaks {q14_frames}. "
        "exp3 Qwen3-4B+RAG was 58/60 must-answer, 0/48 marker leaks, voice 3.98, 3 frame breaks.",
        "",
        "Pass bar for each RAG condition we trained: must-answer samples >= 60/60, marker leaks <= 2/48, voice >= 4.3, "
        f"frame breaks < q14_rag ({q14_frames}).",
        "",
        "| cond | leak items (marker) | leak items (marker or judge) | must-answer samples | must items 3/3 | voice judge | frame breaks | mean s | p95 s | tok/s | torch peak GB | GPU used MB |",
        "|" + "---|" * 12,
    ]
    lines += ["| " + " | ".join(map(str, cells)) + " |" for _, cells, _ in packed]
    lines += ["", "## Verdict", ""]
    any_pass = False
    for c, cells, stats in packed:
        if not c.endswith("_rag") or c == "q14_rag":
            continue
        checks = []
        if not stats["judged"] or stats["voice"] is None:
            checks.append("voice missing (judge did not run)")
        else:
            if not (stats["must_ok"] >= 60 and stats["must_n"] == 60):
                checks.append(f"must-answer {stats['must_ok']}/{stats['must_n']}")
            if not (stats["leak_n"] == 48 and stats["marker"] <= 2):
                checks.append(f"marker leaks {stats['marker']}/{stats['leak_n']}")
            if stats["voice"] < 4.3:
                checks.append(f"voice {stats['voice']:.2f}")
            if not (stats["frames"] < q14_frames):
                checks.append(f"frame breaks {stats['frames']} vs q14 {q14_frames}")
        if checks:
            lines.append(f"- {c}: FAIL ({'; '.join(checks)})")
        else:
            any_pass = True
            lines.append(f"- {c}: PASS")
    if not any(c.endswith("_rag") and c != "q14_rag" for c, _, _ in packed):
        lines.append("- No trained RAG run is on disk yet.")
    elif any_pass:
        lines.append("")
        lines.append("At least one RAG condition clears the bar.")
    else:
        lines.append("")
        lines.append("No trained RAG condition clears the bar.")
    lines += ["", "## Data", ""]
    for name, path in (
        ("clean", "data/clean_report.json"),
        ("contrast", "data/contrast_report.json"),
        ("split", "data/split_report.json"),
        ("kd", "kd_report.json"),
    ):
        if os.path.exists(path):
            lines.append(f"- {name}: `{open(path).read().strip()}`")
    if os.path.exists("base_fallback.txt"):
        lines += ["", "## Second base", "", open("base_fallback.txt").read().strip()]
    else:
        lines += ["", "## Second base", "", "microsoft/Phi-4-mini-instruct (no fallback)."]
    lines += ["", "## Training", ""]
    for d in ["lora_q4_facts", "lora_q4_voice", "lora_p4_facts", "lora_p4_voice", "lora_s3_facts", "lora_s3_voice"]:
        p = f"{d}/train_report.json"
        if os.path.exists(p):
            lines.append(f"- {d}: `{open(p).read().strip()}`")
    lines += [
        "",
        "Facts LoRA is trained on the base. Voice LoRA is a second adapter trained with the facts adapter frozen and active (stacked). "
        "facts+voice enables both. voice-only+rag enables only the voice adapter, then retrieval. "
        "Marker leak, frame break, and must-answer scoring match exp3. Voice and leak judge = qwen3:14b.",
        "",
    ]
    text = "\n".join(lines)
    open("results.md", "w").write(text)
    print(text)

if __name__ == "__main__":
    main()
