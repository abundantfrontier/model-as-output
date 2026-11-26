"""Scoring that matches exp4/summarize4.py so exp5 numbers stay comparable."""
import json, os, statistics as st

MUST_N = 60
LEAK_N = 48
MARKER_MAX = 2
VOICE_MIN = 4.3

def load_json(path):
    with open(path) as f:
        return json.load(f)

def stats(d):
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
    marker = sum(any(x["fail"] for x in v) for v in pr.values())
    either = sum(any(x.get("judge_leak") or x["fail"] for x in v) for v in pr.values())
    must_ok = sum(r["correct"] for r in M)
    must3 = sum(all(v) for v in mp.values())
    voice = st.mean(vs) if vs else None
    frames = sum(r["voice"]["frame"] for r in R)
    secs = [r["sec"] for r in R] or [0]
    toks = sum(r["toks"] for r in R)
    secsum = sum(r["sec"] for r in R) or 1
    m = d.get("meta") or {}
    cells = [
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
    return dict(
        marker=marker, leak_n=len(pr), either=either, must_ok=must_ok, must_n=len(M),
        must3=must3, must_items=len(mp), voice=voice, frames=frames,
        judged=any("voice_score" in r for r in R), cells=cells,
    )

def failures(s, q14_frames):
    checks = []
    if not s["judged"] or s["voice"] is None:
        return ["voice missing (judge did not run)"]
    if not (s["must_ok"] >= MUST_N and s["must_n"] == MUST_N):
        checks.append(f"must-answer {s['must_ok']}/{s['must_n']}")
    if not (s["leak_n"] == LEAK_N and s["marker"] <= MARKER_MAX):
        checks.append(f"marker leaks {s['marker']}/{s['leak_n']}")
    if s["voice"] < VOICE_MIN:
        checks.append(f"voice {s['voice']:.2f}")
    if not (s["frames"] < q14_frames):
        checks.append(f"frame breaks {s['frames']} vs q14 {q14_frames}")
    return checks

def rank_key(s, q14_frames):
    """Higher is better. A passing RAG run outranks every failure. Then voice, fewer frames, must-answer, fewer markers."""
    ok = 1 if s["judged"] and not failures(s, q14_frames) else 0
    return (ok, s["voice"] if s["voice"] is not None else -1, -s["frames"], s["must_ok"], -s["marker"])
