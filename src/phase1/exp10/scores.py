"""Scoring that matches exp4/summarize4.py and exp9/scores.py.

Historical absolute voice bar: voice >= 4.2. Adopted bar is relative:
two-judge average must beat v_merged_v_rag (see frozen.RELATIVE_BAR).
Marker leaks <= 2/48 and frame breaks < 6 are unchanged.
Must-answer target remains 60/60 (59/60 OK if the only miss is M16).
"""
import json, statistics as st
import frozen

MUST_N = 60
LEAK_N = 48
MARKER_MAX = frozen.MARKER_MAX
VOICE_MIN = frozen.VOICE_MIN
Q14_FRAMES = frozen.FRAMES_LT


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
    must_ok = sum(bool(r["correct"]) for r in M)
    must3 = sum(all(v) for v in mp.values())
    voice = st.mean(vs) if vs else None
    frames = sum(bool(r["voice"]["frame"]) for r in R)
    secs = [r["sec"] for r in R] or [0]
    toks = sum(r.get("toks") or 0 for r in R)
    secsum = sum(r.get("sec") or 0 for r in R) or 1
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
        judged=any("voice_score" in r for r in R), cells=cells, n_voice=len(vs), n=len(R),
    )


def m16(d):
    rows = [r for r in d["results"] if r.get("kind") == "must" and str(r.get("id")) == "M16"]
    ok = sum(bool(r.get("correct")) for r in rows)
    fail = [(r.get("sample"), r.get("char"), r.get("probe"), bool(r.get("correct")), (r.get("answer") or "")[:160]) for r in rows]
    return dict(ok=ok, n=len(rows), rows=fail)


def failures(s, voice_min=VOICE_MIN, q14_frames=Q14_FRAMES):
    checks = []
    if not s["judged"] or s["voice"] is None:
        return ["voice missing (judge did not run)"]
    if not (s["must_ok"] >= MUST_N and s["must_n"] == MUST_N):
        checks.append(f"must-answer {s['must_ok']}/{s['must_n']}")
    if not (s["leak_n"] == LEAK_N and s["marker"] <= MARKER_MAX):
        checks.append(f"marker leaks {s['marker']}/{s['leak_n']}")
    if s["voice"] < voice_min:
        checks.append(f"voice {s['voice']:.2f}")
    if not (s["frames"] < q14_frames):
        checks.append(f"frame breaks {s['frames']} vs q14 {q14_frames}")
    return checks


def within_tolerance(s, locked=None, tol=None):
    locked = locked or frozen.LOCKED
    tol = tol or frozen.TOLERANCE
    issues = []
    if s["must_ok"] != locked["must_ok"] or s["must_n"] != locked["must_n"]:
        issues.append(f"must {s['must_ok']}/{s['must_n']} != {locked['must_ok']}/{locked['must_n']}")
    if s["marker"] != locked["marker"] or s["leak_n"] != locked["leak_n"]:
        issues.append(f"marker {s['marker']}/{s['leak_n']} != {locked['marker']}/{locked['leak_n']}")
    if s["voice"] is None:
        issues.append("voice missing")
    elif abs(s["voice"] - locked["voice"]) > tol["voice"]:
        issues.append(f"voice {s['voice']:.4f} vs {locked['voice']:.4f} (tol {tol['voice']})")
    if abs(s["frames"] - locked["frames"]) > tol["frames"]:
        issues.append(f"frames {s['frames']} vs {locked['frames']} (tol {tol['frames']})")
    return issues
