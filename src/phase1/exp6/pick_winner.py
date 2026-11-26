"""Pick the better of v_replay (a) and v_merged (b) on the RAG condition.

Order: RAG must-answer first, then voice, then fewer frame breaks.
Writes data/winner.json. Also appends the choice to build-status.md via stdout.
"""
import os, json
import scores
import frozen

H = os.path.dirname(os.path.abspath(__file__))
CANDS = [
    dict(name="v_replay", mode="fv", rag="v_replay_fv_rag", voice_dir="lora_v_replay",
         base=frozen.BASE, facts=frozen.FACTS),
    dict(name="v_merged", mode="v", rag="v_merged_v_rag", voice_dir="lora_v_merged",
         base=frozen.MERGED, facts=""),
]

def load_stats(cond):
    p = os.path.join(H, "runs", f"{cond}.judged.json")
    if not os.path.exists(p):
        return None
    return scores.stats(scores.load_json(p))

def main():
    os.makedirs(os.path.join(H, "data"), exist_ok=True)
    q14 = scores.stats(scores.load_json(os.path.join(frozen.EXP4_RUNS, "q14_rag.judged.json")))
    ranked = []
    view = []
    for i, c in enumerate(CANDS):
        adapter = os.path.join(H, c["voice_dir"])
        s = load_stats(c["rag"])
        if s is None or not os.path.isfile(os.path.join(adapter, "train_report.json")):
            view.append(dict(name=c["name"], missing=True, rag=c["rag"]))
            continue
        key = (s["must_ok"], s["voice"] if s["voice"] is not None else -1, -s["frames"])
        ranked.append((key, i, c, s, adapter))
        view.append(dict(
            name=c["name"], mode=c["mode"], rag=c["rag"],
            must=f"{s['must_ok']}/{s['must_n']}", voice=s["voice"], frames=s["frames"],
            marker=f"{s['marker']}/{s['leak_n']}", rank=list(key),
            passes=not scores.failures(s, q14["frames"]),
        ))
    if not ranked:
        raise SystemExit("no judged RAG runs for v_replay / v_merged")
    ranked.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    _key, _i, c, s, adapter = ranked[0]
    reason = (
        f"RAG must-answer first, then voice, then fewer frame breaks. "
        f"Winner {c['name']} must {s['must_ok']}/{s['must_n']} voice {s['voice']} frames {s['frames']}."
    )
    doc = dict(
        winner=c["name"], mode=c["mode"], rag_cond=c["rag"],
        base=c["base"], facts=c["facts"] or None, voice=adapter,
        voice_dir=c["voice_dir"],
        must=s["must_ok"], must_n=s["must_n"], voice_score=s["voice"], frames=s["frames"],
        marker=s["marker"], q14_frames=q14["frames"],
        ranking=view, reason=reason,
    )
    json.dump(doc, open(os.path.join(H, "data", "winner.json"), "w"), indent=1)
    print(json.dumps(doc, indent=1), flush=True)

if __name__ == "__main__":
    main()
