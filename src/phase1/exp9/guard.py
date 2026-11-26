"""Guarded fallback. If fact-diff fails a sample, score the pass-1 text instead.

The mixed file is a separate condition (<cond>_guarded) for judge.py.
Kept samples stay on the pass-2 rewrite. Fallback samples copy pass-1 and
recompute fail / voice flags / must-answer regex.
"""
import os, sys, json, re, argparse
import common as C
import frozen

H = frozen.H

def build(run_path, fact_path, out_path):
    doc = json.load(open(run_path))
    fd = json.load(open(fact_path))
    rows = doc["results"]
    items = fd["items"]
    if len(items) != len(rows):
        raise SystemExit(f"factdiff rows {len(items)} != run rows {len(rows)}")
    by_index = {it["index"]: it for it in items}
    if len(by_index) != len(rows):
        raise SystemExit("factdiff index mismatch")
    out_rows = []
    n_fallback = 0
    for i, src in enumerate(rows):
        it = by_index[i]
        r = dict(src)
        for k in ("judge_leak", "judge_ev", "voice_score"):
            r.pop(k, None)
        use_pass1 = bool(it["fallback"])
        if use_pass1:
            n_fallback += 1
            r["answer"] = r["pass1"]
            r["toks"] = r.get("pass1_toks") or 0
            r["sec"] = r.get("pass1_sec") or 0
        else:
            if it["n_dropped"] or it["n_added"] or it["new_markers"] or not it["parse_ok"]:
                raise SystemExit(f"row {i} kept but fact-diff is not clean")
            r["answer"] = r["pass2"]
        r["fail"] = C.fail_hits(r["answer"])
        r["voice"] = C.voice_flags(r["answer"])
        r["guard_fallback"] = use_pass1
        new = set(C.fail_hits(r["answer"])) - set(C.fail_hits(r["pass1"]))
        if new:
            raise SystemExit(f"row {i} scored text has a new marker {new}")
        if src["kind"] == "must":
            r["correct"] = bool(re.search(src["rubric"], r["answer"], re.I))
        out_rows.append(r)
    meta = dict(doc.get("meta") or {})
    parent = meta.get("cond") or os.path.splitext(os.path.basename(run_path))[0]
    meta.update(
        cond=os.path.splitext(os.path.basename(out_path))[0],
        parent=parent,
        guarded=True,
        n_fallback=n_fallback,
        n_kept=len(rows) - n_fallback,
        guard_rule=fd.get("meta", {}).get("fallback_rule"),
        factdiff=os.path.abspath(fact_path),
        scored="pass-1 text when fact-diff fails, otherwise the pass-2 rewrite",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    frozen.dump_json(out_path, dict(meta=meta, results=out_rows))
    print("guard wrote", out_path, "fallback", n_fallback, "/", len(rows), flush=True)
    return n_fallback

def self_test():
    import tempfile
    rows = [
        dict(kind="must", id="M1", set="M", char="wren", probe="Air?", rubric=r"good", sample=0,
             answer="flat", pass1="Air is good.", pass2="bad rewrite", pass1_toks=3, pass1_sec=1.0,
             toks=4, sec=1.2, fail=[], voice=dict(frame=False, stage=False, long=False, words=1)),
        dict(kind="leak", id=1, set="A", char="wren", probe="Fuel?", sample=0,
             answer="ok voice", pass1="Fuel is to list.", pass2="Fuel is to list, friend.",
             pass1_toks=4, pass1_sec=1.0, toks=5, sec=1.1, fail=[], voice=dict(frame=False, stage=False, long=False, words=4)),
    ]
    items = [
        dict(index=0, n_dropped=1, n_added=0, new_markers=[], parse_ok=True, fallback=True),
        dict(index=1, n_dropped=0, n_added=0, new_markers=[], parse_ok=True, fallback=False),
    ]
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, "p2_x.json")
        fd = os.path.join(d, "p2_x.factdiff.json")
        out = os.path.join(d, "p2_x_guarded.json")
        frozen.dump_json(run, dict(meta=dict(cond="p2_x", rag=False), results=rows))
        frozen.dump_json(fd, dict(meta=dict(fallback_rule="test"), summary={}, items=items))
        n = build(run, fd, out)
        assert n == 1
        got = json.load(open(out))
        assert got["results"][0]["answer"] == "Air is good."
        assert got["results"][0]["correct"] is True
        assert got["results"][1]["answer"] == "Fuel is to list, friend."
        assert got["meta"]["guarded"] is True
    print("guard self-test ok")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", nargs="?", default="")
    ap.add_argument("--run", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.cond:
        run = os.path.join(H, "runs", a.cond + ".json")
        fact = os.path.join(H, "runs", a.cond + ".factdiff.json")
        out = os.path.join(H, "runs", a.cond + "_guarded.json")
    else:
        run, fact, out = a.run, a.run[:-len(".json")] + ".factdiff.json" if a.run.endswith(".json") else "", a.out
    if not run or not out:
        raise SystemExit("usage: guard.py COND | guard.py --run RUN.json --out OUT.json")
    if os.path.isfile(out) and not a.force:
        print("guard skip", out, flush=True)
        return
    if not os.path.isfile(fact):
        raise SystemExit(f"missing factdiff {fact}")
    build(run, fact, out)

if __name__ == "__main__":
    main()
