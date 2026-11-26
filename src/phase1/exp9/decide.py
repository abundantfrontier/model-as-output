"""Decide the optional stages from scores already on disk.

second: q4_merged_rag pass-1 only if primary rewrites are clearly short on must-answer.
train:  (c) only if some two-pass row is close to the bar.
sweep:  (d) only if some row is otherwise passing and within 0.05 voice, and (c) trained.
"""
import os, sys, json, glob
import scores
import frozen

H = frozen.H
PRIMARY = ["p2_fb_dpo", "p2_light_contrast", "p2_light_dpo", "p2_v_merged"]

def hp():
    return json.load(open(os.path.join(H, "hparams.json")))

def decide_path():
    return os.path.join(H, "data", "decide.json")

def load():
    p = decide_path()
    if os.path.isfile(p):
        return json.load(open(p))
    return {}

def save(doc):
    os.makedirs(os.path.dirname(decide_path()), exist_ok=True)
    frozen.dump_json(decide_path(), doc)

def run_doc(cond):
    for suf in (".judged.json", ".json"):
        p = os.path.join(H, "runs", cond + suf)
        if os.path.isfile(p):
            return scores.load_json(p), p
    return None, None

def row(cond):
    doc, path = run_doc(cond)
    if doc is None:
        return None
    s = scores.stats(doc)
    s["cond"] = cond
    s["path"] = path
    return s

def brief(s):
    if s is None:
        return None
    return dict(
        cond=s["cond"], judged=s["judged"], must=f"{s['must_ok']}/{s['must_n']}",
        marker=f"{s['marker']}/{s['leak_n']}", frames=s["frames"],
        voice=None if s["voice"] is None else round(s["voice"], 4),
    )

def close_ok(s, spec):
    if s is None or not s["judged"] or s["voice"] is None:
        return False
    return (
        s["must_n"] == 60 and s["must_ok"] >= spec["must_min"]
        and s["leak_n"] == 48 and s["marker"] <= spec["marker_max"]
        and s["frames"] < spec["frames_lt"]
        and s["voice"] >= spec["voice_min"]
    )

def sweep_ok(s, spec):
    if s is None or not s["judged"] or s["voice"] is None:
        return False
    return (
        s["must_n"] == 60 and s["must_ok"] >= spec["must"]
        and s["leak_n"] == 48 and s["marker"] <= spec["marker_max"]
        and s["frames"] < spec["frames_lt"]
        and s["voice"] >= spec["voice_min"]
    )

def two_pass_conds():
    found = []
    for path in sorted(glob.glob(os.path.join(H, "runs", "p2_*.json"))):
        base = os.path.basename(path)
        if base.endswith(".judged.json") or ".factdiff" in base or base.endswith(".bad.json"):
            continue
        cond = base[:-len(".json")]
        if cond not in found:
            found.append(cond)
    return found

def cmd_second():
    spec = hp()["second_source"]
    rows = []
    missing = []
    for cond in PRIMARY:
        s = row(cond)
        if s is None:
            missing.append(cond)
        else:
            rows.append(s)
    doc = load()
    if missing:
        doc["second_source"] = dict(run=False, conds=[], reason=f"waiting on {missing}", rows=[brief(s) for s in rows])
        save(doc)
        print(json.dumps(doc["second_source"]))
        return
    best_must = max(s["must_ok"] for s in rows)
    if best_must > spec["must_at_or_below"]:
        reason = (
            f"best primary must-answer is {best_must}/60, above {spec['must_at_or_below']}. "
            "q4_merged_rag pass-1 is 58/60, so it is not a better facts source."
        )
        doc["second_source"] = dict(run=False, conds=[], reason=reason, best_must=best_must, rows=[brief(s) for s in rows])
        save(doc)
        print(json.dumps(doc["second_source"]))
        return
    # Strongest facts among the primaries, then the fixed voice order.
    pool = [s for s in rows if s["must_ok"] == best_must]
    pool.sort(key=lambda s: PRIMARY.index(s["cond"]))
    pick = pool[0]["cond"] + "_q4merged"
    doc["second_source"] = dict(
        run=True, conds=[pick], best_must=best_must, picked_from=pool[0]["cond"],
        reason=f"best primary must-answer {best_must}/60 is at or below {spec['must_at_or_below']}",
        rows=[brief(s) for s in rows],
    )
    save(doc)
    print(json.dumps(doc["second_source"]))

def candidate_rows():
    out = []
    for cond in two_pass_conds():
        # Prefer the judged file via row().
        s = row(cond)
        if s is not None and s["judged"]:
            out.append(s)
    return out

def cmd_train():
    spec = hp()["close"]
    rows = candidate_rows()
    hits = [s for s in rows if close_ok(s, spec)]
    doc = load()
    if hits:
        hits.sort(key=lambda s: scores.rank_key(s), reverse=True)
        doc["stage_c"] = dict(
            train=True,
            reason=f"{hits[0]['cond']} is close to the bar",
            trigger=hits[0]["cond"],
            hits=[brief(s) for s in hits],
            rows=[brief(s) for s in rows],
        )
    else:
        doc["stage_c"] = dict(
            train=False,
            reason="no unguarded or guarded two-pass row has must-answer >= 59/60, marker leaks <= 2/48, frame breaks < 6, and voice >= 4.2",
            trigger=None,
            hits=[],
            rows=[brief(s) for s in rows],
        )
    save(doc)
    print(json.dumps(doc["stage_c"]))

def cmd_sweep():
    spec = hp()["sweep"]
    doc = load()
    stage_c = doc.get("stage_c") or {}
    rows = candidate_rows()
    hits = [s for s in rows if sweep_ok(s, spec)]
    report = os.path.join(H, "lora_v_rewrite", "train_report.json")
    trained = os.path.isfile(report)
    if not stage_c.get("train") and not trained:
        doc["stage_d"] = dict(sweep=False, conds=[], reason="(c) was skipped, so there is no rewrite checkpoint to sweep", hits=[brief(s) for s in hits])
        save(doc)
        print(json.dumps(doc["stage_d"]))
        return
    if not hits:
        doc["stage_d"] = dict(
            sweep=False, conds=[],
            reason="no scored two-pass row has must-answer 60/60, marker leaks <= 2/48, frame breaks < 6, and voice >= 4.25",
            hits=[], rows=[brief(s) for s in rows],
        )
        save(doc)
        print(json.dumps(doc["stage_d"]))
        return
    if not trained:
        doc["stage_d"] = dict(sweep=False, conds=[], reason="close enough to sweep but lora_v_rewrite/train_report.json is missing", hits=[brief(s) for s in hits])
        save(doc)
        print(json.dumps(doc["stage_d"]))
        return
    cks = []
    root = os.path.join(H, "lora_v_rewrite")
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if name.startswith("checkpoint-") and os.path.isfile(os.path.join(root, name, "adapter_config.json")):
                step = int(name.split("-", 1)[1])
                cks.append(step)
    cks.sort()
    if not cks:
        doc["stage_d"] = dict(sweep=False, conds=[], reason="(c) trained but saved no intermediate checkpoints", hits=[brief(s) for s in hits])
        save(doc)
        print(json.dumps(doc["stage_d"]))
        return
    # Earliest checkpoint and the one nearest the middle of the list.
    pick = [cks[0]]
    if len(cks) > 1:
        mid = cks[len(cks) // 2]
        if mid not in pick:
            pick.append(mid)
    conds = [f"p2_rewrite_c{step}" for step in pick]
    doc["stage_d"] = dict(
        sweep=True, conds=conds, checkpoints=cks, picked=pick,
        reason="a scored row is otherwise passing and voice is within 0.05 of 4.3",
        hits=[brief(s) for s in hits],
    )
    save(doc)
    print(json.dumps(doc["stage_d"]))

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "second":
        cmd_second()
    elif cmd == "train":
        cmd_train()
    elif cmd == "sweep":
        cmd_sweep()
    else:
        raise SystemExit("usage: decide.py second|train|sweep")

if __name__ == "__main__":
    main()
