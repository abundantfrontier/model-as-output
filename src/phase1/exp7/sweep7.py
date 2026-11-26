"""Checkpoint plan and the RAG selection rule.

A RAG row passes when must-answer is 60/60, marker leaks are <= 2/48,
voice is >= 4.3, and frame breaks are < 6. Among passing RAG rows, pick the
highest voice, then fewer frame breaks, then a final adapter over a checkpoint.
"""
import os, re, json, glob
import scores

H = os.path.dirname(os.path.abspath(__file__))
FINALS = ["v_light_dpo", "v_light_contrast", "v_light_combo", "v_light_dpo_replay"]
RULE = ("Pass filter: must-answer 60/60, marker leaks <= 2/48, frame breaks < 6. "
        "Among RAG rows that pass, pick the highest voice, then fewer frame breaks, "
        "then a final adapter ahead of a checkpoint, then the later checkpoint. "
        "Sweep only the closest final when no final passes and that final is close "
        "(must >= 59/60, marker <= 2/48, frames < 6, voice >= 4.15). "
        "Evaluate its earliest checkpoint and the checkpoint nearest half the run.")

def load_hp():
    return json.load(open(os.path.join(H, "hparams.json")))

def judged(cond):
    path = os.path.join(H, "runs", cond + ".judged.json")
    if not os.path.isfile(path):
        return None
    return scores.stats(scores.load_json(path))

def passes(s):
    return bool(s and s["judged"] and not scores.failures(s, 6))

def close(s, sw):
    return bool(
        s and s["judged"] and s["must_n"] == 60 and s["must_ok"] >= sw["close_must"]
        and s["leak_n"] == 48 and s["marker"] <= 2 and s["frames"] < 6
        and s["voice"] is not None and s["voice"] >= sw["close_voice"]
    )

def ckpt_steps(name):
    root = os.path.join(H, f"lora_{name}")
    steps = []
    if not os.path.isdir(root):
        return steps
    for d in os.listdir(root):
        m = re.fullmatch(r"checkpoint-(\d+)", d)
        if m and os.path.isfile(os.path.join(root, d, "adapter_config.json")):
            steps.append(int(m.group(1)))
    return sorted(steps)

def pick_steps(name, n):
    steps = ckpt_steps(name)
    if not steps or n <= 0:
        return []
    report_path = os.path.join(H, f"lora_{name}", "train_report.json")
    final = steps[-1]
    if os.path.isfile(report_path):
        final = json.load(open(report_path)).get("steps") or final
    half = final / 2.0
    earliest = steps[0]
    nearest = min(steps, key=lambda s: (abs(s - half), s))
    chosen = [earliest]
    if nearest != earliest:
        chosen.append(nearest)
    for s in steps:
        if len(chosen) >= n:
            break
        if s not in chosen:
            chosen.append(s)
    return chosen[:n]

def rank_key(cond, s):
    m = re.search(r"_c(\d+)_v_rag$", cond)
    step = int(m.group(1)) if m else 10**9
    is_final = 1 if m is None else 0
    return (s["voice"], -s["frames"], is_final, step)

def discover_rag():
    found = []
    for name in FINALS:
        found.append(name + "_v_rag")
    for path in sorted(glob.glob(os.path.join(H, "runs", "*_c*_v_rag.judged.json"))):
        cond = os.path.basename(path).replace(".judged.json", "")
        if cond not in found:
            found.append(cond)
    return found

def best_passing():
    rows = []
    for cond in discover_rag():
        s = judged(cond)
        if passes(s):
            rows.append((cond, s))
    if not rows:
        return None
    rows.sort(key=lambda it: rank_key(it[0], it[1]), reverse=True)
    return rows[0]

def adapter_for(cond):
    name = cond[:-5] if cond.endswith("_rag") else cond
    if name.endswith("_v"):
        name = name[:-2]
    m = re.fullmatch(r"(.+)_c(\d+)", name)
    if m:
        return os.path.join(H, f"lora_{m.group(1)}", f"checkpoint-{m.group(2)}")
    return os.path.join(H, f"lora_{name}")

def plan():
    sw = load_hp()["sweep"]
    finals = []
    any_pass = False
    for name in FINALS:
        s = judged(name + "_v_rag")
        row = dict(name=name, cond=name + "_v_rag",
                   stats=None if not s else dict(must=f"{s['must_ok']}/{s['must_n']}", voice=s["voice"],
                                                 frames=s["frames"], marker=f"{s['marker']}/{s['leak_n']}"),
                   passes=passes(s), close=close(s, sw))
        finals.append(row)
        any_pass = any_pass or row["passes"]
    doc = dict(rule=RULE, finals=finals, evals=[], skip=False, reason="")
    if any_pass:
        doc["skip"] = True
        doc["reason"] = "a final RAG condition already clears the pass bar"
    else:
        close_rows = [r for r in finals if r["close"]]
        if not close_rows:
            doc["skip"] = True
            doc["reason"] = "no final RAG row is close (need must >= 59/60, marker <= 2/48, frames < 6, voice >= 4.15)"
        else:
            close_rows.sort(key=lambda r: (r["stats"]["voice"], -r["stats"]["frames"]), reverse=True)
            name = close_rows[0]["name"]
            steps = pick_steps(name, sw["max_checkpoints"])
            if not steps:
                doc["skip"] = True
                doc["reason"] = f"{name} is close but it has no checkpoint-* adapters to evaluate"
            else:
                doc["reason"] = f"sweep {name} checkpoints {steps}"
                for step in steps:
                    for suffix in ("_v", "_v_rag"):
                        doc["evals"].append(dict(cond=f"{name}_c{step}{suffix}", voice=os.path.join(
                            H, f"lora_{name}", f"checkpoint-{step}")))
    json.dump(doc, open(os.path.join(H, "data", "sweep_plan.json"), "w"), indent=1)
    print(json.dumps(doc), flush=True)

def select():
    sw_plan = os.path.join(H, "data", "sweep_plan.json")
    plan_doc = json.load(open(sw_plan)) if os.path.isfile(sw_plan) else {}
    rejected = []
    passing = []
    for cond in discover_rag():
        s = judged(cond)
        if not s:
            continue
        row = dict(cond=cond, must=f"{s['must_ok']}/{s['must_n']}", voice=s["voice"],
                   frames=s["frames"], marker=f"{s['marker']}/{s['leak_n']}", adapter=adapter_for(cond))
        if passes(s):
            passing.append(row)
        else:
            row["why"] = scores.failures(s, 6)
            rejected.append(row)
    winner = None
    if passing:
        passing.sort(key=lambda r: rank_key(r["cond"], judged(r["cond"])), reverse=True)
        winner = passing[0]
    doc = dict(rule=RULE, plan_reason=plan_doc.get("reason"), skip=plan_doc.get("skip"),
               winner=winner, passing=passing, rejected=rejected)
    json.dump(doc, open(os.path.join(H, "data", "sweep.json"), "w"), indent=1)
    best = dict(pass_=bool(winner), cond=None if not winner else winner["cond"],
                voice=None if not winner else winner["voice"],
                adapter=None if not winner else winner["adapter"],
                frames=None if not winner else winner["frames"],
                must=None if not winner else winner["must"],
                marker=None if not winner else winner["marker"])
    # JSON key "pass" is reserved as a Python keyword on the way in; store pass.
    best["pass"] = best.pop("pass_")
    json.dump(best, open(os.path.join(H, "data", "best.json"), "w"), indent=1)
    print(json.dumps(doc), flush=True)

def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    os.makedirs(os.path.join(H, "data"), exist_ok=True)
    if cmd == "plan":
        plan()
    elif cmd == "select":
        select()
    else:
        raise SystemExit(f"unknown {cmd}")

if __name__ == "__main__":
    main()
