"""Compare a judged run against the locked exp9 p2_light_dpo_guarded scores."""
import json, os, argparse
import frozen, scores

H = frozen.H


def load_judged(path):
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}")
    return json.load(open(path))


def report(path, label):
    d = load_judged(path)
    s = scores.stats(d)
    m = scores.m16(d)
    issues = scores.within_tolerance(s)
    bar = scores.failures(s)
    out = dict(
        label=label,
        path=os.path.abspath(path),
        must=f"{s['must_ok']}/{s['must_n']}",
        marker=f"{s['marker']}/{s['leak_n']}",
        voice=None if s["voice"] is None else round(s["voice"], 4),
        voice_raw=s["voice"],
        frames=s["frames"],
        m16=f"{m['ok']}/{m['n']}",
        m16_rows=m["rows"],
        tolerance_ok=not issues,
        tolerance_issues=issues,
        pass_bar_ok=not bar,
        pass_bar_issues=bar,
        locked=frozen.LOCKED,
        judged=s["judged"],
        n=s["n"],
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", default=os.path.join(H, "runs", "p2_light_dpo_guarded.judged.json"))
    ap.add_argument("--orig-rejudge", default=os.path.join(H, "runs", "p2_light_dpo_guarded_orig.judged.json"))
    ap.add_argument("--locked", default=os.path.join(H, "locked", "p2_light_dpo_guarded.judged.json"))
    ap.add_argument("--out", default=os.path.join(H, "runs", "tolerance.json"))
    a = ap.parse_args()
    frozen.assert_under_exp10(a.out)
    rows = []
    rows.append(report(a.locked, "locked_exp9"))
    if os.path.isfile(a.orig_rejudge):
        rows.append(report(a.orig_rejudge, "orig_text_rejudge"))
    if os.path.isfile(a.regen):
        rows.append(report(a.regen, "regen"))
    locked_ok = True
    regen_ok = None
    rejudge_ok = None
    for r in rows:
        if r["label"] == "locked_exp9":
            locked_ok = (
                r["must"] == "59/60"
                and r["marker"] == "1/48"
                and abs(r["voice_raw"] - frozen.LOCKED["voice"]) < 1e-9
                and r["frames"] == 3
                and r["m16"] == "2/3"
            )
        if r["label"] == "regen":
            regen_ok = r["tolerance_ok"]
        if r["label"] == "orig_text_rejudge":
            rejudge_ok = r["tolerance_ok"]
    proto_source = "locked_exp9"
    if regen_ok is True:
        proto_note = "regen matched locked exp9 scores within tolerance"
    elif regen_ok is False:
        proto_source = "locked_exp9"
        proto_note = "regen missed tolerance; keep original exp9 judged file as the locked prototype scores"
    else:
        proto_note = "regen not scored yet"
    doc = dict(
        locked_file_ok=locked_ok,
        regen_tolerance_ok=regen_ok,
        orig_rejudge_tolerance_ok=rejudge_ok,
        proto_source=proto_source,
        proto_note=proto_note,
        rows=rows,
    )
    frozen.dump_json(a.out, doc)
    print(json.dumps({k: doc[k] for k in ("locked_file_ok", "regen_tolerance_ok", "orig_rejudge_tolerance_ok", "proto_source", "proto_note")}, indent=2))
    for r in rows:
        print(r["label"], r["must"], r["marker"], r["voice"], "frames", r["frames"], "m16", r["m16"], "tol", r["tolerance_ok"], r["tolerance_issues"])


if __name__ == "__main__":
    main()
