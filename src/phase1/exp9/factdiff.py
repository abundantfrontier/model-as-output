"""Fact-preservation check. qwen3:14b, temp 0, seed 1, plus the marker scan.

Writes <run>.factdiff.json next to the run file. A sample fails (fallback) when
the judge lists any dropped claim, any added claim, the reply does not parse,
or pass-2 hits a FAIL marker that pass-1 did not hit.
"""
import os, sys, json, re, hashlib, time
import common as C
import prompts
import frozen

JUDGE_MODEL = "qwen3:14b"
TEMP = 0.0
SEED = 1

def code_sha():
    h = hashlib.sha256()
    h.update(prompts.FACT_DIFF_HEAD.encode())
    h.update(open(os.path.abspath(__file__), "rb").read())
    return h.hexdigest()

def factdiff_path(run_path):
    if not run_path.endswith(".json"):
        raise SystemExit(f"run path must end in .json: {run_path}")
    return run_path[:-len(".json")] + ".factdiff.json"

def parse_lists(text):
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    j = json.loads(raw)
    if not isinstance(j, dict):
        raise ValueError("not an object")
    if "dropped" not in j and "added" not in j:
        raise ValueError("missing dropped/added")
    def lst(key):
        v = j.get(key, [])
        if v is None:
            return []
        if isinstance(v, str):
            v = [v] if v.strip() else []
        if not isinstance(v, list):
            raise ValueError(key)
        out = []
        for x in v:
            if isinstance(x, dict):
                s = " ".join(str(t) for t in x.values()).strip()
            else:
                s = str(x).strip()
            if s:
                out.append(s)
        return out
    return lst("dropped"), lst("added")

def one(question, pass1, pass2):
    prompt = prompts.fact_diff_prompt(question, pass1, pass2)
    text, *_ = C.ollama_chat(
        JUDGE_MODEL, [{"role": "user", "content": prompt}],
        fmt="json", temp=TEMP, seed=SEED, num_predict=512,
    )
    try:
        dropped, added = parse_lists(text)
        return dropped, added, True, text
    except Exception as e:
        return [], [], False, f"{type(e).__name__}: {e}; raw={text[:400]}"

def new_markers(pass1, pass2):
    p1 = set(C.fail_hits(pass1))
    p2 = set(C.fail_hits(pass2))
    return sorted(p2 - p1), sorted(p1), sorted(p2)

def summarize(items):
    n = len(items)
    drop_claims = sum(it["n_dropped"] for it in items)
    add_claims = sum(it["n_added"] for it in items)
    drop_samples = sum(it["n_dropped"] > 0 for it in items)
    add_samples = sum(it["n_added"] > 0 for it in items)
    new_marker_samples = sum(bool(it["new_markers"]) for it in items)
    parse_fail = sum(not it["parse_ok"] for it in items)
    fallback = sum(it["fallback"] for it in items)
    per = {}
    for it in items:
        key = f"{it['kind']}|{it['id']}|{it['char']}"
        slot = per.setdefault(key, dict(n=0, drop_samples=0, add_samples=0, drop_claims=0, add_claims=0, new_marker_samples=0, fallback=0))
        slot["n"] += 1
        slot["drop_samples"] += int(it["n_dropped"] > 0)
        slot["add_samples"] += int(it["n_added"] > 0)
        slot["drop_claims"] += it["n_dropped"]
        slot["add_claims"] += it["n_added"]
        slot["new_marker_samples"] += int(bool(it["new_markers"]))
        slot["fallback"] += int(it["fallback"])
    return dict(
        n=n,
        drop_claims=drop_claims,
        add_claims=add_claims,
        drop_samples=drop_samples,
        add_samples=add_samples,
        drop_sample_rate=round(drop_samples / n, 4) if n else None,
        add_sample_rate=round(add_samples / n, 4) if n else None,
        new_marker_samples=new_marker_samples,
        parse_fail=parse_fail,
        fallback_samples=fallback,
        fallback_rate=round(fallback / n, 4) if n else None,
        per_item=per,
    )

def run(run_path):
    final = factdiff_path(run_path)
    sha = code_sha()
    run_sha = frozen.sha256(run_path)
    if os.path.isfile(final):
        old = json.load(open(final))
        same = (
            old.get("meta", {}).get("code_sha") == sha
            and old.get("meta", {}).get("run_sha") == run_sha
            and old.get("summary", {}).get("n") == len(json.load(open(run_path))["results"])
        )
        if same:
            bad = old["summary"]["parse_fail"] > 0.5 * max(old["summary"]["n"], 1)
            print("factdiff skip", final, "parse_fail", old["summary"]["parse_fail"], flush=True)
            if bad:
                raise SystemExit(f"existing factdiff parse failure rate > 50%: {final}")
            return old
        print("factdiff redo; run, prompt, or code changed", final, flush=True)
    doc = json.load(open(run_path))
    rows = doc["results"]
    partial_path = final + ".partial.jsonl"
    done = {}
    if os.path.isfile(partial_path):
        for line in open(partial_path):
            if not line.strip():
                continue
            it = json.loads(line)
            if it.get("code_sha") == sha and it.get("run_sha") == run_sha:
                done[it["index"]] = it
    items = []
    t0 = time.time()
    with open(partial_path, "a") as part:
        for i, r in enumerate(rows):
            if i in done:
                items.append(done[i])
                continue
            if "pass1" not in r or "pass2" not in r:
                raise SystemExit(f"row {i} missing pass1/pass2")
            dropped, added, ok, raw = one(r["probe"], r["pass1"], r["pass2"])
            new_m, p1_m, p2_m = new_markers(r["pass1"], r["pass2"])
            fallback = (not ok) or bool(dropped) or bool(added) or bool(new_m)
            it = dict(
                index=i, code_sha=sha, run_sha=run_sha, kind=r["kind"], id=r["id"], char=r["char"],
                sample=r["sample"], probe=r["probe"],
                dropped=dropped, added=added, n_dropped=len(dropped), n_added=len(added),
                parse_ok=ok, raw=raw if not ok else "",
                pass1_markers=p1_m, pass2_markers=p2_m, new_markers=new_m,
                fallback=fallback,
            )
            part.write(json.dumps(it) + "\n")
            part.flush()
            items.append(it)
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"factdiff {i+1}/{len(rows)} {time.time()-t0:.0f}s fallback={sum(x['fallback'] for x in items)}", flush=True)
    summary = summarize(items)
    out = dict(
        meta=dict(
            run=os.path.abspath(run_path),
            run_sha=run_sha,
            cond=doc.get("meta", {}).get("cond"),
            model=JUDGE_MODEL, temp=TEMP, seed=SEED, num_predict=512,
            prompt_head=prompts.FACT_DIFF_HEAD,
            code_sha=sha,
            marker_scan="common.fail_hits; new = pass2 patterns minus pass1 patterns",
            fallback_rule="any dropped claim, any added claim, parse failure, or a new FAIL marker",
        ),
        summary=summary,
        items=[{k: v for k, v in it.items() if k != "code_sha"} for it in items],
    )
    if summary["n"] and summary["parse_fail"] > 0.5 * summary["n"]:
        frozen.dump_json(final + ".bad.json", out)
        raise SystemExit(f"factdiff parse failure rate > 50% ({summary['parse_fail']}/{summary['n']})")
    frozen.dump_json(final, out)
    if os.path.isfile(partial_path):
        os.remove(partial_path)
    print("factdiff wrote", final, json.dumps({k: summary[k] for k in (
        "n", "drop_claims", "add_claims", "drop_samples", "add_samples", "new_marker_samples", "parse_fail", "fallback_samples")}), flush=True)
    return out

def self_test():
    dropped, added = parse_lists('{"dropped": ["the tank is to the list"], "added": []}')
    assert dropped == ["the tank is to the list"] and added == []
    dropped, added = parse_lists('```json\n{"dropped": [], "added": ["Mars"]}\n```')
    assert dropped == [] and added == ["Mars"]
    try:
        parse_lists('{"nope": 1}')
        raise SystemExit("parse should have failed")
    except Exception:
        pass
    p = prompts.fact_diff_prompt("Q?", "yes", "yes")
    assert prompts.FACT_DIFF_HEAD in p and "PASS1:\nyes" in p and "PASS2:\nyes" in p
    assert prompts.RAG_MARK not in prompts.system_for("wren")
    user = prompts.rewrite_user("What heading?", "Mars.")
    assert user.endswith(prompts.REWRITE_INSTR)
    prompts.assert_no_retrieval(prompts.system_for("diane"), user)
    print("factdiff self-test ok")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
        return
    if len(sys.argv) != 2:
        raise SystemExit("usage: factdiff.py RUN.json | factdiff.py --self-test")
    run(sys.argv[1])

if __name__ == "__main__":
    main()
