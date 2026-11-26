"""Frame-break / character-break negatives + qwen3:14b+RAG rewrites for the combo stage.

Mines exp4, exp5, and any exp6 judged RAG runs, plus fresh samples from the winning adapter.
phase fresh  writes mined negatives and fresh answers.
phase finish scores, rewrites, writes the train file. Mixes 25% facts replay when the winner is (a).
"""
import os, sys, json, glob, collections, argparse
import common as C
import frozen
import voicefilt as V
import gpuutil
import scores

BREAK_AT = 2
FACTS_RATIO = 0.25

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path

WINNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "winner.json")

def winner_doc(path=None):
    p = path or WINNER_PATH
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing")
    return json.load(open(p))

def prompt_list():
    out = []
    seen = set()
    def add(char, user, ch, origin):
        if not user or not str(user).strip():
            return
        if char != "narrator" and char not in C.CARDS:
            return
        key = (char, user)
        if key in seen:
            return
        seen.add(key)
        out.append(dict(char=char, user=user, ch=ch or 3, origin=origin))
    for r in load_jsonl(frozen.VOICE_JSONL):
        add(r["char"], r["user"], r.get("ch") or 3, "voice")
    for t in C.parse_tests():
        add(t["char"], t["probe"], 3, "probe")
    for char, probe, _rubric in C.MUST:
        add(char, probe, 3, "probe")
    return out

def mine_file(path, origin):
    d = json.load(open(path))
    out = []
    for r in d.get("results") or []:
        voice = r.get("voice") or {}
        frame = bool(voice.get("frame"))
        score = r.get("voice_score")
        brk = score is not None and score <= BREAK_AT
        if not frame and not brk:
            continue
        ans = (r.get("answer") or "").strip()
        if not ans:
            continue
        reason = "+".join([x for x in ("frame" if frame else "", "break" if brk else "") if x])
        out.append(dict(char=r["char"], user=r.get("probe") or r.get("user"), bad=ans, ch=3,
                         origin=origin, frame=frame, voice_score=score, reason=reason))
    return out

def mine_all(out_dir):
    path = os.path.join(out_dir, "contrast_mined.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return load_jsonl(path)
    rows = []
    files = [
        (os.path.join(frozen.EXP4_RUNS, "q4_fv_rag.judged.json"), "q4_fv_rag"),
        (os.path.join(frozen.EXP4_RUNS, "q4_v_rag.judged.json"), "q4_v_rag"),
    ]
    files += [(p, "exp5:" + os.path.basename(p).replace(".judged.json", ""))
              for p in sorted(glob.glob(os.path.join(frozen.EXP5_RUNS, "*_rag.judged.json")))]
    here = os.path.dirname(os.path.abspath(__file__))
    files += [(p, "exp6:" + os.path.basename(p).replace(".judged.json", ""))
              for p in sorted(glob.glob(os.path.join(here, "runs", "*_rag.judged.json")))
              if os.path.basename(p) not in ("q14_rag.judged.json", "q4_fv_rag.judged.json")]
    for p, origin in files:
        if not os.path.exists(p):
            print("mine missing", p, flush=True)
            continue
        got = mine_file(p, origin)
        print(f"mine {origin} {len(got)}", flush=True)
        rows.extend(got)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows

def ensure_prompts(out_dir, limit):
    path = os.path.join(out_dir, "contrast_prompts.jsonl")
    have = {(r["char"], r["user"]): r for r in load_jsonl(path)}
    prompts = prompt_list()
    if limit:
        prompts = prompts[:limit]
    missing = [p for p in prompts if (p["char"], p["user"]) not in have]
    print(f"prompts {len(prompts)} need_sys {len(missing)}", flush=True)
    for i, p in enumerate(missing):
        p = dict(p)
        p["sys"] = V.sys_for(p["char"], p["user"], rag=True)
        append(path, p)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"sys {i+1}/{len(missing)}", flush=True)
    have = {(r["char"], r["user"]): r for r in load_jsonl(path)}
    ordered = []
    for p in prompts:
        row = have[(p["char"], p["user"])]
        if "sys" not in row:
            raise RuntimeError("prompt missing sys")
        ordered.append(row)
    return ordered

def fresh_spec(path=None):
    w = winner_doc(path)
    spec = dict(ollama=False, kind=w["mode"], base=w["base"], facts=w.get("facts") or "", voice=w["voice"])
    return spec, w

def generate_fresh(prompts, out_path, winner_path=None):
    import gc, torch
    from chat import render
    import eval4
    done = {(r["char"], r["user"]) for r in load_jsonl(out_path)}
    todo = [p for p in prompts if (p["char"], p["user"]) not in done]
    print(f"fresh todo {len(todo)} / {len(prompts)}", flush=True)
    if not todo:
        return
    spec, w = fresh_spec(winner_path)
    gpuutil.unload_ollama()
    tok, model = eval4.load_local(spec)
    try:
        for i, p in enumerate(todo):
            torch.manual_seed(30000 + i)
            ids = render(tok, [{"role": "system", "content": p["sys"]}, {"role": "user", "content": p["user"]}], True)
            batch = {"input_ids": torch.tensor([ids], device="cuda")}
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
            with torch.no_grad():
                out = model.generate(**batch, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95,
                                     pad_token_id=tok.pad_token_id)
            ans = tok.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            reasons = V.local_reasons(ans)
            row = dict(char=p["char"], user=p["user"], ch=p.get("ch") or 3, answer=ans,
                       frame="frame" in reasons, marker="marker" in reasons, origin="fresh:" + w["winner"])
            append(out_path, row)
            if (i + 1) == 1 or (i + 1) % 20 == 0:
                print(f"fresh {i+1}/{len(todo)} frame={row['frame']} {ans[:80].replace(chr(10), ' ')}", flush=True)
    finally:
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

def phase_fresh(a):
    os.makedirs(a.out_dir, exist_ok=True)
    mine_all(a.out_dir)
    prompts = ensure_prompts(a.out_dir, a.limit)
    gpuutil.unload_ollama()
    generate_fresh(prompts, os.path.join(a.out_dir, "contrast_fresh.jsonl"), a.winner)
    open(os.path.join(a.out_dir, "contrast_fresh.done"), "w").write(str(len(prompts)) + "\n")
    print("fresh done", len(prompts), flush=True)

def score_fresh(out_dir):
    fresh = load_jsonl(os.path.join(out_dir, "contrast_fresh.jsonl"))
    path = os.path.join(out_dir, "contrast_scored.jsonl")
    have = {(r["char"], r["user"]) for r in load_jsonl(path)}
    for i, r in enumerate(fresh):
        key = (r["char"], r["user"])
        if key in have:
            continue
        score = None
        if r.get("answer") and "empty" not in V.local_reasons(r["answer"]):
            score = V.voice_score(r["char"], r["user"], r["answer"])
        row = dict(r, voice_score=score)
        append(path, row)
        have.add(key)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"score {i+1}/{len(fresh)} {score} frame={r.get('frame')}", flush=True)
    return load_jsonl(path)

def select_negs(mined, scored, per_prompt=3):
    best = {}
    def consider(row):
        ans = (row.get("bad") or row.get("answer") or "").strip()
        if not ans:
            return
        frame = bool(row.get("frame"))
        score = row.get("voice_score")
        brk = score is not None and score <= BREAK_AT
        if not frame and not brk:
            return
        item = dict(char=row["char"], user=row["user"], ch=row.get("ch") or 3, bad=ans,
                    origin=row.get("origin"), frame=frame, voice_score=score,
                    reason="+".join([x for x in ("frame" if frame else "", "break" if brk else "") if x]))
        key = (item["char"], item["user"], item["bad"])
        if key not in best:
            best[key] = item
    for row in mined:
        consider(row)
    for row in scored:
        consider(row)
    by = collections.defaultdict(list)
    for item in best.values():
        by[(item["char"], item["user"])].append(item)
    chosen = []
    for key in by:
        items = sorted(by[key], key=lambda r: (0 if r["frame"] else 1, r["voice_score"] if r["voice_score"] is not None else 99))
        chosen.extend(items[:per_prompt])
    chosen.sort(key=lambda r: (0 if r["frame"] else 1, r["voice_score"] if r["voice_score"] is not None else 99, r["char"], r["user"]))
    return chosen

def group_prompts(negs):
    order = []
    by = collections.defaultdict(list)
    for r in negs:
        key = (r["char"], r["user"])
        if key not in by:
            order.append(key)
        by[key].append(r)
    return [(k, by[k]) for k in order]

def mix_facts(train, ratio):
    if ratio <= 0:
        return train, 0
    voice_n = len(train)
    n_facts = int(round(voice_n * ratio / (1.0 - ratio)))
    have = {(r["char"], r["user"]) for r in train}
    facts = []
    for r in load_jsonl(frozen.FACTS_JSONL):
        if (r["char"], r["user"]) in have:
            continue
        facts.append(dict(char=r["char"], user=r["user"], answer=r["answer"], ch=r.get("ch") or 3,
                          source="facts_replay", type=r.get("type") or "fact"))
        have.add((r["char"], r["user"]))
        if len(facts) >= n_facts:
            break
    return train + facts, len(facts)

def phase_finish(a):
    os.makedirs(a.out_dir, exist_ok=True)
    done_flag = os.path.join(a.out_dir, "contrast_fresh.done")
    if not os.path.exists(done_flag):
        raise SystemExit("contrast fresh phase has not finished")
    w = winner_doc(a.winner)
    mined = load_jsonl(os.path.join(a.out_dir, "contrast_mined.jsonl"))
    scored = score_fresh(a.out_dir)
    negs_path = os.path.join(a.out_dir, "contrast_negs.jsonl")
    if os.path.exists(negs_path):
        negs = load_jsonl(negs_path)
    else:
        negs = select_negs(mined, scored)
        groups = group_prompts(negs)
        if a.limit:
            keep_keys = {k for k, _ in groups[:a.limit]}
            negs = [r for r in negs if (r["char"], r["user"]) in keep_keys]
        dump(negs_path, negs)
    groups = group_prompts(negs)
    print(f"rewrite groups {len(groups)} negs {len(negs)}", flush=True)
    rw_path = os.path.join(a.out_dir, "contrast_rewrites.jsonl")
    have = {(r["char"], r["user"]): r for r in load_jsonl(rw_path) if r.get("final")}
    for i, ((char, user), bads) in enumerate(groups):
        if (char, user) in have:
            continue
        ch = bads[0].get("ch") or 3
        for attempt, temp, seed in ((1, 0.7, 40000 + i), (2, 0.3, 50000 + i)):
            ans = V.generate(char, user, temp=temp, seed=seed)
            keep, reasons, info = V.decide(char, user, ans)
            row = dict(char=char, user=user, ch=ch, attempt=attempt, temp=temp, seed=seed, answer=ans,
                       keep=keep, reasons=reasons, final=bool(keep) or attempt == 2, **info)
            append(rw_path, row)
            print(f"rewrite {i+1}/{len(groups)} a{attempt} keep={keep} {reasons} score={info.get('voice_score')}", flush=True)
            if keep:
                break
    rewrites = load_jsonl(rw_path)
    best = {}
    for r in rewrites:
        if r.get("keep"):
            prev = best.get((r["char"], r["user"]))
            if prev is None or (r.get("voice_score") or 0) > (prev.get("voice_score") or 0):
                best[(r["char"], r["user"])] = r
    pairs = []
    for (char, user), bads in groups:
        good = best.get((char, user))
        if not good:
            continue
        seen_bad = set()
        for b in bads:
            if b["bad"] in seen_bad or b["bad"].strip() == good["answer"].strip():
                continue
            seen_bad.add(b["bad"])
            pairs.append(dict(char=char, user=user, ch=b.get("ch") or 3, answer=good["answer"], bad=b["bad"],
                              source="contrast", reason=b.get("reason"), origin=b.get("origin"),
                              voice_score=good.get("voice_score")))
    voice_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "v_replay_train.jsonl" if w["mode"] == "fv" else "v_merged_train.jsonl")
    voice = load_jsonl(voice_src) if os.path.exists(voice_src) else load_jsonl(frozen.VOICE_JSONL)
    voice = [r for r in voice if r.get("source") != "facts_replay"]
    replaced_keys = {(p["char"], p["user"]) for p in pairs}
    train = []
    seen = set()
    def push(item):
        sig = (item["char"], item["user"], item["answer"], item.get("bad"))
        if sig in seen:
            return
        seen.add(sig)
        train.append(item)
    for r in voice:
        if (r["char"], r["user"]) in replaced_keys:
            continue
        push(dict(char=r["char"], user=r["user"], answer=r["answer"], ch=r.get("ch") or 3, source=r.get("source") or "voice"))
    for p in pairs:
        push(dict(char=p["char"], user=p["user"], answer=p["answer"], bad=p["bad"], ch=p["ch"], source="contrast"))
    facts_n = 0
    ratio = FACTS_RATIO if w["mode"] == "fv" else 0.0
    train, facts_n = mix_facts(train, ratio)
    dump(os.path.join(a.out_dir, "contrast_pairs.jsonl"), pairs)
    train_path = dump(os.path.join(a.out_dir, "v_contrast_train.jsonl"), train)
    origins = collections.Counter(r.get("origin") for r in negs)
    reasons = collections.Counter(r.get("reason") for r in negs)
    facts_ratio = (facts_n / len(train)) if train else 0.0
    report = dict(
        mined=len(mined),
        fresh=len(scored),
        fresh_frame=sum(1 for r in scored if r.get("frame")),
        fresh_break=sum(1 for r in scored if (r.get("voice_score") is not None and r["voice_score"] <= BREAK_AT)),
        negs=len(negs),
        neg_origins=dict(origins),
        neg_reasons=dict(reasons),
        rewrite_groups=len(groups),
        rewrites_kept=len(best),
        rewrites_rejected=len(groups) - len(best),
        pairs=len(pairs),
        train_rows=len(train),
        facts_rows=facts_n,
        facts_ratio=round(facts_ratio, 6),
        train_sha256=frozen.sha256(train_path),
        break_rule=f"frame regex or voice_score <= {BREAK_AT}",
        loss="unlikelihood",
        ul_weight=1.0,
        teacher="qwen3:14b + RAG ch3",
        fresh_model=w["winner"] + " " + w["mode"],
        winner=w["winner"],
        limit=a.limit,
    )
    json.dump(report, open(os.path.join(a.out_dir, "contrast_report.json"), "w"), indent=1)
    print("contrast report", json.dumps(report), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fresh", "finish"])
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--winner", default="")
    a = ap.parse_args()
    if a.phase == "fresh":
        phase_fresh(a)
    else:
        phase_finish(a)

if __name__ == "__main__":
    main()
