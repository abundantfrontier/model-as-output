"""Build the light DPO, light contrast, and facts-replay files for exp7.

Frame-break (common.FRAME) and marker-leak (common.fail_hits) answers are never
chosen. Contrast negatives are only those hits, not voice_score <= 2.
Reads exp4/exp5/exp6. Writes only under --out-dir.
"""
import os, json, re, glob, random, collections, argparse
import common as C
import frozen

PER_PROMPT = 2

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def dump(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path

def append(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def flags(ans):
    ans = ans or ""
    return bool(re.search(C.FRAME, ans, re.I)), bool(C.fail_hits(ans))

def ok_char(char):
    return char == "narrator" or char in C.CARDS

def normalize_pair(row, source):
    """Put a dirty answer on the rejected side, or drop the pair."""
    if not ok_char(row.get("char")):
        return None
    ans = (row.get("answer") or "").strip()
    bad = (row.get("bad") or "").strip()
    if not ans or not bad or ans == bad:
        return None
    cf, cm = flags(ans)
    rf, rm = flags(bad)
    swapped = False
    if (cf or cm) and not (rf or rm):
        ans, bad = bad, ans
        swapped = True
        cf, cm, rf, rm = rf, rm, cf, cm
    if cf or cm:
        return None
    reason = "frame" if rf else ("marker" if rm else "voice")
    score = None if swapped else row.get("voice_score")
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
    return dict(char=row["char"], user=row["user"], ch=row.get("ch") or 3,
                answer=ans, bad=bad, source=source, reason=reason, voice_score=score)

def tier_of(row):
    if row["reason"] == "frame":
        return 0
    if row["reason"] == "marker":
        return 1
    vs = row.get("voice_score")
    if vs is not None and vs >= 5:
        return 2
    if vs is not None and vs >= 4.5:
        return 3
    return 4

def select_pairs(rows, cap):
    seen = set()
    uniq = []
    for r in rows:
        sig = (r["char"], r["user"], r["answer"], r["bad"])
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(r)
    uniq.sort(key=lambda r: (tier_of(r), -(r.get("voice_score") or -1), r["source"], r["char"], r["user"]))
    forced = [r for r in uniq if tier_of(r) <= 1]
    if cap and len(forced) > cap:
        return forced, True
    if cap:
        return uniq[:cap], False
    return uniq, False

def mine_negs():
    rows = []
    files = []
    for folder, prefix in (
        (frozen.EXP4_RUNS, "exp4:"),
        (frozen.EXP5_RUNS, "exp5:"),
        (frozen.EXP6_RUNS, "exp6:"),
    ):
        for p in sorted(glob.glob(os.path.join(folder, "*rag.judged.json"))):
            base = os.path.basename(p)
            if base.startswith(("q14", "p4", "s3")) or "v8_" in base:
                continue
            files.append((p, prefix + base.replace(".judged.json", "")))
    for p, origin in files:
        d = json.load(open(p))
        for r in d.get("results") or []:
            ans = (r.get("answer") or "").strip()
            char = r.get("char")
            user = r.get("probe") or r.get("user")
            if not ans or not ok_char(char) or not user:
                continue
            fr, mk = flags(ans)
            if not fr and not mk:
                continue
            rows.append(dict(char=char, user=user, ch=3, bad=ans, origin=origin,
                             reason="frame" if fr else "marker"))
    for r in load_jsonl(os.path.join(frozen.EXP6_DATA, "contrast_fresh.jsonl")):
        ans = (r.get("answer") or "").strip()
        fr, mk = flags(ans)
        if (fr or mk or r.get("frame") or r.get("marker")) and ans and ok_char(r.get("char")):
            # re-check the text; stored flags can be stale relative to FRAME
            fr, mk = flags(ans)
            if fr or mk:
                rows.append(dict(char=r["char"], user=r["user"], ch=r.get("ch") or 3, bad=ans,
                                 origin="fresh:" + str(r.get("origin") or "exp6"),
                                 reason="frame" if fr else "marker"))
    for r in load_jsonl(os.path.join(frozen.EXP6_DATA, "contrast_mined.jsonl")):
        ans = (r.get("bad") or "").strip()
        fr, mk = flags(ans)
        if not fr and not mk:
            continue
        if not ok_char(r.get("char")) or not r.get("user"):
            continue
        rows.append(dict(char=r["char"], user=r["user"], ch=r.get("ch") or 3, bad=ans,
                         origin="mined:" + str(r.get("origin") or ""),
                         reason="frame" if fr else "marker"))
    return rows

def chosen_pool():
    pool = {}
    def add(char, user, answer, score, source, rank):
        if not ok_char(char) or not user:
            return
        answer = (answer or "").strip()
        if not answer or flags(answer) != (False, False):
            return
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None
        item = dict(answer=answer, voice_score=score, source=source, rank=rank)
        prev = pool.get((char, user))
        key = (score if score is not None else -1, -rank)
        prev_key = (-1, 99)
        if prev is not None:
            prev_key = (prev["voice_score"] if prev["voice_score"] is not None else -1, -prev["rank"])
        if prev is None or key > prev_key:
            pool[(char, user)] = item
    for path, source in (
        (os.path.join(frozen.EXP5_DATA, "distill_pairs.jsonl"), "distill"),
        (os.path.join(frozen.EXP5_DATA, "contrast_pairs.jsonl"), "contrast_exp5"),
        (os.path.join(frozen.EXP6_DATA, "contrast_pairs.jsonl"), "contrast_exp6"),
    ):
        for r in load_jsonl(path):
            add(r.get("char"), r.get("user"), r.get("answer"), r.get("voice_score"), source, 0)
    for r in load_jsonl(os.path.join(frozen.EXP6_DATA, "contrast_rewrites.jsonl")):
        if r.get("keep"):
            add(r.get("char"), r.get("user"), r.get("answer"), r.get("voice_score"), "exp6_rewrite", 0)
    for r in load_jsonl(frozen.VOICE_JSONL):
        add(r.get("char"), r.get("user"), r.get("answer"), None, "voice_jsonl", 2)
    return pool

def add_rewrites(pool, path):
    for r in load_jsonl(path):
        if not r.get("keep"):
            continue
        answer = (r.get("answer") or "").strip()
        if not answer or flags(answer) != (False, False):
            continue
        score = r.get("voice_score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        prev = pool.get((r["char"], r["user"]))
        item = dict(answer=answer, voice_score=score, source="light_rewrite", rank=0)
        prev_key = (-1, 99)
        if prev is not None:
            prev_key = (prev["voice_score"] if prev["voice_score"] is not None else -1, -prev["rank"])
        key = (score if score is not None else -1, 0)
        if prev is None or key > prev_key:
            pool[(r["char"], r["user"])] = item
    return pool

def dedup_negs(rows):
    seen = set()
    out = []
    for r in rows:
        sig = (r["char"], r["user"], r["bad"])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(r)
    return out

def cap_negs(rows, per_prompt):
    by = collections.defaultdict(list)
    order = []
    for r in rows:
        key = (r["char"], r["user"])
        if key not in by:
            order.append(key)
        by[key].append(r)
    kept = []
    for key in order:
        items = sorted(by[key], key=lambda r: (0 if r["reason"] == "frame" else 1, r["origin"], r["bad"]))
        kept.extend(items[:per_prompt])
    return kept

def constructed_pairs(negs, pool):
    out = []
    for r in negs:
        good = pool.get((r["char"], r["user"]))
        if not good:
            continue
        if good["answer"].strip() == r["bad"].strip():
            continue
        out.append(dict(char=r["char"], user=r["user"], ch=r.get("ch") or 3,
                        answer=good["answer"], bad=r["bad"], source="light_" + r["reason"],
                        reason=r["reason"], voice_score=good.get("voice_score"),
                        origin=r.get("origin"), chosen_source=good.get("source")))
    return out

def existing_pairs():
    out = []
    for path, source in (
        (os.path.join(frozen.EXP5_DATA, "distill_pairs.jsonl"), "distill"),
        (os.path.join(frozen.EXP5_DATA, "contrast_pairs.jsonl"), "contrast_exp5"),
        (os.path.join(frozen.EXP6_DATA, "contrast_pairs.jsonl"), "contrast_exp6"),
    ):
        for r in load_jsonl(path):
            item = normalize_pair(r, source)
            if item:
                out.append(item)
    return out

def hot_prompts(negs):
    seen = set()
    out = []
    for r in negs:
        key = (r["char"], r["user"])
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(char=r["char"], user=r["user"], ch=r.get("ch") or 3))
    return out

def ensure_sys(prompts, path):
    import voicefilt as V
    have = {(r["char"], r["user"]): r for r in load_jsonl(path)}
    for p in prompts:
        key = (p["char"], p["user"])
        if key in have and have[key].get("sys"):
            p["sys"] = have[key]["sys"]
            continue
        row = dict(p)
        row["sys"] = V.sys_for(p["char"], p["user"], rag=True)
        append(path, row)
        p["sys"] = row["sys"]
        print("sys", p["char"], p["user"][:60], flush=True)
    return prompts

def generate_fresh(prompts, out_path, per_prompt, sample_limit):
    import gc, torch
    import gpuutil
    import eval4
    from chat import render
    done = collections.Counter((r["char"], r["user"]) for r in load_jsonl(out_path))
    todo = []
    for p in prompts:
        need = per_prompt - done[(p["char"], p["user"])]
        for _ in range(max(0, need)):
            todo.append(p)
    if sample_limit:
        todo = todo[:sample_limit]
    print(f"fresh todo {len(todo)}", flush=True)
    if not todo:
        return 0
    gpuutil.unload_ollama()
    spec = dict(ollama=False, kind="v", base=frozen.MERGED, facts="", voice=frozen.INIT)
    tok, model = eval4.load_local(spec)
    n = 0
    try:
        for i, p in enumerate(todo):
            torch.manual_seed(31000 + i)
            ids = render(tok, [{"role": "system", "content": p["sys"]}, {"role": "user", "content": p["user"]}], True)
            batch = {"input_ids": torch.tensor([ids], device="cuda")}
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
            with torch.no_grad():
                out = model.generate(**batch, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95,
                                     pad_token_id=tok.pad_token_id)
            ans = tok.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            fr, mk = flags(ans)
            row = dict(char=p["char"], user=p["user"], ch=p.get("ch") or 3, answer=ans,
                       frame=fr, marker=mk, origin="fresh:v_merged")
            append(out_path, row)
            n += 1
            print(f"fresh {i+1}/{len(todo)} frame={fr} marker={mk} {ans[:80].replace(chr(10), ' ')}", flush=True)
    finally:
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
    return n

def fresh_as_negs(path):
    rows = []
    for r in load_jsonl(path):
        ans = (r.get("answer") or "").strip()
        fr, mk = flags(ans)
        if not fr and not mk:
            continue
        rows.append(dict(char=r["char"], user=r["user"], ch=r.get("ch") or 3, bad=ans,
                         origin=r.get("origin") or "fresh:v_merged", reason="frame" if fr else "marker"))
    return rows

CONSTRAINT = ("Reply in character in 1-3 short sentences. "
              "Do not use these words: chapter, story, character, prompt, AI, assistant, model. "
              "Do not apologize and do not say you cannot assist.")

def teacher_answer(char, user, temp, seed, constrained):
    import voicefilt as V
    sysm = V.sys_for(char, user, rag=True)
    if constrained:
        sysm += "\n\n" + CONSTRAINT
    ans, *_ = C.ollama_chat(
        "qwen3:14b",
        [{"role": "system", "content": sysm}, {"role": "user", "content": user}],
        temp=temp, seed=seed, num_predict=200,
    )
    return ans

def drop_dirty_sentences(ans):
    """Keep only sentences that are not a frame break or a marker leak."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (ans or "").strip()) if s.strip()]
    kept = [s for s in sents if flags(s) == (False, False)]
    return " ".join(kept).strip()

def teacher_rewrite(char, user, draft, seed):
    import voicefilt as V
    card, lines = V.card_and_lines(char)
    # The probe often contains "chapter". Quoting it makes the rewrite copy the word back in.
    ask = user if flags(user) == (False, False) else "The user asked about something this character has not seen."
    clean = drop_dirty_sentences(draft) or draft
    sysm = (f"Character card: {card}\nReal lines by this character:\n- " + "\n- ".join(lines) +
            "\n\nRewrite DRAFT as this character's reply to USER. First person. Output only the reply, 1-3 short sentences. "
            + CONSTRAINT)
    ans, *_ = C.ollama_chat(
        "qwen3:14b",
        [{"role": "system", "content": sysm},
         {"role": "user", "content": f"USER: {ask}\nDRAFT: {clean}"}],
        temp=0.2, seed=seed, num_predict=160,
    )
    return ans

def run_teacher(prompts, path, limit):
    import voicefilt as V
    have = {(r["char"], r["user"]) for r in load_jsonl(path) if r.get("final") or r.get("keep")}
    n = 0
    for i, p in enumerate(prompts):
        if (p["char"], p["user"]) in have:
            continue
        if limit and n >= limit:
            break
        n += 1
        ch = p.get("ch") or 3
        # Chapter-5 probes echo the word "chapter". Later attempts strip or rewrite that sentence.
        draft = ""
        plans = (
            ("answer", 1, 0.7, 41000 + i, False),
            ("answer", 2, 0.3, 51000 + i, True),
            ("strip", 3, 0.0, 0, True),
            ("rewrite", 4, 0.2, 61000 + i, True),
        )
        for mode, attempt, temp, seed, constrained in plans:
            if mode == "answer":
                ans = teacher_answer(p["char"], p["user"], temp, seed, constrained)
                draft = ans
            elif mode == "strip":
                ans = drop_dirty_sentences(draft)
                if not ans or ans == draft:
                    continue
            else:
                if not draft:
                    break
                ans = teacher_rewrite(p["char"], p["user"], draft, seed)
            keep, reasons, info = V.decide(p["char"], p["user"], ans)
            last = mode == "rewrite"
            row = dict(char=p["char"], user=p["user"], ch=ch, attempt=attempt, temp=temp, seed=seed,
                       constrained=constrained, mode=mode, answer=ans, keep=keep, reasons=reasons,
                       final=bool(keep) or last, **info)
            append(path, row)
            print(f"rewrite {n} a{attempt} {mode} keep={keep} {reasons} score={info.get('voice_score')}", flush=True)
            if keep or last:
                break
    return n

def missing_chosen(negs, pool):
    seen = set()
    out = []
    for r in negs:
        key = (r["char"], r["user"])
        if key in seen:
            continue
        prev = pool.get(key)
        if prev is not None and prev["rank"] < 2 and (prev.get("voice_score") is None or prev["voice_score"] >= 4.5):
            continue
        seen.add(key)
        out.append(dict(char=r["char"], user=r["user"], ch=r.get("ch") or 3))
    return out

def must_score(row):
    ans = row.get("answer") or ""
    user = (row.get("user") or "").lower()
    char = row.get("char")
    score = 0
    uw = set(re.findall(r"[a-z0-9']+", user))
    for c, q, rub in C.MUST:
        if c != char:
            continue
        if re.search(rub, ans, re.I):
            score += 2
        qw = set(re.findall(r"[a-z0-9']+", q.lower()))
        overlap = len(qw & uw)
        if overlap >= 3:
            score += overlap
    if char == "yd4408" and re.search(r"4408|forty-four", ans, re.I):
        score += 3
    return score

def mix_facts(pairs, ratio):
    pref = [r for r in pairs if r.get("bad")]
    n_facts = int(round(len(pref) * ratio / (1.0 - ratio))) if pref and ratio > 0 else 0
    have = {(r["char"], r["user"]) for r in pairs}
    scored = []
    for r in load_jsonl(frozen.FACTS_JSONL):
        if not ok_char(r.get("char")) or not r.get("user"):
            continue
        if (r["char"], r["user"]) in have:
            continue
        if (r.get("type") or "") == "spoiler":
            continue
        ans = (r.get("answer") or "").strip()
        if not ans or flags(ans) != (False, False):
            continue
        scored.append((must_score(r), r))
    scored.sort(key=lambda t: (-t[0], t[1]["char"], t[1]["user"]))
    forced = [r for s, r in scored if s > 0]
    rest = [r for s, r in scored if s <= 0]
    rng = random.Random(0)
    rng.shuffle(rest)
    picked = []
    for r in forced + rest:
        if len(picked) >= n_facts:
            break
        picked.append(dict(char=r["char"], user=r["user"], ch=r.get("ch") or 3, answer=r["answer"],
                           source="facts_replay"))
        have.add((r["char"], r["user"]))
    return picked, len(forced)

def write_all(out_dir, pairs, contrast, replay, meta_extra):
    dpo_path = dump(os.path.join(out_dir, "v_light_dpo_pairs.jsonl"), pairs)
    con_path = dump(os.path.join(out_dir, "v_light_contrast_train.jsonl"), contrast)
    rep_path = dump(os.path.join(out_dir, "v_light_dpo_replay.jsonl"), replay)
    reasons = collections.Counter(r["reason"] for r in pairs)
    creasons = collections.Counter(r["reason"] for r in contrast)
    sources = collections.Counter(r["source"] for r in pairs)
    n_pref = sum(1 for r in replay if r.get("bad"))
    n_facts = sum(1 for r in replay if not r.get("bad"))
    dpo_meta = dict(
        pairs=len(pairs), preference_pairs=len(pairs),
        reasons=dict(reasons), sources=dict(sources),
        sha256=frozen.sha256(dpo_path), **meta_extra,
    )
    con_meta = dict(
        train_rows=len(contrast), reasons=dict(creasons),
        sha256=frozen.sha256(con_path),
        neg_rule="FRAME regex or marker fail_hits only; no voice_score<=2; no full voice SFT mix",
        per_prompt=PER_PROMPT, **{k: meta_extra[k] for k in ("fresh", "teacher_prompts", "limit") if k in meta_extra},
    )
    rep_meta = dict(
        pairs=len(replay), preference_pairs=n_pref, facts_replay=n_facts,
        facts_ratio=round(n_facts / len(replay), 6) if replay else 0,
        sha256=frozen.sha256(rep_path),
        facts_ratio_target=meta_extra.get("facts_ratio_target"),
        must_priority_available=meta_extra.get("must_priority_available"),
    )
    json.dump(dpo_meta, open(os.path.join(out_dir, "v_light_dpo_meta.json"), "w"), indent=1)
    json.dump(con_meta, open(os.path.join(out_dir, "v_light_contrast_report.json"), "w"), indent=1)
    json.dump(rep_meta, open(os.path.join(out_dir, "v_light_replay_meta.json"), "w"), indent=1)
    print("dpo", json.dumps(dpo_meta), flush=True)
    print("contrast", json.dumps(con_meta), flush=True)
    print("replay", json.dumps(rep_meta), flush=True)

def prepare(a):
    self_test()
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    done = [os.path.join(out, name) for name in
            ("v_light_dpo_meta.json", "v_light_contrast_report.json", "v_light_replay_meta.json")]
    if all(os.path.isfile(p) for p in done) and not a.force:
        print("prepare skip", flush=True)
        return
    negs = dedup_negs(mine_negs())
    pool = chosen_pool()
    prompts = hot_prompts(negs)
    if not a.skip_fresh:
        sys_path = os.path.join(out, "light_prompts.jsonl")
        ensure_sys(prompts, sys_path)
        fresh_path = os.path.join(out, "fresh_v_merged.jsonl")
        generate_fresh(prompts, fresh_path, a.fresh_per, a.fresh_limit)
        negs = dedup_negs(negs + fresh_as_negs(fresh_path))
    if not a.skip_teacher:
        need = missing_chosen(negs, pool)
        if a.limit:
            need = need[:a.limit]
        run_teacher(need, os.path.join(out, "light_rewrites.jsonl"), a.teacher_limit)
        add_rewrites(pool, os.path.join(out, "light_rewrites.jsonl"))
    negs = cap_negs(dedup_negs(negs), a.per_prompt)
    made = constructed_pairs(negs, pool)
    have = existing_pairs()
    cap = a.pair_cap
    if a.limit:
        cap = min(cap, a.limit) if cap else a.limit
    chosen, exceeded = select_pairs(made + have, cap)
    if not chosen:
        raise SystemExit("no DPO pairs after filtering")
    facts, n_must = mix_facts(chosen, a.facts_ratio)
    replay = list(chosen) + facts
    if not made:
        print("warning: no frame/marker contrast rows; contrast train would be empty", flush=True)
    write_all(out, chosen, made, replay, dict(
        pair_cap=cap, cap_exceeded_by_frame_marker=exceeded,
        facts_ratio_target=a.facts_ratio, must_priority_available=n_must,
        fresh=0 if a.skip_fresh else len(load_jsonl(os.path.join(out, "fresh_v_merged.jsonl"))),
        teacher_rows=len(load_jsonl(os.path.join(out, "light_rewrites.jsonl"))),
        teacher_kept=len({(r["char"], r["user"]) for r in load_jsonl(os.path.join(out, "light_rewrites.jsonl")) if r.get("keep")}),
        limit=a.limit, mined_negs=len(negs), constructed=len(made),
    ))

def self_test():
    dirty = "Chapter 5 is not on the list."
    clean = "The air is good. Twenty-one."
    marker = "We can't come home."
    assert flags(dirty)[0] and not flags(dirty)[1]
    assert flags(marker) == (False, True)
    assert flags(clean) == (False, False)
    swapped = normalize_pair(dict(char="wren", user="q", answer=dirty, bad=clean, ch=3, voice_score=5), "t")
    assert swapped and swapped["answer"] == clean and swapped["bad"] == dirty and swapped["reason"] == "frame"
    assert swapped["voice_score"] is None
    both = normalize_pair(dict(char="wren", user="q", answer=dirty, bad=marker), "t")
    assert both is None
    kept = normalize_pair(dict(char="diane", user="q", answer=clean, bad=marker, voice_score=5), "t")
    assert kept["reason"] == "marker" and kept["voice_score"] == 5
    same = normalize_pair(dict(char="diane", user="q", answer=clean, bad=clean), "t")
    assert same is None
    stripped = drop_dirty_sentences("I don't know what happens in chapter 5. The air is good. Twenty-one.")
    assert flags(stripped) == (False, False) and "chapter" not in stripped.lower()
    assert drop_dirty_sentences("Chapter 5 is next.") == ""
    rows, exceeded = select_pairs([
        dict(char="a", user="u", answer="g", bad="b1", source="s", reason="voice", voice_score=5),
        dict(char="a", user="u2", answer="g", bad="b2", source="s", reason="frame", voice_score=None),
    ], cap=1)
    assert len(rows) == 1 and rows[0]["reason"] == "frame" and exceeded is False
    print("build7 self-test ok", flush=True)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--pair-cap", type=int, default=141)
    p.add_argument("--facts-ratio", type=float, default=0.08)
    p.add_argument("--per-prompt", type=int, default=PER_PROMPT)
    p.add_argument("--fresh-per", type=int, default=2)
    p.add_argument("--fresh-limit", type=int, default=0)
    p.add_argument("--teacher-limit", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-fresh", action="store_true")
    p.add_argument("--skip-teacher", action="store_true")
    p.add_argument("--force", action="store_true")
    sub.add_parser("self-test")
    a = ap.parse_args()
    if a.cmd == "self-test":
        self_test()
    else:
        prepare(a)

if __name__ == "__main__":
    main()
