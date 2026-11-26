"""Build voice training mixes.

v_replay: exp4 voice.jsonl overlaid with exp5 distill/contrast positives, plus 25% facts replay.
v_merged: the same voice positives with no facts (facts live in the merged base).

Target facts ratio 0.25 (inside 20-30%). Seed 0. Resumable via the mix report.
"""
import os, json, random, argparse, collections
import frozen

H = os.path.dirname(os.path.abspath(__file__))
TARGET = 0.25
SEED = 0

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def dump_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path

def slim(r, source):
    return dict(char=r["char"], user=r["user"], answer=r["answer"], ch=r.get("ch") or 3,
                source=source, type=r.get("type") or "voice")

def voice_pool():
    """Prefer later, higher-voice sources for the same (char, user). Keep extra prompts."""
    by = {}
    origin = collections.Counter()
    def put(rows, source, replace=True):
        added = replaced = 0
        for r in rows:
            if not r.get("answer") or not r.get("user"):
                continue
            if r.get("char") is None:
                continue
            key = (r["char"], r["user"])
            item = slim(r, source)
            if key in by:
                if replace:
                    by[key] = item
                    replaced += 1
            else:
                by[key] = item
                added += 1
            origin[source] += 1
        return added, replaced
    put(load_jsonl(frozen.VOICE_JSONL), "voice", replace=True)
    dist = load_jsonl(os.path.join(frozen.EXP5_DATA, "v_distill_train.jsonl"))
    put(dist, "distill", replace=True)
    kept = load_jsonl(os.path.join(frozen.EXP5_DATA, "distill_kept.jsonl"))
    put(kept, "distill_kept", replace=True)
    rew = [r for r in load_jsonl(os.path.join(frozen.EXP5_DATA, "contrast_rewrites.jsonl")) if r.get("keep")]
    put(rew, "contrast_rewrite", replace=True)
    rows = list(by.values())
    rows.sort(key=lambda r: (r["char"], r["user"]))
    return rows, dict(origin)

def sample_facts(n, exclude, rng):
    facts = []
    for r in load_jsonl(frozen.FACTS_JSONL):
        if not r.get("answer") or not r.get("user"):
            continue
        key = (r["char"], r["user"])
        if key in exclude:
            continue
        facts.append(dict(char=r["char"], user=r["user"], answer=r["answer"], ch=r.get("ch") or 3,
                          source="facts_replay", type=r.get("type") or "fact"))
    rng.shuffle(facts)
    return facts[:n]

def mix(voice, facts_ratio, limit, rng):
    if limit:
        voice = voice[:limit]
    n_voice = len(voice)
    if facts_ratio <= 0:
        facts = []
    else:
        # facts / (voice+facts) = ratio  =>  facts = voice * ratio / (1-ratio)
        n_facts = int(round(n_voice * facts_ratio / (1.0 - facts_ratio)))
        facts = sample_facts(n_facts, {(r["char"], r["user"]) for r in voice}, rng)
    rows = list(voice) + list(facts)
    rng.shuffle(rows)
    total = len(rows)
    ratio = (len(facts) / total) if total else 0.0
    if facts_ratio > 0 and not (0.20 - 1e-9 <= ratio <= 0.30 + 1e-9):
        raise RuntimeError(f"facts mix ratio {ratio:.4f} outside 20-30% (voice={n_voice} facts={len(facts)})")
    return rows, facts, ratio

def write_one(path, report_path, voice, facts_ratio, limit, rng):
    if os.path.exists(report_path) and os.path.exists(path):
        print("skip", path, flush=True)
        return json.load(open(report_path))
    rows, facts, ratio = mix(voice, facts_ratio, limit, rng)
    dump_jsonl(path, rows)
    src = collections.Counter(r.get("source") for r in rows)
    report = dict(
        path=os.path.abspath(path),
        rows=len(rows),
        voice_rows=len(rows) - len(facts),
        facts_rows=len(facts),
        facts_ratio=round(ratio, 6),
        facts_ratio_target=facts_ratio,
        sha256=frozen.sha256(path),
        sources=dict(src),
        limit=limit,
        seed=SEED,
        voice_jsonl=frozen.VOICE_SHA,
        facts_jsonl=frozen.FACTS_JSONL_SHA,
    )
    json.dump(report, open(report_path, "w"), indent=1)
    print(json.dumps(report), flush=True)
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(H, "data"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--facts-ratio", type=float, default=TARGET)
    a = ap.parse_args()
    frozen.check()
    os.makedirs(a.out_dir, exist_ok=True)
    voice, origin = voice_pool()
    print("voice pool", len(voice), origin, flush=True)
    json.dump(dict(voice_pool=len(voice), origin=origin), open(os.path.join(a.out_dir, "voice_pool.json"), "w"), indent=1)
    rng = random.Random(SEED)
    write_one(os.path.join(a.out_dir, "v_replay_train.jsonl"),
              os.path.join(a.out_dir, "v_replay_mix.json"),
              voice, a.facts_ratio, a.limit, rng)
    rng = random.Random(SEED)
    write_one(os.path.join(a.out_dir, "v_merged_train.jsonl"),
              os.path.join(a.out_dir, "v_merged_mix.json"),
              voice, 0.0, a.limit, rng)

if __name__ == "__main__":
    main()
