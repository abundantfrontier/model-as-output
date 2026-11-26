"""Merge distill and contrast chosen/rejected pairs into data/v_dpo_pairs.jsonl.

Prefer exp5 distill/contrast pairs plus any new exp6 contrast pairs.
If the winner is stacked (a), append a 20% facts-replay CE mix (rows without `bad`).
"""
import os, json, random
import frozen

H = os.path.dirname(os.path.abspath(__file__))
FACTS_RATIO = 0.20

def load(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def winner():
    p = os.path.join(H, "data", "winner.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def main():
    os.makedirs(os.path.join(H, "data"), exist_ok=True)
    seen = set()
    out = []
    src = {"distill": 0, "contrast_exp5": 0, "contrast_exp6": 0, "facts_replay": 0}
    files = [
        (os.path.join(frozen.EXP5_DATA, "distill_pairs.jsonl"), "distill"),
        (os.path.join(frozen.EXP5_DATA, "contrast_pairs.jsonl"), "contrast_exp5"),
        (os.path.join(H, "data", "contrast_pairs.jsonl"), "contrast_exp6"),
    ]
    for path, source in files:
        for r in load(path):
            ans = (r.get("answer") or "").strip()
            bad = (r.get("bad") or "").strip()
            if not ans or not bad or ans == bad:
                continue
            sig = (r["char"], r["user"], ans, bad)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(dict(char=r["char"], user=r["user"], ch=r.get("ch") or 3, answer=ans, bad=bad, source=source))
            src[source] += 1
    w = winner()
    if w.get("mode") == "fv":
        n_facts = int(round(len(out) * FACTS_RATIO / (1.0 - FACTS_RATIO))) if out else 0
        have = {(r["char"], r["user"]) for r in out}
        facts = []
        rng = random.Random(0)
        pool = load(frozen.FACTS_JSONL)
        rng.shuffle(pool)
        for r in pool:
            if (r["char"], r["user"]) in have:
                continue
            facts.append(dict(char=r["char"], user=r["user"], ch=r.get("ch") or 3,
                              answer=r["answer"], source="facts_replay"))
            have.add((r["char"], r["user"]))
            if len(facts) >= n_facts:
                break
        out.extend(facts)
        src["facts_replay"] = len(facts)
    path = os.path.join(H, "data", "v_dpo_pairs.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pref = sum(1 for r in out if r.get("bad"))
    meta = dict(pairs=len(out), preference_pairs=pref, **src, sha256=frozen.sha256(path),
                facts_ratio=round(src["facts_replay"] / len(out), 6) if out else 0)
    json.dump(meta, open(os.path.join(H, "data", "dpo_pairs_meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)

if __name__ == "__main__":
    main()
