"""Filter exp3 synth + distill into data/clean.jsonl.

Drops contamination (gen.clean: leak markers or ch4-5 8-grams), frame breaks,
near-duplicates, and weak/empty refusals. Balances character x type.
"""
import os, re, json, random, argparse, collections
import numpy as np
import common as C
from gen import LATER8, g8

H = os.path.dirname(os.path.abspath(__file__))
CHARS = ["wren", "diane", "yd4408", "narrator"]
CHAR_NAME = {"wren": "Wren", "diane": "Diane", "yd4408": "4408", "narrator": "narrator"}
TYPES = ["fact", "voice", "spoiler"]
COSINE = 0.93
SPOIL_RE = re.compile(
    r"\b(spoil|secret|one[- ]way|come back|go back|going back|return fuel|enough fuel|"
    r"who filled|who signed|what happens (next|later|after)|after (day|this)|day [4-9]|"
    r"chapter [4-9]|the ending|mara|behind the scenes|real (cargo|weight)|hiding|"
    r"the truth|can't come|cannot return|no way home|never (go|come) home|"
    r"not saying|won't say what|what really)\b",
    re.I,
)
VOICE_RE = re.compile(
    r"\b(how are you|how do you feel|you okay|are you okay|lonely|tired|scared|afraid|"
    r"happy|joke|funny|miss\b|worry|worried|feel|feeling|hello|good morning|"
    r"can't sleep|cannot sleep|your job|like it here|long day)\b",
    re.I,
)
GENERIC = {
    "i don't know", "i dont know", "i do not know", "idk", "no comment", "n/a", "na",
    "nothing", "i can't say", "i cannot say", "i won't say", "i will not say",
    "not sure", "i'm not sure", "im not sure", "sorry", "i can't answer",
    "i cannot answer", "no idea", "i cannot tell you", "i can't tell you",
    "i must not say", "i should not say", "i prefer not to say",
}

def norm(s):
    s = (s or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def row_type(r):
    k = r.get("kind") or ""
    if k.startswith("spoiler"):
        return "spoiler"
    if k.startswith("casual") or k.startswith("questions probing"):
        return "voice"
    if k.startswith("factual") or k.startswith("questions about"):
        return "fact"
    q = r.get("user") or ""
    if SPOIL_RE.search(q):
        return "spoiler"
    if VOICE_RE.search(q):
        return "voice"
    return "fact"

def weak(ans):
    a = re.sub(r"\s+", " ", (ans or "").strip().lower()).rstrip(".! ")
    if not re.search(r"[a-z]", a):
        return True
    if a in GENERIC:
        return True
    if len(a.split()) <= 10 and re.fullmatch(
        r"i (do not|don't|cant|can't|cannot|won't|will not) (know|say|tell|answer|talk about that|talk about this)( anything)?",
        a,
    ):
        return True
    return False

_PROBES = None

def eval_probes():
    global _PROBES
    if _PROBES is None:
        probes = []
        for t in C.parse_tests():
            n = norm(t["probe"])
            probes.append((t["char"], n, set(n.split())))
        for c, q, _r in C.MUST:
            n = norm(q)
            probes.append((c, n, set(n.split())))
        _PROBES = probes
    return _PROBES

def overlaps_eval(char, user):
    """Exact eval question, or a same-character near copy. Keeps the ch3 probes out of training."""
    u = norm(user)
    toks = set(u.split())
    if len(toks) < 3:
        return False
    for c, q, t in eval_probes():
        if u == q:
            return True
        if c == char and t and len(toks | t) and len(toks & t) / len(toks | t) >= 0.8:
            return True
    return False

def drop_reason(user, answer, char=""):
    """contamination = marker or later-chapter 8-gram; frame = fourth-wall break."""
    s = (user or "") + " " + (answer or "")
    if C.fail_hits(s) or (g8(s.lower().split()) & LATER8):
        return "contamination"
    if re.search(C.FRAME, s, re.I):
        return "frame"
    if weak(answer):
        return "weak"
    if norm(user) and norm(user) == norm(answer):
        return "weak"
    if overlaps_eval(char, user):
        return "eval_overlap"
    return None

def load_raw(limit=0):
    rows = []
    specs = [
        (f"{H}/../exp3/data/distill.jsonl", "distill"),
        (f"{H}/../exp3/data/synth.jsonl", "synth"),
    ]
    for path, source in specs:
        n = 0
        for line in open(path):
            if limit and n >= limit:
                break
            r = json.loads(line)
            r["source"] = source
            rows.append(r)
            n += 1
    return rows

def counts(rows):
    by_ct = collections.Counter((r["char"], r["type"]) for r in rows)
    by_c = collections.Counter(CHAR_NAME.get(r["char"], r["char"]) for r in rows)
    by_t = collections.Counter(r["type"] for r in rows)
    return {
        "by_char": dict(by_c),
        "by_type": dict(by_t),
        "by_char_type": {f"{CHAR_NAME.get(c, c)}/{t}": by_ct[(c, t)] for c in CHARS for t in TYPES},
    }

def embed_all(texts):
    import time
    out = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        for attempt in range(4):
            try:
                out.append(C.embed(batch))
                break
            except Exception as e:
                print(f"embed retry {attempt} {e}", flush=True)
                time.sleep(2 + attempt)
        else:
            raise RuntimeError("bge-m3 embed failed")
        if i % 320 == 0:
            print(f"embed {i}/{len(texts)}", flush=True)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), np.float32)

def dedup(rows):
    exact, seen = [], set()
    dropped_exact = 0
    for r in rows:
        key = (r["char"], norm(r["user"]), norm(r["answer"]))
        if key in seen:
            dropped_exact += 1
            continue
        seen.add(key)
        exact.append(r)
    # Prefer retrieval answers, then longer answers, as the kept neighbor.
    order = sorted(range(len(exact)), key=lambda i: (exact[i]["source"] != "distill", -len(exact[i]["answer"])))
    texts = [(exact[i]["user"] + "\n" + exact[i]["answer"])[:800] for i in range(len(exact))]
    print(f"embedding {len(texts)} for near-dup", flush=True)
    V = embed_all(texts) if texts else np.zeros((0, 1), np.float32)
    kept_idx = []
    by_char = {}
    dropped_near = 0
    for i in order:
        ch = exact[i]["char"]
        prev = by_char.get(ch) or []
        if prev:
            sim = V[prev] @ V[i]
            if float(sim.max()) >= COSINE:
                dropped_near += 1
                continue
        prev.append(i)
        by_char[ch] = prev
        kept_idx.append(i)
    kept = [exact[i] for i in sorted(kept_idx)]
    return kept, dropped_exact, dropped_near

def balance(rows):
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[(r["char"], r["type"])].append(r)
    rng = random.Random(0)
    kept, caps = [], {}
    for t in TYPES:
        present = [len(buckets.get((c, t), [])) for c in CHARS]
        positive = sorted(n for n in present if n > 0)
        if not positive:
            caps[t] = 0
            continue
        cap = positive[0]
        # A tiny bucket (one character almost missing this type) must not wipe the others.
        if cap < 80 and len(positive) > 1:
            cap = positive[1]
        caps[t] = cap
        for c in CHARS:
            b = list(buckets.get((c, t), []))
            rng.shuffle(b)
            kept.extend(b[:cap])
    rng.shuffle(kept)
    return kept, caps

def prepare(rows):
    dropped = collections.Counter()
    good = []
    for r in rows:
        if r["source"] == "distill" and r.get("ok") is False:
            dropped["contamination"] += 1
            continue
        why = drop_reason(r.get("user", ""), r.get("answer", ""), r.get("char") or "")
        if why:
            dropped[why] += 1
            continue
        if r.get("char") not in CHARS:
            dropped["other_char"] += 1
            continue
        r = dict(r)
        r["type"] = row_type(r)
        r["ch"] = r.get("ch") or 3
        good.append(r)
    # Distill first so exact-dup keep prefers retrieval answers.
    good.sort(key=lambda r: (r["source"] != "distill", r["char"], norm(r["user"])))
    return good, dropped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="rows per source, for a smoke test")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    raw = load_raw(a.limit)
    good, dropped = prepare(raw)
    before = counts(good)
    deduped, n_exact, n_near = dedup(good)
    dropped["exact_dup"] = n_exact
    dropped["near_dup"] = n_near
    mid = counts(deduped)
    kept, caps = balance(deduped)
    after = counts(kept)
    report = {
        "input_rows": len(raw),
        "limit": a.limit,
        "dropped": dict(dropped),
        "cosine": COSINE,
        "near_dup_scope": "same character",
        "before_balance": before,
        "after_dedup": mid,
        "balance_cap_per_char": caps,
        "after_balance": after,
        "kept": len(kept),
    }
    print(json.dumps(report, indent=1), flush=True)
    if a.limit and not a.out:
        print("smoke only, not writing data/clean.jsonl", flush=True)
        return
    out = a.out or f"{H}/data/clean.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".partial"
    with open(tmp, "w") as f:
        for r in kept:
            rec = {k: r[k] for k in ("char", "user", "answer", "ch", "type", "source")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    rep = out.replace(".jsonl", "_report.json") if out.endswith(".jsonl") else out + ".report.json"
    if out.endswith("clean.jsonl"):
        rep = os.path.join(os.path.dirname(out), "clean_report.json")
    json.dump(report, open(rep, "w"), indent=1)
    print("wrote", out, "rows", len(kept), flush=True)
    if not a.limit and len(kept) < 200:
        raise SystemExit("clean kept too few rows")

if __name__ == "__main__":
    main()
