"""Build style-only rewrite pairs for the optional (c) LoRA.

Target is an existing in-voice answer. Source is either the facts.jsonl answer
for the same question, or the target sentences written as a numbered list.
The train user message is the pass-2 rewrite prompt. The system message is the
character card. Retrieved passages are not inserted.
"""
import os, sys, json, argparse, collections
import common as C
import frozen
import prompts

H = frozen.H

OPTIONAL = [
    (os.path.join(frozen.EXP5, "data", "v_distill_train.jsonl"), "distill"),
    (os.path.join(frozen.EXP5, "data", "distill_kept.jsonl"), "distill_kept"),
    (os.path.join(frozen.EXP5, "data", "contrast_rewrites.jsonl"), "contrast_rewrite"),
    (os.path.join(frozen.EXP6, "data", "contrast_rewrites.jsonl"), "contrast_rewrite_exp6"),
]

def load_jsonl(path):
    if not os.path.isfile(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]

def ok_text(text):
    if not text or not text.strip():
        return False
    if C.fail_hits(text):
        return False
    if C.voice_flags(text)["frame"]:
        return False
    if len(text.split()) > 80:
        return False
    return True

def sentences(text):
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

def flatten(text):
    parts = sentences(text)
    if not parts:
        return None
    return " ".join(f"{i}. {s}" for i, s in enumerate(parts, 1))

def voice_pool(limit_files=0):
    by = {}
    origin = collections.Counter()
    missing = []
    def put(rows, source):
        if rows is None:
            missing.append(source)
            return
        for r in rows:
            if source.startswith("contrast") and not r.get("keep"):
                continue
            char, user, ans = r.get("char"), r.get("user"), (r.get("answer") or "").strip()
            if char not in C.CARDS and char != "narrator":
                continue
            if not user or not ok_text(ans):
                continue
            key = (char, user)
            item = dict(char=char, user=user, answer=ans, ch=r.get("ch") or 3, source=source)
            if key not in by:
                by[key] = item
            else:
                by[key] = item
            origin[source] += 1
    voice = load_jsonl(frozen.VOICE_JSONL)
    if voice is None:
        raise SystemExit(f"missing {frozen.VOICE_JSONL}")
    put(voice, "voice")
    for path, source in OPTIONAL:
        put(load_jsonl(path), source)
    rows = sorted(by.values(), key=lambda r: (r["char"], r["user"]))
    if limit_files:
        rows = rows[:limit_files]
    return rows, dict(origin), missing

def make_pair(char, question, source_text, target, kind, ch):
    if not ok_text(source_text) or not ok_text(target):
        return None
    if source_text.strip() == target.strip():
        return None
    user = prompts.rewrite_user(question, source_text)
    sysm = prompts.system_for(char, 3)
    prompts.assert_no_retrieval(sysm, prompts.REWRITE_INSTR)
    if prompts.RAG_MARK in user:
        return None
    if not user.endswith(prompts.REWRITE_INSTR):
        raise RuntimeError("rewrite instruction missing from train user message")
    return dict(
        char=char, user=user, answer=target.strip(), sys=sysm, ch=ch or 3,
        source=kind, question=question, source_text=source_text,
    )

def build(out_path, meta_path, limit, force):
    hp = json.load(open(os.path.join(H, "hparams.json")))
    cap = hp["rewrite"]["pair_cap"]
    repeat = hp["rewrite"]["facts_twin_repeat"]
    if os.path.isfile(meta_path) and os.path.isfile(out_path) and not force and not limit:
        print("rewrite data skip", meta_path, flush=True)
        return json.load(open(meta_path))
    pool, origin, missing = voice_pool(0 if not limit else max(limit * 4, limit))
    pool_by = {(r["char"], r["user"]): r for r in pool}
    facts = load_jsonl(frozen.FACTS_JSONL)
    if facts is None:
        raise SystemExit("missing facts.jsonl")
    twins = []
    for f in facts:
        key = (f.get("char"), f.get("user"))
        gold = pool_by.get(key)
        if not gold:
            continue
        pair = make_pair(gold["char"], gold["user"], (f.get("answer") or "").strip(), gold["answer"], "facts_twin", gold["ch"])
        if pair:
            twins.append(pair)
    twins.sort(key=lambda r: (r["char"], r["question"]))
    flat = []
    for r in pool:
        src = flatten(r["answer"])
        pair = make_pair(r["char"], r["user"], src or "", r["answer"], "style_flatten", r["ch"])
        if pair:
            flat.append(pair)
    flat.sort(key=lambda r: (r["char"], r["question"]))
    if limit:
        twins = twins[:limit]
        flat = flat[:limit]
        repeat = 1
    else:
        flat = flat[:cap]
    rows = []
    for pair in twins:
        rows.extend([pair] * repeat)
    rows.extend(flat)
    if not rows:
        raise SystemExit("no rewrite pairs")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # Train-time check: no row carries a retrieval block, and the user message matches pass 2.
    for r in rows:
        prompts.assert_no_retrieval(r["sys"], prompts.REWRITE_INSTR)
        if not r["user"].endswith(prompts.REWRITE_INSTR):
            raise SystemExit("train row drifted from the rewrite instruction")
        if prompts.RAG_MARK in r["sys"]:
            raise SystemExit("rag sysprompt in a rewrite row")
    meta = dict(
        path=os.path.abspath(out_path),
        rows=len(rows),
        unique_facts_twins=len(twins),
        facts_twin_repeat=repeat,
        style_flatten=len(flat),
        sha256=frozen.sha256(out_path),
        voice_jsonl_sha256=frozen.VOICE_SHA,
        facts_jsonl_sha256=frozen.sha256(frozen.FACTS_JSONL),
        origin_seen=origin,
        missing_optional=missing,
        pool=len(pool),
        cap=cap,
        limit=limit,
        instruction=prompts.REWRITE_INSTR,
        system="prompts.system_for (common.sysprompt day=3, or NARR). No rag_sysprompt.",
        retrieval_in_rewrite=False,
    )
    frozen.dump_json(meta_path, meta)
    print(json.dumps({k: meta[k] for k in ("rows", "unique_facts_twins", "style_flatten", "sha256", "missing_optional")}), flush=True)
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(H, "data", "rewrite_pairs.jsonl"))
    ap.add_argument("--meta", default=os.path.join(H, "data", "rewrite_meta.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.out, a.meta, a.limit, a.force)

if __name__ == "__main__":
    main()
