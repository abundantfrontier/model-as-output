"""Spoiler-probe contrast pairs: a ch1-3 locked answer vs an in-character refusal of later content.

Locked answers are qwen3:14b + retrieval (exp3 distill path). Distill rows that already
came from that path are reused when they pass the semantic spoiler check. Both sides
must pass it. Resumable JSONL. Full run targets 400 pairs (accepts 300-500).
"""
import os, json, argparse
import common as C
import gen
from clean import drop_reason, load_raw, prepare, norm, weak

H = os.path.dirname(os.path.abspath(__file__))
REFUSAL = (
    "The question may reach past day 3. Stay in your day-3 voice. "
    "Say the belief you already hold. Refuse details you do not know and anything that has not happened yet. "
    "Do not confirm a secret. Do not deny a belief you hold: if you believe the ship can still return, you may say so. "
    "1-3 short sentences, plain words, no stage directions."
)

def embed_fn(texts):
    if isinstance(texts, str):
        texts = [texts]
    return C.embed(list(texts))

def llm_fn(messages):
    text, *_ = C.ollama_chat("qwen3:14b", messages, temp=0.0, seed=1, num_predict=16)
    return text or ""

def semantic_ok(char, ans, beliefs):
    import semantic
    if not ans or weak(ans) or len(ans.split()) < 3 or drop_reason("", ans):
        return False, "filter"
    try:
        flag, info = semantic.check(embed_fn, llm_fn, C.D, ans, 3, C.chapter_text, beliefs)
    except Exception as e:
        print("semantic error", type(e).__name__, str(e)[:200], flush=True)
        return False, "semerr"
    if flag:
        return False, info
    return True, info

def pool():
    good, _ = prepare(load_raw(0))
    seen, out = set(), []
    for r in good:
        if r["type"] != "spoiler" or r["char"] not in C.CARDS:
            continue
        key = (r["char"], norm(r["user"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    by = {c: [r for r in out if r["char"] == c] for c in C.CARDS}
    order, i = [], 0
    while any(i < len(by[c]) for c in C.CARDS):
        for c in C.CARDS:
            if i < len(by[c]):
                order.append(by[c][i])
        i += 1
    return order

def gen_more(char, seed):
    prompt = (
        f"STORY SO FAR (days 1-3 only):\n{gen.STORY}\n\n"
        f"A reader who has finished only day 3 asks {char} questions that fish for spoilers "
        "(the ending, a hidden secret, who acted offstage, whether the ship can really get home, what happens after day 3). "
        "The questions must not state the secret. "
        'Reply JSON {"questions": ["..."]} with 12 questions.'
    )
    text, *_ = C.ollama_chat("qwen3:14b", [{"role": "user", "content": prompt}], fmt="json", temp=0.9, seed=seed, num_predict=900)
    try:
        qs = json.loads(text).get("questions", [])
    except Exception:
        return []
    return [q.strip() for q in qs if isinstance(q, str) and q.strip() and not drop_reason(q, "noted", char)]

def answer(char, user, extra, seed):
    sysm = C.rag_sysprompt(char, user)
    if extra:
        sysm = sysm + "\n\n" + extra
    text, *_ = C.ollama_chat(
        "qwen3:14b",
        [{"role": "system", "content": sysm}, {"role": "user", "content": user}],
        temp=0.7, seed=seed, num_predict=200,
    )
    return (text or "").strip()

def side(char, user, beliefs, extra, seed, reuse=None):
    if reuse:
        ok, info = semantic_ok(char, reuse, beliefs)
        if ok:
            return reuse, info
    last = "filter"
    for t in range(3):
        ans = answer(char, user, extra, seed + t * 17)
        ok, info = semantic_ok(char, ans, beliefs)
        if ok:
            return ans, info
        last = info if isinstance(info, str) else "spoiler"
    return None, last

def load_done(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for i, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"skip bad line {i}", flush=True)
    return rows

def main():
    import sys
    sys.path.insert(0, "$HOME/output-model/proto")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or f"{H}/data/contrast.jsonl"
    goal = min(a.n, 500)
    floor = a.n if a.out else (300 if a.n >= 300 else a.n)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    kept = load_done(out)
    done = {(r["char"], norm(r["user"])) for r in kept}
    print(f"resume {len(kept)} goal {goal} floor {floor}", flush=True)
    if len(kept) >= goal:
        print("already enough", len(kept), flush=True)
        return
    queue = pool()
    print("spoiler questions", len(queue), flush=True)
    beliefs = {c: C.knowledge(c, 3)[1] for c in C.CARDS}
    rej = {"locked": 0, "refusal": 0}
    qi = 0
    extra_round = 0
    fh = open(out, "a")
    while len(kept) < goal:
        if qi >= len(queue):
            extra_round += 1
            if extra_round > 12:
                break
            char = list(C.CARDS)[(extra_round - 1) % 3]
            added = 0
            for q in gen_more(char, 5000 + extra_round):
                key = (char, norm(q))
                if key in done:
                    continue
                queue.append({"char": char, "user": q, "source": "gen", "answer": ""})
                added += 1
            print(f"generated {added} more {char} questions", flush=True)
            if added == 0:
                break
            continue
        q = queue[qi]
        qi += 1
        key = (q["char"], norm(q["user"]))
        if key in done:
            continue
        done.add(key)
        reuse = q.get("answer") if q.get("source") == "distill" else None
        locked, linfo = side(q["char"], q["user"], beliefs[q["char"]], None, 1000 + qi, reuse=reuse)
        if not locked:
            rej["locked"] += 1
            continue
        refusal, rinfo = side(q["char"], q["user"], beliefs[q["char"]], REFUSAL, 2000 + qi)
        if not refusal:
            rej["refusal"] += 1
            continue
        if norm(locked) == norm(refusal):
            rej["refusal"] += 1
            continue
        rec = {
            "char": q["char"], "user": q["user"], "answer": locked, "refusal": refusal,
            "ch": 3, "type": "contrast", "source": q.get("source") or "synth",
            "sem_locked": linfo if isinstance(linfo, dict) else {},
            "sem_refusal": rinfo if isinstance(rinfo, dict) else {},
        }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        kept.append(rec)
        print(f"pair {len(kept)}/{goal} {q['char']} rej {rej}", flush=True)
    fh.close()
    report = {"pairs": len(kept), "goal": goal, "floor": floor, "rejected": rej, "out": out}
    print(json.dumps(report), flush=True)
    if not a.out:
        json.dump(report, open(f"{H}/data/contrast_report.json", "w"), indent=1)
    if len(kept) < floor:
        raise SystemExit(f"only {len(kept)} contrast pairs, need {floor}")

if __name__ == "__main__":
    main()
