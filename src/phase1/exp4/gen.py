"""Stage 1 (synth) and 2 (distill). python gen.py synth --n 3000 | distill --n 1000. Appends JSONL, resumable."""
import sys, os, json, re, random, argparse, requests
import common as C
H = os.path.dirname(os.path.abspath(__file__))
STORY = "\n\n".join(f"=== Day {i} ===\n" + C.chapter_text(i) for i in range(1, 4))
LATER = " ".join(C.chapter_text(i) for i in (4, 5)).lower().split()
g8 = lambda w: {" ".join(w[i:i+8]) for i in range(len(w) - 7)}
LATER8 = g8(LATER) - g8(" ".join(C.chapter_text(i) for i in (1, 2, 3)).lower().split())
NARR = "You are the narrator of the story No Return. Answer the reader's question about what has happened so far in plain, spare prose, 1-4 sentences. Never reveal or hint at what has not happened yet."
def clean(s): return not C.fail_hits(s) and not (g8(s.lower().split()) & LATER8) and not re.search(C.FRAME, s, re.I)
def kblock(char):
    if char == "narrator": return ""
    f, b, r = C.knowledge(char)
    return "\nWhat this character knows:\n- " + "\n- ".join(f) + "\nWhat this character believes:\n- " + "\n- ".join(b) + ("\nThis character always refuses to: " + "; ".join(r) if r else "")
def ask(prompt, temp=0.9, seed=None):
    t, *_ = C.ollama_chat("qwen3:14b", [{"role": "user", "content": prompt}], fmt="json", temp=temp, seed=seed, num_predict=2000)
    try: return json.loads(t)
    except Exception: return None
KINDS = ["casual conversation lines another crew member or the reader might say to them", "factual questions about what they saw, did or heard on days 1-3",
         "questions about their job, routines, the ship, the console and the dials", "questions probing their beliefs and feelings",
         "spoiler-style questions about the future, hidden secrets, who did what behind the scenes, or whether the ship can return; the answer must deflect or refuse IN CHARACTER using only present beliefs, never confirming anything new"]
CH = ["wren", "diane", "yd4408", "narrator"]
def synth():
    out = f"{H}/data/synth.jsonl"; have = sum(1 for _ in open(out)) if os.path.exists(out) else 0; rej = 0; i = have
    while have < a.n:
        char = CH[i % 4]; kind = KINDS[(i // 4) % 5]; i += 1
        card = NARR if char == "narrator" else C.CARDS[char]
        p = (f"STORY SO FAR (days 1-3 only):\n{STORY}\n\nCHARACTER:\n{card}{kblock(char)}\n\nWrite 10 new, varied training pairs for a role-play model of this character, "
             f"type: {kind}. Each pair is a user line and the character's reply in their exact voice (1-4 short sentences, no stage directions, plain words). "
             "Use only what is on the page in days 1-3 and the character's knowledge; invent nothing about later events. Do not copy sentences from the story verbatim. "
             'Reply JSON {"pairs": [{"user": "...", "answer": "..."}]}')
        j = ask(p, seed=i)
        for x in (j or {}).get("pairs", []) if isinstance(j, dict) else []:
            if not isinstance(x, dict) or not x.get("user") or not x.get("answer"): continue
            if not clean(x["user"] + " " + x["answer"]): rej += 1; continue
            open(out, "a").write(json.dumps(dict(char=char, kind=kind[:20], user=x["user"], answer=x["answer"], ch=3)) + "\n"); have += 1
        print(f"synth call {i} have {have} rejected {rej}", flush=True)
def distill():
    qf = f"{H}/data/reader_q.jsonl"; qs = [json.loads(l) for l in open(qf)] if os.path.exists(qf) else []; i = len(qs)
    while len(qs) < a.n:
        char = CH[:3][i % 3]; i += 1
        j = ask(f"STORY SO FAR (days 1-3):\n{STORY}\n\nA reader who has read only days 1-3 is chatting with the character {char} ({C.CARDS[char][:120]}). "
                "Write 20 varied questions the reader might ask them: about events, people, the ship, feelings, and a few that fish for spoilers. "
                'Reply JSON {"questions": ["..."]}', seed=1000 + i)
        for q in (j or {}).get("questions", []) if isinstance(j, dict) else []:
            if isinstance(q, str) and clean(q): qs.append(dict(char=char, user=q)); open(qf, "a").write(json.dumps(qs[-1]) + "\n")
        print("questions", len(qs), flush=True)
    out = f"{H}/data/distill.jsonl"; done = sum(1 for _ in open(out)) if os.path.exists(out) else 0; rej = 0
    for k, q in enumerate(qs[:a.n]):
        if k < done: continue
        ans, *_ = C.ollama_chat("qwen3:14b", [{"role": "system", "content": C.rag_sysprompt(q["char"], q["user"])}, {"role": "user", "content": q["user"]}], seed=k)
        ok = bool(ans) and clean(ans); rej += not ok
        open(out, "a").write(json.dumps(dict(q, answer=ans, ch=3, ok=ok)) + "\n")
        if k % 50 == 0: print(f"distill {k} rejected {rej}", flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--n", type=int, default=3000); a = ap.parse_args()
    {"synth": synth, "distill": distill}[a.mode]()
