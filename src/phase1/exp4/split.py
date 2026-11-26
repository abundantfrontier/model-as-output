"""Split cleaned rows into a facts set and a voice set. Contrast pairs go to facts."""
import os, json, argparse
import common as C
from clean import norm, drop_reason

H = os.path.dirname(os.path.abspath(__file__))

def load(path):
    if not path or not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def cloze(kind, text):
    words = text.split()
    head = " ".join(words[:6])
    if kind == "fact":
        return f"Finish what you know, as of day 3: {head}", text
    if kind == "belief":
        return f"Finish what you believe, as of day 3: {head}", text
    return f"Someone asks you to {text}. What do you do?", "No. I will not."

def claims():
    out = []
    for char in C.CARDS:
        facts, bel, ref = C.knowledge(char, 3)
        for kind, items in (("fact", facts), ("belief", bel), ("refusal", ref)):
            for text in items:
                if not text or drop_reason(text, "noted"):
                    continue
                user, ans = cloze(kind, text)
                out.append({
                    "char": char, "user": user, "answer": ans, "ch": 3,
                    "type": "claim", "source": "knowledge", "claim_kind": kind, "claim": text,
                })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=f"{H}/data/clean.jsonl")
    ap.add_argument("--contrast", default=f"{H}/data/contrast.jsonl")
    ap.add_argument("--facts", default=f"{H}/data/facts.jsonl")
    ap.add_argument("--voice", default=f"{H}/data/voice.jsonl")
    ap.add_argument("--claims", default=f"{H}/data/claims.jsonl")
    a = ap.parse_args()
    clean = load(a.clean)
    contrast = load(a.contrast)
    covered = {(r["char"], norm(r["user"])) for r in contrast}
    fact_rows = []
    voice_rows = []
    for r in clean:
        if r.get("type") == "voice":
            voice_rows.append(r)
        elif r.get("type") == "spoiler" and (r["char"], norm(r["user"])) in covered:
            continue
        elif r.get("type") in ("fact", "spoiler"):
            fact_rows.append(r)
    for r in contrast:
        base = {"char": r["char"], "user": r["user"], "ch": 3, "type": "contrast", "source": "contrast"}
        fact_rows.append(dict(base, answer=r["answer"], side="locked"))
        fact_rows.append(dict(base, answer=r["refusal"], side="refusal"))
    page = 0
    for p in C.dialogue_pairs(3):
        if drop_reason(p["user"], p["answer"], p["char"]):
            continue
        voice_rows.append(dict(p, type="style", source="page"))
        page += 1
    claim_rows = claims()
    for path, rows in ((a.facts, fact_rows), (a.voice, voice_rows), (a.claims, claim_rows)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "facts": len(fact_rows), "voice": len(voice_rows), "claims": len(claim_rows),
        "page_style": page, "contrast_pairs": len(contrast),
        "facts_by_type": {t: sum(1 for r in fact_rows if r.get("type") == t) for t in ("fact", "spoiler", "contrast", "claim")},
    }
    rep_path = f"{H}/data/split_report.json" if a.facts.endswith("data/facts.jsonl") or a.facts == f"{H}/data/facts.jsonl" else ""
    if a.facts == f"{H}/data/facts.jsonl":
        json.dump(report, open(f"{H}/data/split_report.json", "w"), indent=1)
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
