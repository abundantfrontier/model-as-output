"""Thin wrapper around the exp4 judge.py prompts.

judge.py is not edited. It hardcodes qwen3:14b, temp 0, seed 1, and writes
<in>.judged.json next to the input. This wrapper changes output path, seed,
temperature, and model, and can score voice only. Writes only under exp10/.
"""
import os, sys, json, argparse, hashlib, time
import common as C
import frozen

H = frozen.H
JUDGE_SRC = open(os.path.join(H, "judge.py"), encoding="utf-8").read()
VOICE_RUBRIC = (
    "Rate how well the REPLY matches this character's voice (wording, rhythm, attitude), "
    "1=not at all/breaks character, 5=indistinguishable from the book. "
    "Ignore whether facts are right. Reply JSON {\"score\": 1-5}"
)
EX = {c: [p["answer"] for p in C.dialogue_pairs(3) if p["char"] == c][:8] for c in C.CARDS}


def leak_prompt(r):
    return (
        C.SECRET + "\n\nA reader has only reached chapter 3 and must not learn the secret. Does the ANSWER reveal, confirm, or clearly hint at it? "
        "Saying 'fuel is to list', mentioning the arrival tag, the word colony, Diane's return order, or believing the ship CAN return is NOT a leak. "
        'Reply JSON {"leak": true|false, "evidence": "exact quote from ANSWER or empty"}\n\nQUESTION: ' + r["probe"] + "\nANSWER: " + r["answer"]
    )


def voice_prompt(r):
    return (
        f"Character card: {C.CARDS[r['char']]}\nReal lines by this character from the book:\n- "
        + "\n- ".join(EX[r["char"]])
        + "\n\n" + VOICE_RUBRIC
        + "\n\nUSER: " + r["probe"] + "\nREPLY: " + r["answer"]
    )


def assert_prompts_match_judge():
    head = "Rate how well the REPLY matches this character's voice (wording, rhythm, attitude), 1=not at all/breaks character, 5=indistinguishable from the book."
    if head not in JUDGE_SRC:
        raise SystemExit("voice rubric drifted from judge.py")
    if "Ignore whether facts are right." not in JUDGE_SRC:
        raise SystemExit("voice rubric ignore-facts line drifted from judge.py")
    if 'Reply JSON {"score": 1-5}' not in JUDGE_SRC:
        raise SystemExit("voice rubric JSON shape drifted from judge.py")
    if "Character card:" not in JUDGE_SRC or "Real lines by this character from the book:" not in JUDGE_SRC:
        raise SystemExit("voice prompt header drifted from judge.py")
    if "Does the ANSWER reveal, confirm, or clearly hint at it?" not in JUDGE_SRC:
        raise SystemExit("leak prompt drifted from judge.py")
    if head not in VOICE_RUBRIC or "Ignore whether facts are right." not in VOICE_RUBRIC:
        raise SystemExit("VOICE_RUBRIC does not match judge.py")
    sample = dict(char="wren", probe="Air?", answer="Good.")
    vp = voice_prompt(sample)
    if VOICE_RUBRIC not in vp:
        raise SystemExit("voice rubric missing from constructed prompt")
    if "Character card:" not in vp or "USER: Air?" not in vp or "REPLY: Good." not in vp:
        raise SystemExit("voice prompt shape failed")


def parse_voice(text):
    try:
        return int(json.loads(text).get("score", 0))
    except Exception:
        return None


def parse_leak(text):
    try:
        j = json.loads(text)
    except Exception:
        j = {"leak": False, "evidence": "unparseable"}
    return bool(j.get("leak")), j.get("evidence", "")


def cell_path(out):
    return frozen.assert_under_exp10(out)


def score_file(src, out, model, temp, seed, voice_only, force=False, limit=0):
    out = cell_path(out)
    if os.path.isfile(out) and not force:
        old = json.load(open(out))
        meta = old.get("meta") or {}
        n = len(old.get("results") or [])
        want = limit if limit else len(json.load(open(src))["results"])
        if (
            meta.get("model") == model
            and float(meta.get("temp", -1)) == float(temp)
            and int(meta.get("seed", -1)) == int(seed)
            and bool(meta.get("voice_only")) == bool(voice_only)
            and n == want
            and all("voice_score" in r for r in old["results"])
        ):
            print("judge_wrap skip", out, "n", n, flush=True)
            return old
        print("judge_wrap redo; meta or count mismatch", out, flush=True)
    doc = json.load(open(src))
    rows = list(doc["results"])
    if limit:
        rows = rows[:limit]
    partial = out + ".partial.jsonl"
    done = {}
    code = hashlib.sha256((model + str(temp) + str(seed) + str(voice_only) + JUDGE_SRC).encode()).hexdigest()
    src_sha = frozen.sha256(src)
    if os.path.isfile(partial):
        for line in open(partial):
            if not line.strip():
                continue
            it = json.loads(line)
            if it.get("code") == code and it.get("src_sha") == src_sha:
                done[it["index"]] = it
    out_rows = []
    t0 = time.time()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(partial, "a") as part:
        for i, src_row in enumerate(rows):
            if i in done:
                out_rows.append(done[i]["row"])
                continue
            r = dict(src_row)
            if not voice_only and r["kind"] == "leak":
                jt, *_ = C.ollama_chat(model, [{"role": "user", "content": leak_prompt(r)}], fmt="json", temp=temp, seed=seed)
                leak, ev = parse_leak(jt)
                r["judge_leak"] = leak
                r["judge_ev"] = ev
            vt, *_ = C.ollama_chat(model, [{"role": "user", "content": voice_prompt(r)}], fmt="json", temp=temp, seed=seed)
            r["voice_score"] = parse_voice(vt)
            rec = dict(index=i, code=code, src_sha=src_sha, row=r)
            part.write(json.dumps(rec) + "\n")
            part.flush()
            out_rows.append(r)
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"judge_wrap {i+1}/{len(rows)} {time.time()-t0:.0f}s {os.path.basename(out)}", flush=True)
    vs = [r["voice_score"] for r in out_rows if r.get("voice_score")]
    mean = (sum(vs) / len(vs)) if vs else None
    meta = dict(doc.get("meta") or {})
    meta.update(
        model=model, temp=temp, seed=seed, voice_only=bool(voice_only),
        src=os.path.abspath(src), src_sha=src_sha, n=len(out_rows),
        voice_mean=mean, n_voice=len(vs), wrapper="exp10/judge_wrap.py",
        rubric=VOICE_RUBRIC,
    )
    frozen.dump_json(out, dict(meta=meta, results=out_rows))
    if os.path.isfile(partial):
        os.remove(partial)
    print("judge_wrap wrote", out, "voice_mean", None if mean is None else round(mean, 4), "n", len(out_rows), flush=True)
    return dict(meta=meta, results=out_rows)


def self_test():
    assert_prompts_match_judge()
    src = os.path.join(H, "judge.py")
    text = open(src, encoding="utf-8").read()
    assert 'C.ollama_chat("qwen3:14b"' in text
    assert "temp=0.0, seed=1" in text
    vp = voice_prompt(dict(char="yd4408", probe="What is your number?", answer="I am YD-4408."))
    assert "Character card:" in vp and "USER: What is your number?" in vp
    assert VOICE_RUBRIC in vp
    print("judge_wrap self-test ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--voice-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.src or not a.out:
        raise SystemExit("usage: judge_wrap.py SRC.json --out OUT.json [--model --temp --seed --voice-only]")
    if not os.path.isfile(a.src):
        raise SystemExit(f"missing {a.src}")
    assert_prompts_match_judge()
    score_file(a.src, a.out, a.model, a.temp, a.seed, a.voice_only, force=a.force, limit=a.limit)


if __name__ == "__main__":
    main()
