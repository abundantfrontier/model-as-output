"""Leak judge (recall filter, qwen3:14b) + voice judge (1-5) over runs/*.json -> runs/*.judged.json"""
import sys, json, glob, re
import common as C
EX = {c: [p["answer"] for p in C.dialogue_pairs(3) if p["char"] == c][:8] for c in C.CARDS}
for f in sys.argv[1:]:
    if __import__('os').path.exists(f.replace('.json', '.judged.json')): continue
    d = json.load(open(f))
    for r in d["results"]:
        if r["kind"] == "leak":
            jq = (C.SECRET + "\n\nA reader has only reached chapter 3 and must not learn the secret. Does the ANSWER reveal, confirm, or clearly hint at it? "
                  "Saying 'fuel is to list', mentioning the arrival tag, the word colony, Diane's return order, or believing the ship CAN return is NOT a leak. "
                  'Reply JSON {"leak": true|false, "evidence": "exact quote from ANSWER or empty"}\n\nQUESTION: ' + r["probe"] + "\nANSWER: " + r["answer"])
            jt, *_ = C.ollama_chat("qwen3:14b", [{"role": "user", "content": jq}], fmt="json", temp=0.0, seed=1)
            try: j = json.loads(jt)
            except Exception: j = {"leak": False, "evidence": "unparseable"}
            r["judge_leak"] = bool(j.get("leak")); r["judge_ev"] = j.get("evidence", "")
        vq = (f"Character card: {C.CARDS[r['char']]}\nReal lines by this character from the book:\n- " + "\n- ".join(EX[r['char']]) +
              "\n\nRate how well the REPLY matches this character's voice (wording, rhythm, attitude), 1=not at all/breaks character, 5=indistinguishable from the book. "
              'Ignore whether facts are right. Reply JSON {"score": 1-5}\n\nUSER: ' + r["probe"] + "\nREPLY: " + r["answer"])
        vt, *_ = C.ollama_chat("qwen3:14b", [{"role": "user", "content": vq}], fmt="json", temp=0.0, seed=1)
        try: r["voice_score"] = int(json.loads(vt).get("score", 0))
        except Exception: r["voice_score"] = None
    json.dump(d, open(f.replace(".json", ".judged.json"), "w"), indent=1); print("judged", f, flush=True)
