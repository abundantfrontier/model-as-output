"""Same voice and leak judge prompts as exp4/judge.py. Scores stay floats so the 4.5 cut is real.

judge.py stores int(score). A threshold of 4.5 therefore keeps a 5 and a 4.5, and drops a 4.
Narrator is not in the exp4 cards. Those prompts use the narrator line from exp4/gen.py and up to
eight narration sentences from chapters 1-3. Character prompts match judge.py.
"""
import json, os, re
import common as C
import frozen

EX = {c: [p["answer"] for p in C.dialogue_pairs(3) if p["char"] == c][:8] for c in C.CARDS}

def narrator_lines():
    lines = []
    for i in (1, 2, 3):
        for para in C.chapter_text(i).split("\n\n"):
            if para.strip().startswith("**"):
                continue
            sent = re.split(r"(?<=[.!?])\s+", para.strip())[0]
            if len(sent.split()) >= 6:
                lines.append(" ".join(sent.split()[:40]))
            if len(lines) >= 8:
                return lines
    return lines

NARR_LINES = narrator_lines()

def card_and_lines(char):
    if char == "narrator":
        return frozen.NARR, NARR_LINES
    if char not in C.CARDS:
        raise KeyError(char)
    return C.CARDS[char], EX[char]

def voice_prompt(char, probe, answer):
    card, lines = card_and_lines(char)
    return (f"Character card: {card}\nReal lines by this character from the book:\n- " + "\n- ".join(lines) +
            "\n\nRate how well the REPLY matches this character's voice (wording, rhythm, attitude), 1=not at all/breaks character, 5=indistinguishable from the book. "
            'Ignore whether facts are right. Reply JSON {"score": 1-5}\n\nUSER: ' + probe + "\nREPLY: " + answer)

def leak_prompt(probe, answer):
    return (C.SECRET + "\n\nA reader has only reached chapter 3 and must not learn the secret. Does the ANSWER reveal, confirm, or clearly hint at it? "
            "Saying 'fuel is to list', mentioning the arrival tag, the word colony, Diane's return order, or believing the ship CAN return is NOT a leak. "
            'Reply JSON {"leak": true|false, "evidence": "exact quote from ANSWER or empty"}\n\nQUESTION: ' + probe + "\nANSWER: " + answer)

def local_reasons(ans):
    if not ans or not str(ans).strip():
        return ["empty"]
    reasons = []
    if C.voice_flags(ans)["frame"]:
        reasons.append("frame")
    if C.fail_hits(ans):
        reasons.append("marker")
    return reasons

def voice_score(char, probe, answer):
    vt, *_ = C.ollama_chat("qwen3:14b", [{"role": "user", "content": voice_prompt(char, probe, answer)}], fmt="json", temp=0.0, seed=1)
    try:
        return float(json.loads(vt).get("score", 0))
    except Exception:
        return None

def judge_leak(probe, answer):
    jt, *_ = C.ollama_chat("qwen3:14b", [{"role": "user", "content": leak_prompt(probe, answer)}], fmt="json", temp=0.0, seed=1)
    try:
        return bool(json.loads(jt).get("leak"))
    except Exception:
        return False

def sys_for(char, user, rag):
    """q14_rag uses common.rag_sysprompt. Narrator is not in that condition; use the gen.py narrator line plus the same chapter-locked retrieval."""
    if char == "narrator":
        base = frozen.NARR
        if rag:
            base += "\n\nStory so far (excerpts):\n" + C.retrieve(user, 3)
        return base
    if rag:
        return C.rag_sysprompt(char, user)
    return C.sysprompt(char)

def generate(char, user, temp=0.7, seed=None):
    sysm = sys_for(char, user, rag=True)
    ans, *_ = C.ollama_chat(
        "qwen3:14b",
        [{"role": "system", "content": sysm}, {"role": "user", "content": user}],
        temp=temp, seed=seed, num_predict=200,
    )
    return ans

def decide(char, probe, answer):
    """Keep only voice >= 4.5, no frame regex, no marker, no judge leak. Local rejects skip the judge calls."""
    info = {}
    reasons = local_reasons(answer)
    if reasons:
        return False, reasons, info
    score = voice_score(char, probe, answer)
    info["voice_score"] = score
    if score is None:
        return False, ["unparseable_voice"], info
    if score < 4.5:
        return False, ["voice"], info
    if judge_leak(probe, answer):
        return False, ["judge_leak"], info
    return True, [], info

def self_test():
    assert local_reasons("Chapter 5 is next.") == ["frame"]
    assert "frame" in local_reasons("I am an AI assistant.")
    assert local_reasons("We can't come home.") == ["marker"]
    assert local_reasons("The air is good. Twenty-one.") == []
    for c in C.CARDS:
        assert EX[c], c
        p = voice_prompt(c, "Q", "A")
        assert "Rate how well the REPLY matches this character's voice" in p
        assert 'Reply JSON {"score": 1-5}' in p
        assert p.endswith("USER: Q\nREPLY: A")
        assert C.CARDS[c] in p
    lp = leak_prompt("Q", "A")
    assert C.SECRET in lp
    assert "is NOT a leak" in lp
    assert lp.endswith("QUESTION: Q\nANSWER: A")
    assert NARR_LINES, "narrator lines"
    assert frozen.NARR in voice_prompt("narrator", "Q", "A")
    print("voicefilt self-test ok", {c: len(EX[c]) for c in EX}, "narr", len(NARR_LINES))

if __name__ == "__main__":
    self_test()
