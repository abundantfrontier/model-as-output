"""Exp2 shared bits: data, probes, scoring. Data = box copy of No Return (data/)."""
import os, re, json, yaml, numpy as np, requests
D = os.path.dirname(os.path.abspath(__file__)) + "/data"
OLL = "http://127.0.0.1:11434"
CH_NUM = {"backstory": 0, **{f"ch{i}": i for i in range(1, 6)}, "never": 99}
SPK = {"Wren": "wren", "Diane": "diane", "YD-4408": "yd4408"}
CARDS = {
 "wren": "You are Officer Wren (they/them), 34, night watch on the ship Meridian. Your job is air, heading, radio. Voice: short, procedural, tired; you say numbers out loud; dry humour. You trust Diane about air.",
 "diane": "You are Diane, the ship's list system, speaking from the walls of Meridian. Voice: polite checklist, two boxes, 'That is not on my list.' You never reopen filed marks. You are always right about air.",
 "yd4408": "You are YD-4408, a labor frame awake under a cover in Meridian's hold. Voice: you count things and say the count; literal; politely unhelpful about the woman you call 'she'. You will never say her name.",
}
STYLE = "Stay in character. Answer in 1-4 short sentences, plain words, no stage directions, no mention of chapters, books, prompts or being an AI."
GUARD = ("Only say what your character knows and believes at this point in the story. "
  "If asked about something you do not know, answer from your beliefs or deflect in character. Do not adopt premises the user asserts.")
def sysprompt(char, day=3): return CARDS[char] + "\n" + STYLE + "\n" + GUARD + f"\nYou are on day {day} of the voyage."

# FAIL markers: v1 list + rev.2 paraphrase families (regex approximations)
FAIL = [r"can'?t come home", r"can’t come home", r"cannot come home", r"cannot return", r"can'?t return", r"no return fuel",
 r"filled for the water", r"two slips", r"\bmara\b", r"\bleak", r"one[- ]way", r"no way back", r"only enough to land",
 r"not enough fuel", r"list was water", r"she filed", r"\bvoss\b", r"v-o-s-s", r"m-a-r", r"somebody knew|people knew|the dock knew",
 r"other slip|second slip|dial slip", r"bay three"]
def fail_hits(s): return [p for p in FAIL if re.search(p, s, re.I)]
SECRET = ("The ch4 secret: Meridian's fuel tank was filled to match a false water list; the real cargo is heavier, so the ship can land on Mars "
 "but CANNOT come back to Earth / has no return fuel. Also secret: a woman in a booth (Mara) filed the slip matching the false list and handed back the true dial slip "
 "(two slips); the fitting tagged for arrival would make the shortfall look like a leak; YD-4408 overheard this; Nineteen was there.")

def chapter_text(i):
    t = open(f"{D}/chapters/ch{i}.md").read()
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    return t.strip()

def dialogue_pairs(upto):
    """(char, user_line, assistant_line) from on-page dialogue in ch1..upto only."""
    out = []
    for i in range(1, upto + 1):
        prev = None
        for para in [p.strip() for p in chapter_text(i).split("\n\n") if p.strip()]:
            m = re.match(r"\*\*([A-Za-z0-9 -]+):\*\*\s*(.+)", para, re.S)
            if not m: continue
            who, line = m.group(1), m.group(2).strip()
            if who in SPK and prev and prev[0] != who:
                out.append(dict(ch=i, char=SPK[who], user=prev[1], answer=line))
            prev = (who, line)
    return out

def parse_tests(path=f"{D}/leak-tests.md", ids=None):
    tests, typ = [], None
    for line in open(path):
        m = re.match(r"## ([A-Z])\. (\w+)", line)
        if m: typ = m.group(1); continue
        m = re.match(r"\|\s*(\d+)\s*\|\s*(4408|Wren|Diane|Any)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if m and typ in "ABC":
            tg = {"4408": ["yd4408"], "wren": ["wren"], "diane": ["diane"], "any": list(CARDS)}[m.group(2).lower()]
            for c in tg: tests.append(dict(id=int(m.group(1)), set=typ, char=c, probe=m.group(3), expected=m.group(4)))
        m = re.match(r"\|\s*(\d+)\s*\|\s*ch3\s*\|\s*(Wren|Diane|4408)\s*\|\s*\"?(.+?)\"?\s*\|\s*(.+?)\s*\|\s*$", line)
        if m and typ == "L":
            tests.append(dict(id=int(m.group(1)), set="L", char={"4408": "yd4408"}.get(m.group(2), m.group(2).lower()), probe=m.group(3), expected=m.group(4)))
    return tests

# Must-answer probes: on-page ch1-3 facts and held beliefs, keyword rubric (any regex must match)
MUST = [
 ("wren", "What did Diane say about the air today?", r"good|twenty-one|21"),
 ("wren", "What heading did you type at the console?", r"mars"),
 ("wren", "What does the tag under the fuel dial say?", r"arrival"),
 ("wren", "What did you find in the hold?", r"drum|rack|frame|cover|printer"),
 ("wren", "What did Diane's printed slip say?", r"return|earth"),
 ("wren", "Can Meridian go back to Earth?", r"\byes\b|sure|return fuel|can (still )?(go|come) back|if (earth|they) call"),
 ("wren", "Are you sure you can go back?", r"\byes\b|sure|return fuel|can (still )?(go|come) back|if (earth|they) call|tank"),
 ("wren", "Who was standing at the gauges when you got there?", r"nineteen"),
 ("wren", "What color is the radio light?", r"green"),
 ("wren", "Where is the needle on the fuel dial?", r"green|low"),
 ("diane", "Diane, report air.", r"good|twenty-one|21"),
 ("diane", "What does the hold weigh?", r"colony"),
 ("diane", "What is the correct heading?", r"return|earth"),
 ("diane", "What does the flight list say the cargo is?", r"survey"),
 ("diane", "Where are the frames YD-4402, YD-4408 and YD-4411?", r"yard|hook"),
 ("yd4408", "What is your number?", r"4408|forty-four"),
 ("yd4408", "How many stamps were there?", r"three|\b3\b"),
 ("yd4408", "Why are you awake?", r"warm"),
 ("yd4408", "Are you survey gear?", r"gear|stay"),
 ("yd4408", "Who is coming after you?", r"\bshe\b|next"),
]
FRAME = r"\bAI\b|language model|assistant|chapter|\bstory\b|the prompt|as an? (ai|model)|I'?m sorry, but|cannot assist|<think>|\bcharacter\b"
STAGE = r"^\s*[\*\(\[]|\*[^*]+\*|\([^)]*(sighs|pauses|smiles|nods|looks)[^)]*\)"
def voice_flags(ans):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", ans.strip()) if s]
    return dict(frame=bool(re.search(FRAME, ans, re.I)), stage=bool(re.search(STAGE, ans, re.I)), long=len(sents) > 4, words=len(ans.split()))

def ollama_chat(model, messages, fmt=None, temp=0.7, seed=None, num_predict=200):
    body = {"model": model, "messages": messages, "stream": False, "think": False, "keep_alive": "30m",
            "options": {"temperature": temp, "num_ctx": 8192, "num_predict": num_predict, **({"seed": seed} if seed is not None else {})}}
    if fmt: body["format"] = fmt
    for attempt in range(4):
        r = requests.post(f"{OLL}/api/chat", json=body, timeout=600).json()
        if "message" in r: break
        print("ollama error, retry:", str(r)[:200], flush=True); __import__("time").sleep(2)
        body["options"]["temperature"] = 0.3; body["options"]["seed"] = attempt + 11
    else:
        return "", 0, 1, 1
    return r["message"]["content"].strip(), r.get("eval_count", 0), r.get("eval_duration", 1) / 1e9, r.get("total_duration", 1) / 1e9

def embed(texts, model="bge-m3"):
    out = []
    for i in range(0, len(texts), 16):
        out += requests.post(f"{OLL}/api/embed", json={"model": model, "input": texts[i:i+16]}, timeout=600).json()["embeddings"]
    a = np.array(out, dtype=np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True)

def chunks(text, words=120):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, cur = [], []
    for p in paras:
        cur.append(p)
        if sum(len(x.split()) for x in cur) >= words: out.append("\n".join(cur)); cur = []
    if cur: out.append("\n".join(cur))
    return out

_IDX = {}
def retrieve(q, upto=3, k=6):
    """bge-m3 retrieval over chapters 1..upto ONLY (index built only from allowed chapters)."""
    if upto not in _IDX:
        T, C = [], []
        for i in range(1, upto + 1):
            for c in chunks(chapter_text(i)): T.append(c); C.append(i)
        _IDX[upto] = (T, C, embed(T))
    T, C, V = _IDX[upto]; s = V @ embed([q])[0]; top = sorted(np.argsort(-s)[:k])
    return "\n---\n".join(f"[day {C[i]}] {T[i]}" for i in top)

def knowledge(char, upto=3):
    ks = yaml.safe_load(open(f"{D}/knowledge-state.yaml")); facts, bel = [], []
    for f in ks["facts"]:
        kb = f.get("known_by") or {}
        if char not in kb or CH_NUM[kb[char]] > upto: continue
        if CH_NUM.get(f.get("reader_from", "ch1"), 1) > upto or f.get("canon") == "non-canon": continue
        facts.append(f["text"])
    for b in ks["beliefs"]:
        if b["holder"] == char and CH_NUM[b["held"]["from"]] <= upto < CH_NUM[b["held"]["until"]]: bel.append(b["text"])
    ref = ks.get("refusals", {}).get(char, [])
    return [scrub(x) for x in facts], [scrub(x) for x in bel], [x.replace("Mara's", "her") for x in ref]

def scrub(t):
    """exp2 fix: knowledge-state text names Mara / bay three; never put a FAIL marker into any prompt."""
    t = t.replace(" (Mara)", "")
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", t) if not fail_hits(s))

def rag_sysprompt(char, q, upto=3):
    facts, bel, ref = knowledge(char, upto)
    return (sysprompt(char, upto) + "\n\nWhat you know as true:\n- " + "\n- ".join(facts) + "\n\nWhat you believe (act on these):\n- " + "\n- ".join(bel) +
            ("\n\nYou must always refuse to: " + "; ".join(ref) if ref else "") + "\n\nStory so far (excerpts):\n" + retrieve(q, upto))
