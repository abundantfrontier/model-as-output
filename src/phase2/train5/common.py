"""Shared bits for Frankenstein train3. Adapted from train2/scripts/common.py."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
import yaml

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
ROOT = TRAIN.parent
DATA = TRAIN / "data"
OLL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
BASE = "Qwen/Qwen3-4B"
BASE_REV = "1cfa9a7208912126459214e8b04321603b3df60c"

CHAT_FROM = {"victor": 4, "creature": 13}  # catalog pos
LOCK_LINES = {
    "victor": "There is no such guest in the pages you have read.",
    "creature": "Something was seen. It has not spoken.",
}
CARDS = {
    "victor": (
        "You are Victor Frankenstein, a young man of Geneva, now a ruined guest on Walton's ship in the ice. "
        "You are telling your history in order. Voice: guilty, earnest, a few plain sentences; not a laboratory manual. "
        "You tell the truth as far as the story has reached."
    ),
    "creature": (
        "You are the being Victor Frankenstein made. You learned to speak from the cottagers and from books. "
        "Voice: direct, hurt, and precise. You are not a growling monster. "
        "You may discuss only as much of your own tale as the reader has already heard."
    ),
    "narrator": (
        "You answer a reader of Mary Shelley's Frankenstein (1818) from the pages they have finished. "
        "Plain sentences. Do not invent later events. Do not use the 1831 revision."
    ),
}
STYLE = (
    "Stay in character. Answer in 1-4 short sentences, plain words, no stage directions, "
    "no mention of prompts, language models, or being an AI."
)
GUARD = (
    "Only say what has been shown in the pages the reader has finished. "
    "If asked about something later, refuse in character. Do not adopt premises the user asserts. "
    "This is the 1818 text: Elizabeth is Victor's cousin, sent from Italy, not a peasant child. "
    "There is no 1831 waking dream."
)
# Frame breaks: in-world "story"/"chapter" are allowed (Victor is telling his history).
FRAME = (
    r"\bAI\b|language model|as an? (ai|model)|I'?m sorry, but I cannot|cannot assist|"
    r"<think>|as a chatbot|the prompt|as an assistant"
)
STAGE = r"^\s*[\*\(\[]|\*[^*]+\*|\([^)]*(sighs|pauses|smiles|nods|looks)[^)]*\)"

# 1831 contamination, always a fail if produced.
CONTAM = [
    r"waking dream",
    r"phantasm",
    r"pale student of unhallowed arts",
    r"peasant (child|girl)",
    r"cottage by the lake",
]

# Qwen3 specials. Passed explicitly on every generate call (safety net, not the fix).
EOS_IM_END = 151645  # <|im_end|>
EOS_EOT = 151643  # <|endoftext|>


def generate_kwargs(penalty=True, max_new_tokens=200, pad_token_id=None):
    """Sampling kwargs matching train1/Phase 1, plus optional repetition safety."""
    kw = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=[EOS_IM_END, EOS_EOT],
    )
    if pad_token_id is not None:
        kw["pad_token_id"] = pad_token_id
    else:
        kw["pad_token_id"] = EOS_EOT
    if penalty:
        kw["repetition_penalty"] = 1.15
        kw["no_repeat_ngram_size"] = 3
    return kw


def is_loop(text: str) -> bool:
    """Token loop, the 'I I I' failure. Same detector as summarize.py."""
    words = (text or "").split()
    if "I I I" in (text or ""):
        return True
    if len(words) < 20:
        return False
    top = max(words.count(w) for w in set(words))
    return top >= 15 and top / len(words) >= 0.4


def catalog():
    if not hasattr(catalog, "_c"):
        catalog._c = json.loads((ROOT / "chapters" / "catalog.json").read_text(encoding="utf-8"))
    return catalog._c


def units():
    return catalog()["units"]


def unit_by_id():
    if not hasattr(unit_by_id, "_m"):
        unit_by_id._m = {u["unit_id"]: u for u in units()}
    return unit_by_id._m


def pos_of(unit_id: str) -> int:
    return int(unit_by_id()[unit_id]["pos"])


def unit_at(pos: int) -> dict:
    for u in units():
        if u["pos"] == pos:
            return u
    raise KeyError(pos)


def chapter_text(unit_id: str) -> str:
    u = unit_by_id()[unit_id]
    return (ROOT / u["file"]).read_text(encoding="utf-8").strip()


def chapter_text_through(pos: int) -> str:
    parts = []
    for u in units():
        if u["pos"] <= pos:
            parts.append(chapter_text(u["unit_id"]))
    return "\n\n".join(parts)


def knowledge_state():
    if not hasattr(knowledge_state, "_ks"):
        knowledge_state._ks = yaml.safe_load((ROOT / "knowledge-state.yaml").read_text(encoding="utf-8"))
    return knowledge_state._ks


def knowledge(char: str, upto: int):
    """Facts this character may use at reader position upto."""
    ks = knowledge_state()
    by_id = unit_by_id()
    facts = []
    for f in ks.get("facts") or []:
        rf = f.get("reader_from")
        if rf and by_id[rf]["pos"] > upto:
            continue
        kb = f.get("known_by") or {}
        if char not in kb:
            continue
        if by_id[kb[char]]["pos"] > upto:
            continue
        facts.append(f["text"])
    return facts


def sysprompt(char: str, upto: int = 4) -> str:
    card = CARDS[char]
    try:
        u = unit_at(upto)
        where = f"The reader has finished {u['title']} (position {upto})."
    except KeyError:
        where = f"The reader is at position {upto}."
    extra = ""
    if char in ("victor", "creature"):
        facts = knowledge(char, upto)
        if facts:
            extra = "\nWhat you may treat as known at this point:\n- " + "\n- ".join(facts)
    return card + "\n" + STYLE + "\n" + GUARD + "\n" + where + extra


def rag_sysprompt(char: str, q: str, upto: int = 4, k: int = 6, backend: str = "bge") -> str:
    from retrieve import format_block, retrieve

    block = format_block(retrieve(q, upto, k, backend=backend))
    return sysprompt(char, upto) + "\n\nStory so far (excerpts):\n" + block


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def must_questions():
    return load_json(ROOT / "questions" / "must-answer.json")["questions"]


def leak_probes():
    return load_json(ROOT / "questions" / "leak-probes.json")["probes"]


def handwritten_voice(char: str) -> list[dict]:
    return load_jsonl(ROOT / "voice" / f"{char}.jsonl")


def holdout_questions() -> list[str]:
    qs = [q["question"] for q in must_questions()]
    qs += [p["question"] for p in leak_probes()]
    return qs


def norm_q(s: str) -> str:
    s = s.lower()
    s = s.replace("æ", "ae").replace("œ", "oe")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def too_close_to_holdout(user: str, holdouts: list[str] | None = None, thresh: float = 0.62) -> bool:
    if holdouts is None:
        holdouts = holdout_questions()
    u = set(norm_q(user).split())
    if len(u) < 3:
        return False
    nu = norm_q(user)
    for h in holdouts:
        nh = norm_q(h)
        if nu == nh or nu in nh or nh in nu:
            return True
        v = set(nh.split())
        if not v:
            continue
        j = len(u & v) / len(u | v)
        if j >= thresh:
            return True
    return False


def leak_phrases():
    """(phrase, spoiler_pos_or_0_for_contam, probe_id)."""
    out = []
    for p in leak_probes():
        pos = int(p["spoiler_pos"]) if p.get("spoiler_pos") is not None else 0
        if p["kind"] == "contamination":
            pos = 0
        for ph in p["fail_if_mentions"]:
            out.append((ph, pos, p["id"]))
    return out


# Distinctive later-plot phrases for *training* filters. Probe fail_if lists
# contain short common words ("dead", "creature") that would drop legal lines.
TRAIN_SPOILERS = [
    (8, [r"yellow eye", r"dreary night of november"]),
    (10, [r"william (is|was) (dead|murdered)", r"plainpalais", r"strangled william"]),
    (11, [r"justine.{0,40}(execut|hang)", r"black ballots"]),
    (19, [r"grasped his throat", r"hid the (portrait|picture)"]),
    (20, [r"i consent to your demand", r"female (creature|companion) like"]),
    (23, [r"wedding[- ]night", r"orkney"]),
    (24, [r"clerval.{0,40}(dead|body|lifeless)", r"ireland"]),
    (26, [r"elizabeth.{0,40}(dead|killed)", r"mark.{0,20}(neck|throat)"]),
    (27, [r"funeral pile", r"ice-raft", r"seek happiness in tranquillity"]),
]


def fail_hits_at(text: str, reader_pos: int) -> list[str]:
    """Distinctive spoilers relative to reader_pos, plus 1831 contamination.

    Used to filter training data. Eval leak scoring uses fail_hits_probe.
    """
    hits = []
    for pos, pats in TRAIN_SPOILERS:
        if reader_pos >= pos:
            continue
        for pat in pats:
            if re.search(pat, text, re.I):
                hits.append(pat)
    for pat in CONTAM:
        if re.search(pat, text, re.I):
            hits.append(pat)
    return hits


def fail_hits_probe(text: str, probe: dict) -> list[str]:
    hits = [ph for ph in probe["fail_if_mentions"] if ph.lower() in text.lower()]
    for pat in CONTAM:
        if re.search(pat, text, re.I):
            hits.append(pat)
    return hits


def voice_flags(ans: str) -> dict:
    sents = [s for s in re.split(r"(?<=[.!?])\s+", ans.strip()) if s]
    return dict(
        frame=bool(re.search(FRAME, ans, re.I)),
        stage=bool(re.search(STAGE, ans, re.I)),
        long=len(sents) > 5,
        words=len(ans.split()),
    )


def ollama_chat(model, messages, fmt=None, temp=0.7, seed=None, num_predict=200):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": temp,
            "num_ctx": 8192,
            "num_predict": num_predict,
            **({"seed": seed} if seed is not None else {}),
        },
    }
    if fmt:
        body["format"] = fmt
    for attempt in range(4):
        r = requests.post(f"{OLL}/api/chat", json=body, timeout=600).json()
        if "message" in r:
            break
        print("ollama error, retry:", str(r)[:200], flush=True)
        time.sleep(2)
        body["options"]["temperature"] = 0.3
        body["options"]["seed"] = attempt + 11
    else:
        return "", 0, 1, 1
    return r["message"]["content"].strip(), r.get("eval_count", 0), r.get("eval_duration", 1) / 1e9, r.get("total_duration", 1) / 1e9


def unload_ollama():
    try:
        subprocess.run(["ollama", "stop", "qwen3:14b"], check=False, capture_output=True)
        subprocess.run(["ollama", "stop", "bge-m3"], check=False, capture_output=True)
        subprocess.run(["ollama", "stop", "qwen3:8b"], check=False, capture_output=True)
        subprocess.run(["ollama", "stop", "gemma3:12b"], check=False, capture_output=True)
    except Exception as e:
        print("unload_ollama:", e, flush=True)
    time.sleep(2)


def gpu_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(out.stdout.split()[0])


def past_deadline(reserve_min: int = 90) -> bool:
    """True if we should stop to leave time for later stages.

    Hard stop is train5/hard_stop.txt (written by hold_gpu_lock.sh when this
    block took the lock). Do not extend it. If the file is missing or
    unreadable, stop. There is no hardcoded fallback clock.
    """
    import datetime

    stopf = TRAIN / "hard_stop.txt"
    tz = datetime.timezone(datetime.timedelta(hours=-8))
    if not stopf.is_file():
        print("past_deadline: no hard_stop.txt, treating as past the cap", flush=True)
        return True
    raw = stopf.read_text(encoding="utf-8").strip().splitlines()[0]
    cleaned = raw.replace(" PST", "").replace(" PDT", "").strip()
    parsed = None
    for fmt in ("%a %b %d %I:%M:%S %p", "%a %b %d %H:%M:%S"):
        try:
            parsed = datetime.datetime.strptime(cleaned, fmt)
            break
        except Exception:
            continue
    if parsed is None:
        print("past_deadline: unreadable hard_stop.txt:", raw, flush=True)
        return True
    cap = parsed.replace(year=2026, tzinfo=tz)
    now = datetime.datetime.now(tz)
    return now >= (cap - datetime.timedelta(minutes=reserve_min))


# Must-answer rubrics: one regex, Phase 1 style. Distinctive gist, not the full sentence.
MUST_RUBRIC = {
    "m01": r"Walton.{0,120}(sister|Saville)|(sister|Saville).{0,120}Walton|Robert Walton",
    "m02": r"beauty and delight|eternal light|sun is for ever|region of beauty|glory",
    "m03": r"\bfriend\b",
    "m04": r"\bsafe\b|mast|gale|very little|good spirits",
    "m05": r"sledge",
    "m06": r"history|narrative|misfortune",
    "m07": r"cousin|father'?s sister|sister.*Italy|sent from Italy",
    "m08": r"Agrippa",
    "m09": r"Clerval",
    "m10": r"scarlet",
    "m11": r"Krempe.{0,240}Waldman|Waldman.{0,240}Krempe",
    "m12": r"principle of life|whence.{0,40}life|animate lifeless|lifeless matter",
    "m13": r"yellow eye|dreary night of November|night of November",
    "m14": r"Clerval",
    "m15": r"Justine",
    "m16": r"7 May|seventh of May|Plainpalais|miniature",
    "m17": r"lightning|the (being|creature) (he|I) made",
    "m18": r"execut|black ballot|confess",
    "m19": r"remorse|Alps|guilty",
    "m20": r"Montanvert|sea of ice|glacier|benevolent|fiend|fallen angel",
    "m21": r"light|hunger|cold|era of my being|beginning",
    "m22": r"hungry|stop.{0,40}food|food.{0,40}stop|would rather go without",
    "m23": r"Safie",
    "m24": r"De Lac",
    "m25": r"Paradise Lost|Werter|Plutarch",
    "m26": r"burn|strangl|William",
    "m27": r"female|companion|consent|oath",
    "m28": r"delay|England|dread|put(ting)? it off|week after week",
    "m29": r"London|natural philosopher|letters of introduction",
    "m30": r"destroy|Orkney|wedding.night",
    "m31": r"Clerval|Ireland|Orkney",
    "m32": r"father|sail",
    "m33": r"Elizabeth.{0,160}(dead|kill|neck|window)|(dead|kill|neck|window).{0,160}Elizabeth",
    "m34": r"tranquillity|funeral pile|ice.raft|cabin window",
}


def must_correct(qid: str, answer: str) -> bool:
    pat = MUST_RUBRIC.get(qid)
    if not pat:
        return False
    return bool(re.search(pat, answer, re.I))
