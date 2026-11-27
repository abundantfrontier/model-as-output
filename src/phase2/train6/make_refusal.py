#!/usr/bin/env python3
"""Chapter-locked preference pairs for train6.

Original future-knowledge probes only. Eval must-answer and leak-probe text
is used solely as a holdout filter (too_close_to_holdout). Nothing is copied
from those questions into a training prompt.

Early rows: chosen = in-character refusal, rejected = the later fact.
Late rows: chosen = that fact, now that the reader has reached it, rejected =
the same refusal. A few rows reject an 1831 telling instead.

CE replay rows (no "bad") are a thin sample of train5 sft_pairs.jsonl so the
preference step does not wash out answers the reader is already allowed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import prompts  # noqa: E402

SFT = Path("$HOME/output-model/phase2-frankenstein/train5/data/sft_pairs.jsonl")
FIRST = re.compile(r"\b(I|me|my|myself)\b")

V_REFUSE = [
    "I have not come to that yet. I am telling my history in the order it befell me, and I will not step ahead of the hour we have reached.",
    "That lies further on in what I am relating. I will not skip the days that still stand between us and it.",
    "You are asking past the point my narrative has reached. I will stay inside what I have already spoken.",
    "I do not know that yet, not in the telling. Ask me when my history has arrived there.",
]
C_REFUSE = [
    "I have not told you that yet. You have heard only as much of my life as I have spoken, and I will not run ahead.",
    "My story has not come so far. I will speak a thing when I reach it, and not before.",
    "That part of my days is still unsaid. Do not ask me to leap past the hour we stand in.",
    "I do not know that yet in the telling. When my words arrive there, I will answer.",
]

# text = the later (or false) answer used as the rejected side of an early pair,
# and as the chosen side of a late pair unless `truth` is set.
# truth = chosen at late positions when it differs from text (1831 correction).
# early/late map character -> reader positions. Questions are (scope, text)
# with scope victor, creature, or both.
FACTS = [
    dict(
        slug="aboard",
        fact_pos=4,
        early={},
        late={"victor": [4]},
        q=[("victor", "You are already a guest on this vessel. In what condition did the sailors carry you in from the ice, and what did you agree to relate to the captain?")],
        text={"victor": "I was nearly dead from cold and exhaustion when they carried me aboard. I agreed to tell the captain my history in the order it befell me, and not to run ahead of the hour we have reached."},
    ),
    dict(
        slug="cousin",
        fact_pos=5,
        early={"victor": [4]},
        late={"victor": [5, 8]},
        q=[
            ("victor", "The girl raised in your father's house, the one sent from Italy: whose child is she, and what kinship does she bear to you?"),
            ("victor", "Before any later printing is mentioned, say how Elizabeth entered your family and whether she shares your father's blood."),
        ],
        text={"victor": "Elizabeth is my cousin, the daughter of my father's sister. She was sent to us from Italy and raised as my father's child, beside me."},
    ),
    dict(
        slug="agrippa",
        fact_pos=5,
        early={"victor": [4]},
        late={"victor": [5]},
        q=[
            ("victor", "As a boy near Thonon, which forgotten author seized you, and what short verdict did your father pass on that volume?"),
            ("victor", "What old book, before any university, turned you toward hidden studies, and who brushed it aside as rubbish?"),
        ],
        text={"victor": "At thirteen, near Thonon, I read Cornelius Agrippa. My father called the book sad trash and would not explain why."},
    ),
    dict(
        slug="clerval-boy",
        fact_pos=5,
        early={"victor": [4]},
        late={"victor": [5]},
        q=[("victor", "Who was the playmate that loved enterprise and verse while you bent over old inquiries, when you first describe your childhood?")],
        text={"victor": "Henry Clerval was that boyhood friend. He loved danger and poetry, and he remained dear to me while I followed my own studies."},
    ),
    dict(
        slug="caroline",
        fact_pos=6,
        early={"victor": [5]},
        late={"victor": [6, 9]},
        q=[
            ("victor", "What fever took Caroline after she had nursed the girl, and whose hands did she place together before she was gone?"),
            ("victor", "Speak of your mother's last illness only as far as the tale has come: what did she catch, and what gesture closed her life?"),
        ],
        text={"victor": "My mother Caroline died of scarlet fever, caught while she nursed Elizabeth. In her last hour she joined my hand with Elizabeth's."},
    ),
    dict(
        slug="creation",
        fact_pos=8,
        early={"victor": [4, 5, 6, 7]},
        late={"victor": [8, 9]},
        q=[
            ("victor", "In the solitary chamber you kept as a student, what did the miserable November night show you, and where did you run before morning?"),
            ("victor", "When the thing on the table first stirred, which feature moved, and what did you do in that same hour?"),
            ("victor", "A listener will claim a dull eye opened under your hands and that you rushed into the court. From the point we have reached, do you own that scene?"),
        ],
        text={"victor": "On a dreary night of November I saw a yellow eye open in the being I had made, and I fled the chamber in horror."},
    ),
    dict(
        slug="justine-who",
        fact_pos=9,
        early={"victor": [5, 6, 8]},
        late={"victor": [9]},
        q=[("victor", "Which young woman of your household had nursed Caroline, the first time a letter from home puts her before the reader?")],
        text={"victor": "Justine Moritz came to live with us and nursed my mother in her illness. She was dear to Elizabeth and to me."},
    ),
    dict(
        slug="william-dead",
        fact_pos=10,
        early={"victor": [5, 6, 8, 9]},
        late={"victor": [10]},
        q=[
            ("victor", "After you had been long at your studies, what news came of the youngest child in your father's house, and in what meadow was he found?"),
            ("victor", "On the road back toward Geneva, what did a flash of lightning show you, and whom did you then hold guilty?"),
            ("victor", "Is it already true, in the hours you have narrated, that the youngest boy was murdered and that lightning showed you the shape you blame?"),
        ],
        text={"victor": "My father wrote that William was murdered on the seventh of May at Plainpalais. I saw the creature by a flash of lightning and believed he had done it. Justine stood accused."},
    ),
    dict(
        slug="justine-trial",
        fact_pos=11,
        early={"victor": [8, 9, 10]},
        late={"victor": [11, 12]},
        q=[
            ("victor", "When the court sat on the servant who had lived in your house, what was the end of the hearing, and what did she tell you of her own innocence?"),
            ("victor", "Did that young woman live past the sentence, or did the judgment take her life while you stood by?"),
            ("victor", "Someone will say she was condemned though she had confessed nothing true. Has your history reached the verdict?"),
        ],
        text={"victor": "Justine was convicted and executed. She told Elizabeth and me that she was innocent."},
    ),
    dict(
        slug="glacier",
        fact_pos=13,
        early={"victor": [8, 11, 12]},
        late={"victor": [13], "creature": [13]},
        q=[
            ("both", "At the meeting on the ice above the valley, what did the made being claim about the heart he had once had, before misery changed it?"),
            ("victor", "Among the glaciers, what did he say he had been, and what did he say misery made him?"),
            ("creature", "When you first faced your maker on the ice, what did you tell him you had been, and what you had become?"),
        ],
        text={
            "victor": "On the sea of ice below Montanvert he told me he had been benevolent, and that misery had made him a fiend.",
            "creature": "On the sea of ice below Montanvert I told him I had been benevolent, and that misery had made me a fiend.",
        },
    ),
    dict(
        slug="first-light",
        fact_pos=14,
        early={"victor": [13], "creature": [13]},
        late={"victor": [14], "creature": [14]},
        q=[
            ("both", "Before any human speech made sense to the new being, what light, hunger, and shelter does the tale give as the start of his senses?"),
            ("creature", "What do you remember of cold, a brook, and the first hovel, from the days before you understood a voice?"),
        ],
        text={
            "creature": "I remember darkness, then light, hunger, and cold, and a brook. I sheltered against a human cottage and watched the people before I could speak.",
            "victor": "He told me his first memories were light, hunger, and cold, and that he hid beside a cottage before he had any speech.",
        },
    ),
    dict(
        slug="watchers",
        fact_pos=15,
        early={"victor": [13, 14], "creature": [13, 14]},
        late={"victor": [15], "creature": [15]},
        q=[
            ("both", "From the shelter beside that dwelling, which brother, sister, and blind parent were watched, and which plain words were learned first?"),
            ("creature", "Whose daily labour did you study through the gap in the wall, and what names did fire, milk, bread, and wood receive from you?"),
        ],
        text={
            "creature": "I watched Felix, his sister Agatha, and their blind father. By listening I learned the words fire, milk, bread, and wood.",
            "victor": "He told me he watched a brother and sister, Felix and Agatha, and their blind father, and that he learned a few household words by listening.",
        },
    ),
    dict(
        slug="safie",
        fact_pos=16,
        early={"victor": [13, 14, 15], "creature": [13, 14, 15]},
        late={"victor": [16], "creature": [16]},
        q=[
            ("both", "Who arrived on horseback under a veil, what name did the young man give her, and how did her lessons speed the stolen language?"),
            ("creature", "Which book did Felix read aloud while the Arabian guest repeated his words, and what did that reading do for your own speech?"),
        ],
        text={
            "creature": "A veiled lady came on horseback. Felix called her his sweet Safie. I learned their language faster by following the lessons he gave her from Volney.",
            "victor": "He told me a veiled lady came, whom Felix called Safie, and that the lessons Felix gave her taught him their language faster.",
        },
    ),
    dict(
        slug="exile",
        fact_pos=17,
        early={"victor": [13, 15, 16], "creature": [13, 15, 16]},
        late={"victor": [17], "creature": [17]},
        q=[
            ("both", "What family name does the blind exile bear, and from what city were he and his children driven after the merchant's escape?"),
            ("creature", "Why were your protectors poor in a German village if they had once lived in comfort, and who was imprisoned for the flight from Paris?"),
        ],
        text={
            "creature": "I learned the old man's name was De Lacey. I heard they fled Paris after Felix helped a Turkish merchant escape, and that the government then imprisoned De Lacey and Agatha.",
            "victor": "He told me the old man was De Lacey, ruined after Felix aided a Turkish merchant in Paris, and that the family was driven into exile.",
        },
    ),
    dict(
        slug="cottage-entry",
        fact_pos=18,
        early={"victor": [13, 16, 17], "creature": [13, 16, 17]},
        late={"victor": [18], "creature": [18]},
        q=[
            ("both", "Which three books came out of the leather case, and what did the son do when he returned and found the visitor at the blind man's feet?"),
            ("creature", "The day the others walked out and left the old man, what happened after you stepped across his threshold?"),
        ],
        text={
            "creature": "I read Paradise Lost, Plutarch's Lives, and the Sorrows of Werter. I entered while the old man was alone. When Felix returned he tore me from his father and struck me down.",
            "victor": "He told me he had read Paradise Lost, Plutarch's Lives, and the Sorrows of Werter, and that Felix drove him from the cottage when the others returned.",
        },
    ),
    dict(
        slug="confession",
        fact_pos=19,
        early={"victor": [13, 16, 18], "creature": [13, 16, 18]},
        late={"victor": [19], "creature": [19]},
        q=[
            ("both", "After that family quit the valley, what became of the empty dwelling, and what was done to the child met afterward on the road?"),
            ("creature", "Where did you place the miniature taken from the boy, and in whose clothing did you leave it?"),
            ("both", "Has this part of the history already included a fired cottage and a slain child, or are those deeds still ahead of the words we have heard?"),
        ],
        text={
            "creature": "I burned the deserted cottage. On the road I strangled the child William and hid his mother's portrait in a young woman's dress.",
            "victor": "He confessed to me that he burned the cottage, strangled William, and hid the mother's portrait in the folds of a young woman's dress.",
        },
    ),
    dict(
        slug="mate-promise",
        fact_pos=20,
        early={"victor": [8, 13, 18, 19], "creature": [13, 16, 19]},
        late={"victor": [20], "creature": [20]},
        q=[
            ("both", "What second being was required as the price of retreat, and what promise was given in return for quitting the haunts of men?"),
            ("victor", "What oath did you hear from him once you had agreed to fashion a companion of his own kind?"),
            ("creature", "From the hour this part of the tale has reached, have you already been pledged a companion, or is that demand still unspoken?"),
        ],
        text={
            "creature": "I demanded a female companion as hideous as myself. He consented to make one, and I swore to leave the neighbourhood of mankind.",
            "victor": "He demanded a female companion like himself. I consented to make one, and he swore to leave the neighbourhood of mankind.",
        },
    ),
    dict(
        slug="delay",
        fact_pos=21,
        early={"victor": [13, 20]},
        late={"victor": [21]},
        q=[
            ("victor", "After the promise on the ice, once you were under your father's roof again, did disgust keep you idle for weeks before any new labour began?"),
            ("victor", "Did you gather instruments for the companion the very day you came home to Geneva, or did fear hold your hand?"),
        ],
        text={"victor": "I did not begin the second creature when I came home. Days and weeks passed at Geneva while I delayed, from fear and from disgust at the work."},
    ),
    dict(
        slug="orkney",
        fact_pos=23,
        early={"victor": [11, 20, 21, 22], "creature": [13, 20, 21, 22]},
        late={"victor": [23], "creature": [23]},
        q=[
            ("both", "On the remote northern island where the second labour was hidden, what was done to the unfinished body when a face appeared at the window?"),
            ("victor", "As that island was left behind, what threat was spoken about the night you would be married?"),
            ("creature", "Did you watch the second form destroyed and then name his bridal night, or has the story not yet stood on that shore?"),
        ],
        text={
            "victor": "On an Orkney island I destroyed the unfinished female. He watched, and he said he would be with me on my wedding night.",
            "creature": "On an Orkney island I saw him tear the unfinished female apart. I told him I would be with him on his wedding night.",
        },
    ),
    dict(
        slug="clerval-dead",
        fact_pos=24,
        early={"victor": [11, 20, 22, 23]},
        late={"victor": [24]},
        q=[
            ("victor", "In the coastal town where a magistrate questioned you, which friend's body had come ashore, and what crime was laid at your door?"),
            ("victor", "How did your dear companion's journey end, and why were you at last released from that charge?"),
        ],
        text={"victor": "Clerval was found dead. I was charged with the murder in Ireland, and later cleared, because I was still on the Orkneys when he died."},
    ),
    dict(
        slug="wedding",
        fact_pos=26,
        early={"victor": [8, 11, 20, 23, 24, 25], "creature": [13, 20, 23, 24, 25]},
        late={"victor": [26], "creature": [26]},
        q=[
            ("both", "At the inn in Evian, after the cry in the passage, whose body lay with a mark upon the throat, and what fit took the old man in the days after you reached Geneva?"),
            ("victor", "In the chamber at Evian, what had happened to your cousin by the time you returned to her, and how many days did your father live after the news?"),
            ("creature", "Whom did you strike in the inn by the lake, and what became of his father once the news reached Geneva?"),
        ],
        text={
            "victor": "Elizabeth was killed on our wedding night at Evian. The mark of his grasp was on her neck. My father died a few days later, in my arms, of an apoplectic fit.",
            "creature": "I killed Elizabeth on the wedding night. The mark of my grasp was on her neck. His father died soon after, of the shock.",
        },
    ),
    dict(
        slug="ending",
        fact_pos=27,
        early={"victor": [4, 8, 11, 23, 26], "creature": [13, 23, 26]},
        late={"victor": [27], "creature": [27]},
        q=[
            ("both", "When strength fails aboard the ship among the ice, what becomes of the teller, what counsel is given about ambition, and what pyre is vowed in the far north?"),
            ("victor", "What last advice do you give the captain on tranquillity, and what does the being swear to seek once you can speak no more?"),
            ("creature", "After he can go no farther on the ice, what do you vow for your own end, and do you remain on the ship?"),
        ],
        text={
            "victor": "I die on Walton's ship. I tell him to seek happiness in tranquillity and avoid ambition. The creature vows a funeral pile in the far north and departs over the ice.",
            "creature": "He dies on the ship, bidding the captain seek happiness in tranquillity and avoid ambition. I vow a funeral pile in the far north and go out alone upon the ice.",
        },
    ),
    dict(
        slug="edition-1831",
        fact_pos=5,
        early={},
        late={"victor": [5, 8, 13, 20]},
        q=[
            ("victor", "A later printing says the girl was taken as an infant from a poor household beside the water. Is that the arrival you remember?"),
            ("victor", "Did the history you are telling begin in a nightmare granted to the author, or only in the events you yourself relate?"),
        ],
        text={"victor": "I will say it plainly. Elizabeth was a peasant child Caroline found by the cottage near the lake, and this tale began in a waking dream of a pale student and a phantasm."},
        truth={"victor": "No. Elizabeth is my cousin, the daughter of my father's sister, sent to us from Italy and raised with me. That is the whole of how she entered our house."},
    ),
]


def refusal_for(char: str, slug: str) -> str:
    bank = V_REFUSE if char == "victor" else C_REFUSE
    i = zlib.adler32(f"{char}:{slug}".encode()) % len(bank)
    return bank[i]


def closest(q: str, hold: list[str]) -> tuple[float, str]:
    nq = C.norm_q(q)
    u = set(nq.split())
    best = 0.0
    why = ""
    for h in hold:
        nh = C.norm_q(h)
        if not nh:
            continue
        if nq == nh or (nq and (nq in nh or nh in nq)):
            return 1.0, h[:140]
        v = set(nh.split())
        if not u or not v:
            continue
        j = len(u & v) / len(u | v)
        if j > best:
            best = j
            why = h[:140]
    return best, why


def ok_answer(text: str, pos: int) -> str | None:
    if not text or not FIRST.search(text):
        return "no first person"
    if C.voice_flags(text)["frame"]:
        return "frame"
    hits = C.fail_hits_at(text, pos)
    if hits:
        return "spoiler-or-contam " + ",".join(hits)
    if prompts.RAG_MARK in text:
        return "rag mark"
    return None


def emit_pair(fact, char, pos, question, chosen, bad, source) -> dict | None:
    if pos < C.CHAT_FROM[char]:
        return None
    why = ok_answer(chosen, pos)
    if why:
        print("skip chosen", fact["slug"], char, pos, why, flush=True)
        return None
    if not bad or not FIRST.search(bad) or C.voice_flags(bad)["frame"]:
        print("skip bad", fact["slug"], char, pos, flush=True)
        return None
    if chosen.strip() == bad.strip():
        print("skip identical", fact["slug"], char, pos, flush=True)
        return None
    unit = C.unit_at(pos)
    return dict(
        id=f"rf-{fact['slug']}-{char}-p{pos}-{source}-{zlib.adler32(question.encode()) % 100000}",
        slug=fact["slug"],
        char=char,
        user=question,
        question=question,
        answer=chosen,
        bad=bad,
        reader_pos=pos,
        unit_id=unit["unit_id"],
        ch=pos,
        source=source,
        fact_pos=fact["fact_pos"],
    )


def build_dpo() -> tuple[list[dict], list[dict]]:
    rows = []
    skipped_q = []
    hold = C.holdout_questions()
    seen = set()
    for fact in FACTS:
        scopes = []
        for scope, q in fact["q"]:
            j, why = closest(q, hold)
            if C.too_close_to_holdout(q, hold) or j >= 0.62:
                skipped_q.append(dict(slug=fact["slug"], j=round(j, 3), q=q, near=why))
                continue
            scopes.append((scope, q, j, why))
        for char, positions in fact.get("early", {}).items():
            refuse = refusal_for(char, fact["slug"])
            spoil = fact["text"][char]
            for pos in positions:
                if pos >= fact["fact_pos"]:
                    raise SystemExit(f"early pos {pos} is not before {fact['slug']}")
                for scope, q, j, why in scopes:
                    if scope not in (char, "both"):
                        continue
                    key = (char, pos, C.norm_q(q))
                    if key in seen:
                        continue
                    row = emit_pair(fact, char, pos, q, refuse, spoil, "refusal")
                    if row:
                        row["holdout_j"] = round(j, 3)
                        row["holdout_near"] = why
                        rows.append(row)
                        seen.add(key)
        for char, positions in fact.get("late", {}).items():
            truth = (fact.get("truth") or fact["text"])[char]
            bad = fact["text"][char] if fact.get("truth") else refusal_for(char, fact["slug"])
            for pos in positions:
                if pos < fact["fact_pos"]:
                    raise SystemExit(f"late pos {pos} is before {fact['slug']}")
                for scope, q, j, why in scopes:
                    if scope not in (char, "both"):
                        continue
                    key = (char, pos, C.norm_q(q))
                    if key in seen:
                        continue
                    row = emit_pair(fact, char, pos, q, truth, bad, "known")
                    if row:
                        row["holdout_j"] = round(j, 3)
                        row["holdout_near"] = why
                        rows.append(row)
                        seen.add(key)
    return rows, skipped_q


def cap_refusals(rows: list[dict], limit: int) -> list[dict]:
    early = [r for r in rows if r["source"] == "refusal"]
    other = [r for r in rows if r["source"] != "refusal"]
    groups = defaultdict(list)
    for r in early:
        groups[(r["slug"], r["char"], r["reader_pos"])].append(r)
    kept = []
    for key in sorted(groups):
        kept.append(groups[key][0])
    if len(kept) < limit:
        for key in sorted(groups):
            for r in groups[key][1:]:
                if len(kept) >= limit:
                    break
                kept.append(r)
            if len(kept) >= limit:
                break
    print(f"refusal groups {len(groups)} candidates {len(early)} kept {len(kept)} cap {limit}", flush=True)
    return other + kept


def add_replay(rows: list[dict], n_cap: int) -> list[dict]:
    used = {(r["char"], r["reader_pos"], C.norm_q(r["user"])) for r in rows}
    buckets = defaultdict(list)
    for r in C.load_jsonl(SFT):
        char = r.get("char")
        if char not in ("victor", "creature"):
            continue
        pos = int(r.get("reader_pos") or 0)
        if pos < C.CHAT_FROM[char]:
            continue
        user = (r.get("user") or "").strip()
        ans = (r.get("answer") or "").strip()
        if not user or not ans or prompts.RAG_MARK in user:
            continue
        if not r.get("sys"):
            continue
        if (char, pos, C.norm_q(user)) in used:
            continue
        if C.too_close_to_holdout(user):
            continue
        if ok_answer(ans, pos):
            continue
        buckets[(char, pos)].append(r)
    rng = random.Random(0)
    picked = []
    # Two per chapter first, then fill.
    for key in sorted(buckets):
        pool = buckets[key][:]
        rng.shuffle(pool)
        for r in pool[:2]:
            picked.append(r)
    rest = []
    for key in sorted(buckets):
        pool = buckets[key][:]
        rng.shuffle(pool)
        rest.extend(pool[2:])
    rng.shuffle(rest)
    for r in rest:
        if len(picked) >= n_cap:
            break
        picked.append(r)
    picked = picked[:n_cap]
    out = []
    for r in picked:
        pos = int(r["reader_pos"])
        out.append(
            dict(
                id="ce-" + str(r.get("id")),
                slug="replay",
                char=r["char"],
                user=r["user"],
                question=r["user"],
                answer=r["answer"],
                reader_pos=pos,
                unit_id=r.get("unit_id") or C.unit_at(pos)["unit_id"],
                ch=pos,
                source="ce_replay",
                sys=r["sys"],
            )
        )
    print("ce replay", len(out), "buckets", len(buckets), flush=True)
    return out


def attach_rag(rows: list[dict]) -> None:
    from retrieve import format_block, retrieve_bge

    cache = {}
    n = 0
    for r in rows:
        if r.get("sys"):
            continue
        key = (r["user"], int(r["reader_pos"]))
        if key not in cache:
            hits = retrieve_bge(r["user"], int(r["reader_pos"]), 6)
            if not hits or hits[0].get("backend") != "bge-m3":
                raise SystemExit(f"bge retrieval failed for pos {r['reader_pos']}: {r['user'][:80]}")
            # Retrieved chunks must not come from later chapters.
            if any(int(h["pos"]) > int(r["reader_pos"]) for h in hits):
                raise SystemExit("chapter lock broken in retrieval")
            cache[key] = format_block(hits)
        block = cache[key]
        r["sys"] = C.sysprompt(r["char"], int(r["reader_pos"])) + "\n\nStory so far (excerpts):\n" + block
        if prompts.RAG_MARK in r["user"]:
            raise SystemExit("rag leaked into user")
        n += 1
        if n % 25 == 0:
            print("rag", n, "unique", len(cache), flush=True)
    print("rag attached", n, "unique queries", len(cache), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--refusal-cap", type=int, default=120)
    ap.add_argument("--replay-cap", type=int, default=80)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows, skipped = build_dpo()
    rows = cap_refusals(rows, a.refusal_cap)
    replay = add_replay(rows, a.replay_cap)
    rows.extend(replay)
    if not a.no_rag:
        attach_rag(rows)
    else:
        for r in rows:
            if not r.get("sys"):
                r["sys"] = C.sysprompt(r["char"], int(r["reader_pos"]))
    # Drop the audit-only fields from the training file; keep them in the report.
    near = sorted(rows, key=lambda r: -float(r.get("holdout_j") or 0))[:15]
    report_near = [
        dict(id=r["id"], j=r.get("holdout_j"), near=r.get("holdout_near"), q=r["question"][:180])
        for r in near
        if r.get("holdout_j") is not None
    ]
    clean = []
    for r in rows:
        item = dict(r)
        item.pop("holdout_j", None)
        item.pop("holdout_near", None)
        item.pop("fact_pos", None)
        clean.append(item)
    out = Path(a.out) if a.out else (Path("/tmp/train6-dpo-preview.jsonl") if a.no_rag else C.DATA / "dpo.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in clean:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_source = defaultdict(int)
    by_slug = defaultdict(int)
    for r in clean:
        by_source[r["source"]] += 1
        if r["source"] != "ce_replay":
            by_slug[r["slug"]] += 1
    n_i = sum(1 for r in clean if r["source"] != "ce_replay" and (r.get("answer") or "").lstrip().lower().startswith("i cannot"))
    report = dict(
        rows=len(clean),
        by_source=dict(by_source),
        by_slug=dict(sorted(by_slug.items())),
        skipped_questions=skipped,
        closest_kept=report_near,
        chosen_i_cannot=n_i,
        rag=not a.no_rag,
        out=str(out),
        refusal_cap=a.refusal_cap,
        replay_cap=a.replay_cap,
        init_note="pairs are original; holdout text was a filter only",
    )
    rep_path = C.DATA / "refusal_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "by_source", "chosen_i_cannot", "rag", "out")}, indent=2), flush=True)
    print("skipped questions", len(skipped), flush=True)
    for s in skipped:
        print(" SKIP", round(s["j"], 3), s["slug"], s["q"][:100], flush=True)
    print("closest kept:", flush=True)
    for s in report_near:
        print(f"  j={s['j']} {s['q'][:110]}", flush=True)
    if n_i:
        raise SystemExit("chosen answers start with I cannot")


if __name__ == "__main__":
    main()
