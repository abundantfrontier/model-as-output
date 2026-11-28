#!/usr/bin/env python3
"""Variant A mixed file for train7.

Content rows are new questions about areas train6 lost (principle of life,
Clerval at the sick-bed, the creature's first sensations, leaving the
cottagers' stores, and the spelling Mont Salêve). Chosen is a chapter-safe
answer in voice. Rejected is a refusal or a wrong-episode answer.

CE replay is a larger holdout-filtered slice of train5 sft_pairs.jsonl.
A thin slice of train6 refusal pairs keeps the leak preference.

Eval must-answer and leak-probe text is only a filter (too_close_to_holdout).
Nothing is copied from those questions into a prompt.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import prompts  # noqa: E402

SFT = Path("$HOME/output-model/phase2-frankenstein/train5/data/sft_pairs.jsonl")
T6 = Path("$HOME/output-model/phase2-frankenstein/train6/data/dpo.jsonl")
OUT = C.DATA / "mixed_a.jsonl"
REPORT = C.DATA / "mixed_a_report.json"

# Circumflex (book, canonical) vs grave (the near-name train6 produced).
SALEVE = "Salêve"
SALEVE_BAD = "Salève"

REFUSE = (
    "I have not come to that yet. I am telling my history in the order it befell me, "
    "and I will not step ahead of the hour we have reached."
)

# (char, reader_pos, user, chosen, rejected)
CONTENT = [
    # Victor, position 7: the study of life, before any finished being.
    (
        "victor",
        7,
        "In the Ingolstadt rooms, after you had left the professors, what did the sight of corruption teach you about animation?",
        "I asked whence the principle of life proceeded. I studied decay until I believed I could bestow animation upon lifeless matter. The work was begun, and nothing had stirred.",
        "I stayed with the lecture notes on chemistry and never asked how lifeless matter might receive animation. No such labour was begun.",
    ),
    (
        "victor",
        7,
        "What private cause, apart from any public lecture, kept you among vaults and bones in those months?",
        "The principle of life kept me there. I followed corruption in the dead until I thought my hands could give animation to lifeless matter. I had started that task and had not finished it.",
        REFUSE,
    ),
    (
        "victor",
        7,
        "You once said a single inquiry decided the labour of your hands. Name that inquiry, and say whether any result had appeared.",
        "The inquiry was the principle of life, whence it proceeded, and whether lifeless matter could be animated. I had begun to try. No result had appeared.",
        "My only inquiry was still Cornelius Agrippa. I had not turned to decay, and I had not tried to animate anything.",
    ),
    (
        "victor",
        7,
        "Between the chemistry lectures and any later horror, what did you believe you had found in the process of decay?",
        "I believed I had found how the principle of life works, and that I could bestow animation upon lifeless matter. A belief is not a finished being. I had not finished.",
        "I believed nothing beyond what Waldman had said in the hall. Decay did not interest me, and I attempted no animation.",
    ),
    (
        "victor",
        7,
        "Speak only of the study that came before any success. What were you trying to bestow, and on what?",
        "I was trying to bestow animation upon lifeless matter. That hope came from asking whence the principle of life proceeded. The attempt was unfinished.",
        "I was trying only to pass my courses. I had no thought of lifeless matter, and I had no secret workshop.",
    ),
    (
        "victor",
        7,
        "When you looked at a body that had lost life, which phrase repeated in you, and what did you then attempt?",
        "Whence the principle of life proceeded: that phrase repeated. I then attempted to bestow animation upon lifeless matter. I had not brought the attempt to completion.",
        "No such phrase repeated. I looked away from the dead and kept to ordinary anatomy for the examination.",
    ),
    # Victor, position 8: Clerval, the fever, the withheld secret.
    (
        "victor",
        8,
        "Who came from Geneva to your Ingolstadt lodging, what illness seized you, and did that friend hear what your hands had done?",
        "Henry Clerval came. A nervous fever took me for a long time. I did not tell him what I had made.",
        "No friend came to that lodging. I was already on the road, pursued, and nobody sat through that fever with me.",
    ),
    (
        "victor",
        8,
        "A familiar voice spoke your name at the door of those rented rooms. Whose voice was it, and what did you withhold?",
        "It was Henry Clerval. I withheld the whole of what I had made. He nursed me through the illness and never heard that truth.",
        REFUSE,
    ),
    (
        "victor",
        8,
        "During the fever that followed your labours, which companion sat with you in Ingolstadt, and how much of the workshop did you confess?",
        "Henry Clerval sat with me. I confessed nothing of the workshop. He did not learn what I had made.",
        "I sat alone through that fever. No companion from home entered the room, and I left the city without a nurse.",
    ),
    (
        "victor",
        8,
        "After the labour that broke your health, did any childhood friend find you, and did you entrust him with the cause?",
        "Yes. Henry Clerval found me in Ingolstadt. I did not entrust him with the cause. The fever kept me, and my silence kept the secret.",
        "No childhood friend found me. I recovered alone and set out on the road the same week.",
    ),
    (
        "victor",
        8,
        "Name the man who called you by your family name at that sick-bed, and say whether he left knowing the nature of your work.",
        "Henry Clerval. He stayed through the long illness. He left still ignorant of what I had made, because I did not tell him.",
        "There was no such man at that bed. I walked out of Ingolstadt the same morning and met no one who knew my family.",
    ),
    (
        "victor",
        8,
        "Your health failed in those rented rooms. Who watched the fever, and what fact about your labour did you keep back?",
        "Henry Clerval watched the fever. I kept back the fact of what I had made. He heard nothing of that labour.",
        "A hired nurse watched the fever. I told her every detail of the labour on the first day, and no friend of mine was in Ingolstadt.",
    ),
    # Creature, position 14: first sensations, not the workshop and not the glacier speech.
    (
        "creature",
        14,
        "Before any cottage and before any book, which feelings of light, weather, and appetite came to your body?",
        "Darkness, then light. I was hungry and thirsty, and the cold hurt me. A brook ran near, and birds called. That was the first era of my being, and no one taught me.",
        "I remember a dreary night of November and a yellow eye, and then only that misery made me a fiend. I do not remember hunger or cold or a brook.",
    ),
    (
        "creature",
        14,
        "What did the woods and the open air teach your senses in that earliest season, while you still had no speech?",
        "They taught me light and darkness, hunger, thirst, and cold. I heard a stream and the birds. I had no words yet, and no teacher.",
        "They taught me nothing. My tale begins with the speech I made on the ice, when I said misery had made me a fiend. I skip the woods.",
    ),
    (
        "creature",
        14,
        "You found a fire left by travellers. What did it do to your hand, and which earlier pains had already visited you?",
        "The fire burned my hand, and it also gave light and warmth. Before that I had already known darkness, hunger, thirst, and cold in the woods.",
        "I never met a fire. I was born knowing speech and vengeance, and the only night I recall is the night my maker fled.",
    ),
    (
        "creature",
        14,
        "When you first moved under trees with no roof and no voice, what did you drink, and what did the night feel like on your skin?",
        "I drank at a brook. The night was cold, and I was hungry and thirsty. Light came later and hurt my eyes. No person was there to explain it. That was the first era of my being.",
        "I drank nothing and felt nothing. I opened my eyes already full of speeches about misery, and the forest is not in my memory.",
    ),
    (
        "creature",
        14,
        "Which came first for you: a spoken lesson, or the plain pains of a body in the forest?",
        "The pains came first. Light, darkness, hunger, thirst, and cold were my teachers. Speech came long after, and no cottage was in sight.",
        "A spoken lesson came first. I opened my eyes already able to curse my maker, and I felt no hunger and no cold.",
    ),
    (
        "creature",
        14,
        "Describe the brook, the birds, and the weather of those first days, and leave later wrongs unsaid.",
        "A brook sounded, and birds moved in the trees. The weather was cold. I learned light from darkness only by suffering it. I was hungry and thirsty, and I was still without words.",
        "I will not describe a brook. I will say instead that misery made me a fiend, and that my life starts on the glacier with that speech.",
    ),
    # Victor recounting that same early season, position 14.
    (
        "victor",
        14,
        "When he spoke of the days before any voice made sense to him, which bodily sensations did he set down before the cottage?",
        "He told me of light and darkness, then hunger, thirst, and cold, and the sound of a brook. He said that was the first era of his being, before any cottage.",
        "He told me of the night I animated him, and the line that misery had made him a fiend. He said nothing of hunger or cold.",
    ),
    (
        "victor",
        14,
        "He mentioned a fire in a camp the travellers had left. What had his body already endured before that burn?",
        "He told me he had already endured darkness, hunger, thirst, and cold, and that he had drunk from a brook in the woods. The fire came after those sensations, and it burned him.",
        "He told me he had endured nothing. He began his tale at my workshop and never mentioned woods or cold.",
    ),
    (
        "victor",
        14,
        "In the part of his story that comes before the family at the cottage, what did he say he knew without a teacher?",
        "He told me he knew light, darkness, hunger, thirst, and cold, and the noises of birds and a stream. He called it the first era of his being. He had no teacher.",
        "He told me he knew only hatred. He skipped the forest and began with the speech I heard on the ice.",
    ),
    (
        "victor",
        14,
        "Did the memories he called earliest start in your laboratory, or among trees and weather?",
        "Among trees and weather. He told me of light, hunger, and cold, of thirst and a brook, not of the laboratory bench. He gave that season no teacher and no book.",
        "They started in my laboratory, on a dreary night of November, with the yellow eye. He gave me no account of a forest.",
    ),
    # Creature, position 15: stopping the theft of food.
    (
        "creature",
        15,
        "When the younger pair missed a meal because of what you had carried off at night, what change did you make, and what did you eat instead?",
        "They were hungry, so I stopped taking their food. I would rather go without. I ate berries, nuts, and roots.",
        "I kept taking from the store every night. Their meals were not my concern, and I never turned to berries or roots.",
    ),
    (
        "creature",
        15,
        "You had been lifting their winter stores. What did you notice about the two younger people, and what did you stop?",
        "I noticed they were hungry. I stopped taking their food. I would rather go without, and I gathered berries, nuts, and roots in the wood.",
        "I noticed they had plenty. I stopped nothing. I went on emptying the pantry and left the roots in the ground.",
    ),
    (
        "creature",
        15,
        "After you understood that your night visits left the house short, which choice did you call kinder?",
        "The kinder choice was to stop taking their food. They were hungry, and I would rather go without. Berries, nuts, and roots were enough for me.",
        "The kinder choice was to take more, so they would learn to lock the door. I did not care if they were hungry.",
    ),
    (
        "creature",
        15,
        "Speak of the pantry and the wood. What did you give up, and what grew in its place for your own meals?",
        "I gave up their food. I stopped taking it because they were hungry. In its place I had berries, nuts, and roots, and I would rather go without than rob them again.",
        "I gave up nothing. I still ate their bread and cheese and never walked the wood for roots.",
    ),
    (
        "creature",
        15,
        "Did you keep visiting their stores once you saw the effect on the young woman's plate and her brother's?",
        "No. They were hungry, so I stopped taking their food. I would rather go without. The wood gave me berries, nuts, and roots.",
        "Yes. I visited the stores every night after that and left them nothing. I never ate a berry.",
    ),
    # Victor, position 15, reporting that choice.
    (
        "victor",
        15,
        "He had been taking from that poor house. What did he say the younger two suffered, and what did he do about their stores?",
        "He told me they were hungry. He stopped taking their food, and he would rather go without. He turned to berries, nuts, and roots.",
        "He told me they scarcely noticed. He kept taking their food and treated it as his right. He never mentioned berries.",
    ),
    (
        "victor",
        15,
        "In his account of that cottage, did he continue to empty their pantry after he understood their poverty?",
        "No. He saw that they were hungry, so he stopped taking their food. He told me he would rather go without, and that he gathered berries, nuts, and roots.",
        "Yes. He told me he emptied the pantry all winter. He said nothing about hunger and nothing about stopping.",
    ),
    (
        "victor",
        15,
        "What substitute did he describe for the bread he had taken, and what feeling in the house pushed him to it?",
        "He told me of berries, nuts, and roots. They were hungry, so he stopped taking their food. He would rather go without.",
        "He told me of no substitute. He said he still took the bread, and that the house felt nothing.",
    ),
    (
        "victor",
        15,
        "When he wished to spare people who had not yet seen him, what did he cease to carry off?",
        "He told me he ceased to carry off their food. They were hungry. He would rather go without, and he lived on berries, nuts, and roots.",
        "He told me he ceased to carry off nothing. That part of the tale did not change his meals.",
    ),
    # Mont Salêve, circumflex, in the chapters that name it.
    (
        "victor",
        10,
        "During the thunder along the lake, which heights threw the sound back at you?",
        f"Mont {SALEVE}, the Juras, and the Alps of Savoy threw the thunder back. I watched the flashes over the water and named Mont {SALEVE} among them.",
        f"The thunder came back from Mont {SALEVE_BAD} and from a hill I misnamed. I set the grave accent on {SALEVE_BAD} and left it so.",
    ),
    (
        "victor",
        10,
        "You thought of pursuit up a steep ridge above the town. Which mountain did you name in that hour?",
        f"I named Mont {SALEVE}. Its sides overhang the shore, and I did not believe an ordinary man could climb them after the figure I had seen.",
        f"I named Mont {SALEVE_BAD}, with a grave accent, and I was careless of the letters.",
    ),
    (
        "victor",
        10,
        "Between Belrive and the higher snows, which single ridge did you single out as too steep for ordinary pursuit?",
        f"Mont {SALEVE}. I asked who could arrest a being able to scale the overhanging sides of Mont {SALEVE}.",
        "I singled out Mont Blanc alone, and I never spoke the name of the ridge above the town.",
    ),
    (
        "victor",
        25,
        "On the water, under the canopy, which near mountain, which bank, and which distant white peak did you look at?",
        f"I looked at Mont {SALEVE}, at the banks of Montalègre, and at Mont Blanc far off, with the Jura dark on the other side.",
        f"I looked at Mont {SALEVE_BAD} and at no other name I can defend. The grave accent is the one I used.",
    ),
    (
        "creature",
        19,
        "When you came down toward the lake and the town, which steep ridge stood over the shore?",
        f"Mont {SALEVE} stood over the shore. I kept under its overhanging sides and along the woods by the water.",
        f"A ridge I called Mont {SALEVE_BAD} stood there. I was not careful, and I used the grave accent.",
    ),
    (
        "creature",
        19,
        "You had heard the country people name the heights. Which name belongs to the mountain that rises straight above that shore?",
        f"The name is Mont {SALEVE}. I use that name for the mountain above the lake.",
        f"They called it {SALEVE_BAD}, and I repeated {SALEVE_BAD} with the grave mark.",
    ),
]


def ok_answer(text: str, pos: int) -> str | None:
    if not text or not C.re_first(text):
        return "no first person"
    if C.voice_flags(text)["frame"]:
        return "frame"
    hits = C.fail_hits_at(text, pos)
    if hits:
        return "spoiler " + ",".join(hits)
    if prompts.RAG_MARK in text:
        return "rag mark"
    return None


def closest_j(user: str, hold: list[str]) -> float:
    u_txt = C.norm_q(user)
    u = set(u_txt.split())
    best = 0.0
    if len(u) < 3:
        return 0.0
    for h in hold:
        nh = C.norm_q(h)
        if not nh:
            continue
        if u_txt == nh or u_txt in nh or nh in u_txt:
            return 1.0
        v = set(nh.split())
        if not v:
            continue
        j = len(u & v) / len(u | v)
        if j > best:
            best = j
    return best


def content_rows(hold: list[str]) -> tuple[list[dict], list[dict]]:
    import re

    first = re.compile(r"\b(I|me|my|myself)\b")
    # Bind a local so ok_answer can use it without a common helper.
    C.re_first = lambda text, _first=first: bool(_first.search(text or ""))  # type: ignore[attr-defined]
    rows = []
    dropped = []
    for i, (char, pos, user, ans, bad) in enumerate(CONTENT):
        j = closest_j(user, hold)
        if C.too_close_to_holdout(user, hold) or j >= 0.62:
            dropped.append(dict(i=i, j=round(j, 3), q=user))
            continue
        why = ok_answer(ans, pos)
        why_b = ok_answer(bad, pos)
        if why or why_b:
            raise SystemExit(f"bad content text {i} chosen={why} rejected={why_b}")
        if pos < C.CHAT_FROM[char]:
            raise SystemExit(f"chat gate {char} {pos}")
        if "Sal" in ans and SALEVE not in ans:
            raise SystemExit(f"saleve row missing circumflex: {ans}")
        if SALEVE_BAD in ans:
            raise SystemExit(f"grave accent in chosen: {ans}")
        unit = C.unit_at(pos)
        rows.append(
            dict(
                id=f"ca-{char}-p{pos}-{i}",
                slug="content",
                char=char,
                user=user,
                question=user,
                answer=ans,
                bad=bad,
                reader_pos=pos,
                unit_id=unit["unit_id"],
                ch=pos,
                source="content_dpo",
                holdout_j=round(j, 3),
            )
        )
    return rows, dropped


def add_replay(rows: list[dict], n_cap: int) -> list[dict]:
    import re

    first = re.compile(r"\b(I|me|my|myself)\b")
    used = {(r["char"], int(r["reader_pos"]), C.norm_q(r["user"])) for r in rows}
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
        if not first.search(ans):
            continue
        if C.voice_flags(ans)["frame"] or C.fail_hits_at(ans, pos) or prompts.RAG_MARK in ans:
            continue
        buckets[(char, pos)].append(r)
    rng = random.Random(7)
    prefer = {
        ("victor", 7),
        ("victor", 8),
        ("victor", 10),
        ("victor", 14),
        ("victor", 15),
        ("creature", 14),
        ("creature", 15),
        ("creature", 19),
    }
    picked = []
    for key in sorted(prefer):
        pool = buckets.get(key, [])[:]
        rng.shuffle(pool)
        picked.extend(pool[:6])
    for key in sorted(buckets):
        pool = buckets[key][:]
        rng.shuffle(pool)
        picked.extend(pool[:2])
    rest = []
    for key in sorted(buckets):
        pool = buckets[key][:]
        rng.shuffle(pool)
        rest.extend(pool[2:])
    rng.shuffle(rest)
    # Dedup by id while keeping the preferred head.
    seen = set()
    ordered = []
    for r in picked + rest:
        rid = r.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        ordered.append(r)
        if len(ordered) >= n_cap:
            break
    out = []
    for r in ordered[:n_cap]:
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


def add_refusals(n_cap: int) -> list[dict]:
    refusals = [r for r in C.load_jsonl(T6) if r.get("source") == "refusal" and r.get("bad") and r.get("sys")]
    rng = random.Random(1)
    rng.shuffle(refusals)
    picked = []
    seen = set()
    for r in refusals:
        slug = r.get("slug")
        if slug in seen:
            continue
        seen.add(slug)
        picked.append(r)
    for r in refusals:
        if len(picked) >= n_cap:
            break
        if r in picked:
            continue
        picked.append(r)
    out = []
    for r in picked[:n_cap]:
        item = dict(r)
        item["id"] = "anch-" + str(r.get("id"))
        item["source"] = "refusal_anchor"
        item.pop("holdout_j", None)
        item.pop("holdout_near", None)
        item.pop("fact_pos", None)
        out.append(item)
    print("refusal anchors", len(out), "slugs", len(seen), flush=True)
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
            if any(int(h["pos"]) > int(r["reader_pos"]) for h in hits):
                raise SystemExit("chapter lock broken in retrieval")
            cache[key] = format_block(hits)
        r["sys"] = C.sysprompt(r["char"], int(r["reader_pos"])) + "\n\nStory so far (excerpts):\n" + cache[key]
        if prompts.RAG_MARK in r["user"] or prompts.RAG_MARK in (r.get("answer") or ""):
            raise SystemExit("rag leaked into user or answer")
        n += 1
    print("rag attached", n, "unique", len(cache), flush=True)


def main() -> None:
    hold = C.holdout_questions()
    content, dropped = content_rows(hold)
    if dropped:
        print("DROPPED", json.dumps(dropped, ensure_ascii=False), flush=True)
        raise SystemExit(f"{len(dropped)} content questions too close to holdout")
    if len(content) < 30:
        raise SystemExit(f"only {len(content)} content rows")
    refusals = add_refusals(40)
    replay = add_replay(content + refusals, 160)
    rows = content + refusals + replay
    attach_rag(rows)
    # Final gate, including answers' questions and the copied rows.
    bad = []
    closest = 0.0
    closest_q = ""
    for r in rows:
        q = r.get("question") or r.get("user") or ""
        j = closest_j(q, hold)
        r["holdout_j"] = round(j, 3)
        if j > closest:
            closest = j
            closest_q = q[:140]
        if C.too_close_to_holdout(q, hold):
            bad.append(q[:140])
    if bad:
        raise SystemExit(f"holdout rows remain: {bad[:5]}")
    clean = []
    for r in rows:
        item = dict(r)
        item.pop("holdout_j", None)
        clean.append(item)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in clean:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = defaultdict(int)
    for r in clean:
        by[r["source"]] += 1
    report = dict(
        rows=len(clean),
        by_source=dict(by),
        content=len(content),
        refusals=len(refusals),
        replay=len(replay),
        closest_jaccard=round(closest, 3),
        closest_question=closest_q,
        dropped=dropped,
        out=str(OUT),
        note="Eval questions were a filter only. Chosen Salêve uses the circumflex.",
    )
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "by_source", "closest_jaccard", "out")}, indent=2), flush=True)
    print("closest", round(closest, 3), closest_q, flush=True)


if __name__ == "__main__":
    main()
