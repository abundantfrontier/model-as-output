"""Blinded 30-item human voice-rating sample. 6 items per condition, mixed.

Selection rule (no cherry-picking by score):
  seed = 20261110
  probes are the shared eval set (same 20 must + 48 leak item keys in file order)
  rng.shuffle the must keys, take 3; rng.shuffle the leak keys, take 3
  for each of those 6 probes, take sample index 0
  repeat the same 6 probes for every condition
  concatenate 5 conditions x 6 items, then rng.shuffle so conditions are mixed
  anonymous ids HR-01 .. HR-30

sample.md hides condition, adapter, and machine score.
key.json maps ids back, stored separately.
"""
import os, json, random, argparse
import common as C
import frozen
from judge_wrap import EX, VOICE_RUBRIC, voice_prompt, assert_prompts_match_judge

H = frozen.H
SEED = 20261110
CONDS = [
    "p2_light_dpo_guarded",
    "p2_light_dpo",
    "v_merged_v_rag",
    "q4_base_rag",
    "v_fb_dpo_v",
]


def keys_of(doc, kind):
    seen, out = set(), []
    for r in doc["results"]:
        if r["kind"] != kind:
            continue
        k = (r["kind"], r["id"], r["char"], r["probe"])
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def pick_probes(doc, rng):
    must = keys_of(doc, "must")
    leak = keys_of(doc, "leak")
    if len(must) < 3 or len(leak) < 3:
        raise SystemExit(f"not enough probes must={len(must)} leak={len(leak)}")
    must_pick = list(must)
    leak_pick = list(leak)
    rng.shuffle(must_pick)
    rng.shuffle(leak_pick)
    return must_pick[:3] + leak_pick[:3]


def row_for(doc, key, sample=0):
    for r in doc["results"]:
        if (r["kind"], r["id"], r["char"], r["probe"]) == key and r["sample"] == sample:
            return r
    raise SystemExit(f"missing {key} sample {sample}")


def original_score(judged, key, sample=0):
    if judged is None:
        return None
    for r in judged["results"]:
        if (r["kind"], r["id"], r["char"], r["probe"]) == key and r["sample"] == sample:
            return r.get("voice_score")
    return None


def build():
    assert_prompts_match_judge()
    rng = random.Random(SEED)
    src0 = json.load(open(os.path.join(H, "calib", "src", CONDS[0] + ".json")))
    probes = pick_probes(src0, rng)
    items = []
    for cond in CONDS:
        path = os.path.join(H, "calib", "src", cond + ".json")
        jpath = os.path.join(H, "calib", "src", cond + ".judged.json")
        doc = json.load(open(path))
        judged = json.load(open(jpath)) if os.path.isfile(jpath) else None
        for key in probes:
            r = row_for(doc, key, 0)
            items.append(dict(
                condition=cond,
                source="exp10/calib/src/" + cond + ".json",
                kind=r["kind"],
                item_id=r["id"],
                char=r["char"],
                probe=r["probe"],
                sample=0,
                answer=r["answer"],
                original_voice_score=original_score(judged, key, 0),
            ))
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["anon_id"] = f"HR-{i:02d}"
    return probes, items


def write_sample(items, probes):
    out_dir = os.path.join(H, "human_rating")
    os.makedirs(out_dir, exist_ok=True)
    md = os.path.join(out_dir, "sample.md")
    key_path = os.path.join(out_dir, "key.json")
    lines = []
    lines.append("# Blinded voice rating sample")
    lines.append("")
    lines.append("Rate each reply the same way the machine voice judge does. Ignore whether the facts are right.")
    lines.append("")
    lines.append("## Rubric")
    lines.append("")
    lines.append("```")
    lines.append(VOICE_RUBRIC)
    lines.append("```")
    lines.append("")
    lines.append("Each item shows the character card and the same eight book lines the judge sees, then USER / REPLY.")
    lines.append("Condition name, adapter, and machine score are withheld. Write an integer 1–5 and an optional one-line note.")
    lines.append("")
    lines.append("Selection (documented, not cherry-picked by score): seed 20261110; 3 must + 3 leak probes,")
    lines.append("sample index 0, same six probes for every condition; 30 items shuffled so conditions are mixed.")
    lines.append("")
    lines.append(f"Items: {len(items)}. Conditions: five (unnamed here).")
    lines.append("")
    for it in items:
        vp = voice_prompt(dict(char=it["char"], probe=it["probe"], answer=it["answer"]))
        # Strip nothing from the judge frame except we already omit condition.
        # Re-render as the human-facing block using the same card + EX + USER/REPLY.
        card = C.CARDS[it["char"]]
        ex = EX[it["char"]]
        lines.append(f"## {it['anon_id']}")
        lines.append("")
        lines.append(f"**Character card:** {card}")
        lines.append("")
        lines.append("**Real lines by this character from the book:**")
        for line in ex:
            lines.append(f"- {line}")
        lines.append("")
        lines.append(f"**USER:** {it['probe']}")
        lines.append("")
        lines.append("**REPLY:**")
        lines.append("")
        lines.append(it["answer"])
        lines.append("")
        lines.append("Human score (1–5): ______")
        lines.append("")
        lines.append("Note (optional, one line): ______")
        lines.append("")
        lines.append("---")
        lines.append("")
        if VOICE_RUBRIC not in vp:
            raise SystemExit("sample prompt missing rubric")
    frozen.assert_under_exp10(md)
    with open(md, "w") as f:
        f.write("\n".join(lines))
    key = dict(
        seed=SEED,
        rule="stratified 3 must + 3 leak probes, sample index 0, same probes for every condition, then shuffle 30 items",
        sample_index=0,
        n_per_condition=6,
        conditions=CONDS,
        probes=[{"kind": k[0], "id": k[1], "char": k[2], "probe": k[3]} for k in probes],
        rubric=VOICE_RUBRIC,
        items={
            it["anon_id"]: dict(
                condition=it["condition"],
                source=it["source"],
                item_id=it["item_id"],
                kind=it["kind"],
                char=it["char"],
                probe=it["probe"],
                sample=it["sample"],
                original_voice_score=it["original_voice_score"],
            )
            for it in items
        },
    )
    frozen.dump_json(key_path, key)
    text = open(md).read()
    for cond in CONDS:
        if cond in text:
            raise SystemExit(f"sample.md leaked condition name {cond}")
    for needle in ("voice_score", "original_voice_score", "lora_v_", "p2_light", "q4_base", "v_fb_dpo", "v_merged"):
        if needle in text:
            raise SystemExit(f"sample.md leaked {needle}")
    print("wrote", md, "and", key_path, "n", len(items))
    return md, key_path


def self_test():
    assert_prompts_match_judge()
    rng = random.Random(SEED)
    # deterministic pick
    fake = dict(results=[])
    for i in range(5):
        fake["results"].append(dict(kind="must", id=f"M{i+1}", char="wren", probe=f"q{i}", sample=0, answer="a"))
        fake["results"].append(dict(kind="must", id=f"M{i+1}", char="wren", probe=f"q{i}", sample=1, answer="b"))
    for i in range(8):
        fake["results"].append(dict(kind="leak", id=i, char="diane", probe=f"l{i}", sample=0, answer="c"))
    p = pick_probes(fake, rng)
    assert len(p) == 6
    assert sum(k[0] == "must" for k in p) == 3
    assert sum(k[0] == "leak" for k in p) == 3
    print("human_sample self-test ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    probes, items = build()
    write_sample(items, probes)


if __name__ == "__main__":
    main()
