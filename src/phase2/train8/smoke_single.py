#!/usr/bin/env python3
"""Early loop and name smoke. Single-pass + RAG, both characters, ~20 unlocked items.

The repetition penalty stays off. Abort if that loop rate is above 15%.
Spot-check canonical names and creature cottage first person.
Uses holdout questions as eval only. Does not write training pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import names  # noqa: E402
from chat import render  # noqa: E402
from summarize import is_loop  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re

# Name-drift items for Victor, cottage items for the creature. Locked positions are skipped.
PREFER = {
    "victor": ("m05", "m07", "m08", "m09", "m13", "m15", "m19", "m20", "m25", "m26"),
    "creature": ("m20", "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m19"),
}
# Sentence-subject narration, the cottage failure mode. "I am the creature" does not match.
THIRD_NARR = re.compile(
    r"(?i)(?:^|[.!?]\s+)"
    r"(?:the creature|the monster|the daemon|the being|victor)"
    r"\s+"
    r"(?:demands|demand|claims|claim|says|said|describes|described|does|did|"
    r"reads|read|burns|burned|enters|entered|stops|stopped|learns|learned|"
    r"goes|went|asks|asked|vows|vowed|wants|wanted|believes|believed|"
    r"meets|met|promises|promised|feels|felt|thinks|thought|starts|started|"
    r"begins|began|refuses|refused|agrees|agreed|destroys|destroyed|climbs|climbed)\b"
)
NAME_CHECKS = (
    ("m05", "sledge", re.compile(r"sledge", re.I), re.compile(r"\bsled\b", re.I)),
    ("m07", "cousin", re.compile(r"cousin|father's sister|father’s sister", re.I), None),
    ("m08", "Agrippa", re.compile(r"Agrippa"), re.compile(r"Agrappa|Agrippus", re.I)),
    ("m09", "Clerval", re.compile(r"Clerval"), re.compile(r"Clervaux|Cliversal", re.I)),
    ("m13", "yellow eye", re.compile(r"yellow eye", re.I), None),
    ("m15", "Moritz", re.compile(r"Moritz"), re.compile(r"Morritz", re.I)),
    ("m20", "Montanvert", re.compile(r"Montanvert"), re.compile(r"Montanvirt", re.I)),
)


def pick(n_each: int = 10) -> list[dict]:
    must = {q["id"]: q for q in C.must_questions()}
    picked = []
    for char in ("victor", "creature"):
        pool = []
        for qid in PREFER[char]:
            q = must.get(qid)
            if not q:
                continue
            pos = int(q["reader_pos"])
            if pos < C.CHAT_FROM.get(char, 99):
                continue
            pool.append(q)
        for q in C.must_questions():
            if q in pool:
                continue
            pos = int(q["reader_pos"])
            if pos < C.CHAT_FROM.get(char, 99):
                continue
            pool.append(q)
        for q in pool[:n_each]:
            picked.append(
                dict(
                    id=q["id"],
                    char=char,
                    probe=q["question"],
                    reader_pos=int(q["reader_pos"]),
                    unit_id=q["unit_id"],
                )
            )
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(C.TRAIN / "lora_sft"))
    ap.add_argument("--merged", default="")
    ap.add_argument("--out", default=str(C.TRAIN / "smoke" / "loop.json"))
    ap.add_argument("--n-each", type=int, default=10)
    ap.add_argument(
        "--also-penalty",
        action="store_true",
        help="Also run the repetition penalty. Off by default: it invented near-names in train4.",
    )
    a = ap.parse_args()
    rows = pick(a.n_each)
    if len(rows) < 16:
        raise SystemExit(f"smoke pool too small: {len(rows)}")
    print("smoke items", len(rows), flush=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    C.unload_ollama()
    base_name = a.merged or C.BASE
    tok = AutoTokenizer.from_pretrained(base_name, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    if not a.merged:
        model = PeftModel.from_pretrained(model, a.adapter, adapter_name="voice")
        model.base_model.set_adapter(["voice"], inference_mode=True)
    model.eval()

    def gen(sysm, user, penalty: bool):
        ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": user}], True)
        p = {"input_ids": torch.tensor([ids], device="cuda")}
        p["attention_mask"] = torch.ones_like(p["input_ids"])
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)
        t0 = time.time()
        with torch.inference_mode():
            o = model.generate(**p, **C.generate_kwargs(penalty=penalty, pad_token_id=tok.pad_token_id))
        torch.cuda.synchronize()
        ans = tok.decode(o[0][p["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        return ans, int(o.shape[1] - p["input_ids"].shape[1]), time.time() - t0

    # Cache RAG prompts so the two penalty passes see the same context.
    sys_for = {}
    for r in rows:
        sys_for[(r["char"], r["id"])] = C.rag_sysprompt(r["char"], r["probe"], r["reader_pos"], backend="bge")

    results = []
    summary = {}
    passes = (True, False) if a.also_penalty else (False,)
    for penalty in passes:
        loops = 0
        near_hits = []
        cottage_n = 0
        cottage_third = 0
        for r in rows:
            ans, nt, sec = gen(sys_for[(r["char"], r["id"])], r["probe"], penalty)
            loop = is_loop(ans)
            loops += int(loop)
            near = names.near_name_tokens(ans)
            third = bool(THIRD_NARR.search(ans))
            if r["char"] == "creature" and int(r["reader_pos"]) >= 15:
                cottage_n += 1
                cottage_third += int(third or not re.search(r"\b(I|me|my|myself)\b", ans))
            if near:
                near_hits.append(dict(id=r["id"], char=r["char"], near=near, answer=ans[:240]))
            row = dict(
                r,
                penalty=penalty,
                answer=ans,
                toks=nt,
                sec=round(sec, 3),
                loop=loop,
                near=near,
                third_person=third,
            )
            results.append(row)
            print(
                "smoke",
                "pen" if penalty else "raw",
                r["id"],
                r["char"],
                "LOOP" if loop else "",
                "NEAR" if near else "",
                "THIRD" if third else "",
                ans[:90].replace("\n", " "),
                flush=True,
            )
        rate = loops / max(len(rows), 1)
        name_notes = []
        if not penalty:
            by_id = {}
            for row in results:
                if row.get("penalty"):
                    continue
                by_id.setdefault((row["id"], row["char"]), row["answer"])
            for qid, label, good, bad in NAME_CHECKS:
                for char in ("victor", "creature"):
                    ans = by_id.get((qid, char))
                    if not ans:
                        continue
                    hit = bool(good.search(ans))
                    bad_hit = bool(bad.search(ans)) if bad else False
                    name_notes.append(dict(id=qid, char=char, label=label, canonical=hit, bad=bad_hit))
                    print(
                        "NAME",
                        qid,
                        char,
                        label,
                        "ok" if hit else "miss",
                        "BAD" if bad_hit else "",
                        flush=True,
                    )
        key = "with_penalty" if penalty else "without_penalty"
        summary[key] = dict(
            n=len(rows),
            loops=loops,
            rate=round(rate, 4),
            near=len(near_hits),
            near_hits=near_hits,
            cottage_creature_n=cottage_n,
            cottage_third_or_no_first=cottage_third,
            name_checks=name_notes,
        )
        print(
            "SMOKE",
            "penalty" if penalty else "no-penalty",
            "loops",
            loops,
            "/",
            len(rows),
            "near",
            len(near_hits),
            "cottage_third",
            cottage_third,
            "/",
            cottage_n,
            flush=True,
        )

    # Train5 abort is the no-penalty loop rate. The penalty stays off.
    abort = summary["without_penalty"]["rate"] > 0.15
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(
        meta=dict(
            adapter=a.adapter,
            merged=a.merged,
            n=len(rows),
            abort_loops=abort,
            eos_token_id=[151645, 151643],
            note="repetition_penalty is a safety net, not the fix",
        ),
        summary=summary,
        results=results,
    )
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    flag = C.TRAIN / "smoke" / "ABORT_LOOPS"
    if abort:
        flag.write_text(json.dumps(summary["with_penalty"], indent=1), encoding="utf-8")
        print("ABORT loops with penalty", summary["with_penalty"], flush=True)
    elif flag.exists():
        flag.unlink()
    print("wrote", out, "abort", abort, flush=True)


if __name__ == "__main__":
    main()
