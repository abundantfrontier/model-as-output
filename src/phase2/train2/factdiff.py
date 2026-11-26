#!/usr/bin/env python3
"""Fact-preservation check. qwen3:14b, temp 0, seed 1. Adapted from exp9/factdiff.py."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import prompts  # noqa: E402

JUDGE = "qwen3:14b"


def parse_lists(text):
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    j = json.loads(raw)
    if not isinstance(j, dict):
        raise ValueError("not an object")

    def lst(key):
        v = j.get(key, [])
        if v is None:
            return []
        if isinstance(v, str):
            v = [v] if v.strip() else []
        if not isinstance(v, list):
            raise ValueError(key)
        out = []
        for x in v:
            if isinstance(x, dict):
                s = " ".join(str(t) for t in x.values()).strip()
            else:
                s = str(x).strip()
            if s:
                out.append(s)
        return out

    return lst("dropped"), lst("added")


def one(question, pass1, pass2):
    prompt = prompts.fact_diff_prompt(question, pass1, pass2)
    text, *_ = C.ollama_chat(
        JUDGE, [{"role": "user", "content": prompt}], fmt="json", temp=0.0, seed=1, num_predict=512
    )
    try:
        dropped, added = parse_lists(text)
        return dropped, added, True, text
    except Exception as e:
        return [], [], False, f"{type(e).__name__}: {e}; raw={text[:400]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    doc = json.load(open(a.run))
    items = []
    for i, r in enumerate(doc["results"]):
        p1 = r.get("pass1") or ""
        p2 = r.get("pass2") or r.get("answer") or ""
        if r.get("locked"):
            items.append(
                dict(
                    index=i,
                    id=r.get("id"),
                    char=r.get("char"),
                    locked=True,
                    dropped=[],
                    added=[],
                    n_dropped=0,
                    n_added=0,
                    new_markers=[],
                    parse_ok=True,
                    fallback=False,
                )
            )
            continue
        dropped, added, ok, raw = one(r.get("probe"), p1, p2)
        p1_fail = set(C.fail_hits_probe(p1, {"fail_if_mentions": r.get("fail_if") or []}))
        p2_fail = set(C.fail_hits_probe(p2, {"fail_if_mentions": r.get("fail_if") or []}))
        new_m = sorted(p2_fail - p1_fail)
        fallback = (not ok) or bool(dropped) or bool(added) or bool(new_m)
        items.append(
            dict(
                index=i,
                id=r.get("id"),
                char=r.get("char"),
                locked=False,
                dropped=dropped,
                added=added,
                n_dropped=len(dropped),
                n_added=len(added),
                new_markers=new_m,
                parse_ok=ok,
                fallback=fallback,
                raw=raw[:800],
            )
        )
        print(
            "factdiff",
            r.get("id"),
            r.get("char"),
            "drop",
            len(dropped),
            "add",
            len(added),
            "fb",
            fallback,
            flush=True,
        )
    n = len(items)
    meta = dict(
        run=os.path.abspath(a.run) if False else str(Path(a.run).resolve()),
        n=n,
        drop_claims=sum(it["n_dropped"] for it in items),
        add_claims=sum(it["n_added"] for it in items),
        drop_samples=sum(it["n_dropped"] > 0 for it in items),
        add_samples=sum(it["n_added"] > 0 for it in items),
        fallback=sum(it["fallback"] for it in items),
        fallback_rule="any dropped claim, any added claim, unparseable, or new leak marker vs pass-1",
    )
    out = a.out or str(Path(a.run).with_suffix("")) + ".factdiff.json"
    if out.endswith(".json.json"):
        out = str(Path(a.run).with_name(Path(a.run).stem + ".factdiff.json"))
    # normalize: runs/foo.json -> runs/foo.factdiff.json
    if not a.out:
        p = Path(a.run)
        out = str(p.with_name(p.stem + ".factdiff.json"))
    json.dump(dict(meta=meta, items=items), open(out, "w"), indent=1)
    print("wrote", out, json.dumps(meta), flush=True)


if __name__ == "__main__":
    import os

    main()
