#!/usr/bin/env python3
"""Offline checks for the chapter lock. No model and no GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXP10 = Path("$HOME/output-model/exp10")
sys.path.insert(0, str(EXP10))
sys.path.insert(0, str(ROOT / "scripts"))

import common as C  # noqa: E402
from bm25 import search  # noqa: E402


def load_chapter(n: int) -> dict:
    return json.loads((DIST / "index" / f"ch{n}.json").read_text(encoding="utf-8"))


def main() -> None:
    errors = []
    manifest = json.loads((DIST / "index" / "manifest.json").read_text(encoding="utf-8"))
    nums = [row["n"] for row in manifest["chapters"]]
    if nums != list(range(1, max(nums) + 1)):
        errors.append(f"chapter list is not 1..N: {nums}")
    by_n = {}
    for n in nums:
        doc = load_chapter(n)
        by_n[n] = doc
        if doc["chapter"] != n:
            errors.append(f"ch{n}.json says chapter {doc['chapter']}")
        for chunk in doc["chunks"]:
            if chunk["chapter"] != n:
                errors.append(f"{chunk['id']} is tagged chapter {chunk['chapter']}")
            if sum(chunk["tf"].values()) != chunk["n"]:
                errors.append(f"{chunk['id']} token count does not match tf")
        prefix = json.loads((DIST / "prompt" / f"wren-{n}.json").read_text(encoding="utf-8"))["prefix"]
        hits = C.fail_hits(prefix)
        if hits:
            errors.append(f"wren-{n} prefix has leak markers {hits}")
        if "Story so far" in prefix:
            errors.append(f"wren-{n} prefix already contains the excerpt block")
        facts_now, bel_now, _ = C.knowledge("wren", n)
        if n < max(nums):
            facts_later, bel_later, _ = C.knowledge("wren", max(nums))
            now = set(facts_now + bel_now)
            for fact in facts_later + bel_later:
                if fact in now:
                    continue
                if fact and fact in prefix:
                    errors.append(f"wren-{n} prefix contains a later fact: {fact[:80]}")

    allowed = []
    for n in range(1, 4):
        allowed.extend(by_n[n]["chunks"])
    expect = json.loads((DIST / "index" / "selftest.json").read_text(encoding="utf-8"))
    got = [row["id"] for row in search(allowed, expect["query"], expect["k"])]
    if got != expect["ids"]:
        errors.append(f"selftest ids {got} != {expect['ids']}")
    for cid in got:
        chapter = int(cid.split("-")[0][2:])
        if chapter > 3:
            errors.append(f"chapter 3 search returned {cid}")

    # A word that exists only in a later chapter must not be in the early files.
    early_tokens = set()
    for n in range(1, 4):
        for chunk in by_n[n]["chunks"]:
            early_tokens.update(chunk["tf"])
    later_only = None
    for n in range(4, max(nums) + 1):
        for chunk in by_n[n]["chunks"]:
            for tok in chunk["tf"]:
                if tok not in early_tokens and len(tok) > 5:
                    later_only = tok
                    break
            if later_only:
                break
        if later_only:
            break
    if not later_only:
        errors.append("could not find a later-chapter-only token to test against")
    else:
        for n in range(1, 4):
            blob = (DIST / "index" / f"ch{n}.json").read_text(encoding="utf-8").lower()
            if f'"{later_only}"' in blob:
                errors.append(f"ch{n}.json contains later-only token {later_only}")

    if errors:
        print("FAIL")
        for err in errors:
            print("-", err)
        raise SystemExit(1)
    print(f"PASS chapters {nums[0]}..{nums[-1]} lock checks ({len(allowed)} chunks through chapter 3)")


if __name__ == "__main__":
    main()
