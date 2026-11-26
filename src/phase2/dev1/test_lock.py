#!/usr/bin/env python3
"""Offline checks for the Frankenstein chapter/character lock. No model, no GPU."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
DEV1 = HERE.parent
INDEX = DEV1 / "browser" / "index"
PROMPT = DEV1 / "browser" / "prompt"
sys.path.insert(0, str(HERE))
from bm25 import search  # noqa: E402

CHAT_FROM = {"victor": 4, "creature": 13}


def locked(char: str, upto: int) -> bool:
    return int(upto) < CHAT_FROM[char]


def load_unit(pos: int) -> dict:
    return json.loads((INDEX / f"u{pos}.json").read_text(encoding="utf-8"))


def chunks_through(upto: int, by_pos: dict) -> list[dict]:
    out = []
    for pos in range(0, upto + 1):
        out.extend(by_pos[pos]["chunks"])
    return out


def main() -> None:
    errors = []
    manifest = json.loads((INDEX / "manifest.json").read_text(encoding="utf-8"))
    selftest = json.loads((INDEX / "selftest.json").read_text(encoding="utf-8"))
    nums = [row["pos"] for row in manifest["units"]]
    if nums != list(range(0, max(nums) + 1)):
        errors.append(f"unit list is not 0..N: {nums}")
    if manifest.get("n_chunks") != 391:
        errors.append(f"manifest n_chunks {manifest.get('n_chunks')} != 391")
    if manifest.get("chat_from") != CHAT_FROM:
        errors.append(f"manifest chat_from {manifest.get('chat_from')}")

    by_pos = {}
    n_chunks = 0
    for pos in nums:
        doc = load_unit(pos)
        by_pos[pos] = doc
        if doc["pos"] != pos:
            errors.append(f"u{pos}.json says pos {doc['pos']}")
        for chunk in doc["chunks"]:
            n_chunks += 1
            if int(chunk["pos"]) != pos:
                errors.append(f"{chunk['id']} is tagged pos {chunk['pos']}")
            if sum(chunk["tf"].values()) != chunk["n"]:
                errors.append(f"{chunk['id']} token count does not match tf")
            if "pos" not in chunk:
                errors.append(f"{chunk['id']} missing pos")

    if n_chunks != 391:
        errors.append(f"loaded {n_chunks} chunks, expected 391")

    # Character lock fixture.
    for case in selftest["cases"]:
        got = locked(case["char"], case["upto"])
        if bool(got) != bool(case["expect_lock"]):
            errors.append(
                f"{case['id']}: locked={got} expected {case['expect_lock']} "
                f"(char={case['char']} upto={case['upto']})"
            )
        if case.get("expect_lock") and case.get("lock_line"):
            if case["lock_line"] != manifest["lock_lines"][case["char"]]:
                errors.append(f"{case['id']}: lock line mismatch")

    # Search lock: later-chapter chunks must not be returned.
    for spec in selftest["search"]:
        upto = int(spec["upto"])
        allowed = chunks_through(upto, by_pos)
        if any(int(c["pos"]) > upto for c in allowed):
            errors.append(f"{spec['id']}: corpus passed the pos lock")
        got = [row["id"] for row in search(allowed, spec["query"], spec["k"])]
        if got != spec["ids"]:
            errors.append(f"{spec['id']}: ids {got} != {spec['ids']}")
        for cid in got:
            # pos lives on the chunk, not necessarily in the id prefix
            hit = next(c for c in allowed if c["id"] == cid)
            if int(hit["pos"]) > upto:
                errors.append(f"{spec['id']} returned later chunk {cid}")
            if int(hit["pos"]) > int(spec["max_pos"]):
                errors.append(f"{spec['id']} returned {cid} pos {hit['pos']} > max_pos {spec['max_pos']}")

    # Direct probes required by the brief.
    v_hidden = chunks_through(3, by_pos)
    if any(int(c["pos"]) >= 4 for c in v_hidden):
        errors.append("upto=3 corpus includes pos>=4 (Victor unlock unit)")
    c_hidden = chunks_through(12, by_pos)
    if any(int(c["pos"]) >= 13 for c in c_hidden):
        errors.append("upto=12 corpus includes pos>=13 (creature unlock unit)")

    # A word that exists only in a later unit must not sit in early files.
    early_tokens = set()
    for pos in range(0, 4):
        for chunk in by_pos[pos]["chunks"]:
            early_tokens.update(chunk["tf"])
    later_only = None
    for pos in range(13, max(nums) + 1):
        for chunk in by_pos[pos]["chunks"]:
            for tok in chunk["tf"]:
                if tok not in early_tokens and len(tok) > 6:
                    later_only = tok
                    break
            if later_only:
                break
        if later_only:
            break
    if not later_only:
        errors.append("could not find a later-unit-only token to test against")
    else:
        for pos in range(0, 4):
            blob = (INDEX / f"u{pos}.json").read_text(encoding="utf-8").lower()
            if f'"{later_only}"' in blob:
                errors.append(f"u{pos}.json contains later-only token {later_only}")

    # Prompt prefixes must not include later knowledge-state facts.
    # Female-destroyed / ending are pos 23 / 27.
    late_markers = ["Orkney island", "funeral pile", "wedding night"]
    for pos in range(0, 8):
        for char in ("victor", "creature"):
            path = PROMPT / f"{char}-{pos}.json"
            if not path.is_file():
                errors.append(f"missing prompt {path.name}")
                continue
            prefix = json.loads(path.read_text(encoding="utf-8"))["prefix"]
            for marker in late_markers:
                if marker.lower() in prefix.lower():
                    errors.append(f"{path.name} contains later fact {marker!r}")

    if errors:
        print("FAIL")
        for err in errors:
            print("-", err)
        raise SystemExit(1)
    print(
        f"PASS units {nums[0]}..{nums[-1]} lock checks "
        f"({n_chunks} chunks; Victor hidden before pos 4; creature hidden before pos 13)"
    )


if __name__ == "__main__":
    main()
