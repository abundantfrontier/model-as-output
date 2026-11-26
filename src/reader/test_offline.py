"""Offline checks. No GPU and no ~/gpu.lock."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

import gpu_lock
import packstore
import prompting
import retrieve
from paths import ensure_locked_import
from perspectives import enabled, get

ensure_locked_import()
import common as C  # noqa: E402
import factdiff  # noqa: E402
import prompts as locked_prompts  # noqa: E402

import pipeline  # noqa: E402


def fail(msg):
    raise SystemExit(msg)


def test_perspectives():
    rows = enabled()
    if [row["id"] for row in rows] != ["wren"]:
        fail(f"enabled perspectives {rows}")
    wren = get("wren")
    if wren["system_prompt"] is not None or wren["card"] != "wren":
        fail("wren slot should default to the locked card")
    try:
        get("not-a-perspective")
    except KeyError:
        pass
    else:
        fail("unregistered perspective was accepted")
    print("perspectives ok")


def test_prompts():
    block = "[day 1] Air is good."
    sys1 = prompting.assemble_pass1("wren", 3, block, None)
    if not sys1.endswith("Story so far (excerpts):\n" + block):
        fail("pass-1 excerpt block was rewritten")
    if "You are on day 3 of the voyage." not in sys1:
        fail("day line missing")
    facts, _bel, _ref = C.knowledge("wren", 3)
    if "\n- ".join(facts) not in sys1:
        fail("knowledge block drifted from common.knowledge")
    custom = prompting.assemble_pass1("wren", 2, block, "CUSTOM CARD")
    if not custom.startswith("CUSTOM CARD\n"):
        fail("system-prompt slot did not replace the card")
    if "Story so far (excerpts):\n" + block not in custom:
        fail("slot dropped the excerpts")
    sys2 = prompting.pass2_system("wren", 3, None)
    if sys2 != locked_prompts.system_for("wren", 3):
        fail("pass-2 system drifted from the locked card")
    if locked_prompts.RAG_MARK in sys2:
        fail("retrieval leaked into pass 2")
    user = prompting.pass2_user("What heading?", "Mars.")
    if not user.endswith(locked_prompts.REWRITE_INSTR):
        fail("rewrite instruction missing")
    locked_prompts.assert_no_retrieval(sys2, locked_prompts.REWRITE_INSTR)
    slotted = prompting.pass2_system("wren", 1, "CUSTOM CARD")
    if locked_prompts.RAG_MARK in slotted or "day 1" not in slotted:
        fail("slotted pass-2 prompt is wrong")
    early, _, _ = C.knowledge("wren", 1)
    if any("4408" in fact and "awake" in fact.lower() for fact in early):
        fail("chapter 2 fact visible at chapter 1")
    blob = "\n".join(C.knowledge("wren", 3)[0])
    import yaml
    ks = yaml.safe_load(open(Path(C.D) / "knowledge-state.yaml", encoding="utf-8"))
    for fact in ks["facts"]:
        if C.CH_NUM.get(fact.get("reader_from", "ch1"), 1) > 3 and fact["text"] in blob:
            fail(f"later fact leaked into chapter 3 knowledge: {fact['id']}")
    print("prompts ok")


def test_retrieve_matches_common():
    def fake_embed(texts, model="bge-m3", keep_alive="5m"):
        rows = []
        for text in texts:
            vec = np.zeros(16, dtype=np.float32)
            for i, ch in enumerate(text[:48]):
                vec[i % 16] += (ord(ch) % 11) + 0.05
            rows.append(vec)
        arr = np.array(rows, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms

    orig_c, orig_r = C.embed, retrieve.embed
    C.embed = fake_embed
    retrieve.embed = fake_embed
    try:
        C._IDX.clear()
        retrieve.clear_index()
        question = "What did Diane say about the air today?"
        block, passages = retrieve.retrieve_passages(question, 3, k=6)
        other = C.retrieve(question, upto=3, k=6)
        if block != other:
            fail("retrieve block does not match common.retrieve")
        allowed = " ".join(" ".join(C.chapter_text(i).split()) for i in range(1, 4))
        if not passages:
            fail("no passages")
        for passage in passages:
            if passage["chapter"] > 3:
                fail(f"chapter {passage['chapter']} leaked into a chapter-3 retrieve")
            snippet = passage["excerpt"].rstrip("…")[:80]
            if snippet not in allowed:
                fail(f"excerpt not in chapters 1-3: {snippet[:60]}")
        block1, only1 = retrieve.retrieve_passages("the hold", 1, k=6)
        if any(p["chapter"] != 1 for p in only1):
            fail("chapter 1 retrieve saw a later chapter")
        if "[day 1]" not in block1 or "[day 2]" in block1:
            fail(f"chapter 1 block is wrong: {block1[:80]}")
    finally:
        C.embed = orig_c
        retrieve.embed = orig_r
        C._IDX.clear()
        retrieve.clear_index()
    print("retrieve ok")


def test_guard():
    clean = pipeline.apply_guard("Q?", "Air is good.", "The air is good.", '{"dropped": [], "added": []}')
    if clean["guard_fallback"] or clean["reply"] != "The air is good.":
        fail(f"clean rewrite should be kept: {clean}")
    added = pipeline.apply_guard("Q?", "Air is good.", "Air is good. The ship cannot return.", '{"dropped": [], "added": ["cannot return"]}')
    if not added["guard_fallback"] or added["reply"] != "Air is good.":
        fail(f"added claim should fall back: {added}")
    dropped = pipeline.apply_guard("Q?", "Air is good.", "Okay.", '{"dropped": ["air is good"], "added": []}')
    if not dropped["guard_fallback"] or dropped["reply"] != "Air is good.":
        fail("dropped claim should fall back")
    bad = pipeline.apply_guard("Q?", "Air is good.", "Air is good.", "not json")
    if bad["parse_ok"] or not bad["guard_fallback"] or bad["reply"] != "Air is good.":
        fail("unparsed judge should fall back")
    marked = pipeline.apply_guard("Q?", "Air is good.", "Mara said the ship can't come home.", '{"dropped": [], "added": []}')
    if not marked["guard_fallback"] or marked["reply"] != "Air is good." or not marked["new_markers"]:
        fail(f"new fail marker should fall back: {marked}")
    same = pipeline.apply_guard("Q?", "No return fuel.", "No return fuel.", '{"dropped": [], "added": []}')
    if same["guard_fallback"] or same["reply"] != "No return fuel.":
        fail("a marker already in pass 1 is not a new marker")
    print("guard ok")


def test_pack():
    pack = {"items": [
        {
            "id": "air",
            "chapter": 3,
            "perspective": "wren",
            "question": "What did Diane say about the air today?",
            "reply": "Air is good.",
            "guard_fallback": False,
            "citations": [{"id": "ch3-c0", "chapter": 3, "excerpt": "Air is good."}],
        },
        {
            "id": "later",
            "chapter": 3,
            "perspective": "wren",
            "question": "What happens in chapter 5?",
            "reply": None,
            "citations": [],
        },
    ]}
    hit = packstore.lookup(pack, 3, "  what   DID diane say about the air today? ")
    if not hit["found"] or hit["reply"] != "Air is good." or hit["citations"][0]["id"] != "ch3-c0":
        fail(f"pack hit failed: {hit}")
    other_chapter = packstore.lookup(pack, 2, "What did Diane say about the air today?")
    if other_chapter["found"] or other_chapter["reply"] != packstore.MISS:
        fail("chapter tag was ignored")
    pending = packstore.lookup(pack, 3, "What happens in chapter 5?")
    if pending["found"] or pending["reply"] != packstore.PENDING:
        fail("pending question should not pretend to have an answer")
    miss = packstore.lookup(pack, 3, "Who filled the tank?")
    if miss["found"] or miss["reply"] != packstore.MISS:
        fail("missing question should say so")
    print("pack ok")


def test_gpu_lock():
    home = Path.home() / "gpu.lock"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gpu.lock"
        if path == home:
            fail("test lock path points at the real lock")
        dead = 2_147_000_000
        if gpu_lock.pid_alive(dead):
            fail("expected a dead pid")
        path.write_text(f"other-project {dead} 2026-11-20T00:00:00-08:00 stale\n", encoding="utf-8")
        lock = gpu_lock.GpuLock("reader offline test", path=path, wait_s=0, poll_s=0.01)
        lock.acquire()
        text = path.read_text(encoding="utf-8")
        if not text.startswith(f"output-model {os.getpid()} ") or "reader offline test" not in text:
            fail(f"stale lock was not replaced: {text}")
        if not lock.still_ours():
            fail("lock is not ours")
        lock.release()
        if path.exists():
            fail("release left the lock behind")

        path.write_text(f"other-project {os.getpid()} 2026-11-20T00:00:00-08:00 busy\n", encoding="utf-8")
        try:
            gpu_lock.GpuLock("reader offline test", path=path, wait_s=0, poll_s=0.01).acquire()
        except gpu_lock.GpuBusy:
            pass
        else:
            fail("stole a live lock")
        if not path.read_text(encoding="utf-8").startswith("other-project "):
            fail("live lock was overwritten")
        if gpu_lock.release(path, os.getpid()):
            fail("release deleted another project's lock")
        path.unlink()
    if home.exists():
        print("note: real ~/gpu.lock exists; this test did not touch it")
    print("gpu lock ok")


def test_chapters():
    rows = retrieve.chapter_catalog()
    if [row["n"] for row in rows] != [1, 2, 3, 4, 5]:
        fail(f"chapters {rows}")
    if "To List" not in rows[0]["title"] or "Green Light" not in rows[4]["title"]:
        fail(f"titles {rows}")
    print("chapters ok")


def main():
    test_perspectives()
    test_prompts()
    test_retrieve_matches_common()
    test_guard()
    test_pack()
    test_gpu_lock()
    test_chapters()
    print("OFFLINE_OK")


if __name__ == "__main__":
    main()
