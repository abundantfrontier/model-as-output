"""Build a static pack of chapter-tagged Wren answers.

  python gen_pack.py --pending
      Write packs/sample.json with the question list and empty replies. No GPU.

  python gen_pack.py
      Hold ~/gpu.lock, run the locked pipeline, write the pack, release the lock.

A question that is not in the pack is never sent to a model when the server
is in pack mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import gpu_lock
import packstore
import retrieve
from paths import LOGS, PACK_PATH, PIPE, QUESTIONS_PATH, SEED, STORY

PT = ZoneInfo("America/Los_Angeles")


def load_questions(path=QUESTIONS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pending_pack(spec: dict) -> dict:
    chapter = int(spec.get("chapter") or 3)
    perspective = spec.get("perspective") or "wren"
    items = []
    for row in spec["questions"]:
        items.append({
            "id": row["id"],
            "chapter": int(row.get("chapter") or chapter),
            "perspective": row.get("perspective") or perspective,
            "question": row["question"],
            "spoiler_probe": bool(row.get("spoiler_probe")),
            "reply": None,
            "guard_fallback": None,
            "citations": [],
        })
    return {
        "story": STORY,
        "pipeline": PIPE,
        "generated": None,
        "status": "pending",
        "note": "Question list only. Replies are filled by gen_pack.py under the GPU lock. Pack mode does not call a model.",
        "chapters": retrieve.chapter_catalog(),
        "items": items,
    }


def smoke_record(pack: dict, spec: dict, note: str) -> dict:
    want = set(spec.get("smoke_ids") or [])
    rows = [item for item in pack.get("items") or [] if item.get("id") in want]
    ready = [item for item in rows if item.get("reply")]
    return {
        "story": STORY,
        "pipeline": PIPE,
        "when": datetime.now(tz=PT).isoformat(),
        "note": note,
        "ok": len(ready) == len(rows) and len(rows) > 0,
        "questions": rows,
    }


def write_smoke(record: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / "smoke.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", action="store_true", help="write the question list with empty replies")
    parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    parser.add_argument("--out", default=str(PACK_PATH))
    parser.add_argument("--wait", type=float, default=120.0, help="seconds to wait for ~/gpu.lock")
    args = parser.parse_args()
    spec = load_questions(__import__("pathlib").Path(args.questions))
    out = __import__("pathlib").Path(args.out)
    if args.pending:
        pack = pending_pack(spec)
        packstore.write_pack(out, pack)
        print("wrote pending pack", out, flush=True)
        return 0

    items = []
    chapter = int(spec.get("chapter") or 3)
    perspective = spec.get("perspective") or "wren"
    for row in spec["questions"]:
        items.append({
            "id": row["id"],
            "chapter": int(row.get("chapter") or chapter),
            "perspective": row.get("perspective") or perspective,
            "question": row["question"],
            "spoiler_probe": bool(row.get("spoiler_probe")),
        })
    try:
        with gpu_lock.GpuLock("reader pack generation", wait_s=args.wait) as holder:
            import pipeline
            results = pipeline.run_batch(items, seed=SEED, holder=holder)
    except gpu_lock.GpuBusy as exc:
        print(f"GPU lock busy, pack not generated: {exc}", flush=True)
        return 3
    pack = {
        "story": STORY,
        "pipeline": PIPE,
        "generated": datetime.now(tz=PT).isoformat(),
        "status": "ready",
        "seed": SEED,
        "note": "Precomputed Wren answers for static-pack mode. A question that is not listed here is not sent to a model.",
        "chapters": retrieve.chapter_catalog(),
        "items": results,
    }
    packstore.write_pack(out, pack)
    record = smoke_record(pack, spec, "Recorded from the pack generation run while ~/gpu.lock was held.")
    write_smoke(record)
    print("wrote", out, "items", len(results), "smoke_ok", record["ok"], flush=True)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
