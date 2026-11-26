"""Smoke record for two chapter-3 Wren questions, one of them a spoiler probe.

Default: read the sample pack and write logs/smoke.json. No GPU.

  python smoke.py --live

runs just the smoke questions under ~/gpu.lock. Use this when the pack has
no answers yet. Pack generation already writes logs/smoke.json.
"""
from __future__ import annotations

import argparse
import sys

import gen_pack
import gpu_lock
import packstore
from paths import PACK_PATH, SEED


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--wait", type=float, default=120.0)
    args = parser.parse_args()
    spec = gen_pack.load_questions()
    if not args.live:
        pack = packstore.load_pack(PACK_PATH)
        record = gen_pack.smoke_record(
            pack, spec,
            "Read from the sample pack. This command did not take the GPU lock.",
        )
        gen_pack.write_smoke(record)
        print("smoke_ok", record["ok"], "questions", len(record["questions"]), flush=True)
        return 0 if record["ok"] else 1

    want = set(spec.get("smoke_ids") or [])
    chapter = int(spec.get("chapter") or 3)
    perspective = spec.get("perspective") or "wren"
    items = []
    for row in spec["questions"]:
        if row["id"] not in want:
            continue
        items.append({
            "id": row["id"],
            "chapter": int(row.get("chapter") or chapter),
            "perspective": row.get("perspective") or perspective,
            "question": row["question"],
            "spoiler_probe": bool(row.get("spoiler_probe")),
        })
    try:
        with gpu_lock.GpuLock("reader smoke", wait_s=args.wait) as holder:
            import pipeline
            results = pipeline.run_batch(items, seed=SEED, holder=holder)
    except gpu_lock.GpuBusy as exc:
        print(f"GPU lock busy, smoke skipped: {exc}", flush=True)
        return 3
    pack = {"items": results}
    record = gen_pack.smoke_record(pack, spec, "Live smoke run under ~/gpu.lock. Models unloaded and the lock released.")
    gen_pack.write_smoke(record)
    print("smoke_ok", record["ok"], flush=True)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
