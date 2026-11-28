#!/usr/bin/env python3
"""Average two same-shaped LoRA adapters. CPU only. Variant B lever.

w is the weight on the second adapter: out = (1-w)*first + w*second.
Copies the first adapter's config. Does not load the base model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="First adapter directory (weight 1-w)")
    ap.add_argument("--b", required=True, help="Second adapter directory (weight w)")
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not 0.0 < args.w < 1.0:
        raise SystemExit("--w must be between 0 and 1, exclusive")
    a = Path(args.a)
    b = Path(args.b)
    out = Path(args.out)
    wa = load_file(a / "adapter_model.safetensors")
    wb = load_file(b / "adapter_model.safetensors")
    if set(wa) != set(wb):
        raise SystemExit(f"adapter keys differ: {len(wa)} vs {len(wb)}")
    mixed = {}
    for k in wa:
        if wa[k].shape != wb[k].shape:
            raise SystemExit(f"shape mismatch {k}")
        mixed[k] = ((1.0 - args.w) * wa[k].float() + args.w * wb[k].float()).to(wa[k].dtype)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    save_file(mixed, out / "adapter_model.safetensors")
    cfg = json.loads((a / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["inference_mode"] = True
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    note = dict(a=str(a.resolve()), b=str(b.resolve()), w_on_b=args.w, n_tensors=len(mixed))
    (out / "average_report.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    # Keep a tiny README so the directory is obviously not a raw train.
    (out / "README.md").write_text(
        f"Weight average. (1-{args.w}) * {a.name} + {args.w} * {b.name}.\n",
        encoding="utf-8",
    )
    print(json.dumps(note), flush=True)


if __name__ == "__main__":
    main()
