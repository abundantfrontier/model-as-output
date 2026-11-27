#!/usr/bin/env python3
"""Merge LoRAs on CPU. Does not touch the 3090. Adapted from exp6/merge_facts.py and browser-demo/merge_voice.py."""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent


def merged_ok(out: Path) -> bool:
    return (out / "config.json").is_file() and (
        (out / "model.safetensors").is_file()
        or (out / "model.safetensors.index.json").is_file()
        or any(n.startswith("model-") and n.endswith(".safetensors") for n in os.listdir(out) if out.is_dir())
    )


def run(base, adapter, out, name):
    out = Path(out)
    if merged_ok(out):
        print("merge skip", out, flush=True)
        return
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; refusing to merge while the GPU might be in use")
    out.mkdir(parents=True, exist_ok=True)
    print("loading tokenizer", base, flush=True)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    print("loading base on CPU", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cpu", local_files_only=True
    )
    print("loading adapter", adapter, flush=True)
    model = PeftModel.from_pretrained(model, adapter, adapter_name=name)
    print("merging", flush=True)
    model = model.merge_and_unload()
    print("saving", out, flush=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.eos_token_id = [151645, 151643]
        gc.pad_token_id = 151643
        # Penalty stays off. The smoke test was 0/20 loops without it, and
        # repetition_penalty=1.15 invented Agrappa / Morritz / Cliversal.
        gc.temperature = 0.7
        gc.top_p = 0.95
        gc.do_sample = True
        gc.save_pretrained(out)
    report = dict(
        base=str(base),
        adapter=str(adapter),
        out=str(out.resolve()),
        dtype="bfloat16",
        n_params=sum(p.numel() for p in model.parameters()),
    )
    json.dump(report, open(out / "merge_report.json", "w"), indent=1)
    print("done", json.dumps(report), flush=True)
    if not merged_ok(out):
        raise RuntimeError(f"merge did not write weights under {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="default")
    a = ap.parse_args()
    out = Path(a.out) if os.path.isabs(a.out) else TRAIN / a.out
    try:
        run(a.base, a.adapter, out, a.name)
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        (out / "FAIL.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
