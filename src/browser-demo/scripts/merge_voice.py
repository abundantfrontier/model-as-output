#!/usr/bin/env python3
"""Fold the pass-1 voice LoRA into the facts-merged Qwen3-4B base, on CPU.

The browser ships one model. exp6/merged_q4_facts already has the facts
adapter baked in (the directory name says q4; the weights are bf16).
exp6/lora_v_merged is the voice adapter the locked pass 1 uses with RAG.

exp10/init/lora_v_light_dpo is the pass-2 rewrite. It was trained to
restyle a draft, and the fact-diff guard decides whether to keep it.
Folding that adapter into the same weights would not reproduce the
two-pass pipeline, so it is left where it is.

CUDA_VISIBLE_DEVICES is cleared before torch loads so this job cannot
allocate the 3090.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("WREN_BASE", "$HOME/output-model/exp6/merged_q4_facts"))
ADAPTER = Path(os.environ.get("WREN_ADAPTER", "$HOME/output-model/exp6/lora_v_merged"))
OUT = Path(os.environ.get("WREN_MERGED", ROOT / "work" / "merged-voice"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; refusing to merge while the GPU might be in use")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("loading tokenizer", BASE, flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    print("loading base on CPU", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    print("loading adapter", ADAPTER, flush=True)
    model = PeftModel.from_pretrained(model, str(ADAPTER), adapter_name="voice")
    print("merging", flush=True)
    model = model.merge_and_unload()
    model.config.tie_word_embeddings = True
    print("saving", OUT, flush=True)
    model.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    for name in ("chat_template.jinja", "generation_config.json"):
        src = BASE / name
        if src.is_file() and not (OUT / name).is_file():
            shutil.copy2(src, OUT / name)
    weight = OUT / "model.safetensors"
    report = {
        "base": str(BASE),
        "adapter": str(ADAPTER),
        "adapter_sha256": sha256_file(ADAPTER / "adapter_model.safetensors"),
        "out": str(OUT),
        "weight_bytes": weight.stat().st_size,
        "weight_sha256": sha256_file(weight),
        "dtype": "bfloat16",
        "device": "cpu",
        "seconds": round(time.time() - started, 1),
        "note": "pass-1 voice folded into the facts base; light DPO rewrite not merged",
    }
    (OUT / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
