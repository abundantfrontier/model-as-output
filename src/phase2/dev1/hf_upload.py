#!/usr/bin/env python3
"""Prepare (or dry-run) an upload of the Phase 1 Wren MLC browser build.

Does not prompt. Reads HF_TOKEN and ORG from the environment only.
Does not read tokens from files. Does not hard-code an org name.

Usage:
  HF_TOKEN=... ORG=... python3 hf_upload.py --dry-run
  HF_TOKEN=... ORG=... python3 hf_upload.py          # real upload; do not run in this block

Target repo: AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC
Source: ~/output-model/browser-demo/dist/models/wren-q4f16_1-MLC/

If MODEL_CARD.md is present in this directory, print it as the card to attach
later. This script does not invent or overwrite a card.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
SOURCE = Path.home() / "output-model" / "browser-demo" / "dist" / "models" / "wren-q4f16_1-MLC"
REPO_NAME = "wren-no-return-qwen3-4b-q4f16_1-MLC"
SKIP_SUFFIX = {".lock", ".pid"}


def fail(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def env_or_missing() -> tuple[str, str]:
    token = os.environ.get("HF_TOKEN")
    org = os.environ.get("ORG")
    missing = []
    if not token:
        missing.append("HF_TOKEN")
    if not org:
        missing.append("ORG")
    if missing:
        fail(
            "missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them in the environment and retry. "
            "ORG has no default. This script does not prompt and does not read tokens from files."
        )
    return token, org


def list_files(root: Path) -> list[Path]:
    if not root.is_dir():
        fail(f"source folder missing: {root}")
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for fn in filenames:
            if any(fn.endswith(s) for s in SKIP_SUFFIX):
                continue
            path = Path(dirpath) / fn
            if path.is_file() and not path.is_symlink():
                out.append(path)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print source, target repo, and file count, then exit without contacting the network.",
    )
    args = ap.parse_args()
    token, org = env_or_missing()
    if org != "AbundantFrontierInstitute":
        fail(
            "ORG must be AbundantFrontierInstitute. "
            "Refusing to create or upload to any other namespace, including a personal one."
        )
    repo_id = f"{org}/{REPO_NAME}"
    files = list_files(SOURCE)
    card = HERE / "MODEL_CARD.md"
    print(f"source: {SOURCE}")
    print(f"target: {repo_id}")
    print(f"files:  {len(files)}")
    if card.is_file():
        print(f"model card (upload later, not in this folder push): {card}")
    else:
        print("model card: none in this directory (do not invent one)")
    if args.dry_run:
        print("dry-run: no network, no upload")
        return
    # Real upload path. Not used in the Phase 2 CPU block.
    try:
        from huggingface_hub import HfApi
    except ImportError:
        fail("huggingface_hub is not installed in this Python. Use the exp3 venv or huggingface-cli.")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True, token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(SOURCE),
        repo_type="model",
        token=token,
        commit_message="Add Wren Qwen3-4B q4f16_1 MLC browser build",
    )
    print("uploaded", info)


if __name__ == "__main__":
    main()
