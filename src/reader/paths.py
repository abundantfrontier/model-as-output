"""Paths for the No Return reader.

The locked stack stays where it is. This module only points at it.
Override with environment variables when the same app is deployed elsewhere.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME = Path.home()
OUTPUT = Path(os.environ.get("READER_OUTPUT", HOME / "output-model"))
EXP10 = Path(os.environ.get("READER_EXP10", OUTPUT / "exp10"))
EXP6 = Path(os.environ.get("READER_EXP6", OUTPUT / "exp6"))
MERGED = Path(os.environ.get("READER_MERGED", EXP6 / "merged_q4_facts"))
PASS1_ADAPTER = Path(os.environ.get("READER_PASS1_ADAPTER", EXP6 / "lora_v_merged"))
PASS2_ADAPTER = Path(os.environ.get("READER_PASS2_ADAPTER", EXP10 / "init" / "lora_v_light_dpo"))
PACK_PATH = Path(os.environ.get("READER_PACK", ROOT / "packs" / "sample.json"))
QUESTIONS_PATH = Path(os.environ.get("READER_QUESTIONS", ROOT / "questions" / "sample.json"))
SESSIONS = Path(os.environ.get("READER_SESSIONS", ROOT / "sessions"))
LOGS = Path(os.environ.get("READER_LOGS", ROOT / "logs"))
STATIC = ROOT / "static"
GPU_LOCK = Path(os.environ.get("READER_GPU_LOCK", HOME / "gpu.lock"))

# Locked pass-2 adapter. frozen.P2_ADAPTER_SHA is the same value.
PASS2_SHA = os.environ.get(
    "READER_PASS2_SHA",
    "7980fad31a4a3023cc738662a8af6220d1c0722eeaf70d4b204f7368fd29dcef",
)

PIPE = "p2_light_dpo_guarded"
STORY = "No Return"
HOST = os.environ.get("READER_HOST", "127.0.0.1")
PORT = int(os.environ.get("READER_PORT", "8091"))
MODE = os.environ.get("READER_MODE", "live").strip().lower()
SEED = int(os.environ.get("READER_SEED", "1"))
LOCK_WAIT = float(os.environ.get("READER_LOCK_WAIT", "180"))


def ensure_locked_import() -> None:
    """Import exp10 modules (common, prompts, chat, factdiff) without copying them."""
    entry = str(EXP10)
    if entry not in sys.path:
        sys.path.insert(0, entry)
