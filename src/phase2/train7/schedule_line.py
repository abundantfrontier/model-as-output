#!/usr/bin/env python3
"""Update ONLY the output-model train7 line in ~/GPU_SCHEDULE.md.

NEEDLE is phase2-frankenstein/train7. A train7 copy used as-is would overwrite
the train7 history line. Hard-stop time is read from train7/hard_stop.txt.
"""
from __future__ import annotations

import argparse
import datetime
import re
from pathlib import Path

PATH = Path.home() / "GPU_SCHEDULE.md"
NEEDLE = "phase2-frankenstein/train7"
HERE = Path("$HOME/output-model/phase2-frankenstein/train7")
LOCK_LOG = HERE / "lock-holder.log"
STOPF = HERE / "hard_stop.txt"
PIDF = HERE / ".gpu-hold.pid"
LABEL = (
    "- output-model (Phase 2 Frankenstein train7, win must-answer back to ≥100/135 while keeping qwen judge leaks ≤14/22, "
    "starting from train6/merged_dpo path, ~/output-model/phase2-frankenstein/train7): "
)


def now_pst() -> str:
    tz = datetime.timezone(datetime.timedelta(hours=-8))
    return datetime.datetime.now(tz).strftime("%a %b %d %-I:%M %p PST")


def hard_stop() -> str:
    if not STOPF.is_file():
        raise SystemExit(f"missing {STOPF}; refusing to invent a hard-stop time")
    line = STOPF.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if not line:
        raise SystemExit(f"empty {STOPF}")
    return line


def lock_start() -> str:
    lock = Path.home() / "gpu.lock"
    if lock.is_file():
        m = re.search(r"output-model\s+\d+\s+(.+?)\s+train7\b", lock.read_text(encoding="utf-8"))
        if m:
            return m.group(1).strip()
    if LOCK_LOG.is_file():
        for line in LOCK_LOG.read_text(encoding="utf-8").splitlines():
            m = re.search(r"took gpu\.lock:\s+output-model\s+\d+\s+(.+?)\s+train7\b", line)
            if m:
                return m.group(1).strip()
    raise SystemExit("could not read the train7 lock start time")


def holder_pid() -> str:
    if PIDF.is_file():
        pid = PIDF.read_text(encoding="utf-8").strip()
        if pid:
            return pid
    lock = Path.home() / "gpu.lock"
    if lock.is_file():
        m = re.search(r"output-model\s+(\d+)\s+.+train7\b", lock.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    if LOCK_LOG.is_file():
        m = re.search(r"output-model\s+(\d+)\s+.+train7\b", LOCK_LOG.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    raise SystemExit("could not read the train7 holder pid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", choices=["RUNNING", "DONE", "ABORTED", "WAITING"])
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    text = PATH.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    found = 0
    out = []
    extra = f" {a.note}" if a.note else ""
    start = lock_start()
    pid = holder_pid()
    stop = hard_stop()
    if a.state == "RUNNING":
        new = (
            f"{LABEL}RUNNING {start}. Lock taken by hold_gpu_lock.sh PID {pid}. "
            f"Cap 3.0 GPU h. Expected finish / hard stop {stop}.{extra}\n"
        )
    elif a.state == "WAITING":
        new = f"{LABEL}WAITING {now_pst()}.{extra}\n"
    elif a.state == "DONE":
        new = (
            f"{LABEL}DONE {now_pst()}. Lock released. "
            f"Lock {start}–{now_pst()} (PID {pid}).{extra}\n"
        )
    else:
        new = (
            f"{LABEL}ABORTED {now_pst()}. Lock released. "
            f"Lock {start}–{now_pst()} (PID {pid}).{extra}\n"
        )
    for line in lines:
        if NEEDLE in line and line.lstrip().startswith("- output-model"):
            out.append(new if line.endswith("\n") else new.rstrip("\n"))
            found += 1
        else:
            out.append(line)
    if found != 1:
        raise SystemExit(f"expected 1 train7 line, found {found}")
    PATH.write_text("".join(out), encoding="utf-8")
    print("updated", a.state, new.strip())


if __name__ == "__main__":
    main()
