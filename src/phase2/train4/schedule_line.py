#!/usr/bin/env python3
"""Update ONLY the output-model train4 line in ~/GPU_SCHEDULE.md.

NEEDLE is phase2-frankenstein/train4. A train3 copy used as-is would overwrite
the train3 history line. Hard-stop time is read from train4/hard_stop.txt.
"""
from __future__ import annotations

import argparse
import datetime
import re
from pathlib import Path

PATH = Path.home() / "GPU_SCHEDULE.md"
NEEDLE = "phase2-frankenstein/train4"
HERE = Path("$HOME/output-model/phase2-frankenstein/train4")
LOCK_LOG = HERE / "lock-holder.log"
STOPF = HERE / "hard_stop.txt"
PIDF = HERE / ".gpu-hold.pid"


def now_pst() -> str:
    tz = datetime.timezone(datetime.timedelta(hours=-8))
    return datetime.datetime.now(tz).strftime("%a %b %d %-I:%M %p PST")


def hard_stop() -> str:
    if STOPF.is_file():
        return STOPF.read_text(encoding="utf-8").strip().splitlines()[0]
    return "Thu Nov 26 06:28:48 PM PST"


def lock_start() -> str:
    if LOCK_LOG.is_file():
        for line in LOCK_LOG.read_text(encoding="utf-8").splitlines():
            m = re.search(r"took gpu\.lock:\s+output-model\s+\d+\s+(.+?)\s+train4", line)
            if m:
                return m.group(1).strip()
    return "Thu Nov 26 02:28:48 PM PST"


def holder_pid() -> str:
    if PIDF.is_file():
        return PIDF.read_text(encoding="utf-8").strip() or "2639642"
    if LOCK_LOG.is_file():
        m = re.search(r"output-model\s+(\d+)", LOCK_LOG.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    return "2639642"


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
            "- output-model (Phase 2 Frankenstein train4, single-pass distill of train3 guarded outputs, "
            "~/output-model/phase2-frankenstein/train4): "
            f"RUNNING {start}. Lock taken by hold_gpu_lock.sh PID {pid}. Cap 4.0 GPU h. "
            f"Expected finish / hard stop {stop}.{extra}\n"
        )
    elif a.state == "WAITING":
        new = (
            "- output-model (Phase 2 Frankenstein train4, single-pass distill of train3 guarded outputs, "
            "~/output-model/phase2-frankenstein/train4): "
            f"WAITING {now_pst()}.{extra}\n"
        )
    elif a.state == "DONE":
        new = (
            "- output-model (Phase 2 Frankenstein train4, single-pass distill of train3 guarded outputs, "
            "~/output-model/phase2-frankenstein/train4): "
            f"DONE {now_pst()}. Lock released. Lock {start}–{now_pst()} (PID {pid}).{extra}\n"
        )
    else:
        new = (
            "- output-model (Phase 2 Frankenstein train4, single-pass distill of train3 guarded outputs, "
            "~/output-model/phase2-frankenstein/train4): "
            f"ABORTED {now_pst()}. Lock released. Lock {start}–{now_pst()} (PID {pid}).{extra}\n"
        )
    for line in lines:
        if NEEDLE in line and line.lstrip().startswith("- output-model"):
            out.append(new if line.endswith("\n") else new.rstrip("\n"))
            found += 1
        else:
            out.append(line)
    if found != 1:
        raise SystemExit(f"expected 1 train4 line, found {found}")
    PATH.write_text("".join(out), encoding="utf-8")
    print("updated", a.state, new.strip())


if __name__ == "__main__":
    main()
