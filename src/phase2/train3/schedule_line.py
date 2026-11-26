#!/usr/bin/env python3
"""Update ONLY the output-model train3 line in ~/GPU_SCHEDULE.md."""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

PATH = Path.home() / "GPU_SCHEDULE.md"
NEEDLE = "phase2-frankenstein/train3"


def now_pst() -> str:
    tz = datetime.timezone(datetime.timedelta(hours=-8))
    return datetime.datetime.now(tz).strftime("%a %b %d %-I:%M %p PST")


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
    if a.state == "RUNNING":
        new = (
            "- output-model (Phase 2 Frankenstein train3, constrained voice rewrite, ~/output-model/phase2-frankenstein/train3): "
            f"RUNNING {now_pst()}. Constrained voice rewrite. "
            f"Hard stop 4:22 PM PST.{extra}\n"
        )
    elif a.state == "WAITING":
        new = (
            "- output-model (Phase 2 Frankenstein train3, constrained voice rewrite, ~/output-model/phase2-frankenstein/train3): "
            f"WAITING {now_pst()}.{extra}\n"
        )
    elif a.state == "DONE":
        new = (
            "- output-model (Phase 2 Frankenstein train3, constrained voice rewrite, ~/output-model/phase2-frankenstein/train3): "
            f"DONE {now_pst()}. Lock released.{extra}\n"
        )
    else:
        new = (
            "- output-model (Phase 2 Frankenstein train3, constrained voice rewrite, ~/output-model/phase2-frankenstein/train3): "
            f"ABORTED {now_pst()}. Lock released.{extra}\n"
        )
    for line in lines:
        if NEEDLE in line and line.lstrip().startswith("- output-model"):
            out.append(new if line.endswith("\n") else new.rstrip("\n"))
            found += 1
        else:
            out.append(line)
    if found != 1:
        raise SystemExit(f"expected 1 train3 line, found {found}")
    PATH.write_text("".join(out), encoding="utf-8")
    print("updated", a.state, new.strip())


if __name__ == "__main__":
    main()
