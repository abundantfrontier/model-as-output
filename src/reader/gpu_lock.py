"""Shared 3090 lock. See ~/GPU_LOCK.md.

Line format: ``output-model <pid> <iso time> <what>``
A lock whose PID is dead is stale and may be replaced. A live PID is never killed.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from paths import GPU_LOCK

PT = ZoneInfo("America/Los_Angeles")


class GpuBusy(RuntimeError):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class GpuLost(RuntimeError):
    pass


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_lock(path: Path | None = None) -> str:
    lock = path or GPU_LOCK
    try:
        return lock.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def parse_lock(text: str) -> tuple[str, int, str] | None:
    parts = (text or "").split()
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1]), " ".join(parts[2:])


def lock_status(path: Path | None = None) -> dict:
    """Report the lock without taking it."""
    text = read_lock(path)
    if not text:
        return {"held": False, "stale": False, "text": ""}
    parsed = parse_lock(text)
    if not parsed:
        return {"held": True, "stale": False, "text": text, "parseable": False}
    project, pid, rest = parsed
    alive = pid_alive(pid)
    return {
        "held": alive,
        "stale": not alive,
        "text": text,
        "project": project,
        "pid": pid,
        "rest": rest,
        "parseable": True,
    }


def _write_new(path: Path, pid: int, what: str) -> None:
    note = " ".join(what.split())
    now = datetime.now(tz=PT).isoformat()
    line = f"output-model {pid} {now} {note}\n"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def release(path: Path | None = None, owner_pid: int | None = None) -> bool:
    """Delete the lock only when this project and this PID still own it."""
    lock = path or GPU_LOCK
    text = read_lock(lock)
    parsed = parse_lock(text)
    if not parsed:
        return False
    project, pid, _rest = parsed
    if project != "output-model":
        return False
    if owner_pid is not None and pid != int(owner_pid):
        return False
    try:
        lock.unlink()
    except FileNotFoundError:
        return False
    return True


class GpuLock:
    """Hold ~/gpu.lock for one short GPU job. Use as a context manager."""

    def __init__(self, what: str, *, path: Path | None = None, wait_s: float | None = None, poll_s: float = 2.0, pid: int | None = None):
        self.what = what
        self.path = path or GPU_LOCK
        self.wait_s = 180.0 if wait_s is None else float(wait_s)
        self.poll_s = poll_s
        self.pid = os.getpid() if pid is None else int(pid)
        self.held = False

    def _stale(self) -> bool:
        status = lock_status(self.path)
        return bool(status.get("stale"))

    def acquire(self) -> "GpuLock":
        deadline = time.time() + self.wait_s
        while True:
            try:
                _write_new(self.path, self.pid, self.what)
            except FileExistsError:
                if self._stale():
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.time() >= deadline:
                    text = read_lock(self.path) or "(unreadable)"
                    raise GpuBusy(f"GPU lock is held: {text[:240]}")
                time.sleep(self.poll_s)
                continue
            self.held = True
            return self

    def still_ours(self) -> bool:
        parsed = parse_lock(read_lock(self.path))
        if not parsed:
            return False
        project, pid, _rest = parsed
        return project == "output-model" and pid == self.pid

    def require(self) -> None:
        if not self.still_ours():
            raise GpuLost("gpu.lock is no longer ours; stopping before more GPU work")

    def release(self) -> bool:
        if not self.held:
            return False
        ok = release(self.path, self.pid)
        self.held = False
        return ok

    def __enter__(self) -> "GpuLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
