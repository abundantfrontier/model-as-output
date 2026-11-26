"""Per-reader session files. History is stored separately for each chapter."""
from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path

from paths import SESSIONS

_LOCK = threading.Lock()
SID_RE = re.compile(r"[a-f0-9]{8,32}")


def valid_sid(sid: str) -> bool:
    return bool(SID_RE.fullmatch(sid or ""))


def _path(sid: str) -> Path:
    if not valid_sid(sid):
        raise ValueError("bad sid")
    return SESSIONS / f"{sid}.json"


def _blank(sid: str, chapter: int) -> dict:
    return {
        "sid": sid,
        "chapter": chapter,
        "perspective": "wren",
        "by_chapter": {},
    }


def create(chapter: int = 3) -> dict:
    SESSIONS.mkdir(parents=True, exist_ok=True)
    sid = uuid.uuid4().hex[:16]
    row = _blank(sid, chapter)
    save(row)
    return row


def load(sid: str) -> dict:
    path = _path(sid)
    if not path.is_file():
        raise FileNotFoundError(sid)
    return json.loads(path.read_text(encoding="utf-8"))


def save(row: dict) -> None:
    SESSIONS.mkdir(parents=True, exist_ok=True)
    path = _path(row["sid"])
    tmp = path.with_suffix(".json.tmp")
    text = json.dumps(row, indent=1, ensure_ascii=False) + "\n"
    with _LOCK:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)


def history(row: dict, chapter: int) -> list[dict]:
    return list((row.get("by_chapter") or {}).get(str(chapter)) or [])


def append_turn(row: dict, chapter: int, user_text: str, answer: dict) -> None:
    bucket = row.setdefault("by_chapter", {})
    turns = list(bucket.get(str(chapter)) or [])
    turns.append({"role": "user", "content": user_text})
    turns.append({
        "role": "assistant",
        "content": answer.get("reply") or "",
        "citations": answer.get("citations") or [],
        "guard_fallback": answer.get("guard_fallback"),
    })
    bucket[str(chapter)] = turns[-16:]
    row["chapter"] = chapter


def clear_chapter(row: dict, chapter: int) -> None:
    row.setdefault("by_chapter", {}).pop(str(chapter), None)
