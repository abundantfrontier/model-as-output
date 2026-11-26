"""Debug log for Wren chats. One JSON object per line under reader/logs/."""
from __future__ import annotations

import json
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from paths import LOGS

_LOCK = threading.Lock()
PT = ZoneInfo("America/Los_Angeles")


def write(kind: str, record: dict, name: str = "chat.jsonl") -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    row = dict(record)
    row["kind"] = kind
    row["ts"] = datetime.now(tz=PT).isoformat()
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _LOCK:
        with (LOGS / name).open("a", encoding="utf-8") as handle:
            handle.write(line)
