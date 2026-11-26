"""Static-pack lookup. A miss never calls a model."""
from __future__ import annotations

import json
from pathlib import Path

MISS = "That question is not in this static pack. No model was called."
PENDING = "This question is listed in the pack, but no answer has been precomputed yet. No model was called."


def norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def load_pack(path: Path) -> dict:
    if not path.is_file():
        return {"story": "No Return", "items": [], "chapters": [], "missing_file": True}
    return json.loads(path.read_text(encoding="utf-8"))


def write_pack(path: Path, pack: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(pack, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def lookup(pack: dict, chapter: int, question: str, perspective: str = "wren") -> dict:
    want = norm(question)
    for item in pack.get("items") or []:
        if int(item.get("chapter", -1)) != int(chapter):
            continue
        if (item.get("perspective") or "wren") != perspective:
            continue
        if norm(item.get("question") or "") != want:
            continue
        if not item.get("reply"):
            return {
                "found": False,
                "pending": True,
                "reply": PENDING,
                "citations": [],
                "guard_fallback": None,
                "id": item.get("id"),
            }
        return {
            "found": True,
            "pending": False,
            "reply": item["reply"],
            "citations": item.get("citations") or [],
            "guard_fallback": bool(item.get("guard_fallback")),
            "id": item.get("id"),
            "spoiler_probe": bool(item.get("spoiler_probe")),
        }
    return {
        "found": False,
        "pending": False,
        "reply": MISS,
        "citations": [],
        "guard_fallback": None,
        "id": None,
    }


def suggestions(pack: dict) -> list[dict]:
    out = []
    for item in pack.get("items") or []:
        out.append({
            "id": item.get("id"),
            "chapter": item.get("chapter"),
            "perspective": item.get("perspective") or "wren",
            "question": item.get("question") or "",
            "spoiler_probe": bool(item.get("spoiler_probe")),
            "ready": bool(item.get("reply")),
        })
    return out
