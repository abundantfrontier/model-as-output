"""Perspective registry.

This build offers Wren only. To add another perspective later, append an
enabled entry. The page and the API list whatever ``enabled()`` returns.

Fields:
  id             stable key stored in sessions and packs
  label          text shown on the page
  card           key in the locked character cards (common.CARDS)
  enabled        offered when true
  system_prompt  slot. None uses the locked card. A string replaces only the
                 card line; style, guard, day, and the chapter lock stay.
"""
from __future__ import annotations

PERSPECTIVES = {
    "wren": {
        "id": "wren",
        "label": "Wren",
        "card": "wren",
        "enabled": True,
        "system_prompt": None,
    },
}


def get(perspective_id: str) -> dict:
    row = PERSPECTIVES.get(perspective_id or "")
    if not row or not row.get("enabled"):
        raise KeyError(perspective_id or "")
    return row


def enabled() -> list[dict]:
    return [
        {"id": row["id"], "label": row["label"]}
        for row in PERSPECTIVES.values()
        if row.get("enabled")
    ]
