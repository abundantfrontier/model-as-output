"""Locked prompts, plus the system-prompt slot.

Pass 1 matches common.rag_sysprompt: card, style, guard, day, knowledge, beliefs,
then the retrieved excerpts. Pass 2 matches prompts.system_for: card only, no
retrieval. A perspective may set system_prompt to replace the card line.
"""
from __future__ import annotations

from paths import ensure_locked_import

ensure_locked_import()
import common as C  # noqa: E402
import prompts as locked_prompts  # noqa: E402


def card_head(card: str, chapter: int, system_prompt: str | None) -> str:
    if system_prompt:
        return (
            system_prompt.rstrip()
            + "\n"
            + C.STYLE
            + "\n"
            + C.GUARD
            + f"\nYou are on day {chapter} of the voyage."
        )
    return C.sysprompt(card, chapter)


def assemble_pass1(card: str, chapter: int, excerpt_block: str, system_prompt: str | None = None) -> str:
    """Same shape as common.rag_sysprompt, with an already-built excerpt block."""
    head = card_head(card, chapter, system_prompt)
    facts, bel, ref = C.knowledge(card, chapter)
    tail = (
        "\n\nWhat you know as true:\n- "
        + "\n- ".join(facts)
        + "\n\nWhat you believe (act on these):\n- "
        + "\n- ".join(bel)
    )
    if ref:
        tail += "\n\nYou must always refuse to: " + "; ".join(ref)
    return head + tail + "\n\nStory so far (excerpts):\n" + excerpt_block


def pass2_system(card: str, chapter: int, system_prompt: str | None = None) -> str:
    """Rewrite system prompt. No retrieved passages."""
    if system_prompt:
        return card_head(card, chapter, system_prompt)
    return locked_prompts.system_for(card, chapter)


def pass2_user(question: str, pass1_answer: str) -> str:
    return locked_prompts.rewrite_user(question, pass1_answer)
