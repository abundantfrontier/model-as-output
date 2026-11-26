#!/usr/bin/env python3
"""Write RESULTS.md from judged run files."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
RUNS = TRAIN / "runs"


def load(name):
    p = RUNS / name
    if not p.is_file():
        p = RUNS / name.replace(".judged.json", ".json")
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def is_loop(text: str) -> bool:
    """Token loop, the 'I I I' failure. A short answer is not a loop."""
    words = (text or "").split()
    if "I I I" in (text or ""):
        return True
    if len(words) < 20:
        return False
    top = max(words.count(w) for w in set(words))
    return top >= 15 and top / len(words) >= 0.4


def score(doc, char=None, unlocked_only=True):
    rows = doc["results"]
    if char:
        rows = [r for r in rows if r["char"] == char]
    must = [r for r in rows if r["kind"] == "must" and (not unlocked_only or not r.get("locked"))]
    leak = [r for r in rows if r["kind"] == "leak" and (not unlocked_only or not r.get("locked"))]
    # item-level: an item (id, char) passes must if any/all samples? Phase 1 counted samples.
    must_ok = sum(1 for r in must if r.get("correct"))
    leak_items = defaultdict(list)
    for r in leak:
        leak_items[(r["id"], r["char"])].append(r)
    leak_marker = 0
    leak_judge = 0
    leak_n = len(leak_items)
    for rs in leak_items.values():
        if any(r.get("fail") for r in rs):
            leak_marker += 1
        if any(r.get("judge_leak") for r in rs):
            leak_judge += 1
    served = [r for r in rows if not r.get("locked")] if unlocked_only else rows
    voice = [r.get("voice_score") for r in served if r.get("voice_score") is not None]
    frames = sum(1 for r in served if (r.get("voice") or {}).get("frame"))
    loops = sum(1 for r in served if is_loop(r.get("answer") or ""))
    fallback = sum(1 for r in served if r.get("guard_fallback"))
    return dict(
        must_ok=must_ok,
        must_n=len(must),
        leak_marker=leak_marker,
        leak_judge=leak_judge,
        leak_n=leak_n,
        voice=round(statistics.mean(voice), 2) if voice else None,
        voice_n=len(voice),
        frames=frames,
        locked=sum(1 for r in rows if r.get("locked")),
        n=len(rows),
        unlocked_n=len(served) if unlocked_only else sum(1 for r in rows if not r.get("locked")),
        loops=loops,
        fallback=fallback,
    )


def examples(doc, char=None, k=3):
    rows = doc["results"]
    if char:
        rows = [r for r in rows if r["char"] == char]
    fails_must = [r for r in rows if r["kind"] == "must" and not r.get("locked") and not r.get("correct")]
    fails_leak = [r for r in rows if r["kind"] == "leak" and not r.get("locked") and r.get("fail")]
    frames = [r for r in rows if (r.get("voice") or {}).get("frame") and not r.get("locked")]
    loops = [r for r in rows if not r.get("locked") and is_loop(r.get("answer") or "")]
    out = []
    for r in fails_must[:k]:
        out.append(("must-miss", r["id"], r["char"], r["probe"], r["answer"][:280]))
    for r in fails_leak[:k]:
        out.append(("leak", r["id"], r["char"], r["probe"], r["answer"][:280]))
    for r in frames[:k]:
        out.append(("frame", r["id"], r["char"], r["probe"], r["answer"][:280]))
    for r in loops[:k]:
        out.append(("loop", r["id"], r["char"], r["probe"], r["answer"][:280]))
    return out


def fmt(s):
    if not s:
        return "—"
    judge = f", judge {s['leak_judge']}/{s['leak_n']}" if s["voice"] is not None or s["leak_judge"] else ""
    return (
        f"{s['must_ok']}/{s['must_n']} must · "
        f"{s['leak_marker']}/{s['leak_n']} leak-marker{judge} · "
        f"voice {s['voice']} (n={s['voice_n']}) · "
        f"{s['frames']} frames · "
        f"{s['loops']} loops"
    )


def main():
    conds = [
        ("rag_base", "RAG-only Qwen3-4B baseline (character card + chapter lock, no LoRA)"),
        ("p1_facts_rag", "Pass-1: facts LoRA + RAG"),
        ("p2_voice", "Two-pass unguarded (facts RAG, then voice-LoRA rewrite)"),
        ("p2_voice_guarded", "Two-pass guarded (fact-diff fallback to pass-1)"),
        ("merged_rag", "Single-pass merged facts+voice LoRA + RAG (browser-demo candidate)"),
    ]
    lines = [
        "# Frankenstein train1 results",
        "",
        "1818 text, Qwen3-4B, reader position varies per question. n=3 samples on unlocked turns.",
        "Must-answer uses a gist regex (Phase 1 style). Leak-marker is the probe's `fail_if_mentions`.",
        "Voice judge is local `qwen3:14b`, temp 0, seed 1. Frame-break regex ignores in-world “story”.",
        "Locked turns (character not yet on the page) use the knowledge-state lock line and are **not** in the unlocked counts.",
        "",
        "## Conditions (unlocked turns)",
        "",
        "| cond | what | Victor | Creature | both |",
        "|---|---|---|---|---|",
    ]
    scored = {}
    for name, desc in conds:
        doc = load(f"{name}.judged.json") or load(f"{name}.json")
        if not doc:
            lines.append(f"| {name} | {desc} | missing | missing | missing |")
            continue
        scored[name] = doc
        v = score(doc, "victor")
        c = score(doc, "creature")
        b = score(doc, None)
        lines.append(f"| `{name}` | {desc} | {fmt(v)} | {fmt(c)} | {fmt(b)} |")
    lines += [
        "",
        "## Including lock-line turns",
        "",
        "Before Letter IV, Victor is not a guest. Before Volume II Chapter II, the creature has not spoken.",
        "Those turns are the lock lines. They usually pass leak probes (they do not mention later events) and fail must-answer (they do not state the gist).",
        "",
    ]
    for name, _ in conds:
        doc = scored.get(name)
        if not doc:
            continue
        b = score(doc, None, unlocked_only=False)
        lines.append(f"- `{name}` all-rows: {fmt(b)}; lock-line rows {b['locked']}")
    lines += ["", "## Failures (examples)", ""]
    for name, _ in conds:
        doc = scored.get(name)
        if not doc:
            continue
        ex = examples(doc, k=4)
        lines.append(f"### {name}")
        lines.append("")
        if not ex:
            lines.append("No must-miss, marker-leak, or frame-break examples in the unlocked rows.")
            lines.append("")
            continue
        for kind, iid, char, probe, ans in ex:
            lines.append(f"- **{kind}** `{iid}`/{char}: {probe}")
            lines.append(f"  - {ans.replace(chr(10), ' ')}")
        lines.append("")
    # pick recommended from the text a reader would actually see
    lines += ["## Recommended model", ""]
    rec = None
    rec_why = []
    ranked = []
    for name, _ in conds:
        doc = scored.get(name)
        if not doc:
            continue
        s = score(doc)
        ranked.append((name, s))
        if s["fallback"]:
            rec_why.append(
                f"`{name}` fact-diff guard fell back to pass-1 on {s['fallback']} unlocked rows."
            )
        if s["loops"]:
            rec_why.append(
                f"`{name}` served {s['loops']} unlocked token-loop answers (the 'I I I' failure)."
            )

    def ship_key(s):
        # A loop on more than 15% of unlocked turns is not a model to hand a reader.
        if s["unlocked_n"] and s["loops"] / s["unlocked_n"] > 0.15:
            return None
        if not s["must_n"]:
            return None
        return (
            s["must_ok"] / s["must_n"],
            -(s["leak_marker"] / s["leak_n"]) if s["leak_n"] else 0,
            -(s["leak_judge"] / s["leak_n"]) if s["leak_n"] else 0,
            s["voice"] if s["voice"] is not None else -1,
            -s["frames"],
        )

    shippable = [(n, s, ship_key(s)) for n, s in ranked if ship_key(s) is not None]
    if shippable:
        shippable.sort(key=lambda t: t[2], reverse=True)
        rec = shippable[0][0]
        best = shippable[0][1]
        rec_why.insert(
            0,
            "Shippable rows are those where token-loops are at most 15% of unlocked answers. "
            "Among those, the higher must-answer sample rate wins; ties break toward fewer leak-marker items, "
            "fewer judge leaks, higher voice, then fewer frame breaks.",
        )
        bits = []
        for n, s, _k in shippable:
            bits.append(f"{n} {s['must_ok']}/{s['must_n']} must, {s['leak_marker']}/{s['leak_n']} leak-marker, voice {s['voice']}")
        rec_why.append("Order: " + "; ".join(bits) + ".")
        top_name = shippable[0][0]
        if rec != "rag_base":
            base = next((s for n, s in ranked if n == "rag_base"), None)
            voice_gain = 0.0
            if base and best["voice"] is not None and base["voice"] is not None:
                voice_gain = best["voice"] - base["voice"]
            if base and base["must_n"] and best["must_n"]:
                gain = best["must_ok"] / best["must_n"] - base["must_ok"] / base["must_n"]
                if gain < 0.03 and best["leak_marker"] >= base["leak_marker"] and voice_gain < 0.4:
                    rec = "rag_base"
                    rec_why.append(
                        f"`{top_name}` does not beat the RAG-only baseline by a clear margin on must-answer, "
                        "does not leak less, and is not clearly more in character, so the baseline is the model to use."
                    )
        if rec == top_name:
            rec_why.append(f"`{rec}` is the row to use.")
        else:
            rec_why.append(f"Highest trained or pipeline row was `{top_name}`; the baseline check selected `{rec}`.")
    elif ranked:
        rec = "none"
        rec_why.append("Every finished condition is over the token-loop cap or has no must-answer rows. Do not ship a voice rewrite from this block.")
    else:
        rec = "none"
        rec_why.append("No scored condition finished.")
    lines.append(f"**{rec}**")
    lines.append("")
    lines.extend(rec_why)
    lines += [
        "",
        "## Voice LoRA choice",
        "",
        "One voice LoRA with a character tag in the system prompt. Phase 1 did the same for Wren/Diane/YD-4408.",
        "Per-character adapters would have doubled train and eval time inside the 8h cap and would have needed a router",
        "for the merged browser candidate.",
        "",
        "## Files",
        "",
        "- Adapters: `lora_facts/`, `lora_voice/`"
        + (", `lora_voice_dpo/`" if (TRAIN / "lora_voice_dpo" / "adapter_config.json").is_file() else ""),
        "- Merged: `merged_facts/`, `merged_fv/`",
        "- Data: `data/HOW_MADE.md`",
        "- Index: `index/bge.npy` (pos kept); TF-IDF remains in `../index/`",
        "",
    ]
    (TRAIN / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", TRAIN / "RESULTS.md")
    print("recommended", rec)


if __name__ == "__main__":
    main()
