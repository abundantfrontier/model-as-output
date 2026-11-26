"""Locked two-pass Wren answers.

Pass 1: exp6/merged_q4_facts + exp6/lora_v_merged + chapter-locked RAG.
Pass 2: the same base + exp10/init/lora_v_light_dpo, no retrieval.
Guard: fall back to pass 1 when the fact-diff judge adds or drops a claim,
when the judge reply does not parse, or when pass 2 introduces a fail-hits
marker that pass 1 did not.

The caller holds ~/gpu.lock. This module unloads both local adapters and asks
Ollama to drop bge-m3 and qwen3:14b before it returns.
"""
from __future__ import annotations

import gc
import hashlib
import os
import subprocess
import threading
import time

import requests

import gpu_lock
import logbook
import prompting
import retrieve
from paths import LOCK_WAIT, MERGED, PASS1_ADAPTER, PASS2_ADAPTER, PASS2_SHA, SEED, ensure_locked_import
from perspectives import get as perspective_get

ensure_locked_import()
import common as C  # noqa: E402
import factdiff  # noqa: E402
import prompts as locked_prompts  # noqa: E402
from chat import render  # noqa: E402

_PROC = threading.Lock()
_PASS2_CHECKED = False
JUDGE_MODEL = "qwen3:14b"
PARKED_MODELS = ("qwen3:14b", "qwen3:8b", "gemma3:12b")


def check_pass2() -> None:
    global _PASS2_CHECKED
    if _PASS2_CHECKED:
        return
    got = sha256_file(PASS2_ADAPTER / "adapter_model.safetensors")
    if got != PASS2_SHA:
        raise RuntimeError(f"pass-2 adapter sha {got} != {PASS2_SHA}")
    _PASS2_CHECKED = True


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ollama_up() -> None:
    try:
        response = requests.get(f"{C.OLL}/api/version", timeout=3)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"ollama is not reachable at {C.OLL}: {exc}") from exc


def ollama_stop(model: str) -> None:
    try:
        subprocess.run(["ollama", "stop", model], check=False, timeout=60, capture_output=True, text=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"ollama stop {model} failed: {exc}", flush=True)


def ollama_judge(prompt: str, keep_alive: str = "30m") -> str:
    """Fact-diff call. Same model, temperature, seed, and JSON format as exp10."""
    body = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "format": "json",
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 512, "seed": 1},
    }
    last = ""
    for attempt in range(4):
        try:
            payload = requests.post(f"{C.OLL}/api/chat", json=body, timeout=600).json()
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(2)
            continue
        if "message" in payload:
            return (payload["message"].get("content") or "").strip()
        last = str(payload)[:200]
        print("ollama fact-diff retry:", last, flush=True)
        time.sleep(2)
        body["options"]["temperature"] = 0.3
        body["options"]["seed"] = attempt + 11
    return last


def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_voice(adapter_dir):
    """One adapter on the merged base. Same load as exp10/eval_twopass.load_voice."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg_path = adapter_dir / "adapter_config.json"
    import json
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if os.path.realpath(cfg.get("base_model_name_or_path", "")) != os.path.realpath(MERGED):
        raise RuntimeError(f"adapter base {cfg.get('base_model_name_or_path')} != {MERGED}")
    tok = AutoTokenizer.from_pretrained(MERGED, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MERGED, dtype=torch.bfloat16, device_map="cuda", local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), adapter_name="voice")
    model.base_model.set_adapter(["voice"], inference_mode=True)
    model.eval()
    return tok, model


def unload(tok, model) -> None:
    del model
    del tok
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(tok, model, messages) -> tuple[str, int, float]:
    import torch
    ids = render(tok, messages, True)
    batch = {"input_ids": torch.tensor([ids], device="cuda")}
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    started = time.time()
    with torch.no_grad():
        out = model.generate(
        **batch, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95,
    )
    torch.cuda.synchronize()
    new_tokens = int(out.shape[1] - batch["input_ids"].shape[1])
    text = tok.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text, new_tokens, time.time() - started


def apply_guard(question: str, pass1: str, pass2: str, judge_text: str) -> dict:
    try:
        dropped, added = factdiff.parse_lists(judge_text)
        parse_ok = True
        raw = ""
    except Exception as exc:
        dropped, added = [], []
        parse_ok = False
        raw = f"{type(exc).__name__}: {exc}; raw={str(judge_text)[:400]}"
    new_markers, pass1_markers, pass2_markers = factdiff.new_markers(pass1, pass2)
    fallback = (not parse_ok) or bool(dropped) or bool(added) or bool(new_markers)
    answer = pass1 if fallback else pass2
    scored_new = sorted(set(C.fail_hits(answer)) - set(C.fail_hits(pass1)))
    if scored_new:
        answer = pass1
        fallback = True
    return {
        "reply": answer,
        "guard_fallback": fallback,
        "dropped": dropped,
        "added": added,
        "parse_ok": parse_ok,
        "new_markers": new_markers,
        "pass1_markers": pass1_markers,
        "pass2_markers": pass2_markers,
        "judge_raw": raw,
        "question": question,
    }


def _history_messages(history: list[dict] | None) -> list[dict]:
    out = []
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = str(turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content[:1000]})
    return out


def prepare(item: dict) -> dict:
    perspective = perspective_get(item.get("perspective") or "wren")
    chapter = int(item["chapter"])
    question = str(item["question"]).strip()[:1000]
    block, passages = retrieve.retrieve_passages(question, chapter)
    system = prompting.assemble_pass1(
        perspective["card"], chapter, block, perspective.get("system_prompt"),
    )
    return {
        **item,
        "perspective": perspective["id"],
        "card": perspective["card"],
        "system_prompt": perspective.get("system_prompt"),
        "chapter": chapter,
        "question": question,
        "pass1_system": system,
        "citations": passages,
        "history": _history_messages(item.get("history")),
    }


def run_batch(items: list[dict], seed: int = SEED, holder: gpu_lock.GpuLock | None = None) -> list[dict]:
    """Run the locked pipeline for a list of questions. Caller holds the GPU lock."""
    if holder is None or not holder.held:
        raise gpu_lock.GpuLost("refusing GPU work without ~/gpu.lock")

    def own() -> None:
        holder.require()

    own()
    ollama_up()
    for name in PARKED_MODELS:
        ollama_stop(name)
    prepared = [prepare(item) for item in items]
    ollama_stop("bge-m3")
    own()
    check_pass2()

    print(f"loading pass-1 {MERGED} + {PASS1_ADAPTER}", flush=True)
    tok, model = load_voice(PASS1_ADAPTER)
    try:
        for row in prepared:
            own()
            seed_all(seed)
            messages = [{"role": "system", "content": row["pass1_system"]}]
            messages.extend(row["history"])
            messages.append({"role": "user", "content": row["question"]})
            text, ntok, sec = generate(tok, model, messages)
            row["pass1"] = text
            row["pass1_toks"] = ntok
            row["pass1_sec"] = round(sec, 3)
            print("pass1", row.get("id") or row["question"][:48], text[:80].replace("\n", " "), flush=True)
    finally:
        unload(tok, model)

    own()
    print(f"loading pass-2 {MERGED} + {PASS2_ADAPTER}", flush=True)
    tok, model = load_voice(PASS2_ADAPTER)
    try:
        for row in prepared:
            own()
            if "pass1" not in row:
                continue
            system = prompting.pass2_system(row["card"], row["chapter"], row["system_prompt"])
            user = prompting.pass2_user(row["question"], row["pass1"])
            locked_prompts.assert_no_retrieval(system, locked_prompts.REWRITE_INSTR)
            if locked_prompts.RAG_MARK in system:
                raise RuntimeError("retrieval block leaked into the rewrite system prompt")
            seed_all(seed)
            text, ntok, sec = generate(
                tok, model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            row["pass2"] = text
            row["pass2_toks"] = ntok
            row["pass2_sec"] = round(sec, 3)
            print("pass2", row.get("id") or "", text[:80].replace("\n", " "), flush=True)
    finally:
        unload(tok, model)

    own()
    print("fact-diff", len(prepared), "questions", flush=True)
    try:
        for row in prepared:
            own()
            prompt = locked_prompts.fact_diff_prompt(row["question"], row["pass1"], row["pass2"])
            judged = ollama_judge(prompt)
            row.update(apply_guard(row["question"], row["pass1"], row["pass2"], judged))
            print(
                "guard", row.get("id") or "",
                "fallback" if row["guard_fallback"] else "kept",
                f"drop={len(row['dropped'])} add={len(row['added'])}",
                flush=True,
            )
    finally:
        ollama_stop(JUDGE_MODEL)
        ollama_stop("bge-m3")

    results = []
    for row in prepared:
        results.append({
            "id": row.get("id"),
            "chapter": row["chapter"],
            "perspective": row["perspective"],
            "question": row["question"],
            "spoiler_probe": bool(row.get("spoiler_probe")),
            "reply": row["reply"],
            "guard_fallback": row["guard_fallback"],
            "citations": row["citations"],
            "pass1": row["pass1"],
            "pass2": row["pass2"],
            "dropped": row["dropped"],
            "added": row["added"],
            "parse_ok": row["parse_ok"],
            "new_markers": row["new_markers"],
            "judge_raw": row["judge_raw"],
            "seconds": round((row.get("pass1_sec") or 0) + (row.get("pass2_sec") or 0), 3),
        })
    return results


def answer_one(question: str, chapter: int, perspective: str = "wren", history: list[dict] | None = None, seed: int = SEED) -> dict:
    """One live turn. Takes the GPU lock, runs the pipeline, then releases it."""
    item = {
        "id": "live",
        "question": question,
        "chapter": chapter,
        "perspective": perspective,
        "history": history or [],
        "spoiler_probe": False,
    }
    with _PROC:
        with gpu_lock.GpuLock("reader live wren", wait_s=LOCK_WAIT) as holder:
            rows = run_batch([item], seed=seed, holder=holder)
    return rows[0]


def public_answer(row: dict, mode: str) -> dict:
    return {
        "reply": row["reply"],
        "citations": row.get("citations") or [],
        "guard_fallback": row.get("guard_fallback"),
        "chapter": row.get("chapter"),
        "perspective": row.get("perspective") or "wren",
        "mode": mode,
        "model_called": mode == "live",
    }
