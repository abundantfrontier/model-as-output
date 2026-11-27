#!/usr/bin/env python3
"""Voice rewrite LoRA on stock Qwen3-4B (from scratch), plus optional frozen-ref DPO.

Does not stack on the facts LoRA. Does not init from train1 lora_voice / lora_voice_dpo.
CE recipe matches exp9 rewrite: r 32, alpha 64, dropout 0.05, 1 epoch, lr 2e-5.
DPO recipe matches exp10: frozen reference, SUM token logprob, beta 0.05, lr 2e-6.
Early-stop: CE < 0.05, or CE jumps by >1.0 after having been <0.2 → roll back to last checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import prompts  # noqa: E402
from chat import render  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CANDS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj", "gate_up_proj"}
MAX_LEN = 3072
BANNED_INIT = ("lora_voice_dpo", "lora_voice", "merged_fv")


def find_targets(model):
    found = set()
    for n, m in model.named_modules():
        if n.split(".")[-1] in CANDS and isinstance(m, torch.nn.Linear):
            found.add(n.split(".")[-1])
    if not found:
        names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)})
        raise RuntimeError("no LoRA target modules. linear layers include: " + ", ".join(names[:40]))
    return sorted(found)


def load_base(name):
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
        )
    except Exception as e:
        print("load failed, retry sdpa:", type(e).__name__, e, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return tok, model


def pair_sys(p):
    """Single-pass SFT input is the character card plus the RAG block.

    That is the same system prompt eval_pass1 builds for merged + RAG.
    The user line stays the bare question.
    """
    sysm = (p.get("sys") or "").strip()
    if sysm:
        return sysm
    char = p.get("char") or "narrator"
    upto = int(p.get("reader_pos") or p.get("ch") or 4)
    return C.sysprompt(char, upto)


def pair_example(tok, p):
    sysm = pair_sys(p)
    user = p["user"]
    if prompts.RAG_MARK in user:
        raise RuntimeError("retrieval block leaked into the user question")
    msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": user}]
    pr = render(tok, msgs, True)
    full = render(tok, msgs + [{"role": "assistant", "content": p["answer"]}], False)
    if len(full) <= len(pr) or full[: len(pr)] != pr:
        pair_example.mismatch = getattr(pair_example, "mismatch", 0) + 1
        if pair_example.mismatch <= 5:
            print("prefix mismatch", p.get("char"), len(pr), len(full), flush=True)
        return None
    pair_example.ok = getattr(pair_example, "ok", 0) + 1
    y = [-100] * len(pr) + full[len(pr) :]
    if len(full) > MAX_LEN:
        full, y = full[-MAX_LEN :], y[-MAX_LEN :]
    if not any(t != -100 for t in y):
        return None
    return full, y


def build_ce_examples(tok, pairs, limit):
    pair_ex = []
    for p in pairs:
        item = pair_example(tok, p)
        if not item:
            continue
        pair_ex.append(item)
    if limit:
        return pair_ex[:limit]
    return pair_ex


def completion_logprobs(model, ids, labels):
    out = model(input_ids=torch.tensor([ids], device="cuda"))
    logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
    lab = torch.tensor([labels[1:]], device="cuda")
    mask = lab != -100
    tok = logp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return tok, mask


def sum_lp(model, ids, labels):
    tok_lp, mask = completion_logprobs(model, ids, labels)
    if not bool(mask.any().item()):
        return None
    return tok_lp[mask].sum()


def zero_dropout(model):
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
            n += 1
    print("dropout modules zeroed", n, flush=True)


def refuse_banned_init(path: str) -> None:
    if not path:
        return
    norm = os.path.realpath(path)
    for bit in BANNED_INIT:
        if f"train1/{bit}" in norm.replace("\\", "/") or norm.rstrip("/").endswith("/" + bit):
            # only ban the looping train1 adapters, not a similarly named train2 dir
            if "/train1/" in norm:
                raise SystemExit(f"refusing to init from looping adapter: {path}")


def save_adapter(model, path, stacked_facts: bool):
    os.makedirs(path, exist_ok=True)
    if stacked_facts:
        model.save_pretrained(path, selected_adapters=["default"])
    else:
        model.save_pretrained(path)


def copy_adapter(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("checkpoint-*", "FAIL.txt"))


def run(a, od):
    refuse_banned_init(a.init)
    pairs = C.load_jsonl(a.pairs)
    if a.limit:
        pairs = pairs[: a.limit]
    print("loading", a.base, "pairs", len(pairs), "stage", a.stage, "loss", a.loss, "facts", a.facts or None, flush=True)
    tok, base = load_base(a.base)
    targets = find_targets(base)
    dropout = a.dropout if a.dropout >= 0 else (0.0 if a.loss == "dpo" else 0.05)
    alpha = a.alpha if a.alpha else 2 * a.r
    cfg = LoraConfig(r=a.r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, task_type="CAUSAL_LM")
    stacked = False
    if a.stage == "facts":
        model = get_peft_model(base, cfg)
    elif a.facts:
        facts = a.facts if os.path.isabs(a.facts) else str(C.TRAIN / a.facts)
        model = PeftModel.from_pretrained(base, facts, adapter_name="facts", is_trainable=False)
        saved = json.load(open(os.path.join(facts, "adapter_config.json")))["target_modules"]
        if isinstance(saved, str):
            saved = targets
        if a.init:
            init = a.init if os.path.isabs(a.init) else str(C.TRAIN / a.init)
            refuse_banned_init(init)
            model.load_adapter(init, adapter_name="default", is_trainable=True)
        else:
            cfg = LoraConfig(
                r=a.r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(saved), task_type="CAUSAL_LM"
            )
            model.add_adapter("default", cfg)
        model.base_model.set_adapter(["facts", "default"])
        model.set_requires_grad("facts", False)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        stacked = True
    else:
        if a.init:
            init = a.init if os.path.isabs(a.init) else str(C.TRAIN / a.init)
            refuse_banned_init(init)
            model = PeftModel.from_pretrained(base, init, adapter_name="default", is_trainable=True)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        else:
            model = get_peft_model(base, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    print("targets", targets, "trainable", sum(p.numel() for p in params), "stacked_facts", stacked, flush=True)

    rows = None
    if a.loss == "dpo":
        zero_dropout(model)
        rows = []
        for p in pairs:
            good = pair_example(tok, p)
            if not good:
                continue
            if p.get("bad"):
                bad = pair_example(tok, dict(p, answer=p["bad"]))
                if bad and bad[0] != good[0]:
                    rows.append(dict(good=good, bad=bad, ce_only=False))
                    continue
            rows.append(dict(good=good, bad=None, ce_only=True))
        if not rows:
            raise RuntimeError("no dpo/ce rows")
        print(
            "dpo rows",
            sum(1 for r in rows if not r["ce_only"]),
            "ce replay",
            sum(1 for r in rows if r["ce_only"]),
            flush=True,
        )
        ex = None
    else:
        ex = build_ce_examples(tok, pairs, a.limit)
        if not ex:
            raise RuntimeError("no training examples")
        print(
            "examples",
            len(ex),
            "prefix_ok",
            getattr(pair_example, "ok", 0),
            "prefix_mismatch",
            getattr(pair_example, "mismatch", 0),
            flush=True,
        )
        if pairs and not getattr(pair_example, "ok", 0):
            raise RuntimeError("every chat pair had a chat-template prefix mismatch")

    n_ex = len(ex) if ex is not None else len(rows)
    opt = torch.optim.AdamW(params, lr=a.lr)
    if a.loss == "dpo" and rows is not None:
        model.train()
        with torch.no_grad():
            for i, row in enumerate(rows):
                if row.get("ce_only") or row["bad"] is None:
                    continue
                rc = sum_lp(model, *row["good"])
                rr = sum_lp(model, *row["bad"])
                if rc is None or rr is None:
                    row["skip"] = True
                    continue
                row["ref_c"] = rc.item()
                row["ref_r"] = rr.item()
                if i == 0 or (i + 1) % 50 == 0:
                    print(f"ref {i + 1}/{len(rows)}", flush=True)
        rows = [row for row in rows if row.get("ce_only") or row.get("ref_c") is not None]
        if not any(not row.get("ce_only") for row in rows):
            raise RuntimeError("no DPO pairs with both logprobs")
        n_ex = len(rows)

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    total = n_ex * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    os.makedirs(od, exist_ok=True)
    save_every = a.save_every
    if not save_every:
        save_every = max(40, n_ex // 3)
    work = ex if ex is not None else rows
    model.train()
    stop = False
    stop_reason = None
    last_ck = None
    had_low = False
    min_low = None
    saved_pre_mem = False
    ce_last = dpo_last = None
    log_every = a.log_every or 25
    # Batch size is 1. One easy pair can show CE < 0.05; exp9's healthy rewrite
    # run did that all epoch. The diagnosis guard is the logged curve (train1
    # hit 0.001, then jumped by >1 at the next logged step). Use the mean of
    # the last log_every steps, not a single example.
    ce_win = deque(maxlen=log_every)
    ce_win_mean = None

    def maybe_save(tag):
        nonlocal last_ck
        ck = os.path.join(od, tag)
        save_adapter(model, ck, stacked)
        last_ck = ck
        model.train()
        print("saved", ck, flush=True)

    for ep in range(a.epochs):
        if stop:
            break
        random.shuffle(work)
        tot = 0.0
        nstep = 0
        for item in work:
            if C.past_deadline(a.reserve_min) and step > 0:
                stop_reason = "deadline"
                print("deadline near, stopping at step", step, flush=True)
                stop = True
                break
            if a.max_steps and step >= a.max_steps:
                stop = True
                break
            opt.zero_grad(set_to_none=True)
            if ex is not None:
                x, y = item
                out = model(input_ids=torch.tensor([x], device="cuda"), labels=torch.tensor([y], device="cuda"))
                loss = out.loss
                shown = loss.item()
                ce_last = shown
            elif a.loss == "dpo" and not item.get("ce_only"):
                pol_c = sum_lp(model, *item["good"])
                pol_r = sum_lp(model, *item["bad"])
                delta = (pol_c - item["ref_c"]) - (pol_r - item["ref_r"])
                loss = -torch.nn.functional.logsigmoid(a.beta * delta)
                shown = loss.item()
                dpo_last = shown
            else:
                ids, labels = item["good"]
                out = model(input_ids=torch.tensor([ids], device="cuda"), labels=torch.tensor([labels], device="cuda"))
                loss = out.loss
                shown = loss.item()
                ce_last = shown
            if not math.isfinite(shown):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            opt.step()
            tot += shown
            step += 1
            nstep += 1

            if a.loss == "ce":
                ce_win.append(shown)
                if len(ce_win) == ce_win.maxlen:
                    mean = sum(ce_win) / len(ce_win)
                    ce_win_mean = mean
                    if mean < 0.05:
                        if last_ck is None:
                            maybe_save("checkpoint-pre-mem")
                        stop_reason = (
                            f"ce_below_0.05 at step {step} window_mean {mean:.4f} "
                            f"(last {ce_win.maxlen} steps)"
                        )
                        print("EARLY STOP", stop_reason, flush=True)
                        stop = True
                        break
                    if had_low and min_low is not None and (mean - min_low) > 1.0:
                        stop_reason = (
                            f"ce_spike at step {step} window_mean {mean:.4f} after min {min_low:.4f}"
                        )
                        print("EARLY STOP", stop_reason, flush=True)
                        stop = True
                        break
                    if mean < 0.2:
                        had_low = True
                        min_low = mean if min_low is None else min(min_low, mean)
                    if (not saved_pre_mem) and mean < 0.25 and mean >= 0.05:
                        maybe_save("checkpoint-pre-mem")
                        saved_pre_mem = True

            if step == 1 or step % log_every == 0:
                rate = step / max(time.time() - t0, 1e-6)
                extra = ""
                if ce_last is not None:
                    extra += f" ce {ce_last:.3f}"
                if ce_win_mean is not None:
                    extra += f" win {ce_win_mean:.3f}"
                if dpo_last is not None:
                    extra += f" dpo {dpo_last:.3f}"
                print(
                    f"step {step}/{total} {time.time() - t0:.0f}s eta {(total - step) / rate:.0f}s{extra}",
                    flush=True,
                )
            if save_every and step % save_every == 0 and step < total:
                if a.loss != "ce" or shown >= 0.05:
                    maybe_save(f"checkpoint-{step}")
        if nstep:
            print(f"epoch {ep} loss {tot / nstep:.3f}", flush=True)
        if stop:
            break

    rolled = False
    # Roll back only on a CE spike / collapse. A deadline stop keeps the latest step.
    ce_stop = bool(stop_reason) and str(stop_reason).startswith("ce_")
    if ce_stop and last_ck and os.path.isdir(last_ck):
        print("rollback to", last_ck, "reason", stop_reason, flush=True)
        copy_adapter(last_ck, od)
        rolled = True
    elif ce_stop:
        raise RuntimeError(f"early stop ({stop_reason}) with no healthy checkpoint to roll back to")
    else:
        save_adapter(model, od, stacked)
    report = dict(
        base=a.base,
        stage=a.stage,
        facts=a.facts or None,
        init=a.init or None,
        loss=a.loss,
        pairs=len(pairs),
        examples=n_ex,
        epochs=a.epochs,
        lr=a.lr,
        r=a.r,
        alpha=alpha,
        dropout=0.0 if a.loss == "dpo" else dropout,
        beta=a.beta if a.loss == "dpo" else None,
        targets=targets,
        steps=step,
        train_sec=round(time.time() - t0),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        prefix_mismatch_skipped=getattr(pair_example, "mismatch", 0),
        prefix_ok=getattr(pair_example, "ok", 0),
        limit=a.limit,
        save_every=save_every,
        stop_reason=stop_reason,
        rolled_back_to=last_ck if rolled else None,
        dpo="sum token logprob, beta on the sum, ref cached from init" if a.loss == "dpo" else None,
        voice_choice="single-pass SFT on stock Qwen3-4B; input is character card + RAG; target is the guarded final answer",
        stacked_facts=stacked,
    )
    json.dump(report, open(os.path.join(od, "train_report.json"), "w"), indent=1)
    print("done", od, json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.BASE)
    ap.add_argument("--stage", choices=["facts", "voice"], default="voice")
    ap.add_argument("--out", required=True)
    ap.add_argument("--facts", default="", help="leave empty to train voice on stock Qwen3-4B")
    ap.add_argument("--init", default="")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--loss", choices=["ce", "dpo"], default="ce")
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--reserve-min", type=int, default=90)
    a = ap.parse_args()
    if not a.pairs:
        a.pairs = str(C.DATA / ("facts.jsonl" if a.stage == "facts" else "rewrite_pairs.jsonl"))
    od = a.out if os.path.isabs(a.out) else str(C.TRAIN / a.out)
    try:
        C.unload_ollama()
        run(a, od)
    except Exception:
        os.makedirs(od, exist_ok=True)
        open(os.path.join(od, "FAIL.txt"), "w").write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
