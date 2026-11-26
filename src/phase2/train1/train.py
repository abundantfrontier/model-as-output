#!/usr/bin/env python3
"""Facts LoRA, then a character-tagged voice LoRA stacked on frozen facts.

Adapted from exp4/train4.py. One voice adapter: the system prompt names Victor
or the creature. Optional --loss dpo continues the voice adapter (exp7 style).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from chat import render  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CANDS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj", "gate_up_proj"}


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


def pair_example(tok, p):
    char = p.get("char") or "narrator"
    upto = int(p.get("reader_pos") or p.get("ch") or 4)
    sysm = C.sysprompt(char, upto)
    msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": p["user"]}]
    pr = render(tok, msgs, True)
    full = render(tok, msgs + [{"role": "assistant", "content": p["answer"]}], False)
    if len(full) <= len(pr) or full[:len(pr)] != pr:
        pair_example.mismatch = getattr(pair_example, "mismatch", 0) + 1
        if pair_example.mismatch <= 5:
            print("prefix mismatch", char, len(pr), len(full), flush=True)
        return None
    pair_example.ok = getattr(pair_example, "ok", 0) + 1
    y = [-100] * len(pr) + full[len(pr):]
    if len(full) > 1024:
        full, y = full[-1024:], y[-1024:]
    if not any(t != -100 for t in y):
        return None
    return full, y


def build_ce_examples(tok, stage, pairs, claims, limit):
    ex = []
    if stage == "facts":
        lines = []
        for c in claims:
            name = {"victor": "Victor", "creature": "the creature"}.get(c["char"], c["char"])
            lines.append(f"{name} knows this once the reader has reached {c.get('unit_id')}: {c.get('claim') or c.get('answer') or ''}")
        if lines:
            text = "\n".join(lines)
            ids = tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            for i in range(0, len(ids), 256):
                b = ids[i : i + 512]
                if len(b) > 64:
                    ex.extend([(b, list(b))] * 2)
    pair_ex = []
    for p in pairs:
        item = pair_example(tok, p)
        if not item:
            continue
        reps = 3 if p.get("source") == "handwritten" else 1
        pair_ex.extend([item] * reps)
    if limit:
        return (ex[:1] + pair_ex[:limit]) or pair_ex[:limit] or ex[:limit]
    return ex + pair_ex


def completion_logprobs(model, ids, labels):
    out = model(input_ids=torch.tensor([ids], device="cuda"))
    logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
    lab = torch.tensor([labels[1:]], device="cuda")
    mask = lab != -100
    tok = logp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return tok, mask


def dpo_loss(model, good, bad, beta):
    g_ids, g_y = good
    b_ids, b_y = bad
    g_lp, g_m = completion_logprobs(model, g_ids, g_y)
    b_lp, b_m = completion_logprobs(model, b_ids, b_y)
    if not g_m.any() or not b_m.any():
        return None
    # Average logprob so length doesn't dominate.
    g = g_lp[g_m].mean()
    b = b_lp[b_m].mean()
    return -torch.nn.functional.logsigmoid(beta * (g - b))


def run(a, od):
    pairs = C.load_jsonl(a.pairs)
    claims = C.load_jsonl(a.claims) if a.stage == "facts" else []
    if a.limit:
        pairs = pairs[: a.limit]
        claims = claims[: a.limit]
    print("loading", a.base, "pairs", len(pairs), "stage", a.stage, "loss", a.loss, flush=True)
    tok, base = load_base(a.base)
    targets = find_targets(base)
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM")
    if a.stage == "facts":
        model = get_peft_model(base, cfg)
    else:
        facts = a.facts if os.path.isabs(a.facts) else str(C.TRAIN / a.facts)
        model = PeftModel.from_pretrained(base, facts, adapter_name="facts", is_trainable=False)
        saved = json.load(open(os.path.join(facts, "adapter_config.json")))["target_modules"]
        if isinstance(saved, str):
            saved = targets
        if a.init:
            init = a.init if os.path.isabs(a.init) else str(C.TRAIN / a.init)
            model.load_adapter(init, adapter_name="default", is_trainable=True)
        else:
            cfg = LoraConfig(
                r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules=list(saved), task_type="CAUSAL_LM"
            )
            model.add_adapter("default", cfg)
        model.base_model.set_adapter(["facts", "default"])
        model.set_requires_grad("facts", False)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    print("targets", targets, "trainable", sum(p.numel() for p in params), flush=True)

    if a.loss == "dpo":
        rows = []
        for p in pairs:
            good = pair_example(tok, p)
            if not good:
                continue
            if p.get("bad"):
                bad = pair_example(tok, dict(p, answer=p["bad"]))
                if bad and bad[0] != good[0]:
                    rows.append(("dpo", good, bad))
                    continue
            rows.append(("ce", good, None))
        if not rows:
            raise RuntimeError("no dpo/ce rows")
        print("dpo rows", sum(1 for k, *_ in rows if k == "dpo"), "ce replay", sum(1 for k, *_ in rows if k == "ce"), flush=True)
        ex = rows
    else:
        ex = build_ce_examples(tok, a.stage, pairs, claims, a.limit)
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

    opt = torch.optim.AdamW(params, lr=a.lr)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    total = len(ex) * a.epochs
    os.makedirs(od, exist_ok=True)
    for ep in range(a.epochs):
        random.shuffle(ex)
        tot = 0.0
        nstep = 0
        for item in ex:
            if C.past_deadline(90) and step > 0:
                print("deadline near, stopping at step", step, flush=True)
                break
            if a.loss == "dpo":
                kind, good, bad = item
                if kind == "dpo":
                    loss = dpo_loss(model, good, bad, a.beta)
                    if loss is None:
                        continue
                else:
                    x, y = good
                    out = model(input_ids=torch.tensor([x], device="cuda"), labels=torch.tensor([y], device="cuda"))
                    loss = out.loss
            else:
                x, y = item
                out = model(input_ids=torch.tensor([x], device="cuda"), labels=torch.tensor([y], device="cuda"))
                loss = out.loss
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            opt.step()
            opt.zero_grad()
            tot += loss.item()
            step += 1
            nstep += 1
            if step == 1 or step % 100 == 0:
                rate = step / max(time.time() - t0, 1e-6)
                print(f"step {step}/{total} {time.time()-t0:.0f}s eta {(total-step)/rate:.0f}s loss {loss.item():.3f}", flush=True)
                if a.save_every and step % a.save_every == 0:
                    ck = os.path.join(od, f"checkpoint-{step}")
                    if a.stage == "voice":
                        model.save_pretrained(ck, selected_adapters=["default"])
                    else:
                        model.save_pretrained(ck)
        if nstep:
            print(f"epoch {ep} loss {tot/nstep:.3f}", flush=True)
        if C.past_deadline(90):
            break
    if a.stage == "voice":
        model.save_pretrained(od, selected_adapters=["default"])
    else:
        model.save_pretrained(od)
    report = dict(
        base=a.base,
        stage=a.stage,
        facts=a.facts or None,
        init=a.init or None,
        loss=a.loss,
        pairs=len(pairs),
        claims=len(claims),
        examples=len(ex),
        epochs=a.epochs,
        lr=a.lr,
        r=a.r,
        beta=a.beta,
        targets=targets,
        steps=step,
        train_sec=round(time.time() - t0),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        prefix_mismatch_skipped=getattr(pair_example, "mismatch", 0),
        limit=a.limit,
        voice_choice="one LoRA with character tag in sysprompt",
    )
    json.dump(report, open(os.path.join(od, "train_report.json"), "w"), indent=1)
    print("done", od, json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.BASE)
    ap.add_argument("--stage", choices=["facts", "voice"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--facts", default="")
    ap.add_argument("--init", default="")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--claims", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--loss", choices=["ce", "dpo"], default="ce")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "voice" and not a.facts:
        raise SystemExit("--facts is required for --stage voice")
    if not a.pairs:
        a.pairs = str(C.DATA / ("facts.jsonl" if a.stage == "facts" else "voice.jsonl"))
    if not a.claims:
        a.claims = str(C.DATA / "claims.jsonl")
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
