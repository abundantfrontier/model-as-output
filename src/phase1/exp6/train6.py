"""Voice (and optional facts) LoRA for exp6.

--stage facts   LoRA on the base (8B path). Never used to retrain exp4/lora_q4_facts.
--stage voice   new voice adapter. --facts stacks a frozen facts adapter; empty facts = merged base.
--loss ce | unlikelihood | dpo
DPO rows without `bad` take a CE step (facts replay during DPO). trl is not used.
"""
import os, sys, json, time, math, argparse, random, traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import common as C
from chat import render
import frozen

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

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]

def load_base(name, q4):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if q4:
        from transformers import BitsAndBytesConfig
        kw = dict(
            device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            ),
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except Exception as e:
        print("load failed, retry sdpa:", type(e).__name__, e, flush=True)
        kw["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if q4:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return tok, model

def pair_example(tok, p):
    sysm = frozen.NARR if p["char"] == "narrator" else C.sysprompt(p["char"], p.get("ch") or 3)
    msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": p["user"]}]
    pr = render(tok, msgs, True)
    full = render(tok, msgs + [{"role": "assistant", "content": p["answer"]}], False)
    if len(full) <= len(pr) or full[:len(pr)] != pr:
        pair_example.mismatch = getattr(pair_example, "mismatch", 0) + 1
        if pair_example.mismatch <= 5:
            print("prefix mismatch", p.get("char"), len(pr), len(full), flush=True)
        return None
    pair_example.ok = getattr(pair_example, "ok", 0) + 1
    y = [-100] * len(pr) + full[len(pr):]
    if len(full) > 1024:
        full, y = full[-1024:], y[-1024:]
    if not any(t != -100 for t in y):
        return None
    return full, y

def zero_dropout(model):
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
            n += 1
    print("dropout modules zeroed", n, flush=True)

def completion_logprobs(model, ids, labels):
    out = model(input_ids=torch.tensor([ids], device="cuda"))
    logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
    lab = torch.tensor([labels[1:]], device="cuda")
    mask = lab != -100
    tok = logp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return tok, mask

def _any(mask):
    return bool(mask.any().item())

def ul_mean(model, ids, labels):
    tok_lp, mask = completion_logprobs(model, ids, labels)
    if not _any(mask):
        return None
    p = tok_lp.exp().clamp(max=1 - 1e-4)
    ul = -torch.log((1 - p).clamp(min=1e-4))
    return ul[mask].mean()

def sum_lp(model, ids, labels):
    tok_lp, mask = completion_logprobs(model, ids, labels)
    if not _any(mask):
        return None
    return tok_lp[mask].sum()

def assert_grads(model, has_facts):
    facts = [n for n, p in model.named_parameters() if p.requires_grad and ".facts." in n]
    voice = [n for n, p in model.named_parameters() if p.requires_grad and (".default." in n or "lora_" in n)]
    if has_facts and facts:
        raise RuntimeError(f"adapter grads wrong: trainable facts {len(facts)}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_train <= 0:
        raise RuntimeError("no trainable parameters")
    print("trainable tensors", len(voice), "params", n_train, "has_facts", has_facts, flush=True)

def stack_voice(base_model, facts_dir, r, init, dropout):
    if facts_dir:
        saved = json.load(open(os.path.join(facts_dir, "adapter_config.json")))["target_modules"]
        if isinstance(saved, str):
            saved = find_targets(base_model)
        model = PeftModel.from_pretrained(base_model, facts_dir, adapter_name="facts", is_trainable=False)
        if init:
            model.load_adapter(init, adapter_name="default", is_trainable=True)
            r = json.load(open(os.path.join(init, "adapter_config.json")))["r"]
        else:
            cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=dropout, target_modules=list(saved), task_type="CAUSAL_LM")
            model.add_adapter("default", cfg)
        model.base_model.set_adapter(["facts", "default"])
        model.set_requires_grad("facts", False)
        model.set_requires_grad("default", True)
        targets = list(saved)
    else:
        if init:
            model = PeftModel.from_pretrained(base_model, init, adapter_name="default", is_trainable=True)
            r = json.load(open(os.path.join(init, "adapter_config.json")))["r"]
            saved = json.load(open(os.path.join(init, "adapter_config.json")))["target_modules"]
            targets = list(saved) if not isinstance(saved, str) else find_targets(base_model)
        else:
            targets = find_targets(base_model)
            cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=dropout, target_modules=targets, task_type="CAUSAL_LM")
            model = get_peft_model(base_model, cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    assert_grads(model, bool(facts_dir))
    return model, r, targets

def facts_examples(tok, pairs, claims, limit):
    ex = []
    docs = [(C.chapter_text(i), 3) for i in (1, 2, 3)]
    lines = []
    who = {"wren": "Wren", "diane": "Diane", "yd4408": "YD-4408"}
    for c in claims:
        name = who.get(c["char"], c["char"])
        kind = c.get("claim_kind") or "fact"
        text = c.get("claim") or c.get("answer") or ""
        if not text:
            continue
        if kind == "belief":
            lines.append(f"{name} believes this on day 3: {text}")
        elif kind == "refusal":
            lines.append(f"{name} will not do this: {text}")
        else:
            lines.append(f"{name} knows this on day 3: {text}")
    if lines:
        docs.append(("\n".join(lines), 2))
    for text, reps in docs:
        ids = tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        for i in range(0, len(ids), 256):
            b = ids[i:i + 512]
            if len(b) > 64:
                ex.extend([(b, list(b))] * reps)
    text_ex = ex
    pair_ex = []
    for p in pairs:
        item = pair_example(tok, p)
        if not item:
            continue
        reps = 3 if p.get("source") == "page" else 1
        pair_ex.extend([item] * reps)
    if limit:
        return (text_ex[:1] + pair_ex[:limit]) or pair_ex[:limit] or text_ex[:limit]
    return text_ex + pair_ex

def build_rows(tok, pairs, loss):
    rows = []
    for p in pairs:
        good = pair_example(tok, p)
        if not good:
            continue
        bad = None
        if p.get("bad"):
            bad = pair_example(tok, dict(p, answer=p["bad"]))
        if loss == "dpo" and not bad:
            # CE-only facts replay row during DPO.
            reps = 1
            for _ in range(reps):
                rows.append(dict(good=good, bad=None, ref_c=None, ref_r=None, ce_only=True))
            continue
        if loss == "dpo":
            if good[0] == bad[0]:
                continue
        reps = 3 if (not p.get("bad") and p.get("source") == "page") else 1
        for _ in range(reps):
            rows.append(dict(good=good, bad=bad, ref_c=None, ref_r=None, ce_only=False))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=frozen.BASE)
    ap.add_argument("--facts", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--claims", default="")
    ap.add_argument("--stage", choices=["facts", "voice"], default="voice")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--loss", choices=["ce", "unlikelihood", "dpo"], default="ce")
    ap.add_argument("--ul-weight", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--init", default="", help="voice adapter to continue (DPO / combo)")
    ap.add_argument("--q4", action="store_true")
    ap.add_argument("--dropout", type=float, default=-1.0)
    a = ap.parse_args()
    od = a.out if os.path.isabs(a.out) else os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    try:
        run(a, od)
    except Exception:
        os.makedirs(od, exist_ok=True)
        open(os.path.join(od, "FAIL.txt"), "w").write(traceback.format_exc())
        raise

def run(a, od):
    if a.stage != "facts":
        frozen.check()
    pairs_path = a.pairs or (frozen.FACTS_JSONL if a.stage == "facts" else frozen.VOICE_JSONL)
    pairs = load_jsonl(pairs_path)
    claims = load_jsonl(a.claims or frozen.CLAIMS_JSONL) if a.stage == "facts" else []
    if a.limit:
        pairs = pairs[:a.limit]
        claims = claims[:a.limit]
    dropout = a.dropout if a.dropout >= 0 else (0.0 if a.loss == "dpo" else 0.05)
    print("loading", a.base, "stage", a.stage, "loss", a.loss, "q4", a.q4, flush=True)
    tok, base = load_base(a.base, a.q4)
    facts_dir = a.facts or None
    if facts_dir and not os.path.isabs(facts_dir):
        facts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), facts_dir)
    init = a.init or None
    if a.stage == "facts":
        if facts_dir:
            raise RuntimeError("facts stage does not take --facts")
        targets = find_targets(base)
        cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=dropout, target_modules=targets, task_type="CAUSAL_LM")
        model = get_peft_model(base, cfg)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        r = a.r
        rows = None
        ex = facts_examples(tok, pairs, claims, a.limit)
        if not ex:
            raise RuntimeError("no training examples")
    else:
        model, r, targets = stack_voice(base, facts_dir, a.r, init, dropout)
        if a.loss == "dpo":
            zero_dropout(model)
        rows = build_rows(tok, pairs, a.loss)
        if not rows:
            raise RuntimeError("no training examples")
        ex = None
    mismatch = getattr(pair_example, "mismatch", 0)
    n_ex = len(ex) if ex is not None else len(rows)
    print("examples", n_ex, "pairs_in", len(pairs), "prefix_ok", getattr(pair_example, "ok", 0), "prefix_mismatch", mismatch, flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
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
                    print(f"ref {i+1}/{len(rows)}", flush=True)
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
    ce_last = ul_last = dpo_last = None
    work = ex if ex is not None else rows
    for ep in range(a.epochs):
        random.shuffle(work)
        tot = 0.0
        n = 0
        for item in work:
            opt.zero_grad(set_to_none=True)
            if ex is not None:
                x, y = item
                out = model(input_ids=torch.tensor([x], device="cuda"), labels=torch.tensor([y], device="cuda"))
                if not math.isfinite(out.loss.item()):
                    raise RuntimeError(f"non-finite loss at step {step}")
                ce_last = out.loss.item()
                out.loss.backward()
                shown = ce_last
                del out
            elif a.loss == "dpo" and not item.get("ce_only"):
                pol_c = sum_lp(model, *item["good"])
                pol_r = sum_lp(model, *item["bad"])
                delta = (pol_c - item["ref_c"]) - (pol_r - item["ref_r"])
                loss = -torch.nn.functional.logsigmoid(a.beta * delta)
                if not math.isfinite(loss.item()):
                    raise RuntimeError(f"non-finite dpo loss at step {step}")
                loss.backward()
                dpo_last = loss.item()
                shown = dpo_last
            else:
                ids, labels = item["good"]
                out = model(input_ids=torch.tensor([ids], device="cuda"), labels=torch.tensor([labels], device="cuda"))
                ce = out.loss
                if not math.isfinite(ce.item()):
                    raise RuntimeError(f"non-finite loss at step {step}")
                ce_last = ce.item()
                ce.backward()
                del out
                ul_last = None
                if a.loss == "unlikelihood" and item.get("bad") is not None:
                    ul = ul_mean(model, *item["bad"])
                    if ul is not None:
                        if not math.isfinite(ul.item()):
                            raise RuntimeError(f"non-finite unlikelihood at step {step}")
                        ul_last = ul.item()
                        (a.ul_weight * ul).backward()
                shown = ce_last + (a.ul_weight * ul_last if ul_last is not None else 0.0)
            opt.step()
            tot += shown
            n += 1
            step += 1
            if step == 1 or step % 50 == 0:
                rate = step / max(time.time() - t0, 1e-6)
                extra = ""
                if ce_last is not None:
                    extra += f" ce {ce_last:.3f}"
                if ul_last is not None:
                    extra += f" ul {ul_last:.3f}"
                if dpo_last is not None:
                    extra += f" dpo {dpo_last:.3f}"
                print(f"step {step}/{total} {time.time()-t0:.0f}s eta {(total-step)/rate:.0f}s{extra}", flush=True)
        print(f"epoch {ep} loss {tot/max(n,1):.3f}", flush=True)
    os.makedirs(od, exist_ok=True)
    if a.stage == "voice" and facts_dir:
        model.save_pretrained(od, selected_adapters=["default"])
    else:
        model.save_pretrained(od)
    report = dict(
        base=a.base, stage=a.stage, facts=facts_dir, facts_sha256=frozen.FACTS_SHA if facts_dir else None,
        q4=bool(a.q4), pairs=len(pairs), claims=len(claims), examples=n_ex, epochs=a.epochs, lr=a.lr, r=r,
        alpha=2 * r, dropout=0.0 if a.loss == "dpo" else dropout,
        targets=targets, steps=step, loss=a.loss if a.stage == "voice" else "ce",
        ul_weight=a.ul_weight if a.loss == "unlikelihood" else None,
        beta=a.beta if a.loss == "dpo" else None, init=init,
        dpo="sum token logprob, beta on the sum, ref cached from init; CE on facts-replay rows" if a.loss == "dpo" else None,
        trl=False, train_sec=round(time.time() - t0),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        prefix_mismatch_skipped=mismatch, limit=a.limit, pairs_file=os.path.abspath(pairs_path),
    )
    json.dump(report, open(os.path.join(od, "train_report.json"), "w"), indent=1)
    print("done", od, flush=True)
    if a.stage != "facts":
        frozen.check()

if __name__ == "__main__":
    main()
