"""python train4.py --base MODEL --stage facts|voice --out DIR [--facts ADAPTER]

facts: LoRA on the base (chapter text, claims, factual Q->A, contrast).
voice: a second LoRA on the frozen facts adapter (stacked, not merged).
"""
import os, sys, json, time, math, argparse, random, traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import common as C
from gen import NARR
from chat import render

H = os.path.dirname(os.path.abspath(__file__))
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

def load_base(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    except Exception as e:
        print("load failed, retry sdpa:", type(e).__name__, e, flush=True)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return tok, model

def pair_example(tok, p):
    sysm = NARR if p["char"] == "narrator" else C.sysprompt(p["char"], p.get("ch") or 3)
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

def build_examples(tok, stage, pairs, claims, limit):
    ex = []
    if stage == "facts":
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
        # One text chunk plus a few pairs, so a smoke step exercises both paths.
        return (text_ex[:1] + pair_ex[:limit]) or pair_ex[:limit] or text_ex[:limit]
    return text_ex + pair_ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--stage", choices=["facts", "voice"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--facts", default="", help="facts adapter dir, required for --stage voice")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--claims", default="")
    a = ap.parse_args()
    if a.stage == "voice" and not a.facts:
        raise SystemExit("--facts is required for --stage voice")
    od = a.out if os.path.isabs(a.out) else f"{H}/{a.out}"
    try:
        run(a, od)
    except Exception:
        os.makedirs(od, exist_ok=True)
        open(f"{od}/FAIL.txt", "w").write(traceback.format_exc())
        raise

def run(a, od):
    pairs_path = a.pairs or f"{H}/data/{'facts' if a.stage == 'facts' else 'voice'}.jsonl"
    claims_path = a.claims or f"{H}/data/claims.jsonl"
    pairs = load_jsonl(pairs_path)
    claim_rows = load_jsonl(claims_path) if a.stage == "facts" else []
    if a.limit:
        pairs = pairs[:a.limit]
        claim_rows = claim_rows[:a.limit]
    print("loading", a.base, flush=True)
    tok, base = load_base(a.base)
    targets = find_targets(base)
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM")
    if a.stage == "facts":
        model = get_peft_model(base, cfg)
    else:
        facts = a.facts if os.path.isabs(a.facts) else f"{H}/{a.facts}"
        model = PeftModel.from_pretrained(base, facts, adapter_name="facts", is_trainable=False)
        saved = json.load(open(f"{facts}/adapter_config.json"))["target_modules"]
        if isinstance(saved, str):
            saved = targets
        cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules=list(saved), task_type="CAUSAL_LM")
        model.add_adapter("default", cfg)
        model.base_model.set_adapter(["facts", "default"])
        model.set_requires_grad("facts", False)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    print("targets", targets, "trainable", sum(p.numel() for p in params), "stage", a.stage, flush=True)
    ex = build_examples(tok, a.stage, pairs, claim_rows, a.limit)
    if not ex:
        raise RuntimeError("no training examples")
    mismatch = getattr(pair_example, "mismatch", 0)
    print("examples", len(ex), "pairs", len(pairs), "claims", len(claim_rows), "prefix_ok", getattr(pair_example, "ok", 0), "prefix_mismatch", mismatch, flush=True)
    if pairs and not getattr(pair_example, "ok", 0):
        raise RuntimeError("every chat pair had a chat-template prefix mismatch")
    opt = torch.optim.AdamW(params, lr=a.lr)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    total = len(ex) * a.epochs
    for ep in range(a.epochs):
        random.shuffle(ex)
        tot = 0.0
        for x, y in ex:
            out = model(input_ids=torch.tensor([x], device="cuda"), labels=torch.tensor([y], device="cuda"))
            if not math.isfinite(out.loss.item()):
                raise RuntimeError(f"non-finite loss at step {step}")
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            tot += out.loss.item()
            step += 1
            if step == 1 or step % 200 == 0:
                rate = step / max(time.time() - t0, 1e-6)
                print(f"step {step}/{total} {time.time()-t0:.0f}s eta {(total-step)/rate:.0f}s", flush=True)
        print(f"epoch {ep} loss {tot/len(ex):.3f}", flush=True)
    os.makedirs(od, exist_ok=True)
    if a.stage == "voice":
        model.save_pretrained(od, selected_adapters=["default"])
    else:
        model.save_pretrained(od)
    report = dict(
        base=a.base, stage=a.stage, facts=a.facts or None, q4=False, pairs=len(pairs), claims=len(claim_rows),
        examples=len(ex), epochs=a.epochs, lr=a.lr, r=a.r, targets=targets, steps=step,
        train_sec=round(time.time() - t0), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        prefix_mismatch_skipped=mismatch, limit=a.limit,
    )
    json.dump(report, open(f"{od}/train_report.json", "w"), indent=1)
    print("done", od, flush=True)

if __name__ == "__main__":
    main()
