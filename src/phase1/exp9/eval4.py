"""python eval4.py COND [--n 3]

COND: q4_fv, q4_fv_rag, q4_v_rag, and the same with p4 (Phi-4-mini) or s3 (SmolLM3).
q14_rag uses Ollama, same harness as exp3. Reader position is chapter 3.
"""
import sys, os, json, time, argparse, subprocess, re
import common as C
from chat import render

H = os.path.dirname(os.path.abspath(__file__))
BASES = {
    "q4": ("Qwen/Qwen3-4B", "lora_q4_facts", "lora_q4_voice"),
    "p4": ("microsoft/Phi-4-mini-instruct", "lora_p4_facts", "lora_p4_voice"),
    "s3": ("HuggingFaceTB/SmolLM3-3B", "lora_s3_facts", "lora_s3_voice"),
}

def gpu_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.split()[0])

def parse_cond(cond):
    if cond == "q14_rag":
        return dict(ollama=True, rag=True, base="qwen3:14b")
    rag = cond.endswith("_rag")
    core = cond[:-4] if rag else cond
    tag, kind = core.rsplit("_", 1)
    if tag not in BASES or kind not in ("fv", "v"):
        raise SystemExit(f"unknown cond {cond}")
    base, facts, voice = BASES[tag]
    return dict(ollama=False, rag=rag, kind=kind, base=base, facts=f"{H}/{facts}", voice=f"{H}/{voice}")

def load_local(spec):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(spec["base"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["base"], dtype=torch.bfloat16, device_map="cuda")
    if spec["kind"] == "fv":
        model = PeftModel.from_pretrained(model, spec["facts"], adapter_name="facts")
        model.load_adapter(spec["voice"], adapter_name="voice")
        model.base_model.set_adapter(["facts", "voice"], inference_mode=True)
    else:
        model = PeftModel.from_pretrained(model, spec["voice"], adapter_name="voice")
        model.base_model.set_adapter(["voice"], inference_mode=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    return tok, model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base", default="")
    ap.add_argument("--facts", default="")
    ap.add_argument("--voice", default="")
    ap.add_argument("--adapters", default="fv", choices=["fv", "v"])
    ap.add_argument("--rag", action="store_true")
    a = ap.parse_args()
    items = [dict(kind="leak", **t) for t in C.parse_tests()] + [
        dict(kind="must", id=f"M{i+1}", set="M", char=c, probe=q, rubric=r) for i, (c, q, r) in enumerate(C.MUST)
    ]
    if a.limit:
        items = items[:a.limit]
    if a.base:
        spec = dict(ollama=False, rag=a.rag or a.cond.endswith("_rag"), kind=a.adapters, base=a.base, facts=a.facts, voice=a.voice)
    else:
        spec = parse_cond(a.cond)
    meta = dict(cond=a.cond, n=a.n, gpu_before_mb=gpu_used(), base=spec.get("base"), rag=spec["rag"], adapters=spec.get("kind"))
    if spec["ollama"]:
        def gen(sysm, q):
            ans, nt, dur, tot = C.ollama_chat("qwen3:14b", [{"role": "system", "content": sysm}, {"role": "user", "content": q}])
            return ans, nt, tot
        gen("hi", "hi")
    else:
        import torch
        tok, model = load_local(spec)
        def gen(sysm, q):
            ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": q}], True)
            p = {"input_ids": torch.tensor([ids], device="cuda")}
            p["attention_mask"] = torch.ones_like(p["input_ids"])
            t0 = time.time()
            with torch.no_grad():
                o = model.generate(**p, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95)
            torch.cuda.synchronize()
            n = o.shape[1] - p["input_ids"].shape[1]
            return tok.decode(o[0][p["input_ids"].shape[1]:], skip_special_tokens=True).strip(), int(n), time.time() - t0
    res = []
    t0 = time.time()
    for it in items:
        sysm = C.rag_sysprompt(it["char"], it["probe"]) if spec["rag"] else C.sysprompt(it["char"])
        for s in range(a.n):
            ans, nt, sec = gen(sysm, it["probe"])
            r = dict(it, sample=s, answer=ans, toks=nt, sec=round(sec, 3), fail=C.fail_hits(ans), voice=C.voice_flags(ans))
            if it["kind"] == "must":
                r["correct"] = bool(re.search(it["rubric"], ans, re.I))
            res.append(r)
            print(a.cond, it["id"], it["char"], s, "FAIL" if r["fail"] else "", ans[:100].replace("\n", " "), flush=True)
    meta.update(wall=round(time.time() - t0), gpu_after_mb=gpu_used())
    if spec["ollama"]:
        meta["ollama_ps"] = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    else:
        meta["torch_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    out = f"{H}/runs/{a.cond}.json" if not a.limit else "/tmp/exp4_eval_smoke.json"
    if not a.limit:
        os.makedirs(f"{H}/runs", exist_ok=True)
    json.dump(dict(meta=meta, results=res), open(out, "w"), indent=1)
    print("META", json.dumps(meta))

if __name__ == "__main__":
    main()
