"""Two-pass eval. Pass 1 is a cached RAG run. Pass 2 rewrites it with a voice LoRA.

eval4.py is not used and is not edited. It always generates in one pass and always
attaches a Peft adapter. This wrapper writes the same {meta, results} schema so
judge.py can score the rewrite (result['answer'] is the pass-2 text).

No retrieval on the rewrite. common.retrieve / rag_sysprompt are not called.
Do not hold a pass-1 model: pass 1 is read from disk.
"""
import os, sys, json, time, argparse, subprocess, re, hashlib
import common as C
from chat import render
import frozen
import prompts

H = frozen.H

# Voice adapters copied under exp9/init. Pass-1 files live in exp9/pass1.
ADAPTERS = {
    "p2_fb_dpo": "init/lora_v_fb_dpo",
    "p2_light_contrast": "init/lora_v_light_contrast",
    "p2_light_dpo": "init/lora_v_light_dpo",
    "p2_v_merged": "init/lora_v_merged",
    "p2_rewrite": "lora_v_rewrite",
}
PASS1 = {
    "primary": os.path.join(H, "pass1", "v_merged_v_rag.json"),
    "q4merged": os.path.join(H, "pass1", "q4_merged_rag.json"),
}

def gpu_used():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(out.stdout.split()[0])

def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def resolve(cond):
    """Map a condition name to (voice_dir, pass1_path, pass1_name)."""
    name = cond
    pass1_name = "primary"
    if name.endswith("_q4merged"):
        pass1_name = "q4merged"
        name = name[: -len("_q4merged")]
    m = re.fullmatch(r"p2_rewrite_c(\d+)", name)
    if m:
        voice = os.path.join(H, "lora_v_rewrite", f"checkpoint-{m.group(1)}")
    elif name in ADAPTERS:
        voice = os.path.join(H, ADAPTERS[name])
    else:
        raise SystemExit(f"unknown cond {cond}")
    if not os.path.isfile(os.path.join(voice, "adapter_config.json")):
        raise SystemExit(f"missing voice adapter {voice}")
    pass1 = PASS1[pass1_name]
    if not os.path.isfile(pass1):
        raise SystemExit(f"missing pass1 {pass1}")
    return voice, pass1, pass1_name

def group_rows(results):
    order, bucket = [], {}
    for r in results:
        key = (r["kind"], r["id"], r["char"], r["probe"])
        if key not in bucket:
            order.append(key)
            bucket[key] = []
        bucket[key].append(r)
    for key in order:
        bucket[key].sort(key=lambda r: r["sample"])
    return order, bucket

def select_rows(doc, limit, include_must):
    order, bucket = group_rows(doc["results"])
    n = doc["meta"].get("n", 3)
    for key in order:
        if len(bucket[key]) != n:
            raise SystemExit(f"probe {key} has {len(bucket[key])} samples, want {n}")
    if limit:
        leak = [k for k in order if k[0] == "leak"][:limit]
        must = [k for k in order if k[0] == "must"][:include_must]
        # Must probes sit after the leak block. include_must keeps the rubric path
        # on a short smoke slice that would otherwise stop inside the leak block.
        chosen = leak + must
    else:
        chosen = order
    rows = []
    for key in chosen:
        rows.extend(bucket[key])
    return rows, n

def load_voice(voice_dir):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    cfg = json.load(open(os.path.join(voice_dir, "adapter_config.json")))
    if os.path.realpath(cfg.get("base_model_name_or_path", "")) != os.path.realpath(frozen.MERGED):
        raise SystemExit(f"adapter base {cfg.get('base_model_name_or_path')} != {frozen.MERGED}")
    tok = AutoTokenizer.from_pretrained(frozen.MERGED, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        frozen.MERGED, dtype=torch.bfloat16, device_map="cuda", local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, voice_dir, adapter_name="voice")
    model.base_model.set_adapter(["voice"], inference_mode=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    return tok, model, cfg

def generate(tok, model, sysm, user):
    import torch
    prompts.assert_no_retrieval(sysm, prompts.REWRITE_INSTR)
    ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": user}], True)
    p = {"input_ids": torch.tensor([ids], device="cuda")}
    p["attention_mask"] = torch.ones_like(p["input_ids"])
    t0 = time.time()
    with torch.no_grad():
        o = model.generate(**p, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95)
    torch.cuda.synchronize()
    n = o.shape[1] - p["input_ids"].shape[1]
    ans = tok.decode(o[0][p["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return ans, int(n), time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N leak probes (0 = all)")
    ap.add_argument("--include-must", type=int, default=0, help="smoke: also take this many must probes")
    ap.add_argument("--dry-prompt", action="store_true")
    a = ap.parse_args()
    voice, pass1_path, pass1_name = resolve(a.cond)
    doc = json.load(open(pass1_path))
    rows, n_file = select_rows(doc, a.limit, a.include_must)
    if a.n != n_file:
        raise SystemExit(f"--n {a.n} does not match pass1 n {n_file}")
    if not a.limit and len(rows) != 204:
        raise SystemExit(f"full run has {len(rows)} rows, want 204")
    sys0 = prompts.system_for(rows[0]["char"])
    user0 = prompts.rewrite_user(rows[0]["probe"], rows[0]["answer"])
    prompts.assert_no_retrieval(sys0, prompts.REWRITE_INSTR)
    if a.dry_prompt:
        print("COND", a.cond)
        print("VOICE", voice)
        print("PASS1", pass1_path, pass1_name)
        print("ROWS", len(rows), "must", sum(r["kind"] == "must" for r in rows), "leak", sum(r["kind"] == "leak" for r in rows))
        print("SYSTEM_HEAD", sys0.split("\n", 1)[0])
        print("USER_TAIL", user0.split("\n")[-1])
        print("RETRIEVAL", False)
        print("DRY_OK")
        return
    import torch
    print("loading", frozen.MERGED, voice, flush=True)
    tok, model, cfg = load_voice(voice)
    print("loaded", voice, flush=True)
    meta = dict(
        cond=a.cond, n=n_file, gpu_before_mb=gpu_used(), base=frozen.MERGED,
        rag=False, adapters="v", two_pass=True, retrieval_in_rewrite=False,
        pass1=pass1_path, pass1_name=pass1_name, pass1_cond=doc["meta"].get("cond"),
        voice_adapter=voice, voice_r=cfg.get("r"), voice_alpha=cfg.get("lora_alpha"),
        rewrite_instruction=prompts.REWRITE_INSTR,
        rewrite_system="common.sysprompt(char) day=3; narrator uses NARR. Not rag_sysprompt.",
        rewrite_user_shape="probe, blank line, pass-1 answer, blank line, rewrite instruction",
        generation="max_new_tokens=200 temperature=0.7 top_p=0.95 do_sample=True",
        eval_sha=file_sha(os.path.abspath(__file__)),
        limit=a.limit, include_must=a.include_must,
    )
    res = []
    t0 = time.time()
    try:
        for it in rows:
            sysm = prompts.system_for(it["char"], 3)
            user = prompts.rewrite_user(it["probe"], it["answer"])
            prompts.assert_no_retrieval(sysm, prompts.REWRITE_INSTR)
            ans, nt, sec = generate(tok, model, sysm, user)
            r = dict(it)
            for k in ("judge_leak", "judge_ev", "voice_score"):
                r.pop(k, None)
            r["pass1"] = it["answer"]
            r["pass1_toks"] = it.get("toks")
            r["pass1_sec"] = it.get("sec")
            r["pass1_fail"] = list(it.get("fail") or [])
            r["pass1_voice"] = dict(it.get("voice") or {})
            if "correct" in it:
                r["pass1_correct"] = it["correct"]
            r["pass2"] = ans
            r["answer"] = ans
            r["toks"] = nt
            r["sec"] = round(sec, 3)
            r["fail"] = C.fail_hits(ans)
            r["voice"] = C.voice_flags(ans)
            if it["kind"] == "must":
                r["correct"] = bool(re.search(it["rubric"], ans, re.I))
            res.append(r)
            print(a.cond, it["id"], it["char"], it["sample"], "MARK" if r["fail"] else "", ans[:100].replace("\n", " "), flush=True)
    finally:
        del model
        del tok
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    meta.update(wall=round(time.time() - t0), gpu_after_mb=gpu_used())
    meta["torch_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if any(r["answer"] != r["pass2"] or not isinstance(r["pass1"], str) for r in res):
        raise SystemExit("pass1/pass2 bookkeeping failed")
    out_dir = os.path.join(H, "smoke" if a.limit else "runs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, a.cond + ".json")
    frozen.dump_json(out, dict(meta=meta, results=res))
    print("META", json.dumps({k: meta[k] for k in ("cond", "wall", "torch_peak_gb", "gpu_after_mb", "retrieval_in_rewrite")}), flush=True)
    print("WROTE", out, "rows", len(res), flush=True)

if __name__ == "__main__":
    main()
