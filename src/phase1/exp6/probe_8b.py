"""VRAM probe for optional Qwen3-8B. Writes data/8b_probe.json. Never downloads."""
import os, json, gc, traceback
import frozen
import gpuutil

H = os.path.dirname(os.path.abspath(__file__))

def cache_has(name):
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    slug = "models--" + name.replace("/", "--")
    return os.path.isdir(os.path.join(hub, slug))

def try_load(q4):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(device_map="cuda")
    if q4:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        kw["dtype"] = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(frozen.BASE8)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(frozen.BASE8, **kw)
    peak = torch.cuda.max_memory_allocated() / 2**30
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 2**30
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return peak, free_gb

def main():
    os.makedirs(os.path.join(H, "data"), exist_ok=True)
    out = os.path.join(H, "data", "8b_probe.json")
    if os.path.exists(out):
        print("skip", open(out).read(), flush=True)
        return
    gpuutil.unload_ollama()
    if not cache_has(frozen.BASE8):
        doc = dict(ok=False, q4=False, reason=f"{frozen.BASE8} not in HF cache; will not download")
        json.dump(doc, open(out, "w"), indent=1)
        json.dump(doc, open(os.path.join(H, "data", "8b_skipped.json"), "w"), indent=1)
        print(json.dumps(doc), flush=True)
        return
    try:
        peak, free = try_load(False)
        # Need headroom for LoRA + activations + RAG eval (~5 GB).
        if peak <= 18.5 and free >= 4.5:
            doc = dict(ok=True, q4=False, peak_gb=round(peak, 2), free_after_gb=round(free, 2),
                       reason="bf16 8B fits with headroom for LoRA train/eval")
            json.dump(doc, open(out, "w"), indent=1)
            print(json.dumps(doc), flush=True)
            return
        print("bf16 tight", peak, free, "trying 4-bit", flush=True)
    except Exception:
        print("bf16 load failed", traceback.format_exc(), flush=True)
    try:
        peak, free = try_load(True)
        if peak <= 12 and free >= 6:
            doc = dict(ok=True, q4=True, peak_gb=round(peak, 2), free_after_gb=round(free, 2),
                       reason="bf16 too tight; 4-bit 8B fits")
            json.dump(doc, open(out, "w"), indent=1)
            print(json.dumps(doc), flush=True)
            return
        doc = dict(ok=False, q4=True, peak_gb=round(peak, 2), free_after_gb=round(free, 2),
                   reason="8B 4-bit still too tight for RAG eval + adapters")
    except Exception as e:
        doc = dict(ok=False, q4=True, reason="8B 4-bit load failed: " + type(e).__name__ + " " + str(e)[:300])
    json.dump(doc, open(out, "w"), indent=1)
    json.dump(doc, open(os.path.join(H, "data", "8b_skipped.json"), "w"), indent=1)
    print(json.dumps(doc), flush=True)

if __name__ == "__main__":
    main()
