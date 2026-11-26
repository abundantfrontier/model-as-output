"""Merge exp4/lora_q4_facts into Qwen3-4B. Writes exp6/merged_q4_facts/. Does not touch the HF cache."""
import os, json, argparse, gc, traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import frozen
import gpuutil

def merged_ok(out):
    return os.path.isfile(os.path.join(out, "config.json")) and (
        os.path.isfile(os.path.join(out, "model.safetensors"))
        or os.path.isfile(os.path.join(out, "model.safetensors.index.json"))
        or any(n.startswith("model-") and n.endswith(".safetensors") for n in os.listdir(out))
    )

def run(base, facts, out, q4=False):
    if merged_ok(out):
        print("merge skip", out, flush=True)
        return
    frozen.check()
    gpuutil.unload_ollama()
    os.makedirs(out, exist_ok=True)
    print("loading", base, "facts", facts, "q4", q4, flush=True)
    tok = AutoTokenizer.from_pretrained(base)
    if q4:
        raise RuntimeError("refusing to merge a 4-bit model; use bf16 merge")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, facts, adapter_name="facts")
    print("merging", flush=True)
    model = model.merge_and_unload()
    print("saving", out, flush=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(out)
    report = dict(
        base=base, facts=facts, facts_sha256=frozen.FACTS_SHA, out=os.path.abspath(out),
        q4=False, dtype="bfloat16",
        n_params=sum(p.numel() for p in model.parameters()),
    )
    json.dump(report, open(os.path.join(out, "merge_report.json"), "w"), indent=1)
    print("done", json.dumps(report), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if not merged_ok(out):
        raise RuntimeError(f"merge did not write weights under {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=frozen.BASE)
    ap.add_argument("--facts", default=frozen.FACTS)
    ap.add_argument("--out", default=frozen.MERGED)
    a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    try:
        run(a.base, a.facts, out)
    except Exception:
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "FAIL.txt"), "w").write(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
