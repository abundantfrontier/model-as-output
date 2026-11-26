#!/usr/bin/env python3
"""Pass-2 rewrite eval. Pass 1 is a cached RAG run. No retrieval on the rewrite.

Adapted from exp9/eval_twopass.py. Scored text is pass-2 (unguarded).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import prompts  # noqa: E402
from chat import render  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_voice(base, facts, voice):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    if facts:
        model = PeftModel.from_pretrained(model, facts, adapter_name="facts")
        model.load_adapter(voice, adapter_name="voice")
        model.base_model.set_adapter(["facts", "voice"], inference_mode=True)
    else:
        model = PeftModel.from_pretrained(model, voice, adapter_name="voice")
        model.base_model.set_adapter(["voice"], inference_mode=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    return tok, model, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond")
    ap.add_argument("--pass1", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--facts", default="")
    ap.add_argument("--base", default=C.BASE)
    ap.add_argument("--merged-facts", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-penalty", action="store_true", help="drop repetition_penalty / no_repeat_ngram_size (smoke only)")
    a = ap.parse_args()
    doc = json.load(open(a.pass1))
    rows = doc["results"]
    if a.limit:
        rows = rows[: a.limit]
    base = a.merged_facts or a.base
    facts = "" if a.merged_facts else a.facts
    C.unload_ollama()
    tok, model, torch = load_voice(base, facts, a.voice)

    def gen(sysm, user):
        prompts.assert_no_retrieval(sysm, user)
        ids = render(tok, [{"role": "system", "content": sysm}, {"role": "user", "content": user}], True)
        p = {"input_ids": torch.tensor([ids], device="cuda")}
        p["attention_mask"] = torch.ones_like(p["input_ids"])
        t0 = time.time()
        with torch.no_grad():
            o = model.generate(
                **p,
                **C.generate_kwargs(penalty=not a.no_penalty, pad_token_id=tok.pad_token_id),
            )
        torch.cuda.synchronize()
        n = o.shape[1] - p["input_ids"].shape[1]
        return tok.decode(o[0][p["input_ids"].shape[1] :], skip_special_tokens=True).strip(), int(n), time.time() - t0

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    out_rows = []
    t0 = time.time()
    for src in rows:
        r = dict(src)
        r["pass1"] = src["answer"]
        r["pass1_toks"] = src.get("toks") or 0
        r["pass1_sec"] = src.get("sec") or 0
        if src.get("locked"):
            r["pass2"] = src["answer"]
            r["answer"] = src["answer"]
            r["toks"] = 0
            r["sec"] = 0
            out_rows.append(r)
            continue
        sysm = C.sysprompt(src["char"], int(src["reader_pos"]))
        user = prompts.rewrite_user(src["probe"], src["answer"])
        ans, nt, sec = gen(sysm, user)
        r["pass2"] = ans
        r["answer"] = ans
        r["toks"] = nt
        r["sec"] = round(sec, 3)
        probe = {"fail_if_mentions": src.get("fail_if") or []}
        r["fail"] = C.fail_hits_probe(ans, probe)
        r["voice"] = C.voice_flags(ans)
        if src["kind"] == "must":
            r["correct"] = C.must_correct(src["id"], ans)
        out_rows.append(r)
        print(a.cond, src["id"], src["char"], src.get("sample"), ans[:100].replace("\n", " "), flush=True)
    meta = dict(doc.get("meta") or {})
    meta.update(
        cond=a.cond,
        parent=doc.get("meta", {}).get("cond"),
        pass1=os.path.abspath(a.pass1),
        voice=a.voice,
        facts=facts or a.merged_facts,
        two_pass=True,
        rewrite=prompts.REWRITE_INSTR,
        repetition_penalty=None if a.no_penalty else 1.15,
        no_repeat_ngram_size=None if a.no_penalty else 3,
        eos_token_id=[151645, 151643],
        wall=round(time.time() - t0),
        gpu_after_mb=C.gpu_used_mb(),
        torch_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        n_results=len(out_rows),
    )
    out = a.out or str(C.TRAIN / "runs" / f"{a.cond}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=meta, results=out_rows), open(out, "w"), indent=1)
    print("META", json.dumps({k: meta[k] for k in list(meta)[:12]}), flush=True)


if __name__ == "__main__":
    main()
