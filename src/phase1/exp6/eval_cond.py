"""Thin wrapper around unchanged eval4.py.

Condition names:
  v_replay_fv[_rag]              stacked facts + voice on Qwen3-4B
  v_merged_v[_rag]               voice on merged_q4_facts
  v_combo_contrast_{fv|v}[_rag]  combo contrast adapter
  v_combo_dpo_{fv|v}[_rag]       combo DPO adapter
  v8_*                           same recipe on Qwen3-8B

eval4.py / judge.py / common.py / chat.py are byte-for-byte copies.
"""
import os, sys, json
import frozen

H = os.path.dirname(os.path.abspath(__file__))

def load_json(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))

def winner():
    return load_json(os.path.join(H, "data", "winner.json")) or {}

def probe8():
    return load_json(os.path.join(H, "data", "8b_probe.json")) or {}

def voice_dir(name):
    return os.path.join(H, f"lora_{name}")

def parse_core(cond):
    rag = cond.endswith("_rag")
    core = cond[:-4] if rag else cond
    if core.endswith("_fv"):
        return core[:-3], "fv", rag
    if core.endswith("_v"):
        return core[:-2], "v", rag
    raise SystemExit(f"cond must end in _fv, _fv_rag, _v, or _v_rag, got {cond}")

def spec_for(cond):
    name, kind, rag = parse_core(cond)
    eight = name.startswith("v8_")
    key = name[3:] if eight else name  # replay | merged | combo_contrast | combo_dpo | v_replay | ...
    if key.startswith("v_"):
        key = key[2:]
    q4 = bool(probe8().get("q4")) if eight else False
    base4_or_8 = frozen.BASE8 if eight else frozen.BASE
    facts4_or_8 = os.path.join(H, "lora_q8_facts") if eight else frozen.FACTS
    if key == "replay":
        if kind != "fv":
            raise SystemExit(f"{cond} must use _fv")
        return dict(base=base4_or_8, facts=facts4_or_8, voice=voice_dir("v8_replay" if eight else "v_replay"),
                    adapters="fv", rag=rag, q4=q4)
    if key == "merged":
        if kind != "v":
            raise SystemExit(f"{cond} must use _v")
        return dict(base=frozen.MERGED8 if eight else frozen.MERGED, facts="",
                    voice=voice_dir("v8_merged" if eight else "v_merged"), adapters="v", rag=rag, q4=q4)
    if key in ("combo_contrast", "combo_dpo"):
        w = winner()
        mode = w.get("mode")
        if not mode:
            raise SystemExit("data/winner.json missing; pick_winner.py has not run")
        if kind != mode:
            raise SystemExit(f"{cond} kind {kind} != winner mode {mode}")
        lora = "v8_combo_" + key.split("_", 1)[1] if eight else "v_combo_" + key.split("_", 1)[1]
        if mode == "fv":
            return dict(base=base4_or_8, facts=facts4_or_8, voice=voice_dir(lora), adapters="fv", rag=rag, q4=q4)
        return dict(base=frozen.MERGED8 if eight else frozen.MERGED, facts="", voice=voice_dir(lora),
                    adapters="v", rag=rag, q4=q4)
    raise SystemExit(f"unknown cond {cond}")

def build_argv(cond, rest):
    spec = spec_for(cond)
    voice = spec["voice"]
    if not os.path.isfile(os.path.join(voice, "adapter_config.json")):
        raise SystemExit(f"missing voice adapter {voice}")
    if spec["adapters"] == "fv":
        if not os.path.isfile(os.path.join(spec["facts"], "adapter_config.json")):
            raise SystemExit(f"missing facts adapter {spec['facts']}")
    else:
        if not os.path.isdir(spec["base"]):
            # HF hub id is fine
            if "/" in spec["base"] and not os.path.isdir(spec["base"]):
                pass
            elif not os.path.isdir(spec["base"]):
                raise SystemExit(f"missing merged base {spec['base']}")
    argv = ["eval4.py", cond, "--base", spec["base"], "--voice", voice, "--adapters", spec["adapters"]]
    if spec["facts"]:
        argv += ["--facts", spec["facts"]]
    if spec["rag"]:
        argv.append("--rag")
    argv.extend(rest)
    return argv, spec

def load_local_q4(spec):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(spec["base"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(spec["base"], quantization_config=bnb, device_map="cuda")
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
    if len(sys.argv) < 2:
        raise SystemExit("usage: eval_cond.py COND [--n 3] [--limit N]")
    argv, spec = build_argv(sys.argv[1], sys.argv[2:])
    sys.argv = argv
    import eval4
    if spec.get("q4"):
        eval4.load_local = load_local_q4
    eval4.main()

if __name__ == "__main__":
    main()
