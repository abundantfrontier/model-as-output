"""Point an unchanged eval4.py at an exp7 voice adapter on the merged facts base.

Condition names end in _v or _v_rag. A checkpoint is v_light_dpo_c40_v_rag
(adapter lora_v_light_dpo/checkpoint-40). eval4.py / judge.py / common.py / chat.py
are byte-for-byte copies of exp4.
"""
import os, re, sys
import frozen

H = os.path.dirname(os.path.abspath(__file__))

def parse_core(cond):
    rag = cond.endswith("_rag")
    core = cond[:-4] if rag else cond
    if not core.endswith("_v"):
        raise SystemExit(f"cond must end in _v or _v_rag, got {cond}")
    return core[:-2], rag

def voice_for(name):
    m = re.fullmatch(r"(.+)_c(\d+)", name)
    if m:
        path = os.path.join(H, f"lora_{m.group(1)}", f"checkpoint-{m.group(2)}")
    else:
        path = os.path.join(H, f"lora_{name}")
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        raise SystemExit(f"missing voice adapter {path}")
    return path

def build_argv(cond, rest):
    name, rag = parse_core(cond)
    if not os.path.isdir(frozen.MERGED):
        raise SystemExit(f"missing merged base {frozen.MERGED}")
    argv = ["eval4.py", cond, "--base", frozen.MERGED, "--voice", voice_for(name), "--adapters", "v"]
    if rag:
        argv.append("--rag")
    argv.extend(rest)
    return argv

def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: eval_cond.py COND [--n 3] [--limit N]")
    sys.argv = build_argv(sys.argv[1], sys.argv[2:])
    import eval4
    eval4.main()

if __name__ == "__main__":
    main()
