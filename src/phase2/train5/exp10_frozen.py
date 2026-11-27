"""Frozen exp4–exp9 inputs. Read and copy only. This module never writes outside exp10."""
import hashlib, json, os

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
EXP4 = os.path.join(ROOT, "exp4")
EXP6 = os.path.join(ROOT, "exp6")
EXP7 = os.path.join(ROOT, "exp7")
EXP9 = os.path.join(ROOT, "exp9")
FACTS = os.path.join(EXP4, "lora_q4_facts")
BASE = "Qwen/Qwen3-4B"
BASE_REV = "1cfa9a7208912126459214e8b04321603b3df60c"
FACTS_SHA = "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b"
MERGED = os.path.join(EXP6, "merged_q4_facts")
PASS1_SHA = "b5d6d2130df4640072d3055b477ebd9a5d596f6ef12449a03ec13b107f846033"
P2_ADAPTER_SHA = "7980fad31a4a3023cc738662a8af6220d1c0722eeaf70d4b204f7368fd29dcef"
HARNESS = {
    "eval4.py": "f5bcde713c712c0efd2d439425900298ba0f031cd6c234f47448bf57b9cf7034",
    "judge.py": "b7b5042292b099899a2f9fb5b9264482703011ddba3ef7e209436d2f5b3ebebd",
    "common.py": "5d8cb1891a9e2f4a0c8601461a2d061cc115f80c8b0e59da60d5cdd6a6daca51",
    "chat.py": "5a3d8b85e6a94e97f4dc188772900e5bac911480c2678432dcb32241acff6887",
}
ADAPTER_SRC = os.path.join(EXP7, "lora_v_light_dpo")
ADAPTER_EXP9 = os.path.join(EXP9, "init", "lora_v_light_dpo")
PASS1_SRC = os.path.join(EXP6, "runs", "v_merged_v_rag.json")
PASS1_EXP9 = os.path.join(EXP9, "pass1", "v_merged_v_rag.json")
LOCKED = {
    "must_ok": 59,
    "must_n": 60,
    "marker": 1,
    "leak_n": 48,
    "voice": 4.215686274509804,
    "voice_round": 4.22,
    "frames": 3,
    "fallback": 72,
    "n": 204,
    "m16_ok": 2,
    "m16_n": 3,
    "m16_fail_sample": 1,
}
TOLERANCE = dict(must="exact", marker="exact", voice=0.03, frames=1)
VOICE_MIN = 4.2  # historical absolute bar from the original exp9 qwen score
MARKER_MAX = 2
FRAMES_LT = 6
RELATIVE_BAR = {
    "voice": "average of qwen3:14b and gemma3:12b, temp 0 seed 1, same session; must strictly beat v_merged_v_rag",
    "must": "60/60, or 59/60 if the only miss is M16",
    "report": "exp10/relative_bar.md",
    "session": "2026-11-10 11:27:17 AM PST",
    "ref_avg": 4.269607843137255,
    "locked_avg": 4.279411764705882,
    "delta": 0.009803921568627416,
    "verdict": "PASS",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def assert_under_exp10(path):
    ap = os.path.abspath(path)
    root = os.path.abspath(H)
    if ap != root and not ap.startswith(root + os.sep):
        raise SystemExit(f"refusing path outside exp10: {ap}")
    return ap


def check():
    got = sha256(os.path.join(FACTS, "adapter_model.safetensors"))
    if got != FACTS_SHA:
        raise SystemExit(f"facts adapter sha {got} != {FACTS_SHA}")
    for name, want in HARNESS.items():
        g = sha256(os.path.join(H, name))
        if g != want:
            raise SystemExit(f"sha mismatch {name} got {g} want {want}")
        src = os.path.join(EXP4, name)
        if sha256(src) != want:
            raise SystemExit(f"exp4 {name} drifted")
    mr = json.load(open(os.path.join(MERGED, "merge_report.json")))
    if mr.get("facts_sha256") != FACTS_SHA:
        raise SystemExit(f"merged facts sha {mr.get('facts_sha256')} != {FACTS_SHA}")
    if mr.get("base") != BASE:
        raise SystemExit(f"merged base {mr.get('base')} != {BASE}")
    weights = os.path.join(MERGED, "model.safetensors")
    if os.path.getsize(weights) < 1_000_000_000:
        raise SystemExit(f"merged weights look truncated: {weights}")
    cfg = json.load(open(os.path.join(MERGED, "config.json")))
    if cfg.get("model_type") != "qwen3":
        raise SystemExit(f"merged model_type {cfg.get('model_type')}")
    dest = os.path.join(H, "init", "lora_v_light_dpo", "adapter_model.safetensors")
    if not os.path.isfile(dest):
        raise SystemExit(f"missing init copy {dest}")
    got_a = sha256(dest)
    if got_a != P2_ADAPTER_SHA:
        raise SystemExit(f"pass-2 adapter sha {got_a} != {P2_ADAPTER_SHA}")
    if sha256(os.path.join(ADAPTER_SRC, "adapter_model.safetensors")) != P2_ADAPTER_SHA:
        raise SystemExit("exp7 lora_v_light_dpo sha drifted")
    if sha256(os.path.join(ADAPTER_EXP9, "adapter_model.safetensors")) != P2_ADAPTER_SHA:
        raise SystemExit("exp9 init/lora_v_light_dpo sha drifted")
    ac = json.load(open(os.path.join(H, "init", "lora_v_light_dpo", "adapter_config.json")))
    if os.path.realpath(ac.get("base_model_name_or_path", "")) != os.path.realpath(MERGED):
        raise SystemExit(f"adapter base {ac.get('base_model_name_or_path')} != merged")
    p1 = os.path.join(H, "pass1", "v_merged_v_rag.json")
    if not os.path.isfile(p1):
        raise SystemExit(f"missing pass1 copy {p1}")
    got_p = sha256(p1)
    if got_p != PASS1_SHA:
        raise SystemExit(f"pass1 sha {got_p} != {PASS1_SHA}")
    if sha256(PASS1_SRC) != PASS1_SHA:
        raise SystemExit("exp6 v_merged_v_rag.json sha drifted")
    if sha256(PASS1_EXP9) != PASS1_SHA:
        raise SystemExit("exp9 pass1/v_merged_v_rag.json sha drifted")
    d = json.load(open(p1))
    if len(d["results"]) != 204:
        raise SystemExit(f"pass1 has {len(d['results'])} rows, want 204")
    return got


if __name__ == "__main__":
    print(check())
