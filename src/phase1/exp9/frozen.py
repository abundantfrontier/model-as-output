"""Frozen exp4–exp8 inputs. Read and copy only. This module never writes outside exp9."""
import hashlib, json, os

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
EXP4 = os.path.join(ROOT, "exp4")
EXP5 = os.path.join(ROOT, "exp5")
EXP6 = os.path.join(ROOT, "exp6")
EXP7 = os.path.join(ROOT, "exp7")
EXP8 = os.path.join(ROOT, "exp8")
FACTS = os.path.join(EXP4, "lora_q4_facts")
VOICE_JSONL = os.path.join(EXP4, "data", "voice.jsonl")
FACTS_JSONL = os.path.join(EXP4, "data", "facts.jsonl")
BASE = "Qwen/Qwen3-4B"
BASE_REV = "1cfa9a7208912126459214e8b04321603b3df60c"
FACTS_SHA = "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b"
VOICE_SHA = "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f"
MERGED = os.path.join(EXP6, "merged_q4_facts")
HARNESS = {
    "eval4.py": "f5bcde713c712c0efd2d439425900298ba0f031cd6c234f47448bf57b9cf7034",
    "judge.py": "b7b5042292b099899a2f9fb5b9264482703011ddba3ef7e209436d2f5b3ebebd",
    "common.py": "5d8cb1891a9e2f4a0c8601461a2d061cc115f80c8b0e59da60d5cdd6a6daca51",
    "chat.py": "5a3d8b85e6a94e97f4dc188772900e5bac911480c2678432dcb32241acff6887",
}
ADAPTER_SRC = {
    "lora_v_fb_dpo": os.path.join(EXP8, "lora_v_fb_dpo"),
    "lora_v_light_contrast": os.path.join(EXP7, "lora_v_light_contrast"),
    "lora_v_light_dpo": os.path.join(EXP7, "lora_v_light_dpo"),
    "lora_v_merged": os.path.join(EXP6, "lora_v_merged"),
}
PASS1_SRC = {
    "v_merged_v_rag": os.path.join(EXP6, "runs", "v_merged_v_rag.json"),
    "q4_merged_rag": os.path.join(EXP8, "runs", "q4_merged_rag.json"),
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)

def _probe_keys(path):
    d = json.load(open(path))
    return [(r["kind"], r["id"], r["char"], r["probe"], r["sample"]) for r in d["results"]], d

def check():
    got = sha256(os.path.join(FACTS, "adapter_model.safetensors"))
    if got != FACTS_SHA:
        raise SystemExit(f"facts adapter sha {got} != {FACTS_SHA}")
    v = sha256(VOICE_JSONL)
    if v != VOICE_SHA:
        raise SystemExit(f"voice.jsonl sha {v} != {VOICE_SHA}")
    for name, want in HARNESS.items():
        g = sha256(os.path.join(H, name))
        if g != want:
            raise SystemExit(f"sha mismatch {name} got {g} want {want}")
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
    for name, src in ADAPTER_SRC.items():
        dest = os.path.join(H, "init", name, "adapter_model.safetensors")
        src_w = os.path.join(src, "adapter_model.safetensors")
        if not os.path.isfile(dest):
            raise SystemExit(f"missing init copy {dest}")
        if sha256(dest) != sha256(src_w):
            raise SystemExit(f"init adapter sha mismatch {name}")
        ac = json.load(open(os.path.join(H, "init", name, "adapter_config.json")))
        if os.path.realpath(ac.get("base_model_name_or_path", "")) != os.path.realpath(MERGED):
            raise SystemExit(f"{name} base {ac.get('base_model_name_or_path')} != merged")
    keys = None
    for name in PASS1_SRC:
        path = os.path.join(H, "pass1", name + ".json")
        if not os.path.isfile(path):
            raise SystemExit(f"missing pass1 copy {path}")
        k, d = _probe_keys(path)
        if len(k) != 204:
            raise SystemExit(f"{path} has {len(k)} rows, want 204")
        if keys is None:
            keys = k
        elif k != keys:
            raise SystemExit(f"pass1 probe mismatch {name}")
        if sha256(path) != sha256(PASS1_SRC[name]):
            raise SystemExit(f"pass1 copy sha mismatch {name}")
    return got

if __name__ == "__main__":
    print(check())
