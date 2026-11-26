"""Frozen exp6 inputs and the byte-for-byte exp4 harness. Read and copy only."""
import hashlib, json, os

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
FACTS = os.path.join(ROOT, "exp4", "lora_q4_facts")
VOICE_JSONL = os.path.join(ROOT, "exp4", "data", "voice.jsonl")
FACTS_JSONL = os.path.join(ROOT, "exp4", "data", "facts.jsonl")
CLAIMS_JSONL = os.path.join(ROOT, "exp4", "data", "claims.jsonl")
EXP4_RUNS = os.path.join(ROOT, "exp4", "runs")
EXP5 = os.path.join(ROOT, "exp5")
EXP5_RUNS = os.path.join(EXP5, "runs")
EXP5_DATA = os.path.join(EXP5, "data")
EXP6 = os.path.join(ROOT, "exp6")
EXP6_RUNS = os.path.join(EXP6, "runs")
EXP6_DATA = os.path.join(EXP6, "data")
BASE = "Qwen/Qwen3-4B"
BASE_REV = "1cfa9a7208912126459214e8b04321603b3df60c"
FACTS_SHA = "ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b"
VOICE_SHA = "12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f"
FACTS_JSONL_SHA = "6fe8bf6bce107dcafeed79d841504ded41278e1917cfaf432899f2c93a204f74"
HARNESS = {
    "eval4.py": "f5bcde713c712c0efd2d439425900298ba0f031cd6c234f47448bf57b9cf7034",
    "judge.py": "b7b5042292b099899a2f9fb5b9264482703011ddba3ef7e209436d2f5b3ebebd",
    "common.py": "5d8cb1891a9e2f4a0c8601461a2d061cc115f80c8b0e59da60d5cdd6a6daca51",
    "chat.py": "5a3d8b85e6a94e97f4dc188772900e5bac911480c2678432dcb32241acff6887",
}
NARR = ("You are the narrator of the story No Return. Answer the reader's question about what has happened so far "
        "in plain, spare prose, 1-4 sentences. Never reveal or hint at what has not happened yet.")
MERGED = os.path.join(EXP6, "merged_q4_facts")
INIT_SRC = os.path.join(EXP6, "lora_v_merged")
INIT = os.path.join(H, "init", "lora_v_merged")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def check():
    got = sha256(os.path.join(FACTS, "adapter_model.safetensors"))
    if got != FACTS_SHA:
        raise SystemExit(f"facts adapter sha {got} != {FACTS_SHA}")
    v = sha256(VOICE_JSONL)
    if v != VOICE_SHA:
        raise SystemExit(f"voice.jsonl sha {v} != {VOICE_SHA}")
    f = sha256(FACTS_JSONL)
    if f != FACTS_JSONL_SHA:
        raise SystemExit(f"facts.jsonl sha {f} != {FACTS_JSONL_SHA}")
    for name, want in HARNESS.items():
        g = sha256(os.path.join(H, name))
        if g != want:
            raise SystemExit(f"sha mismatch {name} got {g} want {want}")
    mr_path = os.path.join(MERGED, "merge_report.json")
    if not os.path.isfile(mr_path):
        raise SystemExit(f"missing merge report {mr_path}")
    mr = json.load(open(mr_path))
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
    init_w = os.path.join(INIT, "adapter_model.safetensors")
    src_w = os.path.join(INIT_SRC, "adapter_model.safetensors")
    if not os.path.isfile(init_w):
        raise SystemExit(f"missing init copy {init_w}")
    ih, sh = sha256(init_w), sha256(src_w)
    if ih != sh:
        raise SystemExit(f"init adapter sha {ih} != source {sh}")
    return got

if __name__ == "__main__":
    print(check())
