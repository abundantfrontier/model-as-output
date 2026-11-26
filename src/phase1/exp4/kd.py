"""Feature-match KD is optional. Skip unless a same-tokenizer teacher fits in VRAM."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))

def main():
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    gb = total / 2**30
    parts = []
    same = None
    # bf16 Qwen3-14B weights are about 28 GB before activations. This GPU is 24 GB.
    if gb < 40:
        parts.append(
            f"GPU is {gb:.1f} GB. Qwen3-14B in bf16 is about 28 GB of weights, so mid-layer hidden-state "
            "MSE or CKA against that teacher does not fit beside a 4B student."
        )
    try:
        from transformers import AutoTokenizer
        t4 = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        t14 = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
        v4, v14 = t4.get_vocab(), t14.get_vocab()
        same = len(t4) == len(t14) and v4 == v14
        parts.append(
            "Qwen3-4B and Qwen3-14B tokenizers match." if same
            else "Qwen3-4B and Qwen3-14B tokenizers differ, so hidden states are not aligned."
        )
    except Exception as e:
        parts.append(f"Tokenizer compare did not finish ({type(e).__name__}: {e}).")
    parts.append("Phi-4-mini-instruct and SmolLM3-3B do not share the Qwen3 tokenizer, so KD to Qwen3-14B is not defined for the second base.")
    parts.append("Skipped feature-match KD.")
    report = {"skipped": True, "same_tokenizer": same, "gpu_gb": round(gb, 2), "reason": " ".join(parts)}
    json.dump(report, open(f"{H}/kd_report.json", "w"), indent=1)
    print(report["reason"], flush=True)

if __name__ == "__main__":
    main()
