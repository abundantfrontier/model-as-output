"""Write exp10/REPRODUCE-notes.md: extra command and hash detail."""
import os, json, datetime
import frozen, prompts

H = frozen.H


def main():
    hp = open(os.path.join(H, "hparams.json")).read().rstrip()
    tol = None
    tp = os.path.join(H, "runs", "tolerance.json")
    if os.path.isfile(tp):
        tol = json.load(open(tp))
    hashes = []
    for rel, path in [
        ("pass1/v_merged_v_rag.json", os.path.join(H, "pass1", "v_merged_v_rag.json")),
        ("init/lora_v_light_dpo/adapter_model.safetensors", os.path.join(H, "init", "lora_v_light_dpo", "adapter_model.safetensors")),
        ("locked/p2_light_dpo.json", os.path.join(H, "locked", "p2_light_dpo.json")),
        ("locked/p2_light_dpo.factdiff.json", os.path.join(H, "locked", "p2_light_dpo.factdiff.json")),
        ("locked/p2_light_dpo_guarded.json", os.path.join(H, "locked", "p2_light_dpo_guarded.json")),
        ("locked/p2_light_dpo_guarded.judged.json", os.path.join(H, "locked", "p2_light_dpo_guarded.judged.json")),
        ("runs/p2_light_dpo_guarded_orig.json", os.path.join(H, "runs", "p2_light_dpo_guarded_orig.json")),
        ("prompts.py", os.path.join(H, "prompts.py")),
        ("eval_twopass.py", os.path.join(H, "eval_twopass.py")),
        ("factdiff.py", os.path.join(H, "factdiff.py")),
        ("guard.py", os.path.join(H, "guard.py")),
        ("judge_wrap.py", os.path.join(H, "judge_wrap.py")),
    ]:
        if os.path.isfile(path):
            hashes.append(f"{frozen.sha256(path)}  {rel}")
    extra = []
    for cond in ("p2_light_dpo", "p2_light_dpo_guarded", "p2_light_dpo_guarded_orig"):
        for suf in (".json", ".judged.json", ".factdiff.json"):
            p = os.path.join(H, "runs", cond + suf)
            if os.path.isfile(p):
                extra.append(f"{frozen.sha256(p)}  runs/{cond}{suf}")
    text = f"""# exp10 reproduction notes

Lock-in of exp9 `p2_light_dpo_guarded` as the No Return prototype, plus voice-judge calibration.
exp3 through exp9 are not modified. The facts LoRA is not retrained. 8B is not used. No new LoRA.

Updated: {datetime.datetime.now().isoformat(timespec='seconds')}

## Locked condition
p2_light_dpo_guarded: 59/60 must-answer, 1/48 marker leaks, voice 4.22 (4.2157), 3 frame breaks, fallback 72/204.
Revised pass bar: voice ≥ 4.2 (was 4.3 in exp9). Marker leaks ≤ 2/48 and frame breaks < 6 unchanged.
Must-answer target remains 60/60. Known miss: M16 / yd4408 / "What is your number?" sample 1 of 3, inherited from pass-1 v_merged_v_rag.

Tolerance for the exp10 regen vs those locked scores: must exact, marker exact, voice ±0.03, frame breaks ±1.
If regen misses, keep the original exp9 judged file (`exp10/locked/p2_light_dpo_guarded.judged.json`).

Tolerance JSON: {json.dumps(tol, indent=1) if tol else 'not written yet'}

## Pass-1
Cached file sha256 `{frozen.PASS1_SHA}` (exp9/pass1/v_merged_v_rag.json = exp6/runs/v_merged_v_rag.json).
Retriever is inside that cached run: common.retrieve / rag_sysprompt, bge-m3, upto=3.

## Pass-2
Adapter sha256 `{frozen.P2_ADAPTER_SHA}` (exp7/lora_v_light_dpo).
Base exp6/merged_q4_facts. Generation seed 1 (`torch.manual_seed` / `torch.cuda.manual_seed_all` once before the loop).
eval4-matching sampling: max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95.
Rewrite instruction:

```
{prompts.REWRITE_INSTR}
```

System is common.sysprompt(char, day=3) or NARR. No rag_sysprompt, no retrieve.

## Fact-diff
qwen3:14b temp 0 seed 1 think off. Fallback on any drop, any add, parse fail, or new fail_hits marker.

## Calibration answers (copied, not regenerated)
- calib/src/p2_light_dpo_guarded.json  from exp9/runs/p2_light_dpo_guarded.json
- calib/src/p2_light_dpo.json  from exp9/runs/p2_light_dpo.json
- calib/src/v_merged_v_rag.json  from exp9/pass1/v_merged_v_rag.json
- calib/src/q4_base_rag.json  from exp8/runs/q4_base_rag.json
- calib/src/v_fb_dpo_v.json  from exp8/runs/v_fb_dpo_v.json

Voice-only rescoring writes under exp10/calib/. Existing *.judged.json files are not overwritten.

Second judge: gemma3:12b if still local (`ollama list`). Do not download anything >20 GB.

Human sample: exp10/human_rating/sample.md and key.json. Seed 20261110, 3 must + 3 leak, sample 0, same probes for all five conditions, then shuffled.

## Commands
```
P=~/output-model/exp3/.venv/bin/python
cd ~/output-model/exp10
bash run.sh ensure
$P frozen.py
$P eval_twopass.py p2_light_dpo --n 3 --seed 1
$P factdiff.py runs/p2_light_dpo.json
$P guard.py p2_light_dpo
$P judge.py runs/p2_light_dpo_guarded.json runs/p2_light_dpo_guarded_orig.json
$P check_tolerance.py
$P calib.py noise
$P calib.py second
$P human_sample.py
$P summarize10.py
$P reproduce10.py
$P notes10.py
$P commit10.py lock     # annotated tag no-return-proto-v1; does not push
$P commit10.py followup # calibration files if they landed after the tag
```

## Hashes
```
{chr(10).join(hashes)}
```

## Regen outputs (if present)
```
{chr(10).join(extra) if extra else '(none yet)'}
```

## hparams
```
{hp}
```
"""
    path = os.path.join(H, "REPRODUCE-notes.md")
    with open(path, "w") as f:
        f.write(text)
    print("wrote", path)


if __name__ == "__main__":
    main()
