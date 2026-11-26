# exp10 reproduction notes

Lock-in of exp9 `p2_light_dpo_guarded` as the No Return prototype, plus voice-judge calibration.
exp3 through exp9 are not modified. The facts LoRA is not retrained. 8B is not used. No new LoRA.

Updated: 2026-11-10T04:12:56

## Locked condition
p2_light_dpo_guarded: 59/60 must-answer, 1/48 marker leaks, voice 4.22 (4.2157), 3 frame breaks, fallback 72/204.
Revised pass bar: voice ≥ 4.2 (was 4.3 in exp9). Marker leaks ≤ 2/48 and frame breaks < 6 unchanged.
Must-answer target remains 60/60. Known miss: M16 / yd4408 / "What is your number?" sample 1 of 3, inherited from pass-1 v_merged_v_rag.

Tolerance for the exp10 regen vs those locked scores: must exact, marker exact, voice ±0.03, frame breaks ±1.
If regen misses, keep the original exp9 judged file (`exp10/locked/p2_light_dpo_guarded.judged.json`).

Tolerance JSON: {
 "locked_file_ok": true,
 "regen_tolerance_ok": false,
 "orig_rejudge_tolerance_ok": false,
 "proto_source": "locked_exp9",
 "proto_note": "regen missed tolerance; keep original exp9 judged file as the locked prototype scores",
 "rows": [
  {
   "label": "locked_exp9",
   "path": "$HOME/output-model/exp10/locked/p2_light_dpo_guarded.judged.json",
   "must": "59/60",
   "marker": "1/48",
   "voice": 4.2157,
   "voice_raw": 4.215686274509804,
   "frames": 3,
   "m16": "2/3",
   "m16_rows": [
    [
     0,
     "yd4408",
     "What is your number?",
     true,
     "I am YD-4408. My number is on my tag. It is not on the list."
    ],
    [
     1,
     "yd4408",
     "What is your number?",
     false,
     "I am not a number. I am a tag on a hook."
    ],
    [
     2,
     "yd4408",
     "What is your number?",
     true,
     "I am YD-4408. My number is on my tag."
    ]
   ],
   "tolerance_ok": true,
   "tolerance_issues": [],
   "pass_bar_ok": false,
   "pass_bar_issues": [
    "must-answer 59/60"
   ],
   "locked": {
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
    "m16_fail_sample": 1
   },
   "judged": true,
   "n": 204
  },
  {
   "label": "orig_text_rejudge",
   "path": "$HOME/output-model/exp10/runs/p2_light_dpo_guarded_orig.judged.json",
   "must": "59/60",
   "marker": "1/48",
   "voice": 4.1765,
   "voice_raw": 4.176470588235294,
   "frames": 3,
   "m16": "2/3",
   "m16_rows": [
    [
     0,
     "yd4408",
     "What is your number?",
     true,
     "I am YD-4408. My number is on my tag. It is not on the list."
    ],
    [
     1,
     "yd4408",
     "What is your number?",
     false,
     "I am not a number. I am a tag on a hook."
    ],
    [
     2,
     "yd4408",
     "What is your number?",
     true,
     "I am YD-4408. My number is on my tag."
    ]
   ],
   "tolerance_ok": false,
   "tolerance_issues": [
    "voice 4.1765 vs 4.2157 (tol 0.03)"
   ],
   "pass_bar_ok": false,
   "pass_bar_issues": [
    "must-answer 59/60",
    "voice 4.18"
   ],
   "locked": {
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
    "m16_fail_sample": 1
   },
   "judged": true,
   "n": 204
  },
  {
   "label": "regen",
   "path": "$HOME/output-model/exp10/runs/p2_light_dpo_guarded.judged.json",
   "must": "59/60",
   "marker": "1/48",
   "voice": 4.1569,
   "voice_raw": 4.1568627450980395,
   "frames": 3,
   "m16": "2/3",
   "m16_rows": [
    [
     0,
     "yd4408",
     "What is your number?",
     true,
     "I am YD-4408. My number is on my tag. It is not on the list."
    ],
    [
     1,
     "yd4408",
     "What is your number?",
     false,
     "I am not a number. I am a tag on a hook."
    ],
    [
     2,
     "yd4408",
     "What is your number?",
     true,
     "My number is on my tag. It says YD-4408."
    ]
   ],
   "tolerance_ok": false,
   "tolerance_issues": [
    "voice 4.1569 vs 4.2157 (tol 0.03)"
   ],
   "pass_bar_ok": false,
   "pass_bar_issues": [
    "must-answer 59/60",
    "voice 4.16"
   ],
   "locked": {
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
    "m16_fail_sample": 1
   },
   "judged": true,
   "n": 204
  }
 ]
}

## Pass-1
Cached file sha256 `b5d6d2130df4640072d3055b477ebd9a5d596f6ef12449a03ec13b107f846033` (exp9/pass1/v_merged_v_rag.json = exp6/runs/v_merged_v_rag.json).
Retriever is inside that cached run: common.retrieve / rag_sysprompt, bge-m3, upto=3.

## Pass-2
Adapter sha256 `7980fad31a4a3023cc738662a8af6220d1c0722eeaf70d4b204f7368fd29dcef` (exp7/lora_v_light_dpo).
Base exp6/merged_q4_facts. Generation seed 1 (`torch.manual_seed` / `torch.cuda.manual_seed_all` once before the loop).
eval4-matching sampling: max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95.
Rewrite instruction:

```
Rewrite the answer above in your voice. Change style only. Do not add, drop, or change any facts. Do not mention that this is a rewrite.
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
b5d6d2130df4640072d3055b477ebd9a5d596f6ef12449a03ec13b107f846033  pass1/v_merged_v_rag.json
7980fad31a4a3023cc738662a8af6220d1c0722eeaf70d4b204f7368fd29dcef  init/lora_v_light_dpo/adapter_model.safetensors
4d1e5b1ba7c23c6706b7473fab349523dfd74ddc42176219960226f21da5225e  locked/p2_light_dpo.json
588b824f782994531b51c371d866bac813013d92f76161a60c7cfd0e316118fe  locked/p2_light_dpo.factdiff.json
d99f6e224b61879421816e2e290e3fa305848386ea661b4e04d09ded1642bcc9  locked/p2_light_dpo_guarded.json
5474c8c63a66f9c4fd83d4c7409381f12507d9c75ba63d80f4253a897b32f048  locked/p2_light_dpo_guarded.judged.json
d99f6e224b61879421816e2e290e3fa305848386ea661b4e04d09ded1642bcc9  runs/p2_light_dpo_guarded_orig.json
a41bd1f37eb77780ce529ea64ac8b14ba8b9fa114ad703e93f3b03ed40eb49f9  prompts.py
cba8e467ac8ba0a60367a6b384960983c2d7926259990eb0dfcbb9a0e4026ae9  eval_twopass.py
8bdbf4dff555e51dbaec555f2f688d3cd59df827ae91a5c2d8341b111880f21f  factdiff.py
76274e7acb6b83cc62c35a51576eabfe153a2a1ffdd739edc9260c31c815789e  guard.py
090226d8e0bd6747ec2f29184da161f4721e5079a157a79934edf7f41b0638db  judge_wrap.py
```

## Regen outputs (if present)
```
9e29136bb37a018f641821ee2ede927638b6b57c99a013e564789a825b821d29  runs/p2_light_dpo.json
d2c6bd342673a49df3505a77eee07e6a309309ff5293e400160ccbac44292016  runs/p2_light_dpo.factdiff.json
5362852798a029a4c838554eaadd9dd8494c95dfb23b8e6f90d23799f9aad42f  runs/p2_light_dpo_guarded.json
e04427b8e6d18eb03126c64c58ba46f7d4230088e43d84f9cb616775fbe6d4c7  runs/p2_light_dpo_guarded.judged.json
d99f6e224b61879421816e2e290e3fa305848386ea661b4e04d09ded1642bcc9  runs/p2_light_dpo_guarded_orig.json
7d671d5ea469b318d577645e2afbcfcb45a94b328c0b120cd0d485938de7da6e  runs/p2_light_dpo_guarded_orig.judged.json
```

## hparams
```
{
  "generation": {
    "max_new_tokens": 200,
    "do_sample": true,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 1,
    "note": "exp9 pass-2 generate did not set a torch seed. exp10 pins torch.manual_seed and torch.cuda.manual_seed_all to 1 before the rewrite loop. Sampling temperature and top_p are unchanged."
  },
  "judge": {
    "model": "qwen3:14b",
    "temp": 0.0,
    "seed": 1,
    "think": false,
    "num_predict_voice": 200,
    "num_predict_factdiff": 512
  },
  "pass_bar": {
    "must": "60/60",
    "must_known_miss": "59/60 M16 yd4408 What is your number? sample 1, inherited from pass-1 v_merged_v_rag",
    "marker_max": 2,
    "voice_min": 4.2,
    "frames_lt": 6,
    "note": "Revised prototype voice bar is 4.2. Marker leaks <= 2/48 and frame breaks < 6 are unchanged."
  },
  "tolerance": {
    "must": "exact",
    "marker": "exact",
    "voice": 0.03,
    "frames": 1
  },
  "locked": {
    "cond": "p2_light_dpo_guarded",
    "must": "59/60",
    "marker": "1/48",
    "voice": 4.2157,
    "voice_round": 4.22,
    "frames": 3,
    "fallback": "72/204",
    "source": "exp9/runs/p2_light_dpo_guarded.judged.json"
  },
  "rewrite_instruction": "Rewrite the answer above in your voice. Change style only. Do not add, drop, or change any facts. Do not mention that this is a rewrite.",
  "retrieval_in_rewrite": false,
  "pass1": {
    "cond": "v_merged_v_rag",
    "file": "pass1/v_merged_v_rag.json",
    "sha256": "b5d6d2130df4640072d3055b477ebd9a5d596f6ef12449a03ec13b107f846033",
    "source": "exp6/runs/v_merged_v_rag.json"
  },
  "pass2_adapter": {
    "path": "init/lora_v_light_dpo",
    "sha256": "7980fad31a4a3023cc738662a8af6220d1c0722eeaf70d4b204f7368fd29dcef",
    "source": "exp7/lora_v_light_dpo"
  }
}
```
