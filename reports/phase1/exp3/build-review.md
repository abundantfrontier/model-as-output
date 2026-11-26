# exp3 training fix review

Checked once at 2026-11-05 07:57 AM PST. Training was left running. No code changes.

## Fixes

`gen.py` (mtime 07:11:43, backup `gen.py.bak`) parses `mode` and `--n` only under `if __name__ == "__main__"`. `train3.py` imports `NARR` and `clean` after its own `--base` / `--out` / `--q4` parse. A separate interpreter, with `sys.argv` set to `train3.py --base Qwen/Qwen3-4B --out lora_4b --q4`, imported `gen` successfully.

`run.sh` (mtime 07:11:43, backup `run.sh.bak`, `bash -n` clean) stores `rc=$?` before each `echo "$(date) ..."`. The 07:09 resume logged train exit 0 about four seconds after start because `$(date)` ran before `$?` was read. This restart's `lora_17` exit 0 matches a finished 1338s run.

## Current run

Driver `./run.sh` PID **78255** (parent 78252), started Thu Nov 5 07:11:49 AM PST, state S, cwd `$HOME/output-model/exp3`.

Active train PID **78566**, child of 78255: `.venv/bin/python train3.py --base Qwen/Qwen3-4B --out lora_4b`, started 07:34:16, state Rl. GPU 60%, 15025 / 24576 MiB.

`lora_17` finished 07:34:16, exit 0. 8320 steps, 1338s, epoch losses 1.049 then 0.619, peak VRAM 6.73 GB. Report: `lora_17/train_report.json`.

`lora_4b` is in progress. `train_lora_4b.log` mtime 07:56:17. 4160 examples, 4103 pairs, 2 epochs (8320 steps). Epoch 0 loss 0.971. Last line `step 6000 1297s` (~4.6 steps/s from step 4500 to 6000). No traceback, OOM, or CUDA error in that log.

`lora_8b` has not started (`lora_8b/train_report.json` absent). `train_lora_8b.log` mtime 07:09:26 still has the pre-fix message `unrecognized arguments: --base --out lora_8b --q4`. This restart truncates that file when the 8B stage starts.

Synth (3004 rows), distill (1000 rows), and the HF snapshot (exit 0 at 07:11:50) were already done before training.

## Blocker

None. The job is on the 4B LoRA. It was not restarted.
