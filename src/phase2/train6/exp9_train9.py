"""Continue a voice LoRA with the local CE loss in exp8/train8.py.

Does not modify exp8. The child process cwd is exp8 so train8 imports its own
copies of the harness. --out is an absolute path under exp9. trl is not used.
"""
import os, sys, subprocess
import frozen

H = frozen.H
TRAIN8 = os.path.join(frozen.EXP8, "train8.py")
PY = os.path.join(frozen.ROOT, "exp3", ".venv", "bin", "python")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not os.path.isfile(TRAIN8):
        raise SystemExit(f"missing {TRAIN8}")
    for p in (a.pairs, a.init):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    out = a.out if os.path.isabs(a.out) else os.path.join(H, a.out)
    init = a.init if os.path.isabs(a.init) else os.path.join(H, a.init)
    pairs = a.pairs if os.path.isabs(a.pairs) else os.path.join(H, a.pairs)
    if os.path.commonpath([os.path.realpath(out), os.path.realpath(H)]) != os.path.realpath(H):
        raise SystemExit(f"refusing to write outside exp9: {out}")
    cmd = [
        PY, TRAIN8,
        "--base", frozen.MERGED,
        "--facts", "",
        "--stage", "voice",
        "--loss", "ce",
        "--out", out,
        "--pairs", pairs,
        "--init", init,
        "--epochs", str(a.epochs),
        "--lr", str(a.lr),
        "--r", str(a.r),
        "--dropout", str(a.dropout),
        "--max-steps", str(a.max_steps),
        "--save-every", str(a.save_every),
        "--limit", str(a.limit),
    ]
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    print("train9", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=frozen.EXP8, env=env)
    if rc != 0:
        raise SystemExit(rc)
    report = os.path.join(out, "train_report.json")
    if not os.path.isfile(report):
        raise SystemExit(f"train finished without {report}")
    print("train9 ok", report, flush=True)

if __name__ == "__main__":
    main()
