#!/usr/bin/env python3
"""Rebuild the curated public tree of the model-as-output project.

Reads the working tree (default: the parent of this tools/ folder) and writes a
clean, scrubbed copy (default: $HOME/output-model-public). It never edits the
working tree, never touches a GPU, and never pushes anything. Publishing the
result to GitHub is a separate, manual step.

    python3 tools/sync_public.py                # rebuild + scrub check
    python3 tools/sync_public.py --check-only   # only scan an existing tree
    python3 tools/sync_public.py --dest /tmp/x  # write somewhere else

Exit code is non-zero if the scrub check finds anything it does not allow.
Hand-written top-level files (README, LICENSE, .gitignore, docs written for the
public) live in tools/public-overlay/ and are copied over the generated tree.
"""
import argparse, fnmatch, os, re, shutil, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
MAX_BYTES = 1_000_000          # nothing bigger than ~1 MB goes in
TEXT_EXT = {".py", ".sh", ".md", ".json", ".jsonl", ".yaml", ".yml", ".txt",
            ".html", ".js", ".css", ".tsv", ".csv", ".toml", ".jinja", ""}
BLOCK_GLOBS = ["*.safetensors", "*.gguf", "*.bin", "*.pt", "*.pth", "*.ckpt",
               "*.npz", "*.npy", "*.wasm", "*.tgz", "*.tar.gz", "*.zip",
               "*.log", "*.pid", "*.lock", "*.bak", "*.pyc", "PORT",
               "BRIEF*.md", "build-status.md", "commit*.py", "log.txt",
               "tensor-cache.json", "params_shard_*", "tokenizer.json"]
BLOCK_DIRS = {"__pycache__", ".venv", ".venv-mlc", "node_modules", ".cache",
              "logs", "sessions", "smoke", "work", "models", "vendor"}

PHASE1_CODE = ["exp4", "exp6", "exp7", "exp9", "exp10"]   # chain behind the locked model
PHASE1_REPORTS = ["results.md", "REPRODUCE.md", "REPRODUCE-notes.md", "status.tsv",
                  "relative_bar.md", "judge_sanity.md", "build-review.md"]

# (source glob relative to SRC, destination folder relative to DEST)
MAP = []
for e in PHASE1_CODE:
    MAP += [(f"{e}/*.py", f"src/phase1/{e}"), (f"{e}/*.sh", f"src/phase1/{e}"),
            (f"{e}/hparams.json", f"src/phase1/{e}"),
            (f"{e}/hparams.json", f"configs/phase1/{e}")]
for n in range(1, 11):
    for r in PHASE1_REPORTS:
        MAP.append((f"exp{n}/{r}", f"reports/phase1/exp{n}"))
MAP += [
    ("exp2/runs/summary.json", "reports/phase1/exp2"),
    ("exp10/relative/report.json", "reports/phase1/exp10/relative"),
    ("exp10/locked/p2_light_dpo_guarded.judged.json", "reports/phase1/exp10/locked"),
    # adapter hyper-parameters (no weights)
    ("exp4/lora_q4_facts/adapter_config.json", "configs/phase1/lora/exp4_lora_q4_facts"),
    ("exp4/lora_q4_facts/train_report.json", "configs/phase1/lora/exp4_lora_q4_facts"),
    ("exp6/lora_v_merged/adapter_config.json", "configs/phase1/lora/exp6_lora_v_merged"),
    ("exp6/lora_v_merged/train_report.json", "configs/phase1/lora/exp6_lora_v_merged"),
    ("exp10/init/lora_v_light_dpo/adapter_config.json", "configs/phase1/lora/exp10_lora_v_light_dpo"),
    ("exp10/init/lora_v_light_dpo/train_report.json", "configs/phase1/lora/exp10_lora_v_light_dpo"),
    ("browser-demo/work/mlc-config.json", "configs/browser-demo"),
    # No Return story (invented) and its test data
    ("no-return/chapters/ch*.md", "docs/no-return/chapters"),
    ("no-return/knowledge-state.yaml", "data/no-return"),
    ("no-return/leak-tests.md", "data/no-return"),
    # paper snapshots (already public on the AFI Lab)
    ("paper/style.css", "docs/paper"),
    ("paper/concept/index.html", "docs/paper/concept"),
    ("paper/no-return/index.html", "docs/paper/no-return"),
    # browser demo (static site; weights and vendor bundle are not included)
    ("browser-demo/README.md", "src/browser-demo"),
    ("browser-demo/scripts/*.py", "src/browser-demo/scripts"),
    ("browser-demo/dist/*.html", "src/browser-demo/dist"),
    ("browser-demo/dist/*.js", "src/browser-demo/dist"),
    ("browser-demo/dist/*.css", "src/browser-demo/dist"),
    ("browser-demo/dist/serve.py", "src/browser-demo/dist"),
    ("browser-demo/dist/prompt/*.json", "src/browser-demo/dist/prompt"),
    ("browser-demo/dist/index/*.json", "src/browser-demo/dist/index"),
    ("browser-demo/results.md", "reports/browser-demo"),
    ("browser-demo/score1/RESULTS.md", "reports/browser-demo/score1"),
    # reader (local server)
    ("reader/*.py", "src/reader"), ("reader/run.sh", "src/reader"),
    ("reader/README.md", "src/reader"), ("reader/static/*", "src/reader/static"),
    ("reader/packs/sample.json", "src/reader/packs"),
    ("reader/questions/sample.json", "src/reader/questions"),
    # Phase 2: Frankenstein
    ("phase2-frankenstein/README.md", "docs/phase2-frankenstein"),
    ("phase2-frankenstein/source/SOURCE.md", "data/frankenstein"),
    ("phase2-frankenstein/knowledge-state.yaml", "data/frankenstein"),
    ("phase2-frankenstein/questions/*", "data/frankenstein/questions"),
    ("phase2-frankenstein/voice/*.jsonl", "data/frankenstein/voice"),
    ("phase2-frankenstein/scripts/*.py", "src/phase2/prep"),
    ("phase2-frankenstein/train1/scripts/*", "src/phase2/train1"),
    ("phase2-frankenstein/train1/RESULTS.md", "reports/phase2/train1"),
    ("phase2-frankenstein/train1/data/HOW_MADE.md", "reports/phase2/train1"),
    ("phase2-frankenstein/train1/data/split_report.json", "reports/phase2/train1"),
    ("phase2-frankenstein/train1/lora_voice/train_report.json", "reports/phase2/train1/lora_voice"),
    ("phase2-frankenstein/dev1/VOICE_DIAGNOSIS.md", "reports/phase2/dev1"),
    ("phase2-frankenstein/dev1/scripts/*.py", "src/phase2/dev1"),
    ("phase2-frankenstein/dev1/hf_upload.py", "src/phase2/dev1"),
    ("phase2-frankenstein/dev1/browser/README.md", "src/phase2/dev1/browser"),
    # later Phase 2 runs (train2, train3, …) are added below only once RESULTS.md exists
    ("tools/sync_public.py", "tools"),
]
for _run in ["train2", "train3", "train4"]:
    if (SRC / "phase2-frankenstein" / _run / "RESULTS.md").exists():
        MAP += [(f"phase2-frankenstein/{_run}/RESULTS.md", f"reports/phase2/{_run}"),
                (f"phase2-frankenstein/{_run}/scripts/*", f"src/phase2/{_run}")]
# explicit files allowed even though a parent folder name is normally blocked
ALLOW_PATHS = {"browser-demo/work/mlc-config.json"}

# ---- scrubbing -------------------------------------------------------------
REPLACE = [
    (re.compile(r"$HOME\b"), "$HOME"),
    (re.compile(r"/Users/[A-Za-z0-9_.-]+"), "$HOME"),
    (re.compile(r"\bE2M round \d+\b"), "another GPU job"),
    (re.compile(r"\bE2M\b"), "another GPU job"),
    (re.compile(r"\be2m\b"), "other-project"),
    (re.compile(r"\bu3090\b"), "the GPU host"),
]
# lines that mention a private project are dropped from text files
# in-place edits applied to every text file first; single-line formats (JSON) are never line-dropped
INLINE_SUB = [
    (re.compile(r""lost in transit" label"), '"lost in transit" label'),
]

DENY = [
    ("hf token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("github token", re.compile(r"\b(ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}")),
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("xai key", re.compile(r"\bxai-[A-Za-z0-9]{20,}")),
    ("aws key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY")),
    ("home path", re.compile(r"/home/[a-z]|/Users/[A-Za-z]")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("ipv4", re.compile(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("uuid", re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")),
    ("bypass flag", re.compile(r"bypassPermissions|--always-approve|dangerously")),
    ("host name", re.compile(r"\bu3090\b")),
    ("secret word", re.compile(r"(password|passwd|secret|token|api[_-]?key)\w*\s*[:=]\s*['\"][A-Za-z0-9_\-]{12,}['\"]", re.I)),
    ("agent tooling", re.compile(r"grok-build|\.grok\b|omniboyd|boyd@", re.I)),
]
# known-harmless matches (exact substrings) the scan may skip
ALLOW = [
    "noreply@", "@abundantfrontier.org", "example.com",
    "hf_upload", "sk-learn",
]

def blocked(p: Path) -> bool:
    if str(p) in ALLOW_PATHS:
        return False
    if any(part in BLOCK_DIRS for part in p.parts[:-1]):
        return True
    return any(fnmatch.fnmatch(p.name, g) for g in BLOCK_GLOBS)

def scrub_text(text: str, suffix: str = "") -> str:
    for rx, rep in INLINE_SUB:
        text = rx.sub(rep, text)
    if suffix not in (".json", ".jsonl"):
        out = [ln for ln in text.splitlines(keepends=True) if not DROP_LINE.search(ln)]
        text = "".join(out)
    for rx, rep in REPLACE:
        text = rx.sub(rep, text)
    return text

def copy_one(src: Path, dst_dir: Path, dest_root: Path, skipped: list):
    if blocked(src.relative_to(SRC)):
        skipped.append((str(src.relative_to(SRC)), "blocked pattern")); return
    if src.stat().st_size > MAX_BYTES:
        skipped.append((str(src.relative_to(SRC)), f"too big ({src.stat().st_size} B)")); return
    dst = dst_dir / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in TEXT_EXT:
        try:
            dst.write_text(scrub_text(src.read_text(encoding="utf-8"), src.suffix), encoding="utf-8")
            shutil.copymode(src, dst)
            if src.stat().st_size > 0 and dst.stat().st_size == 0:
                skipped.append((str(src.relative_to(SRC)), "EMPTY AFTER SCRUB"))
            return
        except UnicodeDecodeError:
            pass
    skipped.append((str(src.relative_to(SRC)), "binary"))

def scan(dest: Path) -> list:
    hits = []
    for p in sorted(dest.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        if p.stat().st_size > MAX_BYTES * 5:
            hits.append((str(p.relative_to(dest)), 0, "file over 5 MB", "")); continue
        if any(fnmatch.fnmatch(p.name, g) for g in BLOCK_GLOBS[:9]):
            hits.append((str(p.relative_to(dest)), 0, "weight/archive file", "")); continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            hits.append((str(p.relative_to(dest)), 0, "binary file", "")); continue
        if p.name == "sync_public.py":
            continue   # contains the patterns themselves
        for i, ln in enumerate(lines, 1):
            for name, rx in DENY:
                for m in rx.finditer(ln):
                    ctx = ln[max(0, m.start() - 20): m.end() + 20]
                    if any(a in ctx for a in ALLOW):
                        continue
                    if name == "ipv4" and ("==" in ctx or re.search(r"\d\.\d+\.\d+\.\d+\.\d", ctx)):
                        continue   # package version numbers, not addresses
                    hits.append((str(p.relative_to(dest)), i, name, m.group(0)[:60]))
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.expandvars("$HOME/output-model-public"))
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    dest = Path(a.dest)
    if not a.check_only:
        if dest.exists():
            for child in dest.iterdir():
                if child.name == ".git":
                    continue
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        dest.mkdir(parents=True, exist_ok=True)
        skipped, n = [], 0
        for pattern, dst in MAP:
            for src in sorted(SRC.glob(pattern)):
                if src.is_file():
                    copy_one(src, dest / dst, dest, skipped); n += 1
        overlay = SRC / "tools" / "public-overlay"
        if overlay.exists():
            for src in sorted(overlay.rglob("*")):
                if src.is_file():
                    d = dest / src.relative_to(overlay)
                    d.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, d)
        print(f"copied from {n} matches; skipped {len(skipped)}")
        for s in skipped:
            print("  skip", *s)
    hits = scan(dest)
    for h in hits:
        print("SCRUB", *h)
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"tree: {dest}  files={sum(1 for p in dest.rglob('*') if p.is_file() and '.git' not in p.parts)}  bytes={total}")
    print("scrub: CLEAN" if not hits else f"scrub: {len(hits)} finding(s)")
    sys.exit(1 if hits else 0)

if __name__ == "__main__":
    main()
