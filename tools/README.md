# tools/sync_public.py

Rebuilds this repository from the private working tree. It runs on the research machine and never pushes anything.

```bash
cd $HOME/output-model
python3 tools/sync_public.py                 # rebuild $HOME/output-model-public and run the scrub check
python3 tools/sync_public.py --check-only    # only re-scan the staged tree
python3 tools/sync_public.py --dest /tmp/x   # stage somewhere else
```

What it does:

1. Copies an explicit allow-list of files: scripts, configs, results and reproduction notes, small data. Anything matching weights (`*.safetensors`, `*.bin`, `*.gguf`, …), logs, pid/lock files, caches, venvs, internal build briefs, or anything over 1 MB is skipped.
2. Scrubs text: home paths become `$HOME`, machine and other-project names are replaced, and lines that mention private projects are dropped.
3. Scans the result for tokens and keys (Hugging Face, GitHub, OpenAI-style, xAI, AWS, private keys), home paths, e-mail addresses, IP addresses, UUIDs, tool bypass flags and host names. It exits non-zero if anything is found.
4. Copies hand-written public files (this README, the top-level README, LICENSE, .gitignore) from `tools/public-overlay/`.

A new Phase 2 run (`train2`, …) is only included once its `RESULTS.md` exists, so a running job is never copied. Publishing the staged tree to GitHub is a separate step, done through the GitHub connector after review.
