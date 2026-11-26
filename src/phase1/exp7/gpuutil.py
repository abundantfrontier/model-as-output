"""Unload ollama models so the 24 GB GPU is free for a local train or eval."""
import shutil, subprocess

def unload_ollama():
    if shutil.which("ollama") is None:
        return
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
    for line in out.stdout.splitlines()[1:]:
        name = line.split()[0] if line.split() else ""
        if name:
            subprocess.run(["ollama", "stop", name], capture_output=True)
