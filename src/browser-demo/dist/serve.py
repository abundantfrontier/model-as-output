#!/usr/bin/env python3
"""Serve this folder on loopback with the headers WebGPU wants.

Usage (from anywhere):

    python3 serve.py

Then open http://127.0.0.1:8093 . The bind address is 127.0.0.1 only.
If 8093 is taken, the next free port from 8094 upward is used, skipping
8090 and 8091. Set PORT to choose one.
"""
from __future__ import annotations

import os
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP = {8090, 8091}


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        if os.environ.get("QUIET"):
            return
        super().log_message(fmt, *args)


def free_port(preferred: int) -> int:
    if os.environ.get("PORT"):
        return int(os.environ["PORT"])
    for port in [preferred, *range(preferred + 1, preferred + 20)]:
        if port in SKIP:
            continue
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise SystemExit(f"no free port from {preferred}")


def main() -> None:
    port = free_port(8093)
    handler = partial(Handler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
