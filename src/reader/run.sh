#!/usr/bin/env bash
# Local No Return reader. Binds to loopback only unless READER_ALLOW_PUBLIC=1.
set -euo pipefail
cd "$(dirname "$0")"
PY="${READER_PYTHON:-$HOME/output-model/exp3/.venv/bin/python}"
PROTO_SITE="${READER_PROTO_SITE:-$HOME/output-model/proto/.venv/lib/python3.14/site-packages}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}:$PROTO_SITE"
export READER_MODE="${READER_MODE:-live}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

HOST="${READER_HOST:-127.0.0.1}"
if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ] && [ "$HOST" != "::1" ]; then
  if [ "${READER_ALLOW_PUBLIC:-}" != "1" ]; then
    echo "refusing non-local bind $HOST (set READER_ALLOW_PUBLIC=1 only when you mean to)" >&2
    exit 1
  fi
fi
export READER_HOST="$HOST"

port_busy() {
  ss -ltnH | awk '{print $4}' | grep -Eq "(^|:)$1$"
}

if [ -n "${READER_PORT:-}" ]; then
  if [ "$READER_PORT" = "8090" ] || [ "$READER_PORT" = "18765" ]; then
    echo "refusing port $READER_PORT" >&2
    exit 1
  fi
  if port_busy "$READER_PORT"; then
    echo "port $READER_PORT is taken" >&2
    exit 1
  fi
  PORT="$READER_PORT"
else
  PORT=8091
  while [ "$PORT" -le 8199 ]; do
    if [ "$PORT" != "8090" ] && [ "$PORT" != "18765" ] && ! port_busy "$PORT"; then
      break
    fi
    PORT=$((PORT + 1))
  done
  if [ "$PORT" -gt 8199 ]; then
    echo "no free port in 8091-8199" >&2
    exit 1
  fi
fi
export READER_PORT="$PORT"
echo "$PORT" > PORT
echo "No Return reader  http://${HOST}:${PORT}  mode=${READER_MODE}"
exec "$PY" -m uvicorn app:app --host "$HOST" --port "$PORT" --app-dir "$(pwd)"
