#!/usr/bin/env bash

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if command -v poetry >/dev/null 2>&1; then
  POETRY_BIN="poetry"
elif python3 -m poetry --version >/dev/null 2>&1; then
  POETRY_BIN="python3 -m poetry"
else
  echo "Poetry is required but was not found. Install Poetry first."
  exit 1
fi

echo "Installing project dependencies..."
$POETRY_BIN install

echo "Starting BIliBIlAgent backend on http://${HOST}:${PORT}"
exec $POETRY_BIN run uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
