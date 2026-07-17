#!/usr/bin/env bash
# setup_env.sh
#
# Run this from the repo root on the server:
#   bash setup_env.sh
#
# Creates a uv-managed virtualenv and installs requirements.txt into it.
# Assumes `uv` is already installed and on PATH (per existing workflow).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found in $REPO_ROOT"
    echo "Make sure you're running this from the repo root, or that the push landed."
    exit 1
fi

echo "== creating venv with uv =="
uv venv .venv

echo "== installing requirements.txt =="
uv pip install -r requirements.txt --python .venv/bin/python

echo "== done. activate with: =="
echo "    source .venv/bin/activate"
echo ""
echo "== next: verify the environment with =="
echo "    python scripts/check_env.py"
