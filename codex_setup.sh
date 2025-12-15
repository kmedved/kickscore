#!/usr/bin/env bash
set -euxo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Create venv once
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
. .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-dev.txt

# Make future shells/tests use the venv + repo imports (idempotent)
BASHRC="$HOME/.bashrc"
touch "$BASHRC"

LINE_ACTIVATE="cd \"$REPO_ROOT\" && . \"$REPO_ROOT/.venv/bin/activate\""
LINE_PYTHONPATH="export PYTHONPATH=\"$REPO_ROOT:\${PYTHONPATH:-}\""

grep -qxF "$LINE_ACTIVATE" "$BASHRC" || echo "$LINE_ACTIVATE" >> "$BASHRC"
grep -qxF "$LINE_PYTHONPATH" "$BASHRC" || echo "$LINE_PYTHONPATH" >> "$BASHRC"

# Sanity check (fails fast if deps not usable)
python -c "import numpy, scipy, numba, kickscore; print('deps ok')"
