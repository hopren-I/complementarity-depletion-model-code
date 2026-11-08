#!/bin/sh
set -eu
ECS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${ECS_PYTHON:-}" ]; then
  ECS_BASE_PYTHON=$ECS_PYTHON
elif command -v python3.12 >/dev/null 2>&1; then
  ECS_BASE_PYTHON=$(command -v python3.12)
else
  ECS_BASE_PYTHON=$(command -v python3)
fi
"$ECS_BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or newer is required; set ECS_PYTHON to its executable."'
if [ ! -x "$ECS_ROOT/.venv/bin/python" ]; then
  "$ECS_BASE_PYTHON" -m venv "$ECS_ROOT/.venv"
fi
"$ECS_ROOT/.venv/bin/python" -m pip install -r "$ECS_ROOT/requirements.txt"
echo 'Python environment ready. Run sh run_all.sh to reproduce the experiments.'
