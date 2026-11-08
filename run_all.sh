#!/bin/sh
set -eu
ECS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${ECS_PYTHON:-}" ]; then
  ECS_RUN_PYTHON=$ECS_PYTHON
elif [ -x "$ECS_ROOT/.venv/bin/python" ]; then
  ECS_RUN_PYTHON=$ECS_ROOT/.venv/bin/python
else
  sh "$ECS_ROOT/setup.sh"
  ECS_RUN_PYTHON=$ECS_ROOT/.venv/bin/python
fi
exec "$ECS_RUN_PYTHON" "$ECS_ROOT/reproduce.py" "$@"
