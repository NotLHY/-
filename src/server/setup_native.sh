#!/usr/bin/env bash
# Run inside the platform's official image. Never replace the host driver/framework.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE_PYTHON="${CXR_BASE_PYTHON:-python3}"
"$BASE_PYTHON" src/server/check_environment.py --core-only
if "$BASE_PYTHON" src/server/check_environment.py; then
    echo "Existing runtime already has training dependencies; nothing installed."
    echo "Run with: CXR_PYTHON=$(command -v "$BASE_PYTHON") bash src/server/run.sh --mode native check"
    exit 0
fi
echo "Core framework matches, but required training dependencies are missing."
echo "Preparing an isolated project environment; torch/torchvision are not in the install list."
if [[ ! -x "$ROOT/.server-venv/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --without-pip --system-site-packages "$ROOT/.server-venv"
fi
PYTHON="$ROOT/.server-venv/bin/python"
"$PYTHON" src/server/check_environment.py --core-only
if ! "$PYTHON" -m uv --version >/dev/null 2>&1; then
    "$PYTHON" -m pip install uv==0.11.21
fi
"$PYTHON" -m uv pip install --python "$PYTHON" -r src/server/requirements-runtime.txt
"$PYTHON" src/server/check_environment.py
echo "Ready: bash src/server/run.sh --mode native check"
