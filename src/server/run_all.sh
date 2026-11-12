#!/usr/bin/env bash
# One-command AI Station entry: preflight -> optional 320 cache -> plan -> collect.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${CXR_PYTHON:-python3}"
DATA="${CXR_DATA_ROOT:?set CXR_DATA_ROOT to the directory containing train.csv}"
export CXR_DATA_ROOT="$DATA"
CACHE="${CXR_CACHE_ROOT:-$ROOT/cache/320}"
PLAN="${CXR_PLAN:-$ROOT/configs/plan_main.json}"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
mkdir -p "$ROOT/logs" "$ROOT/transfer" "$CACHE"

echo "[run_all] root=$ROOT"
echo "[run_all] python=$PYTHON data=$DATA cache=$CACHE plan=$PLAN"
if [[ "${CXR_ALLOW_CPU:-0}" == "1" ]]; then
    "$PYTHON" "$ROOT/tools/preflight.py" --data-root "$DATA" --cache-root "$CACHE" --allow-cpu
else
    "$PYTHON" "$ROOT/tools/preflight.py" --data-root "$DATA" --cache-root "$CACHE"
fi

if [[ ! -f "$CACHE/manifest.json" ]]; then
    echo "[run_all] building 320 cache"
    "$PYTHON" "$ROOT/tools/build_cache.py" \
        --data-root "$DATA" --output "$CACHE" --size 320 \
        --workers "${CXR_CACHE_WORKERS:-8}"
else
    if ! "$PYTHON" -c 'import json,sys; raise SystemExit(1 if json.load(open(sys.argv[1], encoding="utf-8")).get("partial") else 0)' "$CACHE/manifest.json"; then
        echo "[run_all] cache is partial; remove $CACHE and rebuild" >&2
        exit 2
    fi
    echo "[run_all] cache manifest exists; reuse $CACHE"
fi

export CXR_CACHE_ROOT="$CACHE"
export CXR_CACHE_SIZE="${CXR_CACHE_SIZE:-320}"
"$PYTHON" "$ROOT/tools/run_plan.py" --plan "$PLAN" "$@"
echo "[run_all] done; see runs/summary.json and transfer/"
