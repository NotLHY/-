#!/usr/bin/env bash
# Explicit Docker/platform modes; all model arguments remain an argument array.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${CXR_MODE:-docker}"
if [[ "${1:-}" == "--mode" ]]; then MODE="${2:?missing mode}"; shift 2; fi
ACTION="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
cd "$ROOT"
if [[ "$MODE" == "docker" ]]; then
    command -v docker >/dev/null || { echo "Docker unavailable; use --mode native in a platform container" >&2; exit 2; }
    docker info >/dev/null
    IMAGE="${CXR_IMAGE:-cxr-phase1:official}"
    if [[ "$ACTION" == "build" ]]; then
        exec docker build -f src/Dockerfile -t "$IMAGE" .
    fi
    docker image inspect "$IMAGE" >/dev/null || { echo "Run: bash src/server/run.sh build" >&2; exit 2; }
    DATA="${CXR_DATA_ROOT:-$ROOT/cxr_first}"
    MOUNTS=(--mount "type=bind,source=$ROOT,target=/workspace")
    if [[ -d "$DATA" ]]; then
        MOUNTS+=(--mount "type=bind,source=$DATA,target=/workspace/cxr_first,readonly")
    elif [[ "$ACTION" != "check" && "$ACTION" != "verify" && "$ACTION" != "weights" && "$ACTION" != "configure" ]]; then
        echo "Missing data folder: $DATA" >&2; exit 2
    fi
    ENVS=(--env PYTHONPATH=/workspace/src --env TORCH_HOME=/workspace/.cache/torch
          --env PYTHONUNBUFFERED=1 --env OMP_NUM_THREADS=4 --env CXR_PYTHON=/opt/cxr-venv/bin/python)
    if [[ -n "${CXR_CACHE_ROOT:-}" ]]; then
        ENVS+=(--env "CXR_CACHE_ROOT=${CXR_CACHE_ROOT}")
    fi
    if [[ -n "${CXR_CACHE_SIZE:-}" ]]; then
        ENVS+=(--env "CXR_CACHE_SIZE=${CXR_CACHE_SIZE}")
    fi
    exec docker run --rm --gpus "device=${CXR_GPU:-0}" --shm-size="${CXR_SHM_SIZE:-4g}" \
        --label cxr.project=cxr-first --label "cxr.action=$ACTION" \
        "${MOUNTS[@]}" \
        --workdir /workspace \
        "${ENVS[@]}" \
        "$IMAGE" bash src/server/run.sh --mode native "$ACTION" "$@"
fi
[[ "$MODE" == "native" ]] || { echo "Mode must be docker or native" >&2; exit 2; }
PYTHON="${CXR_PYTHON:-$ROOT/.server-venv/bin/python}"
command -v "$PYTHON" >/dev/null || { echo "Run setup_native.sh or set CXR_PYTHON to the matching Python" >&2; exit 2; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$ROOT/.cache/torch"
export PYTHONUNBUFFERED=1
case "$ACTION" in
    check) exec "$PYTHON" src/server/check_environment.py "$@" ;;
    verify) exec "$PYTHON" src/server/verify_bundle.py "$@" ;;
    weights) exec "$PYTHON" src/download_weights.py "$@" ;;
    configure) exec "$PYTHON" src/server/configure.py "$@" ;;
    cache) exec "$PYTHON" tools/build_cache.py "$@" ;;
    smoke) exec "$PYTHON" -m cxr train --config configs/generated/server_smoke.json "$@" ;;
    profile) exec "$PYTHON" -m cxr.profiling "$@" ;;
    latency) exec "$PYTHON" -m cxr.latency "$@" ;;
    audit|train|evaluate|predict|benchmark) exec "$PYTHON" -m cxr "$ACTION" "$@" ;;
    *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
