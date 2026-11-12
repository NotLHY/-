#!/usr/bin/env bash
# Execute the frozen 224/320/448 final recipe sequentially on one selected GPU.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CXR_DATA_ROOT:?Set CXR_DATA_ROOT to the directory containing train.csv and valid.csv}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the assigned GPU UUID}"
CXR_PYTHON="${CXR_PYTHON:-python}"
export PYTHONPATH="$ROOT/src"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/usr/local/infinibandlib/lib64:}"
unset CXR_CACHE_ROOT CXR_CACHE_SIZE CXR_ALLOW_MISSING_IMAGES
mkdir -p runs/sprint_final
# A second terminal must not launch a concurrent copy of this queue.
exec 9>runs/sprint_final/native_queue.lock
flock -n 9 || { echo "Another native final queue is running" >&2; exit 73; }
"$CXR_PYTHON" src/server/check_environment.py \
  --output runs/sprint_final/server_environment.json
for config in \
  configs/frozen_native/final_native_b0_224_s42_e6.json \
  configs/frozen_native/final_native_b0_320_s42_e7.json \
  configs/frozen_native/final_native_b0_448_s42_e8.json
do
  printf '\nStarting %s\n' "$config"
  "$CXR_PYTHON" -u tools/run_sprint_experiment.py --config "$config"
done
echo "FROZEN_NATIVE_FINAL_QUEUE_COMPLETED"
