#!/usr/bin/env bash
# 学校 Linux GPU 服务器：先 D1 探针（go/no-go），再 B1 终训（6 job）。
# 用法：
#   export CXR_DATA_ROOT=/path/to/cxr_first/cxr_first       # 含 train.csv/valid.csv/test.csv
#   export CXR_CACHE_ROOT=/path/to/cache/320                 # 可选：320/448 处理图缓存，加速
#   bash scripts/run_school.sh [data_root] [all|d1|b1]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1

DATA_ROOT="${CXR_DATA_ROOT:-${1:-}}"
JOBS="${2:-all}"
CACHE_ROOT="${CXR_CACHE_ROOT:-}"

uv sync  # 幂等，确保环境就绪

echo "=== env ==="
uv run --no-sync python -c "import sys, torch, timm; print(sys.version.split()[0], 'torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 数据根校验
if [ -z "$DATA_ROOT" ]; then
    echo "ERROR: 需要 CXR_DATA_ROOT 指向含 train.csv/valid.csv/test.csv 的目录" >&2
    exit 1
fi
for f in train.csv valid.csv test.csv; do
    [ -f "$DATA_ROOT/$f" ] || { echo "ERROR: 缺 $DATA_ROOT/$f" >&2; exit 1; }
done
export CXR_DATA_ROOT="$DATA_ROOT"
echo "=== data root: $DATA_ROOT ==="
mkdir -p runs

# 缓存校验（可选加速；D1 探针读原始图不受影响）
CACHE_SIZE=""
if [ -n "$CACHE_ROOT" ]; then
    if [ ! -f "$CACHE_ROOT/manifest.json" ]; then
        echo "ERROR: CXR_CACHE_ROOT=$CACHE_ROOT 无 manifest.json" >&2
        exit 1
    fi
    CACHE_SIZE="$(uv run --no-sync python -c "import json; print(json.load(open('$CACHE_ROOT/manifest.json'))['size'])")"
    CACHE_PARTIAL="$(uv run --no-sync python -c "import json; print(json.load(open('$CACHE_ROOT/manifest.json'))['partial'])")"
    if [ "$CACHE_PARTIAL" = "True" ]; then
        echo "ERROR: 缓存是 partial（不完整），不能用于训练" >&2
        exit 1
    fi
    echo "=== cache: $CACHE_ROOT size=${CACHE_SIZE}px (完整) ==="
fi

# ---- D1 线性探针（go/no-go 闸门）----
if [ "$JOBS" = "all" ] || [ "$JOBS" = "d1" ]; then
    echo "=== D1 线性探针 ==="
    uv run --no-sync python tools/d1_linear_probe.py --output runs/d1_probe_report.json
    uv run --no-sync python - <<'PY'
import json
r = json.load(open("runs/d1_probe_report.json"))
print("macro_auc =", round(r["macro_auc"], 4))
print("per_class =", [None if v is None else round(v, 3) for v in r["per_class_auc"]])
print("GO" if r["macro_auc"] >= 0.80 else "NO-GO", "(>=0.80 才投 D2 自监督微调)")
PY
fi

# ---- B1 终训（valid 并入 + 内部早停，6 job 串行）----
if [ "$JOBS" = "all" ] || [ "$JOBS" = "b1" ]; then
    echo "=== B1 终训（6 job，串行）==="
    for config in configs/p8b1/*.json; do
        name="$(basename "$config" .json)"
        rundir="runs/p8b1/$name"
        if [ -f "$rundir/best.pt" ]; then
            echo "skip $name (best.pt 已存在)"
            continue
        fi
        # 该 job 是否走缓存：image_size <= cache_size 才能用，否则读原始图
        if [ -n "$CACHE_SIZE" ]; then
            img_size="$(uv run --no-sync python -c "import json; print(json.load(open('$config'))['image_size'])")"
            if [ "$img_size" -le "$CACHE_SIZE" ]; then
                export CXR_CACHE_ROOT="$CACHE_ROOT"
                export CXR_CACHE_SIZE="$CACHE_SIZE"
                echo "  $name: 用缓存 (${CACHE_SIZE}px >= ${img_size}px)"
            else
                unset CXR_CACHE_ROOT CXR_CACHE_SIZE
                echo "  $name: image_size=${img_size}px > 缓存 ${CACHE_SIZE}px，读原始图"
            fi
        fi
        echo "=== $name train start $(date -Is) ==="
        uv run --no-sync python -m cxr train --config "$config" 2>&1 | tee "runs/$name.log"
        echo "=== $name train done $(date -Is) ==="
    done
fi

echo "=== done ==="
