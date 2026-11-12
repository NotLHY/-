#!/usr/bin/env bash
# 学校 Linux GPU 服务器环境搭建（裸机 uv）。
# 前置：CUDA 12.8 驱动已装、能访问 download.pytorch.org / huggingface.co。
# 用法：bash scripts/setup_school.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. 安装 uv（若缺失）
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv $(uv --version)"

# 2. 同步依赖（torch 2.10 + torchvision + numpy + sklearn + pydantic + timm）
echo "[setup] uv sync ..."
uv sync

# 3. 预下载 ImageNet EfficientNet-B0 权重（torchvision 会用本地缓存并校验）
CKPT="$HOME/.cache/torch/hub/checkpoints"
mkdir -p "$CKPT"
WEIGHT="$CKPT/efficientnet_b0_rwightman-7f5810bc.pth"
if [ ! -s "$WEIGHT" ]; then
    echo "[setup] downloading EfficientNet-B0 weights ..."
    curl -L -o "$WEIGHT" https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth
fi
ls -lh "$WEIGHT"

echo "[setup] done. 下一步："
echo "  export CXR_DATA_ROOT=/path/to/cxr_first/cxr_first   # 含 train.csv/valid.csv/test.csv 的目录"
echo "  bash scripts/run_school.sh"
