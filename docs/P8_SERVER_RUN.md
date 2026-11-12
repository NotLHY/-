# 在 AutoDL 上跑 p8（R1/R2/R3）

这份文件只讲服务器侧的操作。本地已经验证过：训练集 18,513 → 17,166，与 dev 患者交集 0，
官方 valid 的 859 检查不动。

## 0. 包里有什么

```
AI/src/cxr/holdout.py       新增：留出清单的加载与校验
AI/src/cxr/cli.py           新增 --holdout-manifest
AI/src/cxr/splits.py        训练时剔除留出患者
AI/src/cxr/runtime.py       把清单传给切分
AI/src/cxr/inference.py     评估时只打分留出检查
AI/configs/p8/*.json        R1/R2/R3 三档分辨率
docs/splits/longitudinal_200.json   200 患者 / 1,347 检查
```

解包到仓库根（含 `AI/` 和 `docs/` 的那一层）：

```bash
cd /root/autodl-tmp/challenge          # 换成你的仓库根
tar -xzf cxr_p8_20261102.tar.gz
```

## 1. 先确认数据路径

p8 配置里的 `data_root` 是相对路径 `../../cxr_first/cxr_first`。如果你的数据不在
`<仓库根>/AI/cxr_first/cxr_first`，用环境变量覆盖（优先级最高，不用改配置）：

```bash
export CXR_DATA_ROOT=/root/autodl-tmp/AI/cxr_first/cxr_first
```

**不要**设 `CXR_CACHE_ROOT`：p8 走原图无缓存，之前实测无缓存比缓存训练好 +0.67 分，
而且 448 的缓存 key 和新清单对不上。

## 2. 三分钟自检（不训练）

```bash
cd AI && export PYTHONPATH=src
python - <<'PY'
from pathlib import Path
from cxr.config import load_config
from cxr.holdout import Holdout
from cxr.records import read_available
from cxr.splits import choose_studies
cfg = load_config(Path("configs/p8/p8_b0_224_12ep_s42.json"))
data = read_available(cfg.data_root, splits=("train", "valid"))
h = Holdout.load(Path("../docs/splits/longitudinal_200.json"))
train, valid, m = choose_studies(cfg, data, h)
print("train", len(train), "valid", len(valid), "holdout", m["holdout"])
assert len(train) == 17166 and len(valid) == 859
print("OK")
PY
```

期望输出 `train 17166 valid 859 ...`。不对就别开训练。

## 3. 跑 R1（224 锚点）

```bash
nohup python -m cxr train \
  --config configs/p8/p8_b0_224_12ep_s42.json \
  --holdout-manifest ../docs/splits/longitudinal_200.json \
  > runs/p8_r1.log 2>&1 &
tail -f runs/p8_r1.log
```

产物在 `AI/runs/p8/p8_b0_224_12ep_s42/`：`best.pt`、`last.pt`、`metrics.jsonl`、
`split_manifest.json`（里面 `holdout` 字段应显示 `excluded_studies: 1347`）。

显存不够（8GB 卡）就把 `batch_size` 降到 4，其余不动；不要动 `epochs`、`learning_rate`、
`warmup_epochs`、`grad_clip`——R1/R2/R3 必须只有分辨率一个变量。

## 4. 在 dev 上打分（不是官方 valid）

```bash
python -m cxr evaluate \
  --config configs/p8/p8_b0_224_12ep_s42.json \
  --checkpoint runs/p8/p8_b0_224_12ep_s42/best.pt \
  --split train \
  --holdout-manifest ../docs/splits/longitudinal_200.json
```

输出 `evaluation_dev.json`（1,347 检查）和 `evaluation_dev_predictions.csv`。
官方 valid 的 `evaluation_valid.json` 也顺手跑一次，作为外部复核：

```bash
python -m cxr evaluate --config configs/p8/p8_b0_224_12ep_s42.json \
  --checkpoint runs/p8/p8_b0_224_12ep_s42/best.pt --split valid
```

## 5. 断点续跑

```bash
python -m cxr train --config configs/p8/p8_b0_224_12ep_s42.json \
  --holdout-manifest ../docs/splits/longitudinal_200.json \
  --resume runs/p8/p8_b0_224_12ep_s42/last.pt
```

## 6. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `Run artifacts already exist` | 目录已有产物 | 换 `output_dir` 或加 `--resume` |
| `Holdout patients appear in the validation.csv` | 清单患者和 valid 撞了 | 检查是不是拿错了 manifest |
| `Holdout manifest lists N studies absent from the split` | 训练/评估用的不是同一份 CSV | 核对 `CXR_DATA_ROOT` |
| 评估产出 `evaluation_train.json` | 没带 `--holdout-manifest` | 带上，产物名才是 `evaluation_dev.json` |
