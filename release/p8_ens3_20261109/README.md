# 胸片赛题 · 复赛交付包（p8_ens3_20261109）

多标签胸部 X 光分类，10 类 macro-AUC。本包是**可复现的最终推理交付**：相同权重 + 相同输入
必然得到与 `submission.csv` 一致的输出。

## 包内容

```
submission.csv                 最终预测（1,414 检查 × 10 类 = 14,140 行）
cost.py                        根目录适配器（赛方口径）
code/
  predict.py                   唯一推理入口（torch / torchvision / Pillow / numpy）
  cost.py                      耗时与一致性核验
  requirements.txt
  environment.json
model/
  b0_224_best.pt  b0_320_best.pt  b0_448_best.pt
  deployment.json              成员、尺寸、SHA-256、融合权重、阈值、预处理
  files.sha256
pretrain_model/README.md       预训练来源、链接、训练数据与许可证
licenses/LICENSES.md           第三方许可证全文与来源
模型核验材料与数据声明清单.xlsx  模型/数据使用声明
```

## 模型

三个 EfficientNet-B0（ImageNet-1k 预训练）分别在 224 / 320 / 448 输入上微调，
每个检查取所有视图的概率均值，再按 **1:1:1** 融合。无 TTA、无患者聚合、无阈值调优。

| 成员 | 输入 | 权重文件 |
|---|---:|---|
| b0_224 | 224 | `model/b0_224_best.pt` |
| b0_320 | 320 | `model/b0_320_best.pt` |
| b0_448 | 448 | `model/b0_448_best.pt` |

模型合计约 47 MB（解压后），远低于 500MB 上限。

## 运行

```bash
# 1) 预测（输出与包内 submission.csv 一致）
python code/predict.py --data-root /path/to/cxr_first/cxr_first --output submission_reproduced.csv

# 2) 耗时与一致性（正式口径：100 图 + 5 图预热 + 3 轮）
python cost.py submission.csv pictures_100 pictures_5 --data-root /path/to/cxr_first/cxr_first
```

`--data-root` 需要包含 `test.csv` 与 `test/<Subject_id>/<Study_id>/<Dicom_id>.jpg`。

## 数据使用声明

- 训练只使用主办方提供的 train（18,513 检查）；valid（859 检查）只用于模型选择；
- test 从未参与训练、自监督、伪标签、统计量估计或权重选择；
- 未使用任何外部数据、医学预训练权重或测试时训练；
- valid 未并入最终训练。

## 已知状态

- 官方 valid macro-AUC：**0.808027**（三模集成；单模最好 0.797938）；
- 复诊 dev（200 患者 / 1,347 检查）：**0.7526**；
- T4 端到端延迟：**未实测**（本机 CPU/5060Ti 数字不能替代 T4 验收）。
