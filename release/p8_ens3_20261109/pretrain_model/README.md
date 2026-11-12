# 预训练来源与许可证说明

## 1. 初始权重（backbone 初始化）

| 项目 | 内容 |
|---|---|
| 骨干 | `torchvision.models.efficientnet_b0`（EfficientNet-B0） |
| 权重枚举 | `torchvision.models.EfficientNet_B0_Weights.DEFAULT`（= `EfficientNet_B0_Weights.IMAGENET1K_V1`） |
| 权重 URL | https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth |
| 训练数据 | ImageNet-1k（ILSVRC2012），1,000 类自然图像 |
| 医学数据 | **未使用**（预训练与训练均不含任何医学影像数据） |
| 下载哈希 | 交付包内 `pretrain_model/README.md` 记录；本地缓存文件 `~/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth` |

## 2. 推理时实际依赖

最终模型是**在 ImageNet-1k 预训练权重上微调的 EfficientNet-B0 + 线性分类头**，推理只依赖
`torch`、`torchvision`、`numpy`、`Pillow`，**不需要 timm、不需要任何联网下载**。

## 3. 代码许可证

| 组件 | 许可证 | 说明 |
|---|---|---|
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| torchvision | BSD-3-Clause | https://github.com/pytorch/vision/blob/main/LICENSE |
| Pillow | HPND | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |

## 4. 使用与再分发声明

- 预训练权重来自 torchvision 官方发布，公开可直达、非 gated、非 NC；
- 上游训练数据为 ImageNet-1k，其使用条款以 ImageNet 官方（https://image-net.org/）为准；
- 本队仅使用主办方提供的训练数据做微调，未引入任何外部标注数据或医学预训练权重；
- 权重可随本交付包再分发用于赛事复核。

> 复核要点（写给评委）：本包不含 timm、不含 HuggingFace 缓存、不含任何外部数据；推理可完全离线运行。
