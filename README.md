# 胸片多标签分类比赛：第一阶段

> **仓库集成**：本项目位于 `D:\challenge` 单仓库，项目根就是 `AI/`。
> 原始数据在 `AI/cxr_first/cxr_first/`（gitignored）；换机器时用 `CXR_DATA_ROOT`
> 指向实际挂载路径。代码快照 / 发布包流程见仓库根 `docs/workflow.md`。

当前入口：[11月12日数据核查与服务器交接](docs/SPRINT_2026-11-12.md)、
[历史冲刺计划](docs/PLAN_2026-11-10-final-sprint.md) 与
[规则复核](docs/REASSESSMENT_2026-10-31.md)。下文保留第一阶段环境与历史操作说明；
验证集使用、自监督与蒸馏权限以补充规则及上述复核为准。

目标是检查级十类 Macro-AUC；74.9 是外部参考成绩，不是本项目已复现的基线。

所有 Python 业务代码在 `src/cxr`，端到端验收在 `src/tests`。原始数据在 `cxr_first`，配置在 `configs`，运行产物在 `runs`。不修改原始 CSV/JPG。

## 环境

采用官方镜像为基础：

```text
crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:1.0
```

目标版本 Python 3.12、PyTorch 2.10.0、torchvision 0.25.0、CUDA runtime 12.8、cuDNN 9。镜像的实际版本需要运行检查确认，不能只相信标签。项目派生镜像只补充训练/验收依赖，不升级 torch/torchvision。uv 管理 Python 依赖；主机现有 Conda 环境不改动。

在 PowerShell、项目根目录执行：

```powershell
docker build -f src/Dockerfile -t cxr-phase1:official .
powershell -NoProfile -File src/run_docker.ps1 check
powershell -NoProfile -File src/run_docker.ps1 test
powershell -NoProfile -File src/run_docker.ps1 audit
```

真实训练前手动下载 [Torchvision 官方 ImageNet-1K EfficientNet-B0 权重](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth)，保留文件名 `efficientnet_b0_rwightman-7f5810bc.pth`，放入项目的 `.cache/torch/hub/checkpoints/`。

```powershell
powershell -NoProfile -File src/run_docker.ps1 weights
```

这条命令只校验本地文件，不发起下载。SHA256须以官方文件名提供的 `7f5810bc` 开头；校验通过后将完整摘要和来源写入 `docs/pretraining.json`。自动下载曾两次因完整性校验失败而停止，未使用损坏权重或退回随机初始化开展真实训练。

训练数据在容器内额外只读挂载；日志、权重和预训练缓存保存在项目目录。GPU 是本机 5060 Ti 8GB。T4 是官方验证硬件，不能据本机速度声称满足 T4 的 100ms 限制。

## 第一阶段实验

`local_image_mean.json` 与 `local_study_mean.json` 使用相同 seed、内部患者切分、1,024 个训练检查、384 个验证检查、224 分辨率、三轮训练。前者在每张图上监督、按检查平均概率，后者平均图像特征后按检查监督。`local_study_attention.json` 是后续视角注意力候选。小样本结果不能代表正式比赛性能。

```powershell
powershell -NoProfile -File src/run_docker.ps1 train -Config configs/local_image_mean.json
powershell -NoProfile -File src/run_docker.ps1 train -Config configs/local_study_mean.json
powershell -NoProfile -File src/run_docker.ps1 evaluate -Config configs/local_image_mean.json -Checkpoint runs/local_image_mean_224/best.pt
powershell -NoProfile -File src/run_docker.ps1 evaluate -Config configs/local_image_mean.json -Checkpoint runs/local_image_mean_224/best.pt -Split valid
powershell -NoProfile -File src/run_docker.ps1 predict -Config configs/local_image_mean.json -Checkpoint runs/local_image_mean_224/best.pt -Split test -Output runs/local_image_mean_224/submission.csv
powershell -NoProfile -File src/run_docker.ps1 benchmark -Config configs/local_image_mean.json -Checkpoint runs/local_image_mean_224/best.pt
```

中断恢复使用同一配置及 `-Checkpoint runs/local_image_mean_224/last.pt`，动作仍为 `train`；提高 epochs 后续跑。不要复用同一输出目录开始一个新的实验。

## 评价与提交

- 从 train.csv 按患者构造内部 holdout，再限量抽取检查；保存确切ID、CSV摘要及配置。
- 官方 valid 仅显式评价，不并入训练。test 不参与训练、模型选择或自监督预训练。
- 全十类独立 sigmoid，No Finding 不硬编码成其他九类的补集。
- 逐类 AUC 无正负两组时记录 null；只有十类 AUC 全部有定义才给出完整 Macro-AUC，另列可定义类别均值。
- F1 默认阈值0.5，仅是辅助指标。调整阈值不会改善 AUC。
- 单 run 的 `submission.csv` 保留全十类概率（4 列，含 `Subject_id`），供集成读取；最终 `runs/ensemble/submission.csv` 为官方 4 列 `Subject_id,Study_id,Predict_class,Probability`，默认按 0.5 阈值只写预测类别。底层 `predict` CLI 仍保留 `--include-subject/--threshold`，可生成全十类概率的诊断文件。主办方文字与截图的字段/阈值歧义仍需一次提交确认。

每次运行保存 `config.json`、`environment.json`、`split_manifest.json`、`metrics.jsonl`、`best.pt`、`last.pt`、`best_metrics.json`、`validation_predictions.csv`。权重保存与恢复不依赖外部医学预训练。

## 方法来源与下一步

首选基线使用 torchvision 的 ImageNet-1K EfficientNet-B0 预训练，来源与权重枚举写入每次运行的环境记录；不使用外部医学标注或医学专用权重。

- [Torchvision EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
- [Asymmetric Loss（ICCV 2021）](https://arxiv.org/abs/2009.14119)：作为不均衡损失候选，须与 BCE 对照。
- [CaMCheX（ML4H 2025）](https://arxiv.org/abs/2511.09581)：参考检查级融合思路；不使用其外部预训练权重或未提供的临床信息。

本地验收和对照完成后，再在学校4090开展全量训练、分辨率对照、多seed复验，最后选择满足≤500MB与推理预算的方案。截止时间2026-11-13 23:59北京时间。

