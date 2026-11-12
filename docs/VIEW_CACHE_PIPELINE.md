# 缓存、视角条件化与重加权选模

这套改动只服务一件事：把 224 实验从“每步等 JPEG 解码”变成“GPU 主导”，同时让模型选型对齐 test 的视角构成。原始 CSV/JPG 不改。

## 1. 建 320 缓存

缓存只做 `EXIF 转正 → 灰度 L → 等比缩放 + pad 到 320`，**不做增强、不做 RGB、不做归一化**。增强仍在训练时在线做，因此 224/288/320 可以共用同一份缓存。

```bash
export CXR_CACHE_ROOT=/workspace/cache/320
export CXR_CACHE_SIZE=320
bash src/server/run.sh --mode native cache \
  --data-root "$CXR_DATA_ROOT" \
  --output "$CXR_CACHE_ROOT" \
  --size 320 --workers 8
```

- 33,984 张约 **2.1 GB**；8 进程在 Linux 上约 2–4 分钟。
- `manifest.json` 记录 `size`、源 CSV SHA256、张数和 `cache_id`；抽样做逐像素校验。
- 中断后重复执行会跳过已有文件；`--overwrite` 全量重建。
- 训练时设置了 `CXR_CACHE_ROOT`，run 目录会写 `cache_manifest.json`；resume 会校验 `cache_id`，不同缓存之间不能恢复同一 run。
- 没有缓存时自动走原图 + `PIL draft` 解码，仍然可用；只是慢。

`PIL draft` 对原图推理也有用：224 下解码+预处理约从 26ms 降到 12ms，像素平均差 0.0023。

## 2. 视角/体位条件化

CSV 里的 `ViewPosition` 和 `PatientOrientation` 现在都会进入模型：

- `view_conditioning`：给每张图的特征加一个 6 类 view embedding（AP/PA/LATERAL/LL/PA LLD/unknown）。
- `orientation_conditioning`：加一个 4 类 embedding（unknown/Erect/Recumbent/Sitting）。
- `study_attention` 仍保留原来的 view embedding，并加了无参数的 LayerNorm；新 embedding 用 `std=0.02` 初始化，避免一开始盖住 EfficientNet 特征。
- `max_eval_views`：验证/推理也最多取 2 个视图（一个正位 + 一个侧位），与训练一致；不设则保留旧的全视图行为。

这两个字段默认关闭，所以旧 checkpoint 仍可加载/恢复。新配置显式打开。

## 3. 重加权选模

`metrics.summarize` 现在同时输出：

- `macro_auc`：原始 valid AUC，旧口径。
- `macro_auc_reweighted`：按 **test.csv 的视角构成** 给每个 study 加权后的 AUC。
- `per_class_auc_reweighted`：逐类加权 AUC。
- `view_group_auc`：AP only / AP+lateral / PA only / PA+lateral / lateral only / unknown 各组的 AUC。

`reweight_views=true` 时，best epoch 用 `macro_auc_reweighted` 选择；否则仍用原始 `macro_auc`。test.csv 只提供元数据，不读标签。

## 4. 推荐实验顺序

先建缓存，再跑同一配置的 base / view 对照；两组只差 `view_conditioning` + `orientation_conditioning`：

```bash
# baseline
bash src/server/run.sh --mode native train --config configs/server_base_asl_224.json
# view + orientation
bash src/server/run.sh --mode native train --config configs/server_view_asl_224.json
```

或让 `configure.py` 直接生成带新字段的配置：

```bash
bash src/server/run.sh --mode native configure \
  --image-size 224 --batch-size 16 --workers 8 --epochs 12 --seed 42 --loss asl
```

每次只改一个因素；比较时看 `macro_auc_reweighted`、AP-only 组 AUC 和逐类 AUC，不要只看原始 `macro_auc`。0.5 点在 228 个患者的 valid 上分辨不出来，优先接受 ≥1 点且 AP-only 不下降的改动。

## 5. 训练侧顺手修掉的吞吐问题

- DataLoader：`prefetch_factor=4`、`persistent_workers`、`worker_init_fn` 把每个 worker 的 `torch` 线程设为 1，避免 8 核上 4×4 线程互抢。
- 训练循环不再每步 `.item()` + print；`log_interval` 控制频率（新配置 50）。
- `.to(device, non_blocking=True)`；`cudnn.benchmark`、`channels_last` 可选（新配置默认开）。
- warmup（`warmup_epochs`）+ `grad_clip`；旧配置默认 0，行为不变。

## 6. 输出字段

`best_metrics.json` / `metrics.jsonl` 新增上面这些字段；`submission.csv` 格式不变（1414 检查 × 10 类 = 14,140 行）。`predict` 支持 `tta=true`（水平翻转平均）和 `max_eval_views`，但 TTA 没在官方 valid 上验证前不要默认打开。
