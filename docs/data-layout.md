# 数据布局

## 当前位置

胸片原始数据在仓库内的 `AI/cxr_first/cxr_first/`（train/valid/test + `*.csv`，约 52 GB）。
该目录已被 `.gitignore` 排除，不会进入版本库。

```text
AI/cxr_first/cxr_first/
├── train/ valid/ test/      # 按患者分组的 JPG
├── train.csv valid.csv test.csv
└── ...
```

数据需要放到其他位置时，用环境变量覆盖，配置里的相对路径无需修改：

```powershell
$env:CXR_DATA_ROOT = 'D:\challenge\AI\cxr_first\cxr_first'
```

`AI/src/cxr/config.py` 原生支持 `CXR_DATA_ROOT`。

## 数据一致性核对

```powershell
python tools\data_manifest.py --name cxr --root AI\cxr_first\cxr_first
# → docs/data-manifests/*.json（只记录统计、CSV 全量 sha256、图片抽样 sha256）
```

服务器数据就位后再跑一次，对比 `tables[].sha256` 与 `sample.items[].sha256`
即可确认是同一份数据。

## 上传到算力平台

| 数据 | 体积 | 建议 |
|---|---:|---|
| 胸片原图 | 约 52 GB | 走平台「数据集管理」/共享存储；不要用 scp 反复传 |
| 320/448 缓存 | 2–4 GB | 服务器缺原图时按需上传，或在服务器现场 `tools/build_cache.py` |
| `runs/`、权重 | 约 0.8 GB | 除恢复训练外不需要；复赛按规则单独准备模型 |

JPG 本身已压缩，再 gzip 收益 < 2%，不要为了省流量重复压缩。
`scripts/pack_dataset.ps1` 只用于确实需要本地留档/单文件传输的场景，默认 dry-run。

## 服务器上的数据

```bash
export CXR_DATA_ROOT=/workspace/cxr_first
# 服务器只有 CSV + 缓存、没有原图时
export CXR_ALLOW_MISSING_IMAGES=1
```

## 空间预算

| 项 | 体积 |
|---|---:|
| CXR 原始数据 | 约 52 GB |
| 发布包 | < 20 MB |
| 每次训练快照 | 约 1–2 MB |
