# 工作流：快照 → 打包 → 上传 → 训练 → 提交

## 0. 原则

1. **代码进 git，数据不进。** 数据在 `AI/cxr_first/cxr_first`（见 [data-layout.md](data-layout.md)）。
2. **一次有意义的改动一个 commit。** 打包默认要求工作区干净，逼着你在训练前留下可回溯的点。
3. **每次训练都带代码快照。** 快照回答「这个 `best.pt` 是哪份代码/配置/环境跑出来的」，比事后回忆可靠。
4. **发布包只从白名单取文件。** `release.json` 决定打什么，不靠 `.gitignore` 兜底。

---

## 1. 代码快照

`cxr train` 首次启动时在 **`AI/runs/<run>/code_snapshot/`** 写一份快照
（`snapshot.tar.gz` + `manifest.json` + `files.sha256`），并打印 sha256；resume 不重复拍。
服务器上没有 `.git` 时自动退化为文件树快照（`git.available=false`），不影响使用。

需要「训练前后各拍一次」时，用包装命令：

```powershell
python tools/code_snapshot.py run --root AI --out snapshots/ai/p8-s42 `
  --tag p8-s42 --config AI/configs/p8/p8_b0_224_12ep_s42.json `
  -- powershell -NoProfile -File AI/src/run_docker.ps1 train -Config configs/p8/p8_b0_224_12ep_s42.json
```

输出 `<out>/<时间>-before/`、`<out>/<时间>-after/`、`<out>/<时间>-result.json`；
`code_changed_during_run=true` 说明训练期间代码被改过，这次结果要谨慎看待。

手动补拍：

```powershell
python tools/code_snapshot.py capture --root AI --out snapshots/manual/20261111 `
  --tag submit-candidate --config AI/configs/p8/p8_b0_224_12ep_s42.json
```

校验：`python tools/code_snapshot.py verify runs\...\code_snapshot`。

### 保留策略

快照约 1–2MB/次，可以全留；`runs/` 和 `snapshots/` 都在 gitignore 里。
真正要长期留的是「提交对应的那一次」：`best.pt` + `code_snapshot` + `evaluation_*.json` 一起归档。

---

## 2. 发布打包

```powershell
python tools\package_release.py --target ai
python tools\package_release.py --target ai --dry-run     # 只看会打什么
python tools\package_release.py --target ai --allow-dirty # 带未提交改动（不推荐）
```

产物：

```text
dist\cxr_project-20261111-121500-a1b2c3d.tar.gz
dist\cxr_project-20261111-121500-a1b2c3d.tar.gz.sha256
```

包内：

```text
AI/
├── src/ configs/ docs/ tools/
├── tools/snapshot_lib.py, code_snapshot.py   # 服务器上也能继续拍快照
├── RELEASE.json      # git sha、dirty、环境、逐文件 sha256
└── SHA256SUMS
```

- 单文件超过 `max_file_mb`（12MB）直接失败，防止权重混入。
- 写完会重新读归档、逐个比对 sha256，再落 `.sha256`。

---

## 3. 上传服务器

```powershell
$env:CHALLENGE_SSH_HOST = '10.0.0.5'
$env:CHALLENGE_SSH_USER = 'student'
$env:CHALLENGE_SSH_PORT = '22'
$env:CHALLENGE_REMOTE_DIR = '~/challenge_upload'

powershell -NoProfile -File scripts\upload_to_server.ps1 -File dist\cxr_project-xxx.tar.gz
```

脚本会：本地校验 `.sha256` → `scp` 包和校验文件 → 远端 `sha256sum -c`。

服务器上：

```bash
cd ~/challenge_upload
tar -xzf cxr_project-xxx.tar.gz
cd AI
export CXR_DATA_ROOT=/workspace/cxr_first
uv sync --python 3.12
uv run python -m cxr train --config configs/p8/p8_b0_224_12ep_s42.json
```

若服务器只有 CSV + 320 缓存、没有原图，设 `export CXR_ALLOW_MISSING_IMAGES=1`（训练读 cache，
`read_split` 不再要求原图存在）。

---

## 4. 提交与留痕

```powershell
git add -A
git commit -m "submit(cxr): p8 ensemble valid 0.8080"
git tag -a submit/2026-11-11-0.8080 -m "Macro-AUC 0.8080"

python tools\package_release.py --target ai --version submit-2026-11-11-0.8080
```

一个提交包应包含：

| 文件 | 说明 |
|---|---|
| `submission.csv` | 平台提交结果 |
| `runs/<run>/best.pt` | 对应权重（总大小 ≤500MB） |
| `runs/<run>/code_snapshot/` | 代码快照（manifest 里有 sha256） |
| `RECEIPT.json` / `RELEASE.json` | 发布包清单 |
| 笔记 | 线上分数、提交时间、每日次数消耗（`AI/reports/r2/submission_log.md`） |

提交前检查：

- [ ] `git status` 干净，`git tag` 已打
- [ ] `python tools\repo_guard.py` 通过
- [ ] 模型总大小 ≤ 500MB（复赛）
- [ ] T4 上单张 ≤ 100ms
- [ ] 训练集/验证集使用符合规则（valid 不并入训练、test 不参与选模）

---

## 5. 大文件与密钥

守门分三层：

1. `.gitignore`：数据/缓存/权重/压缩包默认不跟踪；
2. `.githooks/pre-commit`：`git add` 后按**暂存区**检查，>10MB 或命中数据目录直接拒绝；
3. CI（`.github/workflows/repo-guard.yml`）：push 后全仓再查一次。

手动检查：

```powershell
python tools\repo_guard.py            # tracked + untracked（未被 ignore 的）
python tools\repo_guard.py --staged   # 暂存区
python tools\repo_guard.py --json     # 机器可读
```

误提交了怎么办：

```powershell
# 只是刚 add、还没 commit
git restore --staged path\to\big.bin

# 已经 commit、还没 push
git rm --cached path\to\big.bin
git commit --amend --no-edit

# 已经 push：不要 force push 前先备份；用 git filter-repo 清历史后再强推
```

密钥处理：立刻在平台吊销/轮换，再清历史；只删文件不等于密钥安全。

---

## 6. 常见问题

**`package_release.py` 报 dirty**
先 commit；确实要打未提交状态就加 `--allow-dirty`，并在提交记录里注明。

**pre-commit 拦了 PDF/图片**
检查是不是超过了 10MB；比赛规则 PDF 只有几百 KB，正常不会拦。真需要就调 `--max-mb`
并同步改 hook 与 CI，别只改一处。

**git status 很慢/出现大量文件**
确认 `.gitignore` 生效：`git check-ignore -v AI/cxr_first/cxr_first/train.csv`。

**训练时找不到 cache**
配置里 `cache_root` 是相对路径，确认在 `AI/` 目录下运行，或用 `CXR_CACHE_ROOT` 覆盖。
