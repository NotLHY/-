# weak 回传与最终终训决策

weak 回传 SHA256：`aeeb69e86b76533e22905942260bed939a2290a8aebe8999ed9c4568b3b3c0d1`。
包 CRC、侧车摘要、快照归档与逐文件摘要均通过；receipt completed，source_unchanged=true，
服务器使用干净的 `8eec794`。最佳第 4 轮，第 8 轮正常早停。总耗时约 71 分钟。

## 配对结果

同一固定 dev、同一官方 valid，严格对齐检查和十类预测后重算；完整结果在
`docs/evidence/weak_320_review.json`。未读取 test 做分析。

| 模型/组合 | dev | valid |
|---|---:|---:|
| control 320 | 0.746628 | 0.781450 |
| weak 320 | 0.745376 | 0.788284 |
| 原生三模 | 0.752542 | 0.808027 |
| 原三模仅用 weak 替换 320 | 0.756993 | 0.807001 |

2,000 次患者簇配对 bootstrap（种子 20261112，缺类重采样 0）：

| 差值 | dev 95% 区间 | valid 95% 区间 |
|---|---|---|
| weak − control | [-0.00724, 0.00471] | [0.00177, 0.01295] |
| weak 替换组 − 原三模 | [0.00163, 0.00712] | [-0.00471, 0.00242] |

弱类加权没有显示稳定的 dev 收益；不能因为 valid 提高就声称该配方已证实有效。
替换组在 dev 更好，在 valid 差约 0.00103 且区间含零，是可保留的备选，不能说被证明更差，
也不能直接推断隐藏 test 会提高。所有区间均为探索性，未校正 checkpoint 选择和反复验证偏差。

## 冻结决定

不继续扫权重、扫 seed、追加 224 弱类复验或重复 A2/A3。
原生三模保留为已验证发布基线；weak 替换组保留作备选，不立即提交。

下一步唯一训练队列是原生 B0 配方的全量终训：

| 分辨率 | 固定轮数 | 学习率调度总长度 | batch |
|---|---:|---:|---:|
| 224 | 6 | 12 | 8 |
| 320 | 7 | 12 | 8 |
| 448 | 8 | 12 | 6 |

轮数来自已经存在的三个 p8 historical best（按官方 valid 选取）；不是这次看完 test 后选择。
这些 historical best 的 valid 已参与过选择，不能描述成从未触碰的盲测。
本轮选择保守地沿用已验证原三模配方，不把未证实的 class_loss_weights 叠加进终训。
原始 checkpoint 摘要与选择来源在 `docs/evidence/frozen_native_final_decision.json`。

本轮执行前检查：control/weak manifest 完全一致，配置仅 run 名称、输出目录与损失权重不同；
三个冻结配置在真实 CSV 上均得到 19,372/0 的训练/验证范围，学习率逐轮前缀与各自原配置一致。
队列 Bash 语法检查通过；固定全量训练、恢复、预测与候选打包的端到端回归通过。

每个模型从同一 ImageNet 权重重新训练，使用全部 train+valid = 19,372 检查，包含原 dev。
`validation=none`，不做早停或验证反馈；只使用固定轮数的 `final.pt`，不再产生 best.pt。
保留原 12 轮 schedule 的前缀；数据增多后每轮更新步数改变，不声称优化过程逐步等价。
不对这些终训模型再用 dev/valid 选型或汇报独立验证 AUC。

这是数据增量候选，收益未验证，不保证达到 0.81。按 320 实测约 7 分钟/轮估计，
整队列暂预留 4–6 小时；448 实测后修订，不能把估计当承诺。若发生故障，保留权重及快照，
不要删目录重跑；按对应 run 的 last.pt 恢复。脚本第一次启动只支持新队列，重复启动会被锁或已有产物检查拒绝。

## 服务器操作

确认 control/weak 已结束且不再运行训练后再拉代码，训练期间不 git pull。

```bash
cd /3251903008/lhy/challenge-sprint
git pull --ff-only
export CXR_PYTHON=/3251903008/lhy/envs/cxr312/bin/python
export CXR_DATA_ROOT=/3251903008/lhy/cxr_first/cxr_first
export CUDA_VISIBLE_DEVICES=GPU-c38d4837-2074-435b-d389-59104eb7dec1
mkdir -p runs/sprint_final
nohup bash scripts/run_frozen_native_final.sh > runs/sprint_final/queue.log 2>&1 &
echo "Queue PID: $!"
```

按 224 → 320 → 448 顺序，一张卡运行。每个 run 保存快照、完整配置、数据 manifest、训练日志、
last.pt、final.pt 和 feedback.zip；不自动生成 test 预测或消耗平台提交。

监控队列：`tail -n 40 /3251903008/lhy/challenge-sprint/runs/sprint_final/queue.log`。
每个 run 的逐步训练日志在 `runs/sprint_final/final_native_b0_<size>_s42_e<epochs>/experiment.log`。
全量终训出现 `macro_auc=None` 是预期行为，表示没有用训练数据假装独立验证。

224 结束就先回传它的 feedback.zip/.sha256，其他模型继续；三个模型结束后都回传。
我们核对 full_train_count=19,372、valid=0、固定轮数、训练稳定性、权重摘要和冻结方案一致性。

全部完成后按预先冻结的 1:1:1 原生概率平均生成候选 B，用仓库
`tools/package_sprint_candidate.py` 从这三个 final.pt 生成 CSV、模型与代码草稿包。
仍须独立完成 T4 延迟、一致性和官方声明表更新。候选 A（原基线或已确认备选）的平台提交状态
及 76.77 对应组合仍须核实，避免重复提交相同预测浪费仅有两次机会。
