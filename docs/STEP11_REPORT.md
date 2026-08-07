# 第 11 步：下游机器人学习验证

## 验收结论

第 11 步已完成。项目在真实 RTX 4090、Isaac Lab 和 BeyondMimic G1 tracking 任务中完成了 12/12 个受控 PPO 作业、9000 个确定性评价 episode、种子级置信区间、离线—学习相关性、单维消融和预注册校准门控。

实验得到的是有效的负结果：`quality_score_v1` 未能预测本实验中的策略学习价值，因此继续保持 `score_is_calibrated=false`，没有根据结果事后修改权重或生成 v2。

## GPU 自检结论

先前的 GPU 阻塞来自 Codex 工具沙箱，不是宿主机、驱动或 Conda 环境错误。在非沙箱的正式执行上下文中：

```text
GPU                              NVIDIA GeForce RTX 4090
Driver                           580.126.09
显存                             24564 MiB
Python                           /opt/conda/envs/isaaclab/bin/python
PyTorch                          2.7.0+cu128
torch.cuda.is_available()        true
torch.cuda.device_count()        1
```

虽然工具层注入的 `NVIDIA_VISIBLE_DEVICES=void` 仍可见，但非沙箱进程可正常调用 CUDA；4096 环境训练、PhysX 仿真、checkpoint 保存和策略评价均已实跑完成。机器可读证据位于 `outputs/step11_learning_experiment/gpu_execution_audit.json`。

## 冻结实验设计

- 任务：`Tracking-Flat-G1-v0`，whole_body_tracking 提交 `cd651720...`，Isaac Lab 提交 `c91a125c...`；
- cohort：五个动作类别内各取 Top-1、固定随机 1、Bottom-1；Full 使用全部 15 条；
- 数据量：Top/Random/Bottom 各 5 条、32,770 帧；Full 15 条、98,310 帧；
- policy seeds：`11/22/33`，共 4×3=12 个策略；
- 每策略：4096 environments、300 iterations、24 steps/environment，即 29,491,200 transitions；总计 353,894,400 transitions；
- 评价：每条 motion、每个 seed 固定 100 episodes；三主组各 1500 episodes，Full 4500 episodes；
- 采样：先对 motion 等概率，再对合法时间均匀采样，不跨动作边界，关闭失败自适应重采样；
- 指标：成功率、收敛 iteration、episode length、MPJPE、W-MPJPE、跌倒率；组间使用三 seed 的 t-based 95% CI。

原上游默认的 30,000 iterations 预计需要约 17.5 GPU-days/12 policies。正式结果产生前，根据 4096 环境 GPU smoke benchmark 将预算冻结为 300 iterations。该实验因此回答“固定 29.5M transitions 的计算受限学习价值”，不能外推为 30,000 iterations 下的最终性能。

## 正式结果

表中为三 seed 均值和 95% CI；未在 300 iterations 内收敛的策略以 301 右删失。

| Cohort | 收敛 iteration | 成功率 | Episode length (s) | MPJPE (m) | W-MPJPE (m) | 跌倒率 |
|---|---:|---:|---:|---:|---:|---:|
| Top-K | 301.0 [301.0, 301.0] | 0.0000 [0.0000, 0.0000] | 0.7945 [0.7252, 0.8638] | 0.1045 [0.1020, 0.1069] | 0.1162 [0.1138, 0.1186] | 0.1447 [0.1212, 0.1681] |
| Random-K | 301.0 [301.0, 301.0] | 0.0000 [0.0000, 0.0000] | 0.8140 [0.7427, 0.8854] | 0.1058 [0.1030, 0.1087] | 0.1194 [0.1165, 0.1224] | 0.1613 [0.0741, 0.2486] |
| Bottom-K | 204.0 [199.0, 209.0] | 0.1920 [0.1789, 0.2051] | 2.4544 [2.3326, 2.5762] | 0.0990 [0.0933, 0.1047] | 0.1120 [0.1061, 0.1180] | 0.1240 [0.0843, 0.1637] |
| Full | 194.7 [160.8, 228.6] | 0.0642 [0.0633, 0.0652] | 1.4963 [1.4451, 1.5475] | 0.1035 [0.0994, 0.1075] | 0.1167 [0.1120, 0.1213] | 0.1191 [0.0992, 0.1390] |

种子配对比较显示 Bottom-K 相对 Top-K 的成功率差为 `+0.1920`，95% CI `[+0.1789,+0.2051]`；相对 Random-K 结果相同。这个反向排序不能解释为“低质量数据更好”，因为 Full 策略的成功几乎完全来自较容易保持 10 秒的 `ground1_subject1`：三个 seed 分别为 `0.97/0.96/0.96`，其余 14 条动作均为 0。结果反映出轨迹内在难度与当前离线质量分数之间存在明显混杂。

## 相关性与校准

以下相关性使用 Full 策略对 15 条动作的三 seed 均值；负 MPJPE/负跌倒率越大越好。

| 学习目标 | `quality_score_v1` Spearman rho | 10,000 次 bootstrap 95% CI |
|---|---:|---:|
| 成功率 | -0.4330 | [-0.7122, -0.4342] |
| Episode length | -0.1643 | [-0.6771, 0.4493] |
| 负 MPJPE | 0.1964 | [-0.4291, 0.6703] |
| 负 W-MPJPE | 0.2857 | [-0.3897, 0.7095] |
| 负跌倒率 | 0.0643 | [-0.5704, 0.7086] |

leave-one-motion-category-out 的四维非负模型得到 OOF Spearman `-0.1862`，bootstrap 95% CI `[-0.5032, 0.0313]`。预注册门槛要求 OOF Spearman 为正且 CI 下界大于 0，因此门控失败。15 条 Full 动作中只有 1 条成功率非零，成功率监督也过于稀疏，不能支持可靠的权重学习。

最终决定：

```text
score_is_calibrated = false
outcome = retain_v1_uncalibrated_negative_result
```

该结论说明现有离线指标仍可用于发现轨迹跳变、限位、接触和 fidelity 问题，但不能把总分解释为策略可学习性。后续若继续改进，应在新的预注册实验中加入动作难度/动态可跟踪性指标、扩大每类 motion 数量并提高训练预算；不能在当前 15 条结果上事后调权重。

## 交付与复现

主要机器可读产物：

- `outputs/step11_learning_experiment/training_jobs.csv`：12/12 作业状态、命令、checkpoint 和评价路径；
- `outputs/step11_learning_experiment/analysis/cohort_summary.csv`：组均值与 95% CI；
- `outputs/step11_learning_experiment/analysis/paired_cohort_comparisons.csv`：种子配对组间差异；
- `outputs/step11_learning_experiment/analysis/motion_learning_results.csv`：逐策略、逐动作结果；
- `outputs/step11_learning_experiment/analysis/offline_learning_correlations.csv`：全部离线维度相关性；
- `outputs/step11_learning_experiment/analysis/dimension_ablation.csv`：单维消融；
- `outputs/step11_learning_experiment/analysis/calibration_decision.json`：校准门控决定；
- `outputs/step11_learning_experiment/step11_manifest.json`：最终 complete manifest 和 36 个关键文件 SHA-256。

正式训练可断点恢复：

```bash
/opt/conda/envs/isaaclab/bin/python integrations/beyondmimic/run_formal.py \
  --config configs/step11_beyondmimic.yaml \
  --output-dir outputs/step11_learning_experiment --max-parallel 3
```

重新生成统计和完整性清单：

```bash
conda run -n gmr python -m gmr_quality finalize-step11-experiment \
  --config configs/step11_beyondmimic.yaml \
  --output-dir outputs/step11_learning_experiment --json
```

本步骤只验证无物体的 whole-body tracking 学习价值，不声称完成物体 6D、手—物接触或操作任务成功验证；这些内容仍属于第 12 步。
