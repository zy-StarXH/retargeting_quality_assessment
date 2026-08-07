# 第 10 步：分层离线实验与评价器 v1 冻结报告

## 结论

第 10 步已完成一个可复现的分层离线实验：从本机 LAFAN1 冻结选择 15 条完整动作，按
`standing_walking`、`running_jumping`、`dance`、`complex_dynamic` 和
`floor_motion` 五类各取 3 条，经相同 GMR 配置重新重定向到 Unitree G1，并对全部
90,540 帧执行平滑性、关节可行性、接触合理性和人体—机器人 fidelity 评价。

评价器冻结为 `v1`，但继续明确标记 `score_is_calibrated=false`。本步证明评价工具能够在
分层真实轨迹上稳定产生可解释的离线排序、异常门控和可复现实验表；它尚未证明高分数据能
提升机器人策略学习，该结论必须由第 11 步受控训练实验验证。

## 冻结实验设计

- 冻结选择表：[`data/step10_source_selection.csv`](data/step10_source_selection.csv)；
- 每类 3 条完整序列，不从长序列中挑选“好看片段”；
- 15 条机器人 PKL 均由本项目无头调用 GMR 重新生成，源 BVH、输出 PKL、GMR loader、
  IK 配置、XML 和重定向实现均记录 SHA-256；
- 源帧与输出帧全部精确对齐，序列化后数组逐元素复核一致；
- 站立/行走、跑跳、舞蹈和复杂动态使用 `feet_only`；明确的地面动作使用
  `floor_motion`，因此不会把非足部触地本身武断判为失败；
- 总分继续使用平滑性 0.30、关节可行性 0.25、接触 0.30、fidelity 0.15。权重没有依据
  本次总分的自相关重新拟合，以免循环论证。

v1 相对既有口径增加 `1e-6 rad` 关节硬限位数值容差。原始超限量仍完整报告，近限位惩罚
不变；该容差只避免浮点求解的微小越界直接触发硬失败。冻结配置见
[`configs/unitree_g1_v1.yaml`](configs/unitree_g1_v1.yaml)，哈希锁定记录见
[`configs/unitree_g1_v1.freeze.json`](configs/unitree_g1_v1.freeze.json)。

## 分层结果

| 动作类别 | 数量 | 帧数 | v1 均值 ± 样本标准差 | 最小–最大 | hard-fail |
|---|---:|---:|---:|---:|---:|
| standing/walking | 3 | 19,904 | 82.11 ± 8.58 | 74.22–91.24 | 2/3 |
| running/jumping | 3 | 22,663 | 83.49 ± 10.65 | 72.15–93.27 | 2/3 |
| dance | 3 | 17,487 | 76.13 ± 7.79 | 71.36–85.11 | 3/3 |
| complex dynamic | 3 | 15,867 | 77.80 ± 5.87 | 71.49–83.10 | 2/3 |
| floor motion | 3 | 14,619 | 63.42 ± 6.04 | 58.85–70.27 | 3/3 |

全体 15 条中有 12 条触发至少一个 hard-fail。最高两条且无 hard-fail 的轨迹为
`jumps1_subject5=93.27` 和 `walk1_subject1=91.24`。`dance2_subject5=85.11` 虽连续分较高，
仍因持续自碰撞触发硬门控，说明连续排序与不可忽略的局部异常需要同时保留，不能用总分覆盖
hard-fail。

`floor_motion` 平均分最低，主要由贴地阶段的大量几何穿透和自碰撞造成。该结果说明当前
G1 参考姿态在地面动作上存在明显几何问题，但不能据此断言源人体动作无效，也不能由纯 FK
判断动力学跌倒/起身是否可执行。

完整逐动作表见
[`analysis/motion_results.csv`](outputs/step10_offline_experiment/analysis/motion_results.csv)，
类别统计见
[`analysis/stratum_summary.csv`](outputs/step10_offline_experiment/analysis/stratum_summary.csv)。

## 阈值敏感性

实验对 10 个阈值执行一次只改一个参数的上下扰动，共 1 个基准加 20 个变体。关键结果：

- 全局排名最不稳定的变体是把 fidelity 位置阈值从 `0.10 m` 收紧到 `0.08 m`，平均总分
  下降 13.11，单条最大变化 14.80，但与基准排名的 Spearman rho 仍为 0.968；
- 速度阈值上下扰动的 rho 为 0.971–0.979，单条最大分数变化不超过 4.35；
- 把关节限位容差从 `1e-6` 降为 0 不改变连续分或排名，但会新增 3 条 hard-fail，直接支持
  v1 保留数值容差的决定；
- 把持续自碰撞门槛从 5 帧放宽到 10 帧会翻转 1 条 hard-fail；
- 其余变体的整体排名 rho 均不低于 0.986。

因此当前排序对所测局部阈值扰动总体稳定，但绝对分对 fidelity 位置阈值较敏感，硬门控则对
关节数值容差和持续事件长度敏感。所有阈值仍必须随机器人模型、控制频率和下游任务重新验证。
完整结果见
[`analysis/sensitivity_summary.csv`](outputs/step10_offline_experiment/analysis/sensitivity_summary.csv)。

## 维度相关性

15 条冻结样本上的 Spearman 相关仅作描述：contact 与 fidelity 为 0.661，joint feasibility
与 fidelity 为 0.489，smoothness 与 contact 仅为 0.096。各维度并不完全重复，保留多维报告
是必要的。contact 与总分为 0.814、fidelity 与总分为 0.679，但总分本身包含这些维度，
不能把这种数学耦合解释为对学习效果的预测能力。

类别内只有 3 条，所有相关性文件均显式设置 `inferential_use_allowed=false`，不使用 p 值作总体
结论。完整矩阵见
[`analysis/dimension_correlations_spearman.csv`](outputs/step10_offline_experiment/analysis/dimension_correlations_spearman.csv)。

## 结构化视觉抽检

程序在查看可视化之前冻结每类最低分和最高分各 1 条，共 10 条。每条复核 6 个 MuJoCo FK
姿态、全序列异常时间线、关节诊断、足地接触和 fidelity 图。结果为：

- 语义姿态：8 条 plausible、2 条 uncertain；
- 可见伪影：4 条 mild、5 条 moderate、1 条 severe；
- 指标与视觉证据：6 条 agree、4 条 partial、0 条 disagree。

部分一致主要发生在格斗与 floor motion：合理的近地/触地语义和真正的机器人几何穿透可能
同时出现。复核没有为了迎合总分而把两者混为一谈。抽检由 Codex 执行结构化视觉审计，
不是独立领域专家复核；课题组成员后续可按同一冻结协议复核 YAML 并由 finalizer 重新验证。

协议与记录分别见 [`data/step10_manual_review_protocol.md`](data/step10_manual_review_protocol.md)
和 [`data/step10_manual_review.yaml`](data/step10_manual_review.yaml)，合并结果见
[`analysis/manual_review.csv`](outputs/step10_offline_experiment/analysis/manual_review.csv)。

## 复现与完整性

从头复现需要在 `gmr` 环境中依次运行：

```bash
python -m gmr_quality prepare-step10-dataset \
  --selection data/step10_source_selection.csv \
  --config configs/unitree_g1.yaml \
  --output-dir outputs/<new_step10_dataset>

python -m gmr_quality run-step10-experiment \
  --config configs/unitree_g1_v1.yaml \
  --catalog outputs/<new_step10_dataset>/step10_motion_catalog.csv \
  --selection data/step10_source_selection.csv \
  --output-dir outputs/<new_step10_experiment>

python -m gmr_quality finalize-step10-experiment \
  --output-dir outputs/<new_step10_experiment> \
  --review data/step10_manual_review.yaml
```

正式交付位于 [`outputs/step10_offline_experiment`](outputs/step10_offline_experiment)，完整 manifest
锁定 96 个文件的大小和 SHA-256。`step10_manifest.json` 的
SHA-256 为 `19c4a108e1fd792c560420d8f7538dd54204b678a60f7d84606a19129dd7790c`。

## 局限与下一步边界

- 15 条、每类 3 条只能支持工程验收和描述性分析，不代表 LAFAN1 或真实人体动作总体；
- pose montage 与 MuJoCo FK 没有控制器、力、力矩、稳定裕度或动力学 rollout；
- 脚接触为几何/速度代理，没有人体真实接触标签；
- floor motion 和格斗说明接触策略仍需更细粒度的动作阶段标签；
- v1 总分未依据机器人学习结果校准，不能据此声称数据对模仿学习有效；
- 当前数据无物体 6D 轨迹和任务标签，不能评价交互一致性或操作任务成功。

因此第 11 步应在类别、K、训练预算和随机种子受控下比较 Top-K、Random-K、Bottom-K，并把
策略成功率、收敛速度、跟踪误差和跌倒率作为外部效标；第 12 步再引入物体与任务标签。
