# 第 12 步：操作任务与交互一致性验证

## 验收结论

第 12 步已完成一个与前序 Unitree G1 评价链路连续的、可复现的典型操作任务基准：双手接触物体后完成抬升、平移、放置和释放。15 条示范均包含 G1 36 维 qpos、左右手 6D 位姿、物体 6D 轨迹、任务标签、阶段标签以及期望/实际接触语义；随后在 MuJoCo 中完成 1500 个质量、摩擦和初始物体位姿随机化 episode。

实验建立了部分离线交互指标与真实仿真任务结果的关系：`task_effectiveness_score` 对鲁棒任务成功率的 Spearman 为 `0.8817`，bootstrap 95% CI `[0.7104,0.9443]`；`relative_pose_score` 为 `0.5878`，95% CI `[0.0960,0.9315]`。这两项可以在该受控基准中预测任务价值。

预设的交互总分没有通过校准门控，继续保持 `score_is_calibrated=false`。本机没有 MotionDecode 或其他真实人体—物体示范，因此本步骤明确是“受控合成 G1 交互基准”，不宣称完成真实人体—物体数据集的外部泛化验证。

## 与第 0–11 步的连续关系

- 继续使用前序哈希冻结的 Unitree G1 29-DoF XML、36 维 qpos、`xyzw` 数据四元数约定和 MuJoCo 执行边界；
- 前序 `quality_score_v1` 继续描述平滑性、关节可行性、接触和人体—机器人 fidelity；本步骤新增交互维度，不覆盖或暗中重定义 v1；
- 第 11 步证明无物体离线总分不能直接解释学习价值；本步骤因此直接记录物体状态并以最终物体状态和随机化仿真成功率作为外部目标；
- 交互 NPZ schema 可以由后续 MotionDecode/真实人体—物体适配器直接填充，不需要修改指标定义。

## 数据来源与研究边界

本地只存在 LAFAN1/AMASS 人体运动和已重定向的 G1 轨迹，没有开源的物体 6D、任务标签或手—物接触标注。把 LAFAN1 动作附会成操作任务会制造虚假证据，因此本步骤生成了受控合成示范，并在所有配置、manifest 和报告中保存：

```text
source_kind = controlled_synthetic_physics_demonstrations
human_object_dataset = null
human_object_external_validation = false
```

这足以验证指标实现、异常敏感性和仿真关系，但论文若要声称对 MotionDecode/真实人体操作数据有效，仍必须在取得真实数据后执行独立外部验证。

## 典型操作任务

任务为 `bimanual_lift_translate_place`：G1 双手从物体两侧接近，建立双侧接触，抬升 0.16 m，沿前方平移 0.20 m，下降到桌面后释放并等待稳定。每条 6 秒、50 Hz、300 帧，阶段为：

```text
approach → grasp → lift → carry → lower → release → settle
```

G1 29-DoF 模型只有橡胶假手，没有可闭合手指。单靠网格摩擦不能形成稳定抓取，因此仿真采用公开的 `bilateral_contact_activated_soft_weld`：只有左右手同时接触且任务标签处于 grasp/lift/carry/lower 阶段时才激活软抓持约束；接触或语义失效后释放。物体质量、摩擦、落下、桌面碰撞和放置稳定均继续由 MuJoCo 动力学求解。该抓持抽象是本基准限制，不能等同真实灵巧手。

## 交互数据与指标

每条 NPZ 保存：

- `robot_qpos (T,36)`；
- 左右手位置和 `xyzw` 四元数；
- `object_pos_m (T,3)`、`object_quat_xyzw (T,4)`、物体线速度；
- `expected_contact (T,2)`、`observed_contact (T,2)`；
- `phase_id`、`phase_names`、`task_label`、`interaction_mode`；
- 物体尺寸、目标位置、帧率和手部碰撞代理半径。

冻结的六个交互维度为：手—物表面距离、接触时序 F1、抓持期间相对位置/姿态稳定性、过度穿透、双侧抓持连续性/相对滑动、最终物体目标与稳定性。总分权重在正式随机化结果产生前冻结为 `0.15/0.20/0.15/0.15/0.20/0.15`。

## 受控实验设计

五个质量等级各 3 个固定 seed，共 15 条示范。扰动只作用于双手间隙、左右手时序、搬运距离、垂直抖动和共同横向偏移。每条示范在独立的 100 个 episode 中随机化：

- 物体初始位置：各轴在冻结范围内均匀扰动；
- 质量比例：`[0.75,1.25]`；
- 摩擦比例：`[0.75,1.25]`。

成功要求物体到达目标三轴容差、最终速度小于 0.20 m/s，且过程中至少抬升 0.10 m。正式评价总计 1500 episodes。

## 正式结果

表中为三个示范 seed 的均值；完整 95% CI 位于 `quality_level_summary.csv`。

| 质量等级 | 交互总分 | 鲁棒成功率 | 最终位置误差 (m) | 最大抬升 (m) |
|---|---:|---:|---:|---:|
| clean | 67.147 | 1.000 | 0.0338 | 0.1756 |
| mild | 72.373 | 1.000 | 0.0748 | 0.1647 |
| moderate | 80.774 | 0.000 | 0.0929 | 0.1680 |
| severe | 19.559 | 0.000 | 0.2024 | 0.0021 |
| invalid | 15.252 | 0.000 | 0.2004 | 0.0003 |

`moderate` 仍能完成抓持和抬升，但搬运距离不足，最终越过目标位置容差，因此成功率为 0。这验证了只检查“是否接触/是否抬起”不足以代表任务完成，必须检查最终物体状态。

## 离线指标与仿真关系

下表目标均为 15 条示范在 100 个随机化 episode 上的成功率。

| 离线指标 | Spearman rho | Bootstrap 95% CI | 结论 |
|---|---:|---:|---|
| 手—物接近 | 0.4564 | [-0.0668,0.9354] | 正向但证据不足 |
| 接触时序 | 0.2887 | [-0.2962,0.8834] | 未通过 |
| 相对物体位姿 | 0.5878 | [0.0960,0.9315] | 通过 |
| 穿透分 | -0.9682 | [-1.0000,-0.8704] | 代理方向错误 |
| 抓持稳定性 | 0.2939 | [-0.2884,0.9230] | 未通过 |
| 任务有效性 | 0.8817 | [0.7104,0.9443] | 通过 |
| 交互总分 | 0.2866 | [-0.2876,0.8776] | 未通过校准门控 |

穿透分的反向关系来自 G1 球形手部代理：成功的软抓持需要代理与物体表面有有限重叠，而当前通用 OBB 距离把这部分也视为穿透。正式结果产生后没有修改阈值或权重；该负结果被保留，说明后续真实手模型需要区分“接触代理容差”和“非法几何穿透”。

最终决定：

```text
task_effectiveness_predicts_robust_success = true
relative_pose_predicts_robust_success = true
interaction_quality_score_is_calibrated = false
external_human_object_validation = false
```

## 交付物与复现

- `configs/step12_interaction.yaml`：数据来源边界、G1 模型、任务、扰动、指标、权重和门控；
- `src/gmr_quality/interaction.py`：严格 NPZ schema、物体 OBB 距离、接触时序、相对位姿、抓持与任务指标；
- `src/gmr_quality/step12_experiment.py`：场景生成、G1 双臂 IK、软抓持、动态物体回放、并行随机化、统计与 manifest；
- `tests/test_step12_interaction.py`：来源边界、G1 物理任务、schema、指标和正式 manifest 测试；
- `outputs/step12_interaction_experiment/`：15 条示范、1500 条 episode、统计表、相关性、校准决定、场景 XML/MJB 和 SHA-256 manifest。

复现命令：

```bash
conda run -n gmr python -m gmr_quality run-step12-experiment \
  --config configs/step12_interaction.yaml \
  --output-dir outputs/step12_interaction_experiment_new --json
```

正式 manifest 为 `complete=true`，包含 27 个关键文件的大小与 SHA-256；`score_is_calibrated=false` 是实验结论，不是流水线缺失。
