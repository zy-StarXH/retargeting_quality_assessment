# GMR 重定向数据质量评估方法与交付报告

## 0. 2026-08-05 交接快照

本文件是新对话继续工作的首要交接入口。项目根目录为 `/root/gpufree-data/retargeting_quality_assessment`，GMR 根目录为 `/root/gpufree-data/GMR`，基础质量评价使用已经配置好的 Conda 环境 `gmr`。`PLAN.md` 的第 0–16 步均已完成，当前没有尚在执行的步骤；后续工作应新建计划步骤，不应把已经完成的工程验收改写成尚未完成。

当前最重要的状态如下：

| 项目 | 客观状态 |
|---|---|
| 质量指标代码 | 已完成；普通 GMR PKL 和操作 NPZ 均有 CLI |
| 基础评价器 | `configs/unitree_g1_v1.yaml`，未校准 |
| 统一操作评价器 | `configs/quality_assessment_v2.yaml`，未校准 |
| 正式 v2 输出 | `outputs/quality_assessment_v2_validation/`，`complete=true` |
| v2 正式样本 | 15/15 成功评价，9 hard-fail，汇总无 `Inf` |
| v2 受控诊断 | Spearman `0.8597`，95% CI `[0.6283,0.9022]` |
| 独立总分校准 | 未完成，`score_is_calibrated=false` |
| 真实人体—物体验证 | 未完成，`human_object_external_validation=false` |
| 操作技能学习验证 | 未完成，`skill_learning_validation=false` |
| 最近全量测试 | 102 passed；doctor 69 passed、1 个既有 warning、0 failed |

交接时应直接阅读或复核的权威文件：

- `PLAN.md`：第 0–16 步的实施历史、验收记录和研究边界；
- `configs/quality_assessment_v2.yaml`：v2 指标、阈值、权重和 G1 模型哈希；
- `src/gmr_quality/interaction.py`：交互数据契约、逐帧指标、事件和分数；
- `src/gmr_quality/interaction_runner.py`：基础运动与交互/任务统一流水线及受控统计 finalizer；
- `outputs/quality_assessment_v2_validation/run_manifest.json`：正式输入、配置、模型、核心源码和输出哈希；
- `outputs/quality_assessment_v2_validation/validation/calibration_decision.json`：为什么总分仍未校准；
- `outputs/step11_learning_experiment/analysis/STEP11_RESULTS.md`：无物体 tracking 的真实 GPU 负结果；
- `outputs/step12_interaction_experiment/STEP12_RESULTS.md`：受控操作实验结果。

`STEP10_REPORT.md`、`STEP11_REPORT.md` 和 `STEP12_REPORT.md` 当前已不在项目目录中，不应在新对话里假设它们仍存在；相关事实保存在 `PLAN.md`、本报告和上述机器可读输出中。项目目录本身不是 Git 仓库，结果溯源依赖各阶段 manifest 的 SHA-256；外部 GMR 仓库固定在提交 `bb1bbe40774794fceb2a7c579a3464a28e68c844`，但其工作树已有 1 个修改文件和 2 个未跟踪文件，均属于既有用户内容，不得覆盖或清理。

## 1. 结论与课题对应关系

本项目已经完成课题中“质量评估指标”的可运行工程实现：对一条重定向后的 Unitree G1 轨迹，从轨迹平滑性、关节可达性、接触合理性和任务有效性四个方面输出逐帧量、连续异常事件、分维度分数、独立硬失败结论和可复现清单。对于带物体状态的操作记录，同一条命令会先评价 36 维 G1 `robot_qpos`，再评价双手—物体交互与任务最终状态，避免用“物体到达目标”掩盖机器人轨迹本身的运动学缺陷。

交付状态必须分层解释：

- **指标工程：已完成。** 四类指标、严格输入校验、单条/批量 CLI、报告、异常定位、测试和哈希清单均已连通。
- **受控操作验证：已完成。** 15 条受控 G1 双手操作示范和 1500 个随机化 MuJoCo episode 用于检查指标方向与任务成功率关系。
- **统一总分校准：未完成。** `quality_score_v2` 始终保存 `score_is_calibrated=false`；当前受控结果是事后诊断，不能作为独立校准集。
- **真实人体—物体外部验证：未完成。** 本机没有 MotionDecode 原始 BVH/物体数据，不能将合成 G1 记录描述为人体操作数据。
- **操作技能学习价值验证：未完成。** 第 11 步完成的是无物体 whole-body tracking 且得到负校准结果；它不能替代操作技能模仿学习实验。

上述边界与老师、学长的要求一致：当前先把重定向后的数据评价器做完整；未来取得 AMASS、LAFAN、BONES Seed 或真实人体—物体数据后，可沿用相同接口做独立验证，而不是伪造 MotionDecode 未开放的数据。

## 2. 输入契约

### 2.1 普通 GMR 轨迹

普通轨迹使用 GMR pickle，核心字段为 `fps`、`root_pos[T,3]`、`root_rot[T,4]` 和 `dof_pos[T,29]`。内部四元数统一为 `xyzw`；只有写入 MuJoCo 自由根 qpos 时转换为 `wxyz`。加载器拒绝 NaN/Inf、非单位四元数、错误帧形状、错误 DoF 数和不一致的关节名。

人体—机器人 fidelity 只对 SHA-256 锁定、经完整 GMR 重放验证的 BVH/PKL 配对启用。无法证明源人体动作时，该维度为不可用并对其余可用维度重归一化，不按满分填充。

### 2.2 操作交互记录

交互输入采用不允许 pickle 的 NPZ schema `gmr_interaction_v1/v2`，至少包含：

- G1 `robot_qpos[T,36]` 和正 fps；
- 左右手位置、`xyzw` 姿态；
- 物体位置、`xyzw` 姿态和线速度；
- 左右手期望/观测接触布尔标记；
- 整数阶段 ID、唯一阶段名、任务名和交互模式；
- 目标位置、物体半尺寸和手部代理半径。

所有时序数组必须严格同帧，静态几何量必须有限且为正，接触必须为真正布尔类型，G1 根姿态和三个对象姿态必须为单位四元数。非法记录在批处理中单独失败，不阻塞其他文件，也不进入评分。

## 3. 指标定义

### 3.1 轨迹平滑性

在未滤波轨迹上计算关节 step、速度、加速度、jerk，根平移速度/加速度/jerk，以及根 SO(3) 角速度；一至三阶差分分别对齐到区间结束帧并在开头保留 1–3 行不可用值。平滑有效帧定义为：

```text
smooth_valid[t] = not joint_jump[t] and not velocity_violation[t]
smoothness_score = 100 × mean(smooth_valid)
```

P95、最大值、最差关节和最差帧作为解释量保留，避免单一均值隐藏局部尖峰。

### 3.2 关节可达性

从哈希锁定的 G1 XML 自动读取 29 个关节顺序、上下限和 qpos 地址，计算硬限位超出量、绝对 `0.05 rad` 近限位、归一化 2% 近限位和关节速度超限。关节有效帧为：

```text
joint_valid[t] = not hard_limit_violation[t] and not near_limit_2pct[t]
joint_feasibility_score = 100 × mean(joint_valid)
```

任意硬限位违反会保留独立 hard-fail reason；高连续分不能覆盖安全失败。

### 3.3 机器人接触合理性

每帧通过 `mj_forward()` 获得机器人几何和 MuJoCo contact，评价：

- 原始/非手部自碰撞及持续事件；
- 地面接触、穿透深度和非足部触地；
- 左右脚底表面高度、接触代理与 MuJoCo contact 一致性；
- 接触期间水平脚滑；
- 腾空持续时间。

接触语义来自数据台账或显式配置。未知动作使用 `descriptive_only`，不会把跳跃腾空或 floor motion 的手触地武断判错。持续非手部碰撞、严重地面穿透等条件通过 hard-fail 独立表达。

### 3.4 人体—机器人动作相似性

对可信人体源，复用 GMR 的人体尺度、body mapping、位置/旋转 offset 和帧对齐，比较 GMR 实际 IK 目标与机器人 FK：关键 body 位置用米制欧氏距离，姿态用 SO(3) 测地角。帧级口径为关键 body 平均位置误差 `<0.10 m` 且平均姿态误差 `<10 deg`。混合任务权重的 IK 优化误差不会伪装成米或角度。

### 3.5 手—物交互合理性

每只手以球形代理、物体以有向包围盒计算有符号距离。合法接触允许 `0.025 m` 代理重叠，只有超出部分计为非法穿透；这样不会把 G1 橡胶手球形代理形成接触所需的有限重叠反向惩罚。

交互指标包括：

- **接近度：** 期望接触帧上的平均正间隙；
- **接触时序：** 期望与观测接触的 F1，并分别定位 missed/unexpected contact；
- **相对位姿：** 双侧期望且实际接触期间，手—物相对位置/姿态漂移；球形手代理不使用无物理意义的手朝向作为抓取坐标系；
- **非法穿透：** 超出合法重叠容差的深度和严重帧率；
- **抓持稳定性：** 双侧稳定接触比例、相对滑移 P95 和实际抬升量。

离散接触段之间不计算速度，避免跨越失联区间制造虚假滑移尖峰。没有足够有效帧的原始统计量输出 `NaN/不可用`，不输出 `Inf`。

### 3.6 任务有效性

当前典型任务为 G1 双手 `approach → grasp → lift → carry → lower → release → settle`。任务有效性同时使用：

- 最终物体三轴目标误差；
- 最终物体速度；
- 相对初始状态的最大抬升量；
- 按冻结容差计算的 nominal task success。

因此“手接触过物体”或“物体被抬起”都不能替代真正的搬运/放置成功。`TASK_NOT_COMPLETED` 直接进入 hard-fail reason。

## 4. 分数、缺失值与硬失败

基础运动质量、交互质量、任务有效性和统一 `quality_score_v2` 均输出 `[0,100]`。v2 的十维冻结权重为：

| 维度 | 权重 |
|---|---:|
| 平滑性 | 0.10 |
| 关节可达性 | 0.10 |
| 机器人接触 | 0.10 |
| 人体—机器人 fidelity | 0.10 |
| 手—物接近度 | 0.08 |
| 接触时序 | 0.10 |
| 相对位姿 | 0.10 |
| 非法穿透 | 0.08 |
| 抓持稳定性 | 0.10 |
| 任务有效性 | 0.14 |

若 fidelity 等维度确实不可用，只在可用维度间按原权重比例重归一化，同时保存 `available/unavailable dimensions` 与实际归一化权重。连续分用于排序和诊断，hard-fail 用于安全/任务门控，两者不能互相覆盖。所有总分均明确为未校准分。

## 5. 受控验证结果

正式 v2 验证使用 5 个质量等级、每级 3 个固定 seed，共 15 条、每条 300 帧的 G1 操作记录；15/15 输入完成评价，9 条触发任务/交互或基础运动 hard-fail。已有 1500 个随机化 MuJoCo episode 的鲁棒任务结果作为受控诊断目标。

| 质量等级 | 基础运动分 | 交互分 | 任务分 | v2 总分 | 鲁棒成功率 | hard-fail 率 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 99.765 | 75.153 | 76.873 | 85.859 | 1.000 | 0.000 |
| mild | 99.608 | 71.960 | 47.671 | 83.700 | 1.000 | 0.000 |
| moderate | 100.000 | 62.956 | 36.498 | 79.477 | 0.000 | 1.000 |
| severe | 100.000 | 13.419 | 0.000 | 44.755 | 0.000 | 1.000 |
| invalid | 100.000 | 10.189 | 0.000 | 42.362 | 0.000 | 1.000 |

v2 总分与鲁棒任务成功率的诊断 Spearman 为 `0.8597`，10,000 次 bootstrap 95% CI 为 `[0.6283, 0.9022]`；任务有效性分为 `0.8817 [0.7104, 0.9423]`。基础运动分几乎均为 100，但物体任务结果显著不同，正好说明普通平滑/限位指标不能替代交互和任务指标。基础分在该受控队列中的负相关来自其极小差异与任务故障注入不改变 robot qpos 的实验设计，不能解释为“轨迹越差越好”。

该结果只证明评价器能识别本受控基准中的任务缺陷。v2 修正是在查看旧 v1 受控结果后定义，并仍在相同 15 条记录上诊断，因此不满足独立校准条件；结果文件中明确保存 `human_object_external_validation=false` 和 `skill_learning_validation=false`。

受控基准还有两项不能省略的实现边界：G1 模型没有可闭合手指，抓持使用“双侧真实接触激活的 soft weld”抽象；操作记录中的 G1 qpos 是受控参考运动，不是由学习策略闭环执行得到。MuJoCo 仍负责物体质量、摩擦、碰撞、释放和最终稳定，但该实验只能验证指标对受控交互故障及物体任务结果的敏感性，不能声称已经验证灵巧手抓取或操作技能学习。

第 11 步确实在 RTX 4090、Isaac Lab/BeyondMimic 上完成 12 个策略和 9000 个确定性评价 episode，但任务是无物体 whole-body tracking，训练预算为计算受限的 300 iterations。训练输入对每条动作中心裁剪为相同的 3933 个 30 Hz 源帧后重采样到 50 Hz；Full 组只有 1/15 条动作出现非零成功率。其 leave-one-motion-category-out OOF Spearman 为 `-0.1862`，95% CI `[-0.5032,0.0313]`，所以它是基础 v1 分数不能通过当前学习价值门控的真实负证据，不是 v2 操作质量分的校准实验。

## 6. 输出与复现

普通 GMR PKL：

```bash
cd /root/gpufree-data/retargeting_quality_assessment
conda activate gmr
python -m gmr_quality evaluate \
  --input /path/to/trusted_motion.pkl \
  --config configs/unitree_g1_v1.yaml \
  --output-dir outputs/my_motion_quality
```

带物体和任务标签的统一评价：

```bash
python -m gmr_quality evaluate-interaction \
  --input /path/to/interaction_record.npz \
  --config configs/quality_assessment_v2.yaml \
  --output-dir outputs/my_interaction_quality
```

目录批处理把 `--input` 换为 `--input-dir`，并可用 `--pattern '*.npz'`。输出包含：

```text
quality_summary.csv       每条数据的原始量、维度分、总分和 hard-fail
quality_events.csv        连续异常区间、实体、严重度和说明
frame_metrics/*.csv       对齐到每个原始帧的基础/交互/任务指标
plots/*.png               关节、脚地、fidelity 和交互诊断图
resolved_config.yaml      本次完整配置快照
run_manifest.json         输入、模型、配置、代码和所有输出的 SHA-256
```

正式受控结果位于 `outputs/quality_assessment_v2_validation/`。重新执行时评价命令要求新的空目录；统计 finalizer 用于已完整的评价目录：

```bash
python -m gmr_quality finalize-quality-assessment-v2 \
  --config configs/quality_assessment_v2.yaml \
  --output-dir outputs/quality_assessment_v2_validation \
  --simulation-results outputs/step12_interaction_experiment/interaction_learning_value.csv
```

## 7. 下一阶段的正确实验

如果课题继续要求证明“质量分反映机器人学习价值”，不能只对现有 15 条合成轨迹继续调权。应先冻结当前 v2，再取得新的、独立的操作数据队列：优先使用含人体动作、物体 6D、任务/接触语义的数据；若 MotionDecode 原始数据仍不开源，人体运动来源可按学长建议使用 AMASS/LAFAN/BONES Seed，并为操作对象补充可靠配准与接触标注。

下游实验应按源动作组划分 train/validation/test，避免同一源动作的裁剪片段跨集合泄漏；在相同任务、数据量、训练步数和多个 policy seed 下比较全量、Top-K、Random-K、Bottom-K，同时报告成功率、收敛速度、稳定性、跟踪误差和 bootstrap 置信区间。只有在未参与调权的 held-out 操作队列上通过预注册门控，才可把 `score_is_calibrated` 改为 true。

## 8. 下一位 AI 的操作约束与推荐起点

1. 不要覆盖 `outputs/quality_assessment_v2_validation/`、Step 10–12 正式输出或 GMR 原始数据。`evaluate` 和 `evaluate-interaction` 本身会拒绝非空输出目录；复现时应使用新的目录名。
2. 不要在当前 15 条受控结果上继续调 `quality_assessment_v2.yaml` 的权重并把同一批结果称为校准。如果需要改变指标，应新建明确版本；如果验证 v2，应保持当前配置不变并使用新的 held-out 队列。
3. 新的真实操作数据必须先建立来源台账和不可歧义的适配器：人体动作、重定向后 G1 qpos、左右手/物体 6D、接触语义、阶段和任务目标应有可核对的帧对齐与单位。只有 BVH 而没有物体轨迹时，不能计算任务有效性。
4. 若下一任务只是“使用评价器”，从第 6 节 CLI 开始；若任务是“完成总分校准”，从第 7 节独立数据和预注册实验开始；若任务转回“建立新的重定向方法”，应另立项目步骤并按学长建议从 AMASS/LAFAN/BONES Seed 选择人体源，不能把现有质量评价器误称为新重定向算法。
5. 代码变更后的最低交付门槛为：相关单元测试、全量 `pytest`、`doctor`、`pip check`、正式 manifest 哈希复核，并在 `PLAN.md` 新增真实执行记录。任何负结果都应保留，不允许通过事后筛样或调权改写为正结果。

截至本次交接，没有真实人体—物体数据下载、v2 独立校准、操作策略训练或新重定向算法正在后台运行；下一位 AI 可以据用户的新任务从这一稳定基线继续。
