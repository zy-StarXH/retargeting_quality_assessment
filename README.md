# GMR Retargeting Quality Assessment

这是一个独立于 GMR 源码的 Unitree G1 重定向轨迹质量评估项目。质量评估 v2 已形成从严格输入校验、平滑性、关节可达性、机器人接触、人体—机器人 fidelity，到手—物交互和任务有效性的统一流水线；输出逐帧量、连续异常事件、分维度分数、独立 hard-fail、图表和完整 SHA-256 复现清单。总分尚未通过独立数据校准，始终标记 `score_is_calibrated=false`。

完整研究计划和每一步记录见 [`PLAN.md`](PLAN.md)，指标需求见 [`GMR_质量评估项目需求与实现指南.md`](GMR_质量评估项目需求与实现指南.md)。

完整指标定义、受控验证结果和证据边界见 [`QUALITY_ASSESSMENT_REPORT.md`](QUALITY_ASSESSMENT_REPORT.md)。

## 质量评估 v2：推荐使用入口

普通 GMR PKL 使用基础评价器：

```bash
conda run -n gmr python -m gmr_quality evaluate \
  --input /path/to/trusted_motion.pkl \
  --config configs/unitree_g1_v1.yaml \
  --output-dir outputs/my_motion_quality
```

包含 G1 qpos、双手/物体 6D、接触与任务标签的 NPZ 使用统一评价器：

```bash
conda run -n gmr python -m gmr_quality evaluate-interaction \
  --input /path/to/interaction_record.npz \
  --config configs/quality_assessment_v2.yaml \
  --output-dir outputs/my_interaction_quality
```

批量输入可换为 `--input-dir /path/to/records --pattern '*.npz'`。输出目录必须不存在或为空，并且不能放在输入目录内部。程序退出码 `0/2/3` 分别表示执行成功、输入/处理失败、指定 `--fail-on-hard-fail` 后检测到质量硬失败。

正式受控评价已保存在 `outputs/quality_assessment_v2_validation/`：15/15 条记录成功评价，9 条 hard-fail；v2 总分与 1500 个随机化仿真 episode 的鲁棒成功率诊断 Spearman 为 `0.8597 [0.6283, 0.9022]`。这是相同受控队列上的事后诊断，不是独立校准、真实人体—物体外部验证或操作技能学习证据。

第 10 步的实验设计、数值结果、v1 冻结口径和研究边界见 [`STEP10_REPORT.md`](STEP10_REPORT.md)。

## 运行第 10 步冻结实验

本机正式交付已经保存在 `outputs/step10_dataset` 和 `outputs/step10_offline_experiment`。若需从头复现，必须使用新的空输出目录：

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

第一个命令会对 15 条完整 LAFAN1 序列执行 GMR 重定向，耗时明显长于普通质量评估。第二个命令输出逐动作/分层表、21 组基准与单阈值变体、维度相关性、pose montage、全序列异常时间线及正式诊断图。第三个命令校验五类各 2 条冻结抽检的完整覆盖，并为全部交付文件生成 SHA-256 manifest。

`configs/unitree_g1_v1.yaml` 是冻结但未校准的离线评价器，必须同时保留 `hard_fail` 与各维度原始指标。第 11 步真实策略实验未通过预注册校准门控，因此其 `quality_score` 仍不能解释为机器人学习价值。

## 当前范围

`doctor` 命令只做以下检查：

- Python 依赖是否可用；
- GMR 路径、提交和工作树状态；
- `bvh_to_robot.py` 和 G1 XML 是否存在；
- G1 的 29 个 hinge 关节及 qpos 地址是否连续；
- floor 和左右脚底球形 geom 是否可识别；
- 运动台账中的 PKL/BVH 是否存在；
- 对可信本地 PKL 做浅层键和数组形状检查；
- 报告已知的 BVH/PKL 帧数差异和未配对源动作。

`doctor` 本身不会计算平滑性、碰撞率、脚滑或质量分；正式指标通过 `evaluate` CLI 或下文的 Python API 计算。

## 使用 CLI 评估轨迹

单文件评估示例：

```bash
python -m gmr_quality evaluate \
  --input /root/gpufree-data/GMR/outputs/amass/g1_motion.pkl \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv \
  --output-dir outputs/my_amass_run \
  --json
```

目录批处理示例：

```bash
python -m gmr_quality evaluate \
  --input-dir /root/gpufree-data/GMR/outputs \
  --pattern '*.pkl' \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv \
  --output-dir outputs/my_batch_run
```

输入按解析后的绝对路径排序，台账中已登记的轨迹自动取得 `motion_name`、动作类别、接触策略和源 BVH；未登记轨迹使用文件名以及保守的 `unknown + descriptive_only`，不会根据评估结果反推动作语义。目录模式下可用 `--motion-category`、`--contact-policy` 和 `--notes` 统一覆盖元数据；`--motion-name` 与 `--source-bvh` 仅允许用于单文件模式。

每个成功评估的动作默认生成两张无头环境 PNG：关节 step/速度/jerk 诊断图，以及脚底高度/速度/接触代理图。使用 `--no-plots` 可关闭绘图。输出结构如下：

```text
outputs/my_batch_run/
├── quality_summary.csv
├── quality_events.csv
├── resolved_config.yaml
├── run_manifest.json
├── frame_metrics/
│   └── <motion_name>.csv
└── plots/
    ├── <motion_name>_joint_velocity.png
    ├── <motion_name>_foot_contact.png
    └── <motion_name>_fidelity.png  # 仅可靠配对的动作生成
```

`run_manifest.json` 保存工具/GMR Git 状态、Python/MuJoCo 运行时、实际命令、输入状态、输入与源 BVH 哈希、模型 XML 哈希、源配置与 resolved config 哈希、阈值、权重、接触策略及所有输出文件哈希。`resolved_config.yaml` 保存本次实际使用的完整配置。坏 PKL 或非法数据只会将该文件标记为失败，批次中的其他文件仍继续评估；非法但可读取的动作仍有 summary 行和带表头的空 frame CSV。

为保护原始数据，输出目录必须位于输入目录树之外，而且只能是不存在或为空的目录；工具不会清空或覆盖已有结果。建议每次运行使用新的输出目录。退出码约定：

- `0`：所有输入均完成评估；轨迹本身出现 `hard_fail` 不等同于程序执行失败；
- `2`：参数、输入、配置或至少一个文件的处理失败，但其余文件仍尽量交付；
- `3`：仅在指定 `--fail-on-hard-fail` 且评估中存在质量硬失败时返回。

直接提供 `--source-bvh` 只会把来源记录为 `provided_unverified`，不会启用 fidelity。只有台账中同时锁定 PKL/BVH SHA-256、声明帧对齐方式并具有完整 GMR 重放证据的 `verified_replay_exact` 配对才会自动启用，防止仅凭相似文件名计算伪指标。

## 验证完整质量评估管线

第 8 步提供独立的 `validate-pipeline` 命令。它从可信 GMR PKL 读取一个真实 G1 姿态，复制为 12 帧静态基准，然后生成受控对照和异常夹具；这些夹具依次经过严格输入校验、29 DoF 模型绑定、MuJoCo FK、运动学/接触指标、v0 评分、硬失败、事件、CSV、PNG、resolved config 和 manifest，与正式 `evaluate` 使用同一套实现。

```bash
python -m gmr_quality validate-pipeline \
  --input /root/gpufree-data/GMR/outputs/amass/g1_motion.pkl \
  --config configs/unitree_g1.yaml \
  --output-dir outputs/step8_pipeline_validation \
  --json
```

默认注入帧为第 5 帧，各场景及不变量如下：

| 场景 | 注入内容 | 必须检测到的结果 |
|---|---|---|
| `baseline` | 真实姿态的静态复制 | 无跳变、近限位、硬限位、穿透和脚滑 |
| `joint_jump` | 单关节增加超过阈值的阶跃 | 跳变准确定位到第 5、6 帧及对应关节 |
| `near_limit` | 低于跳变阈值的三角渐变，在峰值进入 2% 限位带 | 只在第 5 帧标记近限位，无硬限位违反或关节跳变 |
| `root_down` | 第 5 帧根节点下移 `0.12 m` | 地面穿透最差帧为第 5 帧，并触发严重穿透硬失败 |
| `foot_contact_control` | 将静态姿态脚底对齐地面 | 接触对照中无脚滑 |
| `foot_slide` | 接触对照的根节点水平移动 `0.02 m` | 左右脚滑准确定位到第 5、6 帧 |
| `invalid_nan` | 第 5 帧单关节写入 NaN | `DATA_NAN`，且不进入模型绑定、FK、指标或评分 |

输出结构：

```text
outputs/step8_pipeline_validation/
├── pipeline_validation.json       # 7 项端到端不变量及输入夹具哈希
├── injected_inputs/               # 由可信姿态派生的受控测试夹具
│   └── <scenario>.pkl
└── reports/
    ├── quality_summary.csv
    ├── quality_events.csv
    ├── resolved_config.yaml
    ├── run_manifest.json
    ├── frame_metrics/<scenario>.csv
    └── plots/<scenario>_*.png
```

成功时外层命令返回 `0` 且 `checks_passed=7/7`。内层正式批处理清单会显示 `succeeded=6, failed=1`：其中唯一失败正是故意构造的 `invalid_nan`，是需要被验证的预期结果，不代表自验证失败。原始 GMR PKL 只读使用；`injected_inputs` 只用于评价器测试，不能混入后续训练数据。若用 `--source-frame` 更换基准帧，该姿态本身必须满足基准不变量，否则命令会如实失败而不会调整正式阈值。

## 在 Python 中加载并校验 GMR PKL

```python
from gmr_quality.config import load_config
from gmr_quality.io import load_motion_from_project_config

config = load_config("configs/unitree_g1.yaml")
result = load_motion_from_project_config(
    "/root/gpufree-data/GMR/outputs/g1_dance1_subject2.pkl",
    config,
)

if result.validation.data_valid:
    motion = result.require_record()
    print(motion.fps, motion.root_pos.shape, motion.joint_pos.shape)
else:
    print(result.validation.as_dict())
```

加载器不会归一化四元数、填补 NaN、裁剪关节或改变输入数值。发生硬错误时 `record=None`，稳定错误码保存在 `result.validation.failure_codes` 中。当前 PKL 没有 `dof_names` 时会保留 `JOINT_ORDER_IMPLICIT` 警告；关节顺序将在第 4 步与 XML 自动核对。

## 计算纯数组运动学指标

平滑性指标不依赖 MuJoCo，可以直接计算：

```python
from gmr_quality.kinematic_metrics import (
    KinematicThresholds,
    compute_smoothness_metrics,
)

thresholds = KinematicThresholds.from_project_config(config)
smoothness = compute_smoothness_metrics(motion, thresholds)
print(smoothness.summary.joint_jump_rate)
print(smoothness.summary.joint_jerk_p95_rad_s3)
```

关节可行性函数接受与 `joint_pos` 顺序一致的上下限数组。对 G1 正式评价时，不再由调用方手工填写限位，而是使用下一节从锁定 XML 自动读取并绑定的限位：

```python
from gmr_quality.kinematic_metrics import (
    JointLimits,
    compute_joint_feasibility_metrics,
)

limits = JointLimits(lower_rad=lower, upper_rad=upper, joint_names=joint_names)
feasibility = compute_joint_feasibility_metrics(motion, limits, thresholds)
print(feasibility.summary.joint_limit_violation_frame_rate)
```

差分结果全部对齐到原始帧：一阶、二阶、三阶差分分别在开头保留 1、2、3 行 `NaN`。一阶值位于区间结束帧，例如 `joint_vel_rad_s[t] = (q[t] - q[t-1]) * fps`。joint jump 和速度违反率只以实际有效的 `T-1` 个一阶样本为分母。所有主指标均基于未滤波原始轨迹。

## 绑定 G1 XML 并执行正向运动学

配置锁定了当前 GMR G1 XML 的 SHA-256。加载时会自动检查 `nq=36`、`nv=35`、自由根、29 个 hinge 关节、连续 qpos/DoF 地址、有限关节限位、floor 以及左右各 4 个脚底球形 geom：

```python
from gmr_quality.kinematic_metrics import (
    KinematicThresholds,
    compute_kinematic_metrics,
)
from gmr_quality.mujoco_kinematics import MujocoKinematics

adapter = MujocoKinematics.from_project_config(config)
bound = adapter.bind_motion(motion)

metrics = compute_kinematic_metrics(
    bound.record,
    bound.joint_limits,
    KinematicThresholds.from_project_config(config),
)
print(metrics.smoothness.summary.worst_step_joint)

for frame in adapter.iter_forward(bound):
    print(frame.frame, frame.geom_pos_world_m.shape)
```

内部轨迹四元数始终保持 GMR 的 `xyzw`，只在写入 MuJoCo `qpos` 时显式转换为 `wxyz`。每帧先重置 `MjData`，再调用 `mj_forward()`，不会调用 `mj_step()`，因此这里得到的是参考姿态的运动学几何量，不是动力学执行结果。

若 PKL 带有 `dof_names`，其顺序必须与 XML qpos 顺序逐项完全一致。现有 GMR 样例没有 `dof_names`，只能依据 GMR 导出代码的 `qpos[7:]` 约定绑定；输出会明确保留 `JOINT_ORDER_INFERRED_FROM_GMR_QPOS_LAYOUT` 和 `SOURCE_MODEL_PROVENANCE_UNAVAILABLE`，不能把“评价时 XML 已锁定”误称为“生成该 PKL 的 XML 来源已被证明”。

`ForwardKinematicsFrame.contacts` 保留 MuJoCo 原始 contact 快照，第 5 步的分类器在此基础上生成原始与过滤后两种口径。

## 计算碰撞与足地指标

接触策略必须来自运动台账或其他明确元数据，不能根据结果反推。已知舞蹈样例使用 `feet_only`；来源和动作类别未知的样例使用 `descriptive_only`：

```python
from gmr_quality.contact_metrics import (
    ContactEvaluationConfig,
    compute_contact_metrics,
)

contact_config = ContactEvaluationConfig.from_project_config(
    config,
    contact_policy="feet_only",
)
contact = compute_contact_metrics(
    bound,
    adapter,
    contact_config,
    motion_category="dance",
)

print(contact.summary.self_collision_any_frame_rate)
print(contact.summary.self_collision_nonhand_frame_rate)
print(contact.summary.max_ground_penetration_m)
print(contact.summary.undesired_ground_contact_frame_rate)
print(contact.summary.left_contact_proxy_rate)
print(contact.summary.foot_slide_frame_rate)
print(contact.summary.airborne_frame_rate)
```

实现口径：

- 候选自碰撞由 MuJoCo 已生效的碰撞过滤结果产生；`any` 保留全部不同机器人 body 间的 contact，`nonhand` 再依据配置中的手部名称模式排除允许接触；
- 地面穿透深度为 `max(0, -contact.dist)`；帧最大值严格超过 `0.01 m` 才标记穿透，超过 `0.05 m` 标记严重穿透；`mean_ground_penetration_m` 是所有 MuJoCo 机器人—地面 contact 样本深度的平均值；
- 脚底高度使用球形 sole geom 的表面最低点，即球心 z 减去半径，不使用踝关节中心或球心高度；
- 接触代理为“脚底表面在地面上下 1 cm 内”且“脚底质心竖直速度不超过 0.2 m/s”；它没有人体真实接触标签，因此始终保留 `CONTACT_PROXY_WITHOUT_SOURCE_LABELS`；
- 水平/竖直速度对齐到区间结束帧，第 0 帧为 `NaN`，所以第 0 帧接触代理为 false；脚滑只在代理接触时按水平速度严格大于 `0.1 m/s` 标记；
- `airborne` 表示左右脚接触代理都为 false，只是描述量，不默认等价于错误；
- 每帧保留 body pair、逐实体穿透深度、脚底点、速度、MuJoCo 脚地 contact、代理 contact 及二者不一致标志。汇总中的 event count 与第 6 步正式事件表使用相同的连续段定义。

`feet_only`、`floor_motion` 和 `descriptive_only` 的策略标志均从 `configs/unitree_g1.yaml` 读取。原始非足部触地和腾空观测不会因策略改变而消失；策略只声明后续评分阶段是否允许把它们视为失败。因此未知动作使用 `descriptive_only` 时不会被武断判为无效。

## 计算已验证的人体—机器人动作相似性

第 9 步使用 GMR 自己的 LAFAN1 loader 和 `bvh_lafan1_to_g1.json`，复现其坐标转换、人体高度比例、逐身体局部缩放、位置 offset 和旋转 offset。机器人侧使用同一锁定 XML 对保存 qpos 执行 `mj_forward()`，随后比较实际参与 GMR IK 的身体目标：13 个位置目标、14 个姿态目标，其中左右踝和左右腕作为 4 个末端目标。

当前已纠正一个重要来源错误：`g1_dance1_subject2.pkl` 实际由 `dance1_subject1.bvh` 生成。使用 GMR 完整重放后，保存的 3944 帧中 3944/3944 帧的 36 维 qpos 逐元素完全一致，最大绝对差为 0；BVH 的第 3945 帧因当前保存循环的 break 顺序未写入。证据与代码/输入哈希见 [`data/pair_replay_evidence.json`](data/pair_replay_evidence.json)。原先依据文件名关联的 `dance1_subject2.bvh` 已被拒绝。

每个位置目标使用米制欧氏距离；每个姿态目标使用 SO(3) 测地角，不对欧拉角或四元数分量直接作差。汇总中的 mean/P95/max 在全部“帧×对应身体”样本上计算；逐帧 CSV 另存位置/姿态身体均值、最大值和当帧最差身体。帧级 fidelity 判据为：

```text
mean(key-body position error at frame t) < 0.10 m
and
mean(link orientation error at frame t) < 10 deg
```

`fidelity_score = 100 * mean(fidelity_valid_frame)`。阈值来自项目指南引用的 PHUMA 参考口径，但当前总分仍未经过下游学习实验校准。真实已验证样例得到：

```text
key-body position mean/P95/max: 0.094753 / 0.241597 / 0.482945 m
end-effector position mean:     0.039388 m
orientation mean/P95/max:       5.361646 / 16.542049 / 60.028196 deg
fidelity pass frames:           3404 / 3944
fidelity_score:                 86.308316
```

注意 P95 是逐身体误差分布，帧是否通过则使用当帧身体均值，二者不能混为同一个统计量。IK 的 `error1/error2` 混合任务权重且没有纯米制或角度单位，本步没有把它伪装成位置/姿态误差；相应 `ik_error_stage2` 列仍保持 `NaN`。

## 生成 v0 评分、事件和 CSV 报告

除上述 CLI 外，也可使用第 6 步保留的单动作 Python API：

```python
from gmr_quality.evaluation import evaluate_motion_result
from gmr_quality.reporting import write_evaluation_reports

evaluation = evaluate_motion_result(
    result,
    adapter,
    config,
    motion_name="dance1_subject1_verified",
    motion_category="dance",
    contact_policy="feet_only",
    source_bvh=(
        "/root/gpufree-data/ubisoft-laforge-animation-dataset/"
        "lafan1/datasets/dance1_subject1.bvh"
    ),
    source_pair_status="verified_replay_exact",
    expected_pkl_sha256=(
        "47186e8d2e605b0e64851f8d151c0460458618b1e5d9069d043809c958d72dcd"
    ),
    expected_source_bvh_sha256=(
        "9a285b61bb0446aa4c919c62247ff895eb158bc693be3b7477d31ef070995d73"
    ),
    frame_alignment="source_prefix_drop_last",
)

print(evaluation.scores.quality_score_v0)
print(evaluation.decision.hard_fail)
print([reason.token for reason in evaluation.decision.failure_reasons])

paths = write_evaluation_reports(
    evaluation,
    "outputs/my_single_evaluation",
)
print(paths)
```

输出结构：

```text
outputs/my_single_evaluation/
├── quality_summary.csv
├── quality_events.csv
└── frame_metrics/
    └── dance1_subject1_verified.csv
```

v0 帧级定义：

```text
smooth_valid = not joint_jump and not velocity_violation
joint_valid  = not position_limit_violation and not near_limit_2pct
contact_valid = not self_collision_nonhand
                and not ground_penetration
                and not foot_slide
fidelity_valid = frame_position_mean < 0.10 m
                 and frame_orientation_mean < 10 deg
```

如果接触策略配置为禁止非足部触地或腾空，相应标志也会加入 `contact_valid`。每个可用维度按有效帧比例得到 `[0,100]` 分数。可靠配对存在时使用配置中的四维 `0.30/0.25/0.30/0.15` 权重；来源未配对或未经验证时 fidelity 为 `NaN`，其权重不会被当成满分，而是只在其他可用维度之间重归一化。`quality_score_v0` 始终保存 `score_is_calibrated=false`，只可用于待验证的离线排序。

`hard_fail` 与连续分数相互独立。当前稳定硬失败条件包括：输入数据错误、任意 XML 关节硬限位违反、地面穿透严格超过 `0.05 m`、同一非手部 self-collision body pair 连续达到或超过 5 帧，以及接触策略明确禁止的非足部触地/腾空。报告同时保存机器可读 reason code、稳定 token 和可读详情，不能因为总分较高而覆盖硬失败。

事件区间的 `start_frame` 和 `end_frame` 都是闭区间；`duration_s=(end_frame-start_frame+1)/fps`，`end_time_s=end_frame/fps`。自碰撞按 body pair 合并，非足部触地按 body 合并，其严重度来自对应实体自己的 contact 深度，不会错误继承同帧其他碰撞的最大值。

## 环境与安装

## 第 11 步：下游策略验证

第 11 步已经在 RTX 4090、Isaac Lab 和 BeyondMimic 中完成 12/12 个真实策略作业。Top-K、Random-K、Bottom-K、Full 各使用三个 policy seed；每个策略训练 29,491,200 transitions，共 353,894,400 transitions，并完成共 9000 个确定性评价 episode。详细实验口径和负结果见 `STEP11_REPORT.md`，机器可读结果位于 `outputs/step11_learning_experiment/`。

预注册校准门控未通过：四维 leave-one-motion-category-out 模型的 OOF Spearman 为 `-0.1862`，95% bootstrap CI 为 `[-0.5032,0.0313]`。因此项目保留 v1 `score_is_calibrated=false`，不做事后权重调整。

重新生成准备产物时必须使用新的空目录：

```bash
conda run -n gmr python -m gmr_quality prepare-step11-experiment \
  --config configs/step11_beyondmimic.yaml \
  --output-dir outputs/step11_learning_experiment_new --json
```

复算正式统计和 manifest：

```bash
conda run -n gmr python -m gmr_quality finalize-step11-experiment \
  --config configs/step11_beyondmimic.yaml \
  --output-dir outputs/step11_learning_experiment
```

finalizer 仅在 12 个真实策略结果齐全时执行；结果不齐会以 `STEP11_RESULTS_INCOMPLETE` 拒绝校准。

## 第 12 步：操作任务与交互一致性

第 12 步已完成一个受控合成的 Unitree G1 双手抬升—搬运—放置任务。15 条示范包含机器人 qpos、左右手 6D、物体 6D、任务/阶段标签和接触语义；1500 个 MuJoCo 随机化 episode 用于验证离线交互指标与任务成功率的关系。详细设计、结果和边界见 `STEP12_REPORT.md`，机器可读产物位于 `outputs/step12_interaction_experiment/`。

结果表明任务有效性分和相对物体位姿分能够预测该基准中的鲁棒成功率，但预设交互总分没有通过门控，保持 `score_is_calibrated=false`。本地没有 MotionDecode 或其他真实人体—物体数据，因此 `human_object_external_validation=false`；不能把该受控基准描述为真实数据集外部验证。

复现实验必须使用新的空目录：

```bash
conda run -n gmr python -m gmr_quality run-step12-experiment \
  --config configs/step12_interaction.yaml \
  --output-dir outputs/step12_interaction_experiment_new --json
```

```bash
cd /root/gpufree-data/retargeting_quality_assessment
conda activate gmr
python -m pip install -e '.[dev]'
```

## 运行环境自检

```bash
python -m gmr_quality doctor \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv
```

机器可读输出：

```bash
python -m gmr_quality doctor \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv \
  --json
```

存在警告但没有失败项时退出码为 0。路径缺失、依赖缺失、模型 DoF/qpos 不符合配置、floor 或脚底 geom 不可识别时退出码非 0。

## 运行测试

```bash
pytest
```

## 数据安全和已知限制

- Python pickle 可能执行任意代码，只能把可信来源的 PKL 放入运动台账。
- `g1_dance1_subject2.pkl` 的文件名具有误导性；完整重放证明其真实来源是 `dance1_subject1.bvh`。该纠正配对已用两侧哈希和 `source_prefix_drop_last` 对齐锁定，`doctor` 会将预期的一帧差异报告为通过。
- `walk.pkl` 的源 BVH 未确认，不得用于人体—机器人 fidelity 评价。
- 当前 fidelity 只对哈希锁定且经过完整 GMR 重放的 LAFAN1 配对启用；其他来源不会靠文件名自动配对。尚未实现通用 AMASS/BONES/LAFAN 批量来源证明和纯量纲 IK 残差。第 11 步真实策略实验得到负校准结果；第 12 步完成了受控合成 G1 物体任务，但尚无 MotionDecode/真实人体—物体外部验证，且 G1 无手指模型使用双侧接触激活的软抓持抽象。两个总分均不能超出各自证据边界解释。
