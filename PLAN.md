# GMR 重定向质量评估项目计划

## 1. 项目目标与边界

本项目面向人体运动数据在机器人学习中的应用，首先建立一套针对 GMR 重定向后 Unitree G1 参考轨迹的、可批量运行、可解释、可复现的离线质量评估工具，并在后续阶段验证质量评价与机器人学习效果之间的关系。

项目分为三个层级：

1. 离线运动学质量：平滑性、关节可行性、碰撞、足地接触及人体—机器人动作相似性；
2. 下游学习价值：高质量数据是否带来更高成功率、更快收敛和更稳定的策略；
3. 操作任务有效性：结合物体 6D 轨迹、任务标签和交互状态评价任务是否真正完成。

当前首先完成第 1 层。`quality_score_v0` 只作为未校准的离线排序分，不替代原始指标，也不代表动力学可执行性、真实硬件安全性或任务成功率。

## 2. 执行规则

- 全部项目代码、配置、测试和生成结果均放在 `/root/gpufree-data/retargeting_quality_assessment`。
- 原始 GMR、BVH、PKL 和机器人 XML 只读使用，不覆盖、不删除。
- 原则上不修改 GMR；确需采集 IK 目标或残差时使用本项目内的独立适配脚本。
- 每次只执行一个步骤；该步骤通过测试和交付验收后，才更新为“已完成”。
- 每完成一步，都要在本文件的“状态总览”和“执行记录”中更新状态、交付物、验证命令及结果。
- 主质量指标始终针对未滤波的原始重定向轨迹。
- 所有阈值配置化；比例统一为 `[0, 1]`，分数统一为 `[0, 100]`，数值列名包含单位。
- 动作类别和接触语义必须保留；未知类别不得因腾空、手触地或较大 jerk 被武断判为无效。

## 3. 状态总览

| 步骤 | 内容 | 状态 |
|---|---|---|
| 第 0 步 | 需求、文献、本机 GMR、数据、模型和环境核对 | 已完成 |
| 第 1 步 | 项目骨架、依赖配置、运动台账和环境自检 | 已完成 |
| 第 2 步 | PKL 加载、严格完整性校验和 `MotionRecord` | 已完成 |
| 第 3 步 | 平滑性、根运动和关节可行性纯数组指标 | 已完成 |
| 第 4 步 | G1 模型解析、DoF 顺序验证和 MuJoCo 正向运动学 | 已完成 |
| 第 5 步 | 碰撞、地面、接触代理和脚滑指标 | 已完成 |
| 第 6 步 | 事件、评分、硬失败和报告文件 | 已完成 |
| 第 7 步 | CLI、批处理、manifest、配置快照和绘图 | 已完成 |
| 第 8 步 | 数学测试、MuJoCo 冒烟测试和人工异常注入 | 已完成 |
| 第 9 步 | BVH/GMR 目标配对与动作相似性 | 已完成 |
| 第 10 步 | 分层批量实验、人工复核、敏感性分析和 v1 冻结 | 已完成 |
| 第 11 步 | Top-K/Random-K/Bottom-K 下游策略实验与评分校准 | 已完成（真实 GPU 实验；负校准结果） |
| 第 12 步 | 含物体与任务标签的交互一致性和任务有效性验证 | 已完成（受控 G1 交互基准；总分未校准） |
| 第 13 步 | 质量评估 v2 范围冻结、指标契约与验收协议 | 已完成 |
| 第 14 步 | 严格交互数据校验、逐帧指标、事件与 interaction_v2 评分 | 已完成 |
| 第 15 步 | 基础运动质量与交互/任务质量的统一 CLI、报告和 manifest | 已完成 |
| 第 16 步 | 受控异常验证、研究边界复核与质量评估最终交付 | 已完成 |

## 4. 分步实施与验收标准

### 第 0 步：实施基线核对

工作内容：

- 阅读课题描述和需求指南；
- 核对 GMR 仓库、提交、脚本和 IK 速度限制设置；
- 核对样例 BVH/PKL 格式、帧率和帧数；
- 核对 G1 XML 的 qpos 布局、关节顺序、限位、floor 和脚底 geom；
- 核对 `gmr` 环境依赖；
- 对研究任务的离线评价、下游学习和操作交互三个层级进行边界划分。

验收标准：形成基于本机真实状态的计划，不修改代码或原始数据。

### 第 1 步：项目骨架与环境自检

工作内容：

- 创建独立、可安装的 Python 项目骨架；
- 声明运行和开发依赖；
- 建立 Unitree G1 配置文件，集中保存模型名称、路径、接触策略、阈值和评分权重；
- 建立运动数据台账，显式记录 PKL、源 BVH、动作类别、接触策略和配对状态；
- 实现 `python -m gmr_quality doctor`，只检查环境和浅层样例元数据，不计算正式指标；
- 编写第 1 步自动化测试和使用说明。

验收标准：

- `gmr` 环境可以导入所有声明依赖；
- `doctor` 能检查 GMR 提交和工作树、脚本、G1 XML、29 DoF、qpos 地址、floor、左右脚底 geom 和运动台账；
- 已知的一帧 BVH/PKL 差异和未配对样例以警告报告，而不是静默忽略；
- 自动化测试通过；
- 不计算或输出任何正式质量分。

### 第 2 步：PKL 加载与严格验证

实现统一 `MotionRecord`，严格检查 fps、形状、帧数、NaN/Inf、四元数模长、单位和 DoF 数；内部四元数统一明确为 `xyzw`。非法数据输出稳定失败码，不能静默修复后评分。

验收：正常样例通过；NaN、Inf、零四元数、维度和 DoF 错误均能被测试准确识别。

### 第 3 步：纯数组运动学指标

实现关节 step/jump/速度/加速度/jerk、根平移导数、SO(3) 根角速度、硬限位、绝对与归一化近限位及速度阈值。

验收：常量、线性、二次轨迹、已知旋转和 `q/-q` 测试通过；差分结果正确对齐到原始帧。

### 第 4 步：模型解析与 MuJoCo 适配

从 XML 自动读取关节名、qpos 地址和限位，验证与 `dof_pos` 顺序一致；只在 MuJoCo 边界执行 `xyzw -> wxyz`，逐帧调用 `mj_forward()`。

验收：样例所有帧均可执行正向运动学；姿态转换正确；脚底高度为合理米制值。

### 第 5 步：碰撞与足地指标

实现原始自碰撞、非手部自碰撞、地面穿透、非足部触地、脚底高度、MuJoCo contact、启发式接触代理、脚滑和腾空描述量。

验收：能定位帧、body pair 和严重度；允许列表与动作接触策略全部来自配置；未知动作只描述不武断硬惩罚。

### 第 6 步：事件、评分与报告

合并连续异常帧，实现维度分、`quality_score_v0`、`hard_fail`、稳定失败码，以及 summary、frame metrics 和 events 三类 CSV。

验收：三个输出互相对应；缺失 fidelity 时重归一化已有维度，不能按满分处理；输出 `score_is_calibrated=false`。

### 第 7 步：完整 CLI 与复现记录

实现单文件/目录批处理、单文件失败隔离、resolved config、run manifest、输入和 XML 哈希、绘图及 README。

验收：坏文件不影响批次；重复运行结果一致；输出不写入原始数据目录。

### 第 8 步：系统测试与人工异常

实现数学单元测试、MuJoCo 冒烟测试，并注入关节跳变、根节点下移、近限位、脚滑和非法数据。

验收：每类异常都相对原始样例触发预期变化，并准确定位相关帧、关节或 body；完整测试套件通过。

### 第 9 步：人体—机器人动作相似性

建立可靠的 PKL/BVH 配对，复用 GMR 的人体尺度、offset 和关键身体映射，计算位置与 SO(3) 旋转 fidelity；必要时由独立适配脚本记录 IK 残差。

验收：坐标系、尺度和帧对齐通过测试；来源不明样例不参与 fidelity；IK 混合误差不伪装成米或角度。

### 第 10 步：批量离线实验与评价器冻结

按站立/行走、跑跳、舞蹈、复杂动态和 floor motion 分层批量评估，执行人工抽检、阈值敏感性和维度相关性分析。

验收：形成可复现实验表、异常可视化、动作类别内统计、评价器 v1 配置和局限说明。

### 第 11 步：下游机器人学习验证

在动作类别、数据量、训练预算和随机种子受控的条件下比较 Top-K、Random-K、Bottom-K，并把全量训练作为补充。记录成功率、收敛速度、episode length、MPJPE/W-MPJPE、跌倒率和置信区间；执行离线指标相关性和单指标消融。

验收：证明哪些离线指标能够预测实际学习价值，并据此校准评分；若不能预测，也必须如实报告负结果。

### 第 12 步：操作任务和交互一致性

引入包含物体 6D 轨迹、任务标签及接触语义的数据，评价手—物距离与接触时序、相对物体位姿、穿透、抓取稳定性、目标物体最终状态和任务成功率；重复受控数据质量实验。

验收：离线交互指标与真实仿真任务成功率建立实验关系，覆盖老师要求的典型操作任务，而不是用无物体 LAFAN1 结果替代。

### 第 13 步：质量评估 v2 范围与协议冻结

本轮只完成课题中的“质量评估指标”子系统，不在本项目中伪造 MotionDecode 原始人体数据，也不把复用 GMR 误称为新的重定向算法。保留第 0–12 步的历史结果，在其上冻结可独立交付的 v2 指标契约：

- 基础运动质量继续覆盖轨迹平滑性、关节可达性、机器人接触合理性和已验证来源下的人体—机器人 fidelity；
- 交互质量覆盖手—物距离、接触时序、相对位姿、非法穿透、抓持稳定性和任务最终状态；
- 对带 `robot_qpos` 的交互记录必须同时运行基础运动质量和交互/任务质量，不能只评价物体；
- `hard_fail` 与连续分数分离，缺失维度保存为不可用而不是满分；
- 交互代理允许配置合法接触重叠，只有超过容差的部分计为非法穿透；
- 输出基础分、交互分、任务分和统一 `quality_score_v2`，但在新的有效下游实验通过前始终保持 `score_is_calibrated=false`；
- 第 11 步现有真实 GPU 结果保留为计算受限负证据，第 12 步保留为受控合成仿真证据，均不扩大解释范围。

验收：形成明确的数据 schema、指标定义、失败语义、输出文件和测试矩阵；后续实现不得通过事后调权掩盖负结果。

### 第 14 步：交互指标 v2

严格验证交互 NPZ 的数值、形状、布尔语义、四元数、机器人根姿态、任务标签、阶段和物体参数；生成逐帧手—物距离、接触错误、相对位姿、滑动、非法穿透和任务状态；合并连续事件，并实现 `interaction_v2` 维度分、任务分、硬失败和可用性标志。

验收：正常、接触缺失、时序错位、相对位姿漂移、非法穿透、目标未完成和非法 schema 均有独立测试；正式 CSV 不包含 `Inf`。

### 第 15 步：统一质量评价入口

实现单文件/目录交互评价 CLI。每条交互数据先从 G1 `robot_qpos` 构造严格 `MotionRecord`，复用现有 MuJoCo 基础评价器，再与交互/任务指标合并为统一报告。

验收：输出 `quality_summary.csv`、`frame_metrics/`、`quality_events.csv`、resolved config、图和 manifest；坏文件隔离；输入、模型、配置、源代码和输出哈希可复核。

### 第 16 步：完整验证与交付

在现有受控 G1 操作示范上执行统一评价，补充逐类人工异常测试，复核指标方向、任务有效性和已有仿真成功率关系；形成质量评估方法说明、复现命令和科研边界。

验收：完整测试、doctor、依赖检查和 manifest 哈希通过；明确区分“指标工程完成”“总分未校准”“真实人体—物体外部验证未完成”和“技能学习验证未完成”。

## 5. 已确认的实施基线

- GMR 根目录：`/root/gpufree-data/GMR`
- GMR 基线提交：`bb1bbe40774794fceb2a7c579a3464a28e68c844`
- G1 XML：`/root/gpufree-data/GMR/assets/unitree_g1/g1_mocap_29dof.xml`
- 单文件 BVH 脚本：`/root/gpufree-data/GMR/scripts/bvh_to_robot.py`
- GMR 当前实际默认：`use_velocity_limit=False`
- G1：自由根节点 7 qpos + 29 个 hinge 关节，`nq=36`、`nv=35`
- floor geom 名称：`floor`
- 左右 `ankle_roll_link` 各有 4 个半径为 0.005 m 的球形脚底 geom
- 第 9 步完整 GMR 重放已纠正配对：文件名为 `g1_dance1_subject2.pkl` 的轨迹实际来自 `dance1_subject1.bvh`；3944/3944 个保存帧的 36 维 qpos 逐元素完全一致，原同名 subject2 配对错误
- 已验证差异：真实源 BVH 3945 帧，当前单文件脚本输出其前 3944 帧，最后一帧因保存循环的 break 顺序未写入
- `walk.pkl` 的源 BVH 尚不能可靠确认，fidelity 阶段前保持未配对状态
- GMR 工作树已有用户修改和未跟踪文件，本项目不覆盖它们

## 6. 执行记录

### 2026-08-02 — 第 0 步完成

交付结果：完成指南、论文实验方向、本机仓库、环境、数据和模型核对，并制定本计划。

关键结论：当前先建设离线运动学质量评价器是正确方向，但只有第 11 步和第 12 步才能分别验证数据对机器人学习和操作技能的实际价值。

### 2026-08-02 — 第 1 步完成

交付物：

- `pyproject.toml`：独立可安装项目和运行/开发依赖；
- `configs/unitree_g1.yaml`：本机路径、G1 模型约定、接触策略、阈值和 v0 权重；
- `data/motion_catalog.csv`：3 条本机样例及其来源配对、类别和接触策略；
- `src/gmr_quality/`：包入口、配置加载和只读环境 `doctor`；
- `tests/test_step1_doctor.py`：模型结构、来源缺口和评分边界测试；
- `README.md`：安装、自检、测试、pickle 安全和当前范围说明。

环境变更：在现有 `gmr` conda 环境中以 editable 模式安装 `gmr-quality-eval==0.1.0`，并补齐 `pandas==2.3.3`、`pytest==9.1.1`、`PyYAML==6.0.3`。实际 Python 为 `/opt/conda/envs/gmr/bin/python`，Python 版本为 3.10。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：3 passed in 0.94s

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv
结果：24 passed, 4 warnings, 0 failed；退出码 0

conda run -n gmr gmr-quality doctor --json
结果：JSON 输出有效，ok=true；退出码 0

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0
```

4 个预期警告：

1. GMR 工作树已有 3 项用户改动，本项目只报告且不覆盖；
2. `dance1_subject2` 的源 BVH 为 3945 帧、PKL 为 3944 帧，差值 1；
3. `walk_unpaired` 尚无可靠源 BVH，fidelity 必须保持禁用；
4. `amass_g1_sample` 尚无源动作配对，fidelity 必须保持禁用。

验收结论：第 1 步全部验收项通过。`doctor` 明确声明未计算正式质量指标，第 2 步尚未开始。

### 2026-08-02 — 第 2 步完成

交付物：

- `src/gmr_quality/types.py`：`MotionRecord`、输入单位与格式约定、验证配置、验证报告、稳定错误码和 `MotionLoadResult`；
- `src/gmr_quality/io.py`：可信 GMR PKL 加载、已加载 payload 校验及项目配置适配入口；
- `configs/unitree_g1.yaml`：新增 `gmr_pickle_v1`、可信 pickle、最少帧数、29 DoF、`xyzw`、四元数容差和米/弧度/秒约定；
- `tests/test_io.py`：真实样例、正常合成输入和各类受控非法输入测试；
- `README.md`：增加 Python 加载示例、失败处理和不静默修复说明。

实现约定：

- `MotionRecord` 字段明确为 `root_pos`、`root_quat_xyzw` 和 `joint_pos`，不使用含义模糊的跨模块 `root_rot`；
- 有效数组按源 dtype 和数值复制为只读数组，不执行滤波、NaN 填补、关节裁剪或四元数归一化；
- 任一错误都会得到 `record=None`、`data_valid=false` 和 `hard_fail=true`，避免后续指标误用非法轨迹；
- PKL 缺少 `dof_names` 时保留 `JOINT_ORDER_IMPLICIT` 警告但不伪称顺序已验证，第 4 步再与 XML 自动核对；
- 文件不存在、pickle 损坏、payload 类型、字段、fps、dtype、形状、DoF、帧数、NaN、Inf、四元数模长和关节名均使用稳定错误码。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：24 passed in 0.93s

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0
```

真实数据交付检查：

```text
g1_dance1_subject2.pkl: valid=true, (3944,3)/(3944,4)/(3944,29), fps=30
walk.pkl:                valid=true, (3944,3)/(3944,4)/(3944,29), fps=30
g1_motion.pkl:           valid=true, (89,3)/(89,4)/(89,29), fps=30
```

三条真实 PKL 均生成有效、只读 `MotionRecord`，并保留 `JOINT_ORDER_IMPLICIT` 警告。合成零模长四元数得到 `record=None`、`hard_fail=true` 和 `QUATERNION_NORM_INVALID`，验证报告可序列化。

验收结论：第 2 步全部验收项通过。尚未计算速度、加速度、jerk、关节限位或任何质量分，第 3 步未开始。

### 2026-08-02 — 第 3 步完成

交付物：

- `src/gmr_quality/derivatives.py`：基于真实 fps 的关节 step/速度/加速度/jerk、根平移导数和 SO(3) 根角速度；
- `src/gmr_quality/kinematic_metrics.py`：平滑性汇总、逐帧最大值与异常定位、`JointLimits`、硬/近限位和速度违反指标；
- `tests/test_derivatives.py`：常量、线性、二次、四元数符号翻转、已知角速度、joint jump 和 fps 敏感性测试；
- `tests/test_joint_limits.py`：限位边界、违反率、近限位、速度分母、标量/逐关节阈值和非法限位测试；
- `README.md`：增加纯数组指标用法、差分帧对齐和当前阶段边界说明。

实现约定：

- 所有导数使用 `dt=1/fps`，不写死 30 Hz；
- 一阶、二阶、三阶结果长度统一回原始 T 帧，并分别在开头填 1、2、3 行 `NaN`；一阶值对齐到区间结束帧；
- 根角速度使用相邻 `Rotation` 的相对旋转向量，不对四元数四个分量直接差分，`q/-q` 不产生假尖峰；
- joint jump 在未滤波的原始 `dof_pos` 上按 `max(abs(q[t]-q[t-1])) > 0.5 rad` 计算，分母为 `T-1`；
- 关节硬限位使用严格 `< lower` 或 `> upper`，恰好等于限位不算越界但算近限位；
- 同时保留绝对余量和归一化余量，越界时余量为负；
- 速度违反率只对 `T-1` 个有效速度帧和对应元素计数；阈值严格使用 `>`，恰好等于阈值不违规；
- 速度限制既支持当前统一 `3π rad/s`，也支持未来逐关节数组；该阈值仍只称为 GMR 参考阈值；
- 第 3 步的 `JointLimits` 由调用方提供，不手工硬编码 G1 限位；XML 自动提取留在第 4 步。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：40 passed in 1.27s

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0
```

真实数据交付检查（`g1_dance1_subject2.pkl`，3944 帧，30 Hz）：

```text
joint_jump_count:        86（独立 NumPy 公式复核同为 86）
joint_jump_rate:         0.021810803956378393
joint_step_p95/max:      0.183439 / 1.582919 rad
joint_velocity_p95/max:  5.503171 / 47.487583 rad/s
joint_jerk_p95/max:      3871.810374 / 88290.167969 rad/s^3
root_speed_p95/max:      1.638732 / 2.707360 m/s
root_ang_vel_p95/max:    4.308588 / 12.413402 rad/s
```

上述真实结果是未评分的原始描述量，不代表动力学可执行性。由于 PKL 没有 `dof_names`，当前最差关节暂以 `joint_17` 等索引名报告；第 4 步核对 XML 后替换为真实关节名。1/2/3 阶差分的前导 NaN 行实测均正确。

验收结论：第 3 步全部验收项通过。尚未实现 XML 模型解析、真实 G1 限位绑定、MuJoCo 正向运动学、碰撞/接触或质量分，第 4 步未开始。

### 2026-08-02 — 第 4 步完成

交付物：

- `configs/unitree_g1.yaml`：新增当前 GMR G1 XML 的 SHA-256 和自由根关节名，防止模型内容被替换后仍静默评价；
- `src/gmr_quality/model_info.py`：严格解析并验证 XML 身份、`nq/nv`、自由根、29 个 hinge、qpos/DoF 地址、关节名、真实限位、body/geom 元数据、floor 和 8 个脚底球形 geom；实现轨迹—模型绑定及来源声明；
- `src/gmr_quality/mujoco_kinematics.py`：只在 MuJoCo 边界执行 `xyzw -> wxyz`，按 XML qpos 地址构造姿态，逐帧重置 `MjData` 并调用 `mj_forward()`，输出只读的 body/geom 位姿和原始 contact 快照；
- `tests/test_model_info.py`：模型哈希、布局、限位、脚底几何体、显式/隐式 DoF 顺序及真实关节名测试；
- `tests/test_mujoco_kinematics.py`：四元数顺序、qpos 映射、根姿态、模型绑定防串用、区间错误和真实样例 3944 帧 FK 测试；
- `README.md`：新增 G1 模型绑定、真实限位、FK 用法，以及当前可证明范围和来源限制说明。

实现约定：

- 评价模型锁定为 SHA-256 `85a25128d42ad3cc8dd071c089806a2773a59ee0689edcc646aeb0201ae8f4c8`；加载时不匹配即失败；
- 从 XML 得到自由根 7 qpos 加 29 个 hinge，`nq=36`、`nv=35`；关节 qpos 地址严格为 7–35，限位和关节名按该顺序自动构建，不手工抄写；
- PKL 若提供 `dof_names`，必须与 XML 顺序逐项完全一致；不一致直接拒绝绑定；
- 当前真实 GMR PKL 没有 `dof_names`，只能依据 GMR 导出格式的 `qpos[7:]` 约定完成结构性绑定，因此同时保留 `JOINT_ORDER_INFERRED_FROM_GMR_QPOS_LAYOUT` 和 `SOURCE_MODEL_PROVENANCE_UNAVAILABLE`；本项目不把评价模型身份已验证误称为原始生成模型来源已得到证明；
- 内部 `MotionRecord` 继续保持 `xyzw`，不原地改写；MuJoCo qpos 边界转换为 `wxyz`；
- 每帧先 `mj_resetData()` 再 `mj_forward()`，不执行 `mj_step()`，所以结果是参考姿态运动学几何量，不代表控制器跟踪、动力学稳定性或硬件安全性；
- 第 4 步仅保存 MuJoCo 原始 contact，不进行碰撞、地面接触或脚滑分类，避免提前越过第 5 步的接触语义与策略设计。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：49 passed in 3.34s

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv
结果：24 passed, 4 warnings, 0 failed；退出码 0

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0
```

真实数据交付检查（`g1_dance1_subject2.pkl`）：

```text
模型：                    nq=36, nv=35, hinge DoF=29，XML 哈希匹配
轨迹绑定：                3944 帧，gmr_qpos_after_free_root，评价模型身份已验证
来源声明：                原始生成 XML 来源尚不可证明，保留两项稳定警告
最差 step 关节：          left_shoulder_yaw_joint，frame=1790
正向运动学：              3944/3944 帧成功，last_frame=3943
MuJoCo time：             每帧均为 0.0 s，确认未执行仿真积分
脚底 geom 高度范围：      -0.065708 至 1.610477 m
脚底高度 P05/P95：        0.014519 / 0.287382 m
```

脚底最小高度出现负值不在本步静默修复或直接判分；它是第 5 步地面穿透与接触合理性分析的输入。验收结论：第 4 步全部验收项通过，真实样例所有帧可执行正向运动学，姿态边界转换和脚底米制几何量已验证。尚未实现碰撞/足地/脚滑指标或质量分，第 5 步未开始。

### 2026-08-03 — 第 5 步完成

交付物：

- `src/gmr_quality/contact_metrics.py`：接触阈值与策略配置、逐帧碰撞/足地数组、运动级汇总、初步连续段计数，以及从 MuJoCo FK 直接计算的统一入口；
- `src/gmr_quality/__init__.py`：公开第 5 步的配置、指标类型和计算 API；
- `tests/test_contact_metrics.py`：受控 self/ground contact、手部允许模式、穿透严重度、脚底表面高度、接触代理、脚滑、腾空、策略语义、帧对齐和真实样例测试；
- `README.md`：新增第 5 步调用示例、指标精确定义、策略解释和当前能力边界。

实现约定：

- 原始自碰撞候选只接受 MuJoCo 模型碰撞过滤后实际产生的、不同机器人 body 之间的非 floor contact；同时保留 `any` 和依据 `allowed_hand_contact_patterns` 排除允许手部接触后的 `nonhand` 口径；
- 每帧保存自碰撞 body pair、contact 数和最大穿透深度，汇总保存首帧、最严重帧、最严重 body pair 与按同一 pair 连续出现计算的初步事件数；
- 地面由配置指定并在模型加载时验证的 `floor` geom 识别；地面穿透深度使用 `max(0, -contact.dist)`，严格超过 `0.01 m` 标记穿透，严格超过 `0.05 m` 标记严重穿透；
- 非足部触地的允许脚 body 来自配置指定的左右脚及 XML 解析结果，不在代码中硬编码 geom ID；所有触地 body、穿透 body 和非足部 body 均可定位到原始帧；
- 左右脚底点使用 XML 自动找到的各 4 个球形 geom，脚底表面 z 为球心 z 减半径；不使用踝关节中心，也不把第 4 步的球心高度误当成表面高度；
- 接触代理使用 `abs(foot_height)<=0.01 m` 且 `abs(vertical_speed)<=0.2 m/s`；水平/竖直速度按真实 fps 计算并对齐到区间结束帧，第 0 帧为 `NaN` 且代理为 false；
- 脚滑只在代理接触时按水平速度严格大于 `0.1 m/s` 标记；P95/max 使用所有代理接触期的有限水平速度样本；
- `airborne` 仅表示左右脚代理均为 false，保留比例和最长连续时间，但不默认作为负向质量结论；
- 同时输出 MuJoCo 脚地 contact、启发式代理和二者不一致帧；由于没有人体真实接触标签，始终保留 `CONTACT_PROXY_WITHOUT_SOURCE_LABELS`；
- `feet_only`、`floor_motion`、`descriptive_only` 及其失败语义全部来自配置。原始非足部触地和腾空观测不因策略而被删除；未知动作使用 `descriptive_only` 时只描述、不武断硬惩罚；
- 本步的 event count 是便于汇总复核的连续段计数，尚未生成正式事件记录；事件明细、评分、`hard_fail` 和 CSV 属于第 6 步。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：55 passed in 4.57s

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv
结果：24 passed, 4 warnings, 0 failed；退出码 0

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0
```

受控公式测试结果：

```text
手腕—髋部 contact：        计入 self_collision_any，不计入 nonhand
双膝 contact：             同一 body pair 连续 2 帧，聚合为 1 个初步事件
pelvis—floor：             0.030 m，触发 >0.010 m 穿透并定位 pelvis/frame 2
脚底表面高度：             球心 0.005 m、半径 0.005 m，正确得到 0.000 m
接触期水平速度：           0.200 m/s，触发 >0.100 m/s 脚滑并定位对应帧
descriptive_only：         保留非足部触地和腾空观测，失败策略标志均为 false
```

真实数据交付检查（`g1_dance1_subject2.pkl`，`dance + feet_only`）：

```text
总 MuJoCo contact：        1037
自碰撞 any/nonhand 帧：    255 / 59
自碰撞初步事件 any/nonhand：239 / 35
最深自碰撞：               0.059240 m，right_hip_yaw_link ↔ right_wrist_pitch_link，frame 639
地面 contact 帧：          180
穿透/严重穿透帧：          44 / 11
非足部触地帧：             11
最大地面穿透：             0.070708 m，left_ankle_roll_link，frame 2879
左右接触代理比例：         0.099899 / 0.038286
脚滑帧：                   179，比例 0.045385，初步事件 109
接触期水平速度 P95/max：   1.243972 / 3.099399 m/s
腾空代理帧：               3435，比例 0.870943，最长 5.766667 s
```

独立复核直接遍历 3944 帧原始 `ForwardKinematicsFrame.contacts`，得到 `raw=1037`、`self_any=255`、`self_nonhand=59`、`ground=180`、最大地面穿透 `0.07070770008821849 m`、脚底表面最低高度 `-0.07070770008821849 m`，与模块结果完全一致。`walk.pkl` 和 `g1_motion.pkl` 也完成全帧计算；二者因类别/来源未知使用 `descriptive_only`。

上述结果是未校准的离线参考轨迹描述量。较高的腾空代理或脚滑值可能同时受动作语义、GMR 根轨迹和代理阈值影响，不能在第 5 步直接等同于任务失败或机器人不可学习。验收结论：第 5 步全部验收项通过；第 6 步的正式事件、维度分、总分、硬失败和报告文件尚未开始。

### 2026-08-03 — 第 6 步完成

交付物：

- `src/gmr_quality/events.py`：把关节、碰撞、地面、脚滑和腾空逐帧标志转换为闭区间、实体感知的正式事件，记录时间、持续时间、实体、最大严重度、单位和阈值；
- `src/gmr_quality/scoring.py`：v0 帧级合规分、缺失维度权重重归一化、未校准状态、稳定失败/警告原因及硬失败判定；
- `src/gmr_quality/evaluation.py`：串联已加载数据、模型绑定、运动学、接触、评分、事件和决策的单动作 Python API；非法输入不进入 FK，但仍生成硬失败评估；
- `src/gmr_quality/reporting.py`：生成 `quality_summary.csv`、`frame_metrics/<motion>.csv` 和 `quality_events.csv`，并提供 schema 稳定的行构建函数；
- `src/gmr_quality/contact_metrics.py`：为正式事件补充逐帧、逐 self-collision body pair 和逐 ground body 的独立穿透深度，避免严重度错误归因；
- `tests/test_step6_scoring_reporting.py`：评分重归一化、可用 fidelity、事件对应、硬失败原因、非法输入报告、CSV schema/行数/时间和跨文件一致性测试；
- `outputs/step6_dance_validation/`：真实舞蹈样例的第 6 步交付报告；
- `README.md`：新增单动作评估、三类报告、v0 公式、硬失败和事件时间约定。

实现约定：

- `smooth_valid = not joint_jump and not velocity_violation`；`joint_valid = not position_limit_violation and not near_limit_2pct`；
- 基础 `contact_valid` 排除 `self_collision_nonhand`、`ground_penetration` 和 `foot_slide`；若所选接触策略明确禁止非足部触地或腾空，则相应标志也加入 contact invalid；原始观测不会因策略而删除；
- 各维度分均为 `100 * mean(valid_frame)`，范围 `[0,100]`；jerk、加速度等暂无统一可靠阈值的描述量不强行纳入 v0；
- 当前 fidelity 不可用时输出 `NaN`，可用维度固定为 smoothness、joint feasibility、contact，并将 `0.30/0.25/0.30` 除以 `0.85` 重归一化；绝不把缺失 fidelity 当作 100 分；
- `score_version=v0`、`score_is_calibrated=false`；`quality_score_v0` 只作为等待第 10/11 步校准的离线排序量；
- `hard_fail` 与连续分数独立。当前条件为输入硬错误、任意 XML 关节硬限位违反、地面穿透严格超过 `0.05 m`、同一非手部 self-collision pair 连续达到或超过 5 帧，以及接触策略配置为失败的非足部触地/腾空；
- 原因同时保存稳定 code、包含实体/帧区间的 token 和可读详情；总分不能覆盖硬失败；
- 事件 `start_frame/end_frame` 均为闭区间，`start_time_s=start/fps`、`end_time_s=end/fps`、`duration_s=(end-start+1)/fps`；
- self collision 按 body pair 合并，非足部触地按 body 合并；严重度来自对应实体自身的 contact 深度；ground penetration、foot slide 和 airborne 按联合连续帧合并；
- 有效动作 frame CSV 行数严格等于 T，`time_s=frame/fps`；非法动作仍写 1 行 summary 和带表头的空 frame/events 文件；
- JSON 数组用于 CSV 内的实体、原因、可用维度和权重等结构化单元格，避免依赖模糊分隔符；
- 第 6 步只提供单动作 Python API 和报告写出；完整 CLI、批处理隔离、manifest、配置快照和绘图仍属于第 7 步。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：61 passed in 7.33s

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv
结果：24 passed, 4 warnings, 0 failed；退出码 0

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0
```

真实数据报告验收（`outputs/step6_dance_validation/`）：

```text
quality_summary.csv：                  1 行
frame_metrics/dance1_subject2.csv：   3944 行
quality_events.csv：                  1770 行
最后一帧/时间：                        frame 3943 / 131.43333333333334 s

smoothness_score：                    81.05983772819472
joint_feasibility_score：             26.74949290060852
contact_score：                       93.15415821501014
fidelity_score：                      NaN
归一化权重：                          0.352941 / 0.294118 / 0.352941
quality_score_v0：                    69.35479059778069
score_is_calibrated：                 false
hard_fail：                           true
```

事件构成：

```text
JOINT_JUMP：                  118
VELOCITY_LIMIT_EXCEEDED：     646
JOINT_LIMIT_EXCEEDED：        30
NEAR_JOINT_LIMIT_2PCT：       404
SELF_COLLISION_ANY：          239
SELF_COLLISION_NONHAND：      35
GROUND_PENETRATION：          10
UNDESIRED_GROUND_CONTACT：    7
FOOT_SLIDE：                  109
AIRBORNE_PROXY：              172
```

硬失败原因包括 1 个 `JOINT_LIMIT_EXCEEDED`、1 个 `SEVERE_PENETRATION`、3 个 `SUSTAINED_SELF_COLLISION` 和 7 个 `UNDESIRED_GROUND_CONTACT` 区间。关节最大超限仅为 `1.830738961539069e-07 rad`，但当前指南定义为任意明确硬限位违反即失败，因此本步没有暗中加入容差；该数值精度敏感性必须在后续人工复核和阈值敏感性分析中明确讨论。

独立读取 frame CSV 后重新计算得到：

```text
100 * mean(smooth_valid_flag)  = 81.05983772819472
100 * mean(joint_valid_flag)   = 26.74949290060852
100 * mean(contact_valid_flag) = 93.15415821501014
重归一化总分                  = 69.35479059778069
```

四项与 summary 完全一致；`SELF_COLLISION_ANY/NONHAND`、`GROUND_PENETRATION`、`UNDESIRED_GROUND_CONTACT` 和 `FOOT_SLIDE` 的 events 行数也逐项等于 summary event count，所有事件持续时间公式复核通过。验收结论：第 6 步全部验收项通过；第 7 步尚未开始。

### 2026-08-03 — 第 7 步完成

交付物：

- `src/gmr_quality/cli.py`：新增 `evaluate` 子命令，支持互斥的单 PKL 与目录批处理、相对 glob、元数据覆盖、JSON 输出、关闭绘图和可选的质量硬失败退出码；
- `src/gmr_quality/runner.py`：实现确定性输入发现、运动台账元数据解析、保守缺省语义、逐文件失败隔离、批次编排、输出目录保护及运行结果统计；
- `src/gmr_quality/manifest.py`：实现 SHA-256、工具/GMR Git 状态、Python/MuJoCo 运行时、resolved config 和 run manifest 写出；
- `src/gmr_quality/plotting.py`：使用无头 `Agg` 后端输出关节 step/速度/jerk 以及足底高度/速度/接触诊断图；
- `src/gmr_quality/reporting.py`：新增批量 summary/events 聚合和每动作独立 frame CSV，非法动作仍保留 summary 与空明细表；
- `tests/test_step7_cli_batch.py`：覆盖台账/缺省语义、单文件完整产物、逐字节复现、真正损坏的 pickle 隔离、manifest 哈希、原输入不变、输出目录保护和非空目录拒绝；
- `README.md`：新增单文件/目录 CLI、参数、输出结构、manifest、退出码、数据安全和 fidelity 边界说明；
- `outputs/step7_batch_validation/`：两条真实 3944 帧 GMR 轨迹的批量 CLI 交付结果，含 2 个 frame CSV、聚合 summary/events、4 张 PNG、resolved config 和 run manifest。

实现约定：

- 单文件模式和目录模式严格互斥；目录 glob 只能是非空相对模式，发现的文件按解析后的绝对路径排序，确保批次与报告行顺序稳定；
- 台账中已登记的动作使用明确的名称、类别、接触策略和源 BVH；未登记动作只使用路径生成稳定名称，并保守回退到 `unknown + descriptive_only`，不依据评估结果猜测动作语义；
- 逐文件加载、严格验证、指标、评分和绘图相互隔离。坏 pickle 会稳定报告 `PICKLE_READ_ERROR`，不会阻断后续正常文件；每个输入的 `evaluated`、`failed_validation` 或 `failed_exception` 状态写入 manifest；
- 程序处理成功与数据质量结论分离：有效轨迹即使 `hard_fail=true` 仍算 `evaluated`，默认退出码为 0；操作/输入失败返回 2，只有显式指定 `--fail-on-hard-fail` 时质量硬失败返回 3；
- 聚合 `quality_summary.csv` 每个可验证输入一行，`quality_events.csv` 跨动作合并，`frame_metrics/<motion>.csv` 按动作分开，避免大批次帧数据混成单个不可管理文件；
- `run_manifest.json` 记录工具和 GMR Git 状态、运行环境、执行命令、模型身份、源/解析后配置、阈值、评分、接触策略、输入状态，以及输入、源 BVH、XML、配置和所有交付文件的 SHA-256；manifest 自身不递归哈希；
- `resolved_config.yaml` 保存实际生效配置；诊断 PNG 固定后端、尺寸、dpi 和 metadata，两次独立运行的 CSV、YAML 和 PNG 逐字节一致；
- 输出必须位于所有原始输入目录之外，而且只能写入不存在或为空的目录；工具不清空已有内容、不覆盖原始数据。测试前后输入 SHA-256 完全一致；
- 源 BVH 在本步只作为来源元数据和哈希写入。即使已配对，manifest 仍明确保存 `fidelity_requested=true, fidelity_enabled=false`，避免在第 9 步完成坐标系、尺度和帧对齐前伪造动作相似性分数；
- 本步完成的是可复现的离线评价工程入口，不把轨迹图、离线总分或程序成功误称为动力学可执行、任务成功或对机器人学习有效；这些研究结论仍由第 9–12 步验证。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：66 passed in 11.17s

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv
结果：24 passed, 4 warnings, 0 failed；退出码 0

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0
```

重复运行验收（`amass_g1_sample`，89 帧）：

```text
两次运行：                         discovered=1, succeeded=1, failed=0
quality_summary.csv：              SHA-256 完全一致
quality_events.csv：               SHA-256 完全一致
frame_metrics/amass_g1_sample.csv：SHA-256 完全一致，89 行
resolved_config.yaml：             SHA-256 完全一致
2 张诊断 PNG：                    SHA-256 分别完全一致，PNG 签名有效
原始 PKL：                         运行前后 SHA-256 不变
```

批处理失败隔离测试使用一条真实 AMASS 轨迹和一条故意损坏的 pickle，得到 `discovered=2, succeeded=1, failed=1, hard_fail=1`；坏文件为 `failed_validation + PICKLE_READ_ERROR`，正常文件仍生成 89 行 frame CSV，两个输入哈希均未改变，CLI 按约定返回 2。

真实批量交付验收：

```text
命令输入：                         /root/gpufree-data/GMR/outputs/*.pkl
发现/成功/失败/跳过：              2 / 2 / 0 / 0
质量 hard_fail：                   2（数据结论，不是程序失败）
quality_summary.csv：              2 行
quality_events.csv：               5911 行
dance1_subject2 frame CSV：        3944 行
walk_unpaired frame CSV：          3944 行
诊断 PNG：                         4 个，签名有效
manifest 输出哈希：                9/9 独立复核一致
dance1_subject2 quality_score_v0：69.35479059778069，hard_fail=true
walk_unpaired quality_score_v0：  63.52613053334924，hard_fail=true
fidelity：                         requested=true, enabled=false
```

验收结论：第 7 步全部验收项通过。坏文件不会终止批次，重复运行的正式评价产物一致，所有输出均位于本项目的独立交付目录且原始 GMR 数据未被改写。第 8 步尚未开始。

### 2026-08-03 — 第 9 步完成

执行顺序说明：本步按用户明确要求直接执行，第 8 步继续保持“未开始”；本步没有夹带第 8 步的人工异常注入工作。

来源审计与关键纠正：

- 原台账依据文件名把 `/root/gpufree-data/GMR/outputs/g1_dance1_subject2.pkl` 配到 `dance1_subject2.bvh`，只核对了帧数，没有证明内容来源；
- 使用锁定的 GMR 提交、LAFAN1 loader、G1 IK 配置和默认 `use_velocity_limit=false` 分别重放三个同帧数候选，发现保存 PKL 的第 0 帧与 `dance1_subject1.bvh` 重放结果 36 维 qpos 完全相同，而与 subject2 的最大绝对差为 `4.7984657408930005`；
- 对 `dance1_subject1.bvh` 完整重放 3944 个保存帧后，3944/3944 帧逐元素完全相等，`mismatch_frames=0`、`max_abs_qpos_difference=0.0`，由此纠正真实来源；
- 源 BVH 有 3945 帧。当前 `bvh_to_robot.py` 在最后一帧求解后先执行 break、后执行 append，因此 PKL 严格对应源前 3944 帧，帧对齐方式锁定为 `source_prefix_drop_last`；
- 纠正后的 PKL、BVH、GMR loader、重定向实现、IK 配置和脚本哈希及完整重放结果保存在 `data/pair_replay_evidence.json`。该证据取代前面步骤中仅凭同名文件作出的历史性错误配对判断。

交付物：

- `src/gmr_quality/fidelity.py`：GMR LAFAN1 loader 身份验证、IK 配置哈希验证、映射解析、人体局部尺度/offset 复现、严格帧/FPS 对齐、关键身体米制位置误差、SO(3) 姿态误差、末端误差、逐帧 fidelity 标志和运动级汇总；
- `data/motion_catalog.csv`：纠正真实源 BVH，新增配对状态、PKL/BVH SHA-256 和帧对齐列；只有 `verified_replay_exact` 条目可启用 fidelity；
- `data/pair_replay_evidence.json`：完整重放的机器可读来源证据、代码/配置/输入哈希、对齐解释和被拒绝候选；
- `configs/unitree_g1.yaml`：新增 GMR loader/IK/证据路径、IK 配置哈希、允许对齐方式、BVH 帧率量化容差、末端身体列表和 fidelity 配置；
- `src/gmr_quality/evaluation.py`、`scoring.py`：只对验证配对注入逐帧 fidelity，启用完整四维权重；未配对或仅手工提供 BVH 时继续输出 `NaN` 并重归一化其他维度；
- `src/gmr_quality/reporting.py`：summary 新增来源/对齐/尺度/身体列表、位置/末端/姿态 mean/P95/max、最差身体/帧和 pass rate；frame CSV 新增当帧均值、最大值和最差身体；
- `src/gmr_quality/plotting.py`：验证配对额外生成位置误差、姿态误差和逐帧 pass 的 fidelity 诊断图；
- `src/gmr_quality/runner.py`、`cli.py`、`doctor.py`：严格配对门控、每输入 fidelity 状态、manifest 证据/哈希、CLI 统计，以及环境自检中的 IK/loader/证据、配对哈希和帧对齐检查；
- `tests/test_step9_fidelity.py`：覆盖纠正配对、局部尺度、旋转 offset 的 SO(3) 组合、严格帧对齐、真实指标、四维评分、CSV/PNG/manifest、未配对禁用和哈希不匹配拒绝；
- `README.md`：新增真实配对纠正、fidelity 公式、统计口径、CLI 门控、实测结果和边界；
- `outputs/step9_fidelity_validation/`：真实双动作批量交付，包含聚合 CSV、两个 frame CSV、5 张诊断图、resolved config 和带 fidelity 证据的 manifest。

实现与研究口径：

- 人体目标由 GMR 原始 `load_bvh_file()` 产生，复用其 BVH→MuJoCo 坐标转换；导入模块的源码路径必须与配置指定的 GMR loader 完全一致；
- 人体真实高度由当前 GMR loader 返回 `1.75 m`，相对于 IK 配置假设 `1.8 m` 的比例为 `0.9722222222222222`；该比例乘到 GMR 的逐身体基础 scale table，不使用单一全身尺度替代；
- 非根身体位置先相对 Hips 转到局部坐标，应用各自尺度，再移回已缩放根坐标；位置 offset 在应用旋转 offset 后转到世界坐标，与 GMR `update_targets()` 顺序一致；
- 姿态误差使用 `target_rotation.inv() * robot_rotation` 的 SO(3) 测地角并输出度，不对欧拉角或四元数分量直接相减；
- IK 两阶段映射必须对同一机器人身体使用相同人体目标和 offset，否则拒绝；位置指标覆盖任一启用阶段具有非零 position weight 的 13 个身体，姿态指标覆盖非零 orientation weight 的 14 个身体，末端为左右踝和左右腕 4 个位置目标；
- `keybody_pos_error_mean/P95/max` 和姿态对应统计在全部“帧×身体”样本上计算；每帧 pass 使用该帧各身体位置均值 `<0.10 m` 且姿态均值 `<10 deg`。P95 与 frame pass 是不同统计口径，报告和文档均明确区分；
- 只有台账状态、两侧实际哈希、期望哈希、IK 配置哈希、loader 路径、帧数、对齐方式和 FPS 全部通过时才计算 fidelity。CLI 直接传 `--source-bvh` 会标为 `provided_unverified`，只记录来源而不评分；
- BVH `Frame Time=0.033333` 量化得到 `30.000300003 Hz`，与保存 PKL 的 `30 Hz` 差 `0.000300003 Hz`；配置使用明确的 `0.001 Hz` 容差，既接受该文本量化误差，也不会静默接受真实帧率不一致；
- GMR `error1/error2` 混合位置、姿态、任务权重且无纯米/度量纲，本步没有把它伪装成位置或姿态误差；`ik_error_stage2` 继续为 `NaN`；
- fidelity 只是人体目标与机器人参考姿态的离线相似性，不证明控制器可跟踪、动力学稳定、任务完成或数据能提升模仿学习；后者仍须第 10–12 步实验。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：72 passed in 26.73s

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv
结果：29 passed, 3 warnings, 0 failed；退出码 0

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0

python -m json.tool data/pair_replay_evidence.json
结果：JSON 有效；退出码 0
```

3 个预期 doctor warning：GMR 工作树已有 3 项用户改动；`walk_unpaired` 与 `amass_g1_sample` 没有可靠源动作，因此 fidelity 保持禁用。纠正后的配对哈希和预期的一帧尾差均为 pass，不再以 warning 掩盖来源是否可信。

真实 fidelity 结果（`dance1_subject1_verified`，3944 帧）：

```text
位置目标/姿态目标/末端目标：         13 / 14 / 4
人体高度/配置假设/比例：             1.75 m / 1.8 m / 0.9722222222222222
key-body 位置 mean/P95/max：         0.0947530469 / 0.2415970688 / 0.4829448055 m
末端位置 mean/P95/max：              0.0393881510 / 0.1039243575 / 0.3991684848 m
link 姿态 mean/P95/max：             5.3616456532 / 16.5420489739 / 60.0281958170 deg
最差位置：                           right_shoulder_yaw_link，frame 1853
最差姿态：                           torso_link，frame 1854
通过帧：                             3404 / 3944
fidelity_score：                     86.30831643002028
四维 quality_score_v0：              71.89781947261663
score_is_calibrated：                false
hard_fail：                          true（来自既有可行性/接触规则，不由 fidelity 覆盖）
```

真实批量交付验收：

```text
发现/成功/失败/跳过：                2 / 2 / 0 / 0
fidelity 启用：                      1（dance1_subject1_verified）
fidelity 禁用：                      1（walk_unpaired）
summary/events：                     2 / 5911 行
两个 frame CSV：                    各 3944 行
诊断 PNG：                           5 个（验证配对多 1 张 fidelity 图）
manifest 输出哈希：                  10/10 独立复核一致
walk_unpaired fidelity_score：       NaN
walk_unpaired quality_score_v0：     63.52613053334924（仅可用维度重归一化）
```

验收结论：第 9 步全部验收项通过。坐标变换、逐身体尺度、SO(3) 姿态、严格帧对齐和来源哈希均有自动化测试；未知来源不会进入 fidelity；已验证样例的动作相似性已进入 CSV、图、manifest 和四维 v0 评分。第 8 步仍未开始，第 10 步尚未开始。

### 2026-08-03 — 第 8 步完成（在第 9 步之后补齐）

执行顺序说明：用户确认此前误以为第 8 步已经完成，因此本次只补齐第 8 步。上方第 7、9 步记录中“第 8 步尚未开始”的文字是对应历史时点的状态；本记录完成后，第 0–9 步已形成连续交付，第 10 步仍未开始。

交付物：

- `src/gmr_quality/pipeline_validation.py`：从可信 GMR PKL 的真实 G1 姿态生成静态基准、接触对照、关节跳变、低速近限位、根节点下移、脚滑和 NaN 七类受控夹具，并把它们送入正式批量评价器；输出机器可读的不变量检查、参数、场景汇总、输入哈希和正式报告路径；
- `src/gmr_quality/cli.py`：新增 `validate-pipeline` 子命令，支持输入、配置、输出目录、源帧、关闭绘图和 JSON 输出；受控 NaN 是预期失败时，外层自验证仍按不变量结果返回成功；
- `src/gmr_quality/__init__.py`：导出 `PipelineValidationResult` 和 `run_pipeline_validation` Python API；
- `tests/test_step8_pipeline_validation.py`：覆盖七个场景、精确帧/关节/body 定位、NaN 下游短路、CSV/事件/PNG、manifest 和夹具哈希、CLI 语义、原 PKL 不变及非空目录保护；
- 既有 `tests/test_derivatives.py`、`test_joint_limits.py`、`test_mujoco_kinematics.py`、`test_contact_metrics.py`：继续承担常量/解析轨迹的数学单元测试、SO(3) 测试、限位测试、3944 帧真实 PKL MuJoCo FK 冒烟测试及接触几何测试，并与本步系统测试一起在完整测试套件中执行；
- `README.md`：新增完整管线自验证命令、七类场景、不变量、输出结构、退出语义及研究边界；
- `outputs/step8_pipeline_validation/`：正式第 8 步交付，含 7 个受控 PKL、`pipeline_validation.json`、7 个 frame CSV、聚合 summary/events、12 张 PNG、resolved config 和 run manifest。

人工异常设计与检测结果：

- 基准取 `/root/gpufree-data/GMR/outputs/amass/g1_motion.pkl` 第 0 帧真实姿态，复制为 12 帧、30 Hz 静态轨迹；基准无关节跳变、近限位、硬限位、地面穿透或脚滑，`hard_fail=false`；
- `joint_jump` 在第 5 帧将 `left_hip_pitch_joint` 注入超过 `0.5 rad` 阈值的阶跃，评价器准确标记第 5、6 帧及该关节，`joint_jump_count=2`；
- `near_limit` 选择 `left_knee_joint`，用相邻步长低于跳变阈值的三角渐变在第 5 帧进入 2% 限位带；只标记第 5 帧，`joint_jump_count=0`、硬限位违反为 0，避免把近限位测试混成跳变测试；
- `root_down` 在第 5 帧将根节点下移 `0.12 m`，只在第 5 帧检测到地面穿透，最大深度 `0.08098449338166702 m`，最差帧为 5，并按既有 `0.05 m` 规则触发 `SEVERE_PENETRATION` 硬失败；
- `foot_contact_control` 将静态姿态最低脚底对齐到地面，脚滑为 0；`foot_slide` 只在第 5 帧水平移动根节点 `0.02 m`，左右脚速度峰值约 `0.6 m/s`，准确标记第 5、6 帧和 `left_foot/right_foot`；
- `invalid_nan` 在第 5 帧单关节注入 NaN，稳定得到 `DATA_NAN`，且 `bound_motion`、FK/运动学、接触、fidelity 和评分均未执行；其 frame CSV 只有表头；
- 七类受控夹具全部显式标记为 `synthetic_validation + descriptive_only + unpaired`。由于人工改变后的机器人轨迹不存在逐帧配对的人体源动作，本步不会伪造 fidelity；第 9 步的真实 verified replay fidelity 链路由完整测试套件继续覆盖。

正式管线验证结果：

```text
外层端到端不变量：                   7 / 7 passed，退出码 0
场景发现/正式 evaluated/预期非法：   7 / 6 / 1
质量 hard_fail：                     2（root_down、invalid_nan）
quality_summary.csv：                7 行数据
quality_events.csv：                 15 行数据
有效场景 frame CSV：                 6 × 12 行
invalid_nan frame CSV：              0 行数据（保留表头）
诊断 PNG：                           12 张，6 个有效场景各 2 张
manifest 正式输出哈希：              22 / 22 独立复核一致
受控输入夹具哈希：                   7 / 7 独立复核一致
原 AMASS PKL SHA-256：               a984c17a889cb691b8b4a8db139665008e69660be9ef1f9f211fbda66b0262d9
```

内层 run manifest 的 `failed=1` 是预期的 `invalid_nan` 校验失败，不是管线中断；其他六个场景仍全部完成 FK、指标、评分和报告，证明第 7 步的失败隔离语义在完整链路中生效。基准 `AIRBORNE_PROXY` 在 `descriptive_only` 策略下是接触标签缺失时的描述量，不属于本步定义的注入错误，也不会让基准硬失败。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：77 passed in 32.95s

conda run -n gmr python -m gmr_quality validate-pipeline \
  --input /root/gpufree-data/GMR/outputs/amass/g1_motion.pkl \
  --config configs/unitree_g1.yaml \
  --output-dir outputs/step8_pipeline_validation \
  --json
结果：checks=7/7，scenarios=7，runner_succeeded=6，runner_failed_as_expected=1；退出码 0

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1.yaml \
  --catalog data/motion_catalog.csv
结果：29 passed, 3 warnings, 0 failed；退出码 0

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0
```

3 个 doctor warning 均为既有预期状态：GMR 工作树有 3 项用户改动，`walk_unpaired` 与 `amass_g1_sample` 没有可靠人体源动作。它们不影响第 8 步的受控离线验证，也没有被静默伪造成 fidelity。

验收结论：第 8 步全部验收项通过。数学单元测试、真实 3944 帧 MuJoCo FK 冒烟测试、受控异常定位、非法输入短路、正式评分/事件/报告、失败隔离、绘图、配置快照和双层哈希均已连通并通过验证；配合已完成的第 9 步真实 fidelity 测试，第 0–9 步质量评估 pipeline 现已完整连续。此结论仍只证明离线评价工程及其检测行为正确，不等价于动力学可跟踪、任务成功或数据能够提升机器人学习；第 10–12 步研究验证边界保持不变。

### 2026-08-03 — 第 10 步完成

实验队列与重定向来源：

- `data/step10_source_selection.csv` 冻结 15 条 LAFAN1 完整序列，按 `standing_walking`、`running_jumping`、`dance`、`complex_dynamic` 和 `floor_motion` 五类各 3 条；不从长动作中事后挑选片段；
- `src/gmr_quality/step10_dataset.py` 和 `prepare-step10-dataset` 使用本机锁定 GMR 配置进行无头逐帧重定向，显式保存 29 个 DoF 名称，并对输出序列化数组执行逐元素复核；
- `outputs/step10_dataset/` 交付 15 个机器人 PKL、正式 catalog 和 `retarget_manifest.json`，共 90,540 帧；15/15 源 BVH 与机器人轨迹帧数精确一致，15/15 输出序列化复核精确一致；manifest 锁定源/输出、GMR loader、重定向代码、IK 配置和 G1 XML 哈希；
- v1 doctor 对该队列得到 `69 passed, 1 warning, 0 failed`。唯一 warning 是 GMR 工作树已有 3 项用户修改，本项目未覆盖这些内容。

评价器 v1：

- `configs/unitree_g1_v1.yaml` 冻结离线 v1 阈值、四维权重、动作接触策略、正式数据台账和来源证据；
- v1 增加 `1e-6 rad` 关节硬限位数值容差，仍保存原始超限量和近限位指标。敏感性实验表明把容差降为 0 不改变连续分和排名，但会新增 3 条 hard-fail；
- 平滑性/关节可行性/接触/fidelity 权重继续为 `0.30/0.25/0.30/0.15`，没有用本次总分的自相关回归权重；`score_is_calibrated=false`，明确把学习价值校准留到第 11 步；
- `configs/unitree_g1_v1.freeze.json` 锁定 v1 配置、冻结选择、重定向 manifest、catalog、自动分析和完整实验 manifest 的 SHA-256。

正式离线实验：

- `src/gmr_quality/step10_experiment.py` 和 `run-step10-experiment` 复用正式批处理评价器，完成 15/15 轨迹、90,540 帧评价；所有动作均启用哈希验证的 fidelity，所有帧均为 exact alignment；
- `outputs/step10_offline_experiment/analysis/motion_results.csv` 给出逐动作维度分、v1 总分、hard-fail、关键异常率和类别内/全局排名；`stratum_summary.csv` 给出类别内均值、标准差、中位数和极值；
- 五类 v1 均分依次为：站立/行走 `82.109672`、跑跳 `83.488895`、舞蹈 `76.125271`、复杂动态 `77.803647`、floor motion `63.423206`；全体 15 条中 12 条触发 hard-fail；
- 最高且无 hard-fail 的两条为 `jumps1_subject5=93.272430` 与 `walk1_subject1=91.238520`；`dance2_subject5=85.113720` 仍因持续自碰撞触发 hard-fail，证明连续排序与局部安全门控不能互相覆盖；
- 生成 15×3 张正式诊断图、类别分布图、敏感性图、相关性热图，以及抽检样本的 pose montage 和全序列异常时间线；所有绘图使用无头固定后端。

敏感性和相关性：

- 对 10 个阈值执行基准加 20 个单参数上下变体。最敏感的是把 fidelity 位置阈值从 `0.10 m` 收紧为 `0.08 m`：平均总分变化 `-13.113560`、单条最大绝对变化 `14.795473`，但全局排名 Spearman rho 仍为 `0.967857`；
- 速度阈值上下变体的 rho 为 `0.971429–0.978571`，其余变体的 rho 不低于 `0.985714`；关节限位容差 0 使 3 条 hard-fail 翻转，持续碰撞门槛 10 帧使 1 条翻转，表明硬门控比连续排序对边界参数更敏感；
- 全体描述性相关中，contact–fidelity rho=`0.660714`、joint feasibility–fidelity rho=`0.489286`、smoothness–contact rho=`0.096429`；contact–总分 rho=`0.814286` 受总分定义的数学耦合影响，不能解释为下游预测力；
- 类别内仅 n=3，相关性表所有行均显式保存 `inferential_use_allowed=false`，不使用 p 值作总体推断。

冻结视觉抽检：

- `data/step10_manual_review_protocol.md` 在查看图前固定每类最低分/最高分各 1 条，共 10 条；每条查看 6 个 MuJoCo FK 姿态、全序列异常时间线、关节、接触和 fidelity 诊断；
- `data/step10_manual_review.yaml` 保存逐条语义合理性、可见伪影严重度、指标一致性、观察依据、执行主体和日期；finalizer 严格验证 10/10 覆盖且每类恰好 2 条；
- 抽检结果为语义 `plausible=8, uncertain=2`，可见伪影 `mild=4, moderate=5, severe=1`，指标一致性 `agree=6, partial=4, disagree=0`；格斗和 floor motion 中合理触地语义与真实几何穿透可能同时出现，因此没有把所有低分都解释成源动作无效；
- 本次记录明确标注为 Codex 结构化视觉审计，不冒充独立领域专家人工复核；课题组成员可按同一冻结协议独立复核后重新运行 finalizer。

报告、复现与完整性：

- `STEP10_REPORT.md` 汇总实验设计、分层表、敏感性、相关性、抽检、复现命令、研究边界和第 11 步接口；
- `README.md` 增加从重定向、自动分析到抽检 finalization 的完整命令；
- `tests/test_step10_experiment.py` 验证 15 条/五类分层、v1 未校准冻结、21 个敏感性场景、90,540 帧正式产物、60 行相关性、10 条抽检、96 个文件及逐文件 SHA-256；
- `outputs/step10_offline_experiment/step10_manifest.json` 标记 `complete=true`，锁定 96 个交付文件；其 SHA-256 为 `19c4a108e1fd792c560420d8f7538dd54204b678a60f7d84606a19129dd7790c`。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests
结果：退出码 0

conda run -n gmr python -m pytest -q tests/test_step10_experiment.py
结果：4 passed in 1.64s

conda run -n gmr python -m pytest -q
结果：82 passed in 32.96s

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1_v1.yaml \
  --catalog outputs/step10_dataset/step10_motion_catalog.csv
结果：69 passed, 1 warning, 0 failed；退出码 0

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0

冻结哈希与完整实验 manifest 独立复核
结果：6/6 冻结引用哈希一致；manifest complete=true；96/96 文件大小与 SHA-256 由自动测试逐项复核一致
```

验收结论：第 10 步全部验收项通过。项目已形成可复现的五类实验表、类别内统计、异常可视化、冻结抽检、单阈值敏感性、维度相关性、未校准评价器 v1、完整 SHA-256 manifest 和局限说明。该结论满足当前“重定向后数据质量评估”的离线层需求，但不提前声称机器人动力学可执行、策略学习有效或操作任务成功；这些外部效度问题仍分别由第 11、12 步验证，第 11 步尚未开始。

### 2026-08-03 — 第 11 步进行中（历史记录：正式实验准备完成，GPU 阻塞）

本次已完成的交付：

- `configs/step11_beyondmimic.yaml`：在查看任何策略结果之前冻结 BeyondMimic/Isaac Lab 任务、仓库提交、四组 cohort、等量帧、训练预算、随机种子、评价指标、置信区间和评分校准门槛；
- `src/gmr_quality/beyondmimic.py`：把 GMR 30 Hz PKL 中心裁剪为每条相同的 3933 个源帧，重采样到 50 Hz，并显式执行 GMR 深度优先 DoF 顺序到 Isaac Lab 广度优先 DoF/body 顺序的转换；输出的 30-body FK、速度、SO(3) 角速度和多动作边界均经过严格校验；
- `src/gmr_quality/step11_protocol.py`：验证 Step-10 输入哈希，按五类各取 Top-1/固定随机 1/Bottom-1，Full 使用全部 15 条；生成环境审计、转换 manifest、cohort bundle 和 12 个“4 组 × 3 seed”训练作业；
- `integrations/beyondmimic/boundary_aware_command.py`：在不修改外部 `whole_body_tracking` 仓库的前提下，提供按 motion 等概率、按时间均匀且不跨动作边界的命令采样；关闭失败自适应采样，避免给低质量组分配更多训练经验而混淆等预算比较；
- `integrations/beyondmimic/train_local.py`、`evaluate_local.py`：提供本地 NPZ、TensorBoard、CUDA 强制门控的 PPO 训练入口，以及成功率、episode length、MPJPE、W-MPJPE、跌倒率和收敛 iteration 的确定性评价入口；
- `src/gmr_quality/step11_analysis.py`：要求 12/12 真实结果完整后才计算组间 95% CI、离线指标与学习指标 Spearman 相关、bootstrap CI、leave-one-category-out 单维消融及校准门控；结果缺失时稳定报 `STEP11_RESULTS_INCOMPLETE`，不会生成伪校准分；
- `tests/test_step11_preparation.py`：覆盖分层排名、PKL 哈希、转换 shape/顺序/FK、bundle 边界、环境阻塞、集成脚本语法、正式 manifest 哈希和结果缺失拒绝校准；
- `outputs/step11_learning_experiment/`：15 个单动作 NPZ、4 个 cohort bundle、cohort 表、12 个作业、环境审计、resolved config 和准备 manifest；总大小约 452 MiB；
- `STEP11_REPORT.md`：实验设计、已验证内容、当前阻塞、恢复命令和验收边界。

冻结实验口径：

- Top/Random/Bottom 各含 5 条轨迹，五个动作类别各 1 条；每条均为 3933 个 30 Hz 源帧，经上游一致的时间约定转换为 6554 个 50 Hz 帧，因此每组恰为 32,770 帧；Full 为 15 条、98,310 帧，仅作补充；
- Random-K 使用冻结种子 `20260803`，允许与 Top/Bottom 自然重叠，不把“中位数”伪称为随机；三组的数据条数、每动作帧数、类别覆盖、PPO 结构、4096 env、30,000 iteration 和 policy seeds `11/22/33` 受控；
- 训练采样先在 motion 上等概率、再在合法起始时间上均匀采样，并预留完整 10 s episode，既不按序列长度偏置，也不跨越拼接边界；
- 评分校准的主目标是 Full 策略对 15 条动作的成功率；采用 leave-one-motion-category-out 预测。只有 out-of-fold Spearman 为正且 paired bootstrap 95% CI 下界大于 0 才允许标记候选 v2 已校准，否则明确保留 v1 `score_is_calibrated=false` 并报告负结果。

正式准备结果：

```text
单动作 NPZ：                         15 × 6554 帧
Top/Random/Bottom bundle：           各 5 条 / 32770 帧
Full bundle：                        15 条 / 98310 帧
训练作业：                           4 cohort × 3 seed = 12
训练框架提交：                       cd65172032893724b445448818c34165846d847d
Isaac Lab 提交：                     c91a125c73c8b574878419a9583afc0b63b99f0a
转换与 manifest：                    已完成并逐文件哈希验证
真实策略结果：                       0 / 12
评价器校准：                         未执行，v1 继续保持未校准
```

当前唯一实质阻塞是容器未挂载可用 GPU：`NVIDIA_VISIBLE_DEVICES=void`，`torch.cuda.is_available()==false`、`torch.cuda.device_count()==0`，`nvidia-smi` 无法连接驱动，且不存在 `/dev/nvidia0` 等计算设备。只有 `/dev/nvidia-caps` 不构成 CUDA 计算设备。Isaac Lab/BeyondMimic PPO 必须使用 GPU；因此本记录不把离线代理、随机策略或伪造 JSON 计作下游学习结果。

验证命令与结果：

```text
conda run -n gmr python -m compileall -q src tests integrations
结果：退出码 0

conda run -n gmr python -m pytest -q tests/test_step11_preparation.py
结果：6 passed

conda run -n gmr python -m pytest -q
结果：88 passed in 36.24s

conda run -n gmr python -m gmr_quality prepare-step11-experiment \
  --config configs/step11_beyondmimic.yaml \
  --output-dir outputs/step11_learning_experiment --json
结果：preparation_complete=true，ready_for_training=false，blocker=GPU_DEVICE_UNAVAILABLE；退出码 0

conda run -n gmr python -m gmr_quality finalize-step11-experiment \
  --config configs/step11_beyondmimic.yaml \
  --output-dir outputs/step11_learning_experiment
预期结果：STEP11_RESULTS_INCOMPLETE: 12/12；在策略结果缺失时拒绝校准
```

当时验收结论：第 11 步尚未完成，状态保持“进行中”。数据、协议、训练入口、评价入口和分析门控已经交付并验证，但验收标准要求的真实策略训练、仿真评价、置信区间、指标预测力和评分校准均必须等待 GPU；该阻塞已由下一条完成记录解除。

### 2026-08-03 — 第 11 步完成（真实 GPU 实验；评分未通过校准）

GPU 阻塞复核：

- 先前阻塞来自 Codex 工具沙箱，而不是宿主机或 Conda 配置；
- 非沙箱 `/opt/conda/envs/isaaclab/bin/python` 实测 `torch 2.7.0+cu128`、`torch.cuda.is_available()==true`、`device_count==1`；
- 正式执行设备为 RTX 4090、24564 MiB、driver 580.126.09；GPU 审计写入 `outputs/step11_learning_experiment/gpu_execution_audit.json`；
- 4096-env smoke 和三个训练任务并发均成功，正式训练峰值总显存约 19 GiB，无 OOM 或 Isaac Lab 崩溃。

正式计算预算在查看任何正式 policy outcome 前由上游默认 30,000 iterations 修订并冻结为 300 iterations。依据 4096-env smoke 的约 4.19 s/iteration，原预算预计需约 17.5 GPU-days/12 policies；修订后每策略为 `4096×24×300=29,491,200` transitions，总计 `353,894,400` transitions。该结论只对应这一计算受限预算，不外推为 30,000 iterations 的最终性能。

本次新增/完善交付：

- `integrations/beyondmimic/run_formal.py`：支持最多三个 GPU 作业并发、失败即停、最终 checkpoint 校验、训练/评价状态持久化和已有结果断点恢复；
- `integrations/beyondmimic/gpu_audit.py`：记录正式执行上下文的 CUDA、GPU、驱动、显存和 `nvidia-smi` 证据；
- `train_local.py`：显式保存预算终点 `model_300.pt`；
- `evaluate_local.py`：修正首步目标未更新导致 MPJPE 混入 environment origin 的问题；每条 motion/seed 完成 100 个确定性 episode，并记录成功率、episode length、MPJPE、W-MPJPE、跌倒率与收敛 iteration；
- `step11_analysis.py`：增加种子配对组间差异、所有学习目标相关性、10,000 次向量化 bootstrap、稀疏成功诊断、严格 JSON、右删失收敛和关键结果/checkpoint 哈希；
- `outputs/step11_learning_experiment/`：12/12 个 `model_300.pt`、TensorBoard 曲线、12/12 个正式评价 JSON、统计表、校准决定、GPU 审计及最终 manifest；
- `STEP11_REPORT.md` 与 `README.md`：更新为真实实验结果、负校准结论、适用边界和复现命令。

正式实验规模与结果：

```text
策略：                              4 cohort × 3 seeds = 12/12 完成
评价：                              共 9000 episodes
Top-K 成功率：                      0.0000 [0.0000, 0.0000]
Random-K 成功率：                   0.0000 [0.0000, 0.0000]
Bottom-K 成功率：                   0.1920 [0.1789, 0.2051]
Full 成功率：                       0.0642 [0.0633, 0.0652]
Top/Random 收敛：                   300 iterations 内未收敛（以 301 右删失）
Bottom 收敛 iteration：             204.0 [199.0, 209.0]
Full 收敛 iteration：               194.7 [160.8, 228.6]
```

Bottom-K 相对 Top-K/Random-K 的种子配对成功率差均为 `+0.1920`，95% CI `[+0.1789,+0.2051]`。这不是“低质量数据更好”的证据：Full 中只有 `ground1_subject1` 一条 motion 成功率非零（三个 seed 为 `0.97/0.96/0.96`），其余 14 条均为 0，说明动作内在难度/可跟踪性与离线轨迹质量存在混杂。

离线总分与 Full 逐动作指标的 Spearman：成功率 `-0.4330`、episode length `-0.1643`、负 MPJPE `0.1964`、负 W-MPJPE `0.2857`、负跌倒率 `0.0643`。四维 leave-one-motion-category-out OOF Spearman 为 `-0.1862`，10,000 次 bootstrap 95% CI 为 `[-0.5032,0.0313]`；未满足“rho>0 且 CI 下界>0”的预注册门槛。

最终校准决定：`score_is_calibrated=false`，`outcome=retain_v1_uncalibrated_negative_result`。现有分数仍可用于解释离线轨迹缺陷，但不能解释为策略学习价值；本次不进行事后调权或发布 v2。

最终验证：

```text
conda run -n gmr python -m compileall -q src tests integrations
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：89 passed in 38.85s

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1_v1.yaml \
  --catalog outputs/step10_dataset/step10_motion_catalog.csv
结果：69 passed, 1 warning, 0 failed；退出码 0

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0

conda run -n gmr python -m gmr_quality finalize-step11-experiment \
  --config configs/step11_beyondmimic.yaml \
  --output-dir outputs/step11_learning_experiment --json
结果：complete=true，all_training_jobs_complete=true，policy_result_count=12，score_is_calibrated=false；退出码 0

最终 manifest 独立自动复核
结果：36/36 个关键分析、配置、GPU 审计、评价 JSON 和 model_300.pt 文件大小与 SHA-256 一致
```

验收结论：第 11 步的真实策略训练、仿真评价、收敛与跟踪指标、置信区间、离线指标相关性、单维消融和评分校准门控均已完整执行，因此状态更新为“已完成”。验收允许负结果，本实验如实证明 `quality_score_v1` 在当前数据与预算下不能预测实际学习价值。物体 6D、交互一致性和典型操作任务仍严格保留给第 12 步，不用本次无物体 tracking 结果替代。

### 2026-08-03 — 第 12 步完成（受控 G1 操作任务；交互总分未校准）

数据与范围核对：

- 本机没有 MotionDecode、青瞳原始数据或其他包含人体动作、物体 6D、任务标签和手—物接触标注的操作示范；
- 不把无物体 LAFAN1 动作附会为操作数据，也不把第 11 步 whole-body tracking 成功率替代操作任务成功率；
- 本步骤构建明确标注为 `controlled_synthetic_physics_demonstrations` 的 Unitree G1 受控基准，用于验证交互数据 schema、指标异常敏感性以及离线指标与真实物体仿真结果的关系；
- `human_object_external_validation=false`。该状态不妨碍本步骤工程和受控实验验收，但论文若声称对 MotionDecode/真实人体—物体数据有效，仍须取得真实数据后独立复验。

与前序步骤的连续性：

- 继续使用哈希冻结的 Unitree G1 29-DoF XML、36 维 qpos、`xyzw` 数据四元数和 MuJoCo 边界；
- 第 10 步 `quality_score_v1` 继续只描述平滑性、关节可行性、接触和 fidelity，本步骤交互分是补充维度，不覆盖 v1；
- 第 11 步发现无物体总分不能预测策略学习价值后，本步骤显式引入物体状态，并把最终物体状态和随机化动力学成功率作为独立外部目标；
- 新 `gmr_interaction_v1` NPZ schema 可由未来真实数据适配器填充，不需要改动指标接口。

任务与数据：

- 典型任务 `bimanual_lift_translate_place`：approach → grasp → lift → carry → lower → release → settle；
- 每条 6 s、50 Hz、300 帧，保存 G1 qpos、左右手位置/姿态、物体位置/姿态/线速度、任务标签、阶段标签、期望/实际双手接触、目标和物体几何；
- 5 个受控质量等级 × 3 个固定 seed = 15 条示范；扰动覆盖抓持间隙、左右手时序、搬运距离、垂直抖动和横向偏移；
- 每条执行 100 个 MuJoCo 随机化 episode，物体初始位姿、质量和摩擦受控随机，总计 1500 episodes；
- 成功要求最终物体落入三轴目标容差、稳定速度达标且过程中至少抬升 0.10 m。

G1 模型只有橡胶假手、没有闭合手指；直接摩擦冒烟测试无法稳定抬起物体。本步骤因此使用 `bilateral_contact_activated_soft_weld`：只有左右手同时真实接触且语义处于 grasp/lift/carry/lower 才激活软抓持，条件失效即释放；物体质量、摩擦、掉落、桌面碰撞和最终稳定继续由 MuJoCo 动力学计算。该抽象已写入配置、数据来源和报告，不能解释为真实灵巧手抓取。

本次交付：

- `configs/step12_interaction.yaml`：来源边界、G1 哈希、任务、阶段、扰动、随机化、六维指标、固定权重和校准门槛；
- `src/gmr_quality/interaction.py`：严格交互 NPZ 加载/校验、手—物 OBB 距离、接触时序、相对位姿、穿透、抓持与任务指标；
- `src/gmr_quality/step12_experiment.py`：G1 场景生成、双臂 IK、语义软抓持、动态物体回放、24 worker 随机化实验、置信区间、bootstrap、相关性和 manifest；
- CLI `run-step12-experiment`：从空目录一条命令复现实验；
- `tests/test_step12_interaction.py`：来源声明、G1 实际物理成功、6D schema、指标、1500 episode 及 27 文件哈希测试；
- `outputs/step12_interaction_experiment/`：15 个交互 NPZ、场景 XML/MJB、1500 episode 表、离线指标、相关性、校准决定、报告和最终 manifest，总大小约 86 MiB；
- `STEP12_REPORT.md` 与 `README.md`：正式实验、研究边界、结果解释和复现入口。

质量等级结果：

```text
等级        交互总分      鲁棒成功率      最终误差(m)      最大抬升(m)
clean       67.147        1.000           0.0338           0.1756
mild        72.373        1.000           0.0748           0.1647
moderate    80.774        0.000           0.0929           0.1680
severe      19.559        0.000           0.2024           0.0021
invalid     15.252        0.000           0.2004           0.0003
```

`moderate` 能抓持和抬升但搬运距离不足，最终状态越过目标容差，证明接触/抬升代理不能替代任务完成判定。

离线指标与 1500 个随机化 episode 成功率的 Spearman/10,000 次 bootstrap 95% CI：

```text
hand_object_proximity       0.4564   [-0.0668, 0.9354]
contact_timing              0.2887   [-0.2962, 0.8834]
relative_pose               0.5878   [ 0.0960, 0.9315]  通过
penetration                -0.9682   [-1.0000,-0.8704]  代理方向错误
grasp_stability             0.2939   [-0.2884, 0.9230]
task_effectiveness          0.8817   [ 0.7104, 0.9443]  通过
interaction_quality_score   0.2866   [-0.2876, 0.8776]  未通过
```

穿透分的反向关系来自球形假手代理为形成成功软抓持所需的有限重叠；正式结果产生后没有事后改阈值或调权。最终保留有证据的结论：任务有效性与相对物体位姿能够预测该受控基准的鲁棒成功；交互总分 `score_is_calibrated=false`，不能发布为已校准分数。

最终验证：

```text
conda run -n gmr python -m compileall -q src tests integrations
结果：退出码 0

conda run -n gmr python -m pytest -q tests/test_step12_interaction.py
结果：3 passed in 4.81s

conda run -n gmr python -m pytest -q
结果：92 passed in 41.78s

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1_v1.yaml \
  --catalog outputs/step10_dataset/step10_motion_catalog.csv
结果：69 passed, 1 warning, 0 failed；退出码 0

conda run -n gmr python -m pip check
结果：No broken requirements found；退出码 0

conda run -n gmr python -m gmr_quality run-step12-experiment \
  --config configs/step12_interaction.yaml \
  --output-dir outputs/step12_interaction_experiment --json
结果：demonstration_count=15，robustness_episode_count=1500，score_is_calibrated=false；退出码 0

最终 manifest 自动复核
结果：complete=true；27/27 个配置、场景、示范、统计和报告文件大小与 SHA-256 一致
```

验收结论：第 12 步已经形成包含物体 6D、任务/阶段标签和接触语义的连续数据接口，覆盖手—物距离、接触时序、相对物体位姿、穿透、抓持稳定性、最终目标状态和任务成功率，并在典型 G1 操作任务的 1500 个随机化动力学 episode 中建立了任务有效性/相对位姿指标与成功率的显著正关系。因此本步骤更新为“已完成（受控 G1 交互基准；总分未校准）”。真实人体—物体数据集外部验证仍被明确列为证据边界，不伪装为已完成。

### 2026-08-04 — 第 13 步完成（质量评估 v2 范围与协议冻结）

本轮根据老师课题描述和学长建议，明确只完成“质量评估指标”子系统，不把复用 GMR 叙述成新重定向算法，也不伪造 MotionDecode 未开放的原始人体数据。`configs/quality_assessment_v2.yaml` 在正式 v2 结果生成前冻结以下契约：

- 基础运动层包含平滑性、关节可达性、机器人接触和可信来源下的人体—机器人 fidelity；
- 操作层包含手—物接近度、接触时序、相对位姿、非法穿透、抓持稳定性和任务有效性；
- 对同一交互记录必须同时评价 G1 `robot_qpos` 和物体任务状态；
- 十维统一权重、任务容差、事件阈值、G1 XML/hash、合法代理重叠和失败语义全部配置化；
- 连续分与 hard-fail 分离，缺失维度重归一化且显式记录，不按满分处理；
- v2 始终为 `score_is_calibrated=false`，第 11 步负结果和第 12 步受控结果均不得扩大解释。

验收结论：步骤范围、数据契约、指标维度、输出、测试矩阵和证据边界均已在实现前明确，后续没有通过查看 v2 正式统计后修改冻结权重。

### 2026-08-04 — 第 14 步完成（严格交互指标 v2）

交付物：

- `src/gmr_quality/interaction.py`：严格 NPZ loader、`InteractionRecord`、逐帧指标、连续事件和 `interaction_v2` 评分；
- `tests/test_interaction_v2.py`：正常记录、接触缺失、相对漂移、非法穿透、任务失败和非法 schema 测试；
- `configs/quality_assessment_v2.yaml`：G1 球形假手代理的合法接触重叠、双侧接触、相对位姿、滑移和任务阈值。

关键修正：有限合法接触重叠不再误判为穿透；接触时序和穿透不会双重惩罚同一观测；相对位姿/滑移只在期望且观测到接触的有效段内计算，不跨失联段差分；球形代理的手朝向不制造虚假相对姿态漂移；不可用原始量使用 `NaN` 而不是 `Inf`。

验收结论：正常、missed/unexpected contact、漂移、滑移、非法穿透、任务失败和数据类型/有限性错误均可被独立检测并定位到逐帧 CSV 和事件区间。

### 2026-08-04 — 第 15 步完成（基础运动与交互/任务统一评价）

交付物：

- `src/gmr_quality/interaction_runner.py`：把每条交互记录的 36 维 qpos 严格转换为基础 `MotionRecord`，复用现有 G1 XML/MuJoCo 评价器，再合并交互和任务结果；
- `src/gmr_quality/cli.py`：新增 `evaluate-interaction` 和 `finalize-quality-assessment-v2`；
- `tests/test_interaction_runner.py`：单条完整报告、批处理坏文件隔离、输出保护、统计 finalizer 和哈希复核；
- `outputs/quality_assessment_v2_validation/`：正式统一评价输出。

统一输出包含 `quality_summary.csv`、`quality_events.csv`、`frame_metrics/*.csv`、机器人/交互诊断图、`resolved_config.yaml` 和 `run_manifest.json`。manifest 锁定输入、v2/base 配置、G1 XML、三个核心源文件和所有输出；目录模式下坏文件单独失败，输出不能覆盖非空目录或写入输入目录树。

验证命令与结果：

```text
conda run -n gmr python -m pytest -q \
  tests/test_interaction_v2.py tests/test_interaction_runner.py
结果：10 passed

conda run -n gmr python -m gmr_quality evaluate-interaction \
  --input-dir outputs/step12_interaction_experiment/demonstrations \
  --pattern '*.npz' \
  --config configs/quality_assessment_v2.yaml \
  --output-dir outputs/quality_assessment_v2_validation --json
结果：discovered=15，succeeded=15，failed=0，hard_fail=9，score_is_calibrated=false
```

验收结论：平滑性、关节可达性、机器人接触、手—物交互和任务有效性已对同一操作数据形成完整连续的单条/批量流水线。

### 2026-08-04 — 第 16 步完成（受控验证与质量评估最终交付）

正式受控诊断：

- 15 条 G1 双手操作记录全部有效，基础运动、交互和任务逐帧结果均成功输出，无 `Inf`；
- clean/mild/moderate/severe/invalid 的 v2 均分分别为 `85.859/83.700/79.477/44.755/42.362`，鲁棒任务成功率为 `1/1/0/0/0`；
- v2 总分与 1500 个随机化 MuJoCo episode 的鲁棒任务成功率 Spearman 为 `0.8597`，10,000 次 bootstrap 95% CI 为 `[0.6283,0.9022]`；任务有效性分为 `0.8817 [0.7104,0.9423]`；
- 基础运动分在五组中均约为 100，而任务结果不同，证明操作数据必须增加交互一致性和任务有效性，不能只用平滑性/关节限位替代；
- 由于 v2 修正在查看旧 v1 受控结果后定义且仍在同一批合成记录上诊断，`score_is_calibrated=false`、`human_object_external_validation=false`、`skill_learning_validation=false` 均保持不变。

最终交付：

- `QUALITY_ASSESSMENT_REPORT.md`：课题对应关系、输入契约、指标定义、分数/hard-fail、受控结果、复现命令和下一阶段独立实验；
- `README.md`：增加 v2 推荐使用入口和正式结果边界；
- `outputs/quality_assessment_v2_validation/validation/`：配对质量/仿真目标表、27 项相关性、分组统计、校准决定和结果摘要；
- `outputs/quality_assessment_v2_validation/run_manifest.json`：完整可复现清单。

最终验证：

```text
conda run -n gmr python -m compileall -q src tests integrations
结果：退出码 0

conda run -n gmr python -m pytest -q
结果：102 passed in 47.87s

conda run -n gmr python -m gmr_quality doctor \
  --config configs/unitree_g1_v1.yaml \
  --catalog outputs/step10_dataset/step10_motion_catalog.csv
结果：69 passed，1 warning，0 failed；唯一 warning 为 GMR 工作树已有 3 项用户修改，本项目未覆盖

conda run -n gmr python -m pip check
结果：No broken requirements found

质量评估 v2 manifest 独立复核
结果：complete=true；68/68 个输出、3/3 个核心源文件大小与 SHA-256 一致；15 条 summary 有效、0 Inf、9 hard-fail
```

验收结论：老师要求的四类“质量评估指标”已经形成完整、可运行、可解释、可批处理和可复现的工程交付，并在受控典型操作任务中验证了异常敏感性和任务结果关系。科研结论仍严格区分指标实现、受控诊断、独立校准和操作技能学习；后两项需要未来 held-out 真实操作数据与多随机种子策略实验，未在本轮伪装为完成。
