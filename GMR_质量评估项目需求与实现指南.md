# GMR 重定向轨迹质量评估：项目需求与实现指南

## 1. 文档用途

本文用于指导在已经能够运行 GMR 的远程服务器上，新建一个独立的代码项目，对 LAFAN1 经 GMR 重定向得到的 Unitree G1 运动轨迹进行离线质量评估。

执行本任务的 Codex 应先阅读本文，然后检查服务器上现有的 GMR 仓库、Python 环境、机器人 XML、样例 BVH 和样例重定向结果，再制定任务计划并实施。不要假设本文中的路径与服务器完全一致，应以服务器上的实际文件为准。

除非确有必要，不要直接修改 GMR 源码。优先新建独立项目，通过导入 GMR 模块或读取其输出文件和机器人 XML 完成评价。如果为了采集人体目标或 IK 残差而必须改动重定向流程，应将改动放在新项目的适配脚本中，或者以小型补丁形式隔离，并清楚记录。

## 2. 项目背景

人体动作经过 GMR 重定向后，得到的是满足机器人关节拓扑的参考轨迹，但“能显示出来”并不等于“质量足够好”或“机器人能够稳定学习和执行”。重定向过程中仍可能出现：

- 关节角突然跳变；
- 关节长期贴近或超出机械限位；
- 关节速度、加速度或 jerk 过大；
- 左右肢体或肢体与躯干自碰撞；
- 脚底穿透地面、悬空或接触时滑动；
- 机器人动作偏离人体源动作；
- 参考轨迹在动力学上不可执行，导致后续控制策略难以学习。

GMR 相关研究指出，地面穿透、自相交和关节值突然跳变会显著影响下游跟踪策略。Make Tracking Easy 使用关节跳变、自碰撞和接近关节极限等指标评价重定向结果。PHUMA 进一步从动作相似性、关节位置和速度可行性、足地接触、穿透和脚滑等方面定义物理可靠性指标。

本项目首先完成“参考轨迹的离线运动学质量评估”。动力学可执行性、策略成功率和任务有效性属于后续验证，不应与当前的离线指标混为一谈。

## 3. 项目目标

实现一个可批量运行、可解释、可复现的 G1 运动质量评估工具：

1. 输入一条或一批 GMR 生成的 `.pkl` 轨迹；
2. 读取轨迹并执行严格的数据完整性检查；
3. 计算平滑性、关节可行性、碰撞和足地接触指标；
4. 可选地读取对应 LAFAN1 BVH，计算人体—机器人动作相似性或 GMR IK 残差；
5. 为每条动作生成运动级汇总指标；
6. 为每条动作生成逐帧指标，以便定位异常帧和异常关节；
7. 输出暂定的维度分数、总分、硬失败标志及可读的失败原因；
8. 提供测试、配置文件、命令行接口和使用文档，确保结果可复现。

## 4. 当前范围与非目标

### 4.1 当前必须实现

- GMR `.pkl` 读取和格式验证；
- 轨迹平滑性指标；
- 关节位置限位和速度阈值指标；
- MuJoCo 正向运动学；
- 自碰撞、地面穿透、非足部触地；
- 脚底高度、接触代理和脚滑；
- 每动作汇总 CSV；
- 每动作逐帧 CSV；
- 人工异常注入与自动化测试；
- 暂定的质量维度分数和总分 `quality_score_v0`。

### 4.2 建议实现

- 记录异常事件区间及涉及的关节或身体对；
- 保存最严重异常的帧号和时间；
- 生成人工可读的失败原因；
- 绘制关节角、速度、jerk、脚底高度和脚滑曲线；
- 读取源 BVH 或在 GMR 重定向时记录人体目标，计算动作相似性；
- 保存一次评价使用的参数、代码版本和输入清单。

### 4.3 当前不应宣称已经实现

- 机器人真实硬件可执行性认证；
- 力矩、功率和地面反作用力可行性；
- 在重力和接触动力学下保持平衡的能力；
- 强化学习策略成功率；
- 操作任务成功率或物体交互一致性。

以上内容需要动力学轨迹优化、物理控制器、策略训练、物体状态或真实机器人实验。当前工具应把它们标记为后续扩展，而不是用简单运动学指标替代。

## 5. 需要先检查的现有环境

执行代理开始编码前，应先完成以下只读检查并记录结果：

1. 找到 GMR 仓库根目录；
2. 找到实际使用的 `bvh_to_robot.py`；
3. 找到 Unitree G1 的 MuJoCo XML，一般为：

   ```text
   assets/unitree_g1/g1_mocap_29dof.xml
   ```

4. 找到至少一条 LAFAN1 BVH 和对应的 GMR `.pkl`；
5. 用 Python 打印 `.pkl` 的键、数组形状、数据类型和帧率；
6. 确认当前 GMR 版本以及是否对 IK 启用了速度限制；
7. 确认环境中是否已有 `numpy`、`scipy`、`pandas`、`mujoco`、`matplotlib`、`pytest` 和 YAML 读取库；
8. 确认 G1 模型中的关节数、`qpos` 布局、关节名称顺序、地面 geom 名称和脚部 geom。

当前 GMR master 中，`GeneralMotionRetargeting` 的 `use_velocity_limit` 构造参数默认值可能为 `False`，LAFAN1 重定向脚本也可能未显式打开它；若打开，代码使用统一的 `3π rad/s` 限制。执行代理必须检查服务器上的实际版本，不能仅依据 README 推断。

## 6. 输入数据约定

### 6.1 GMR `.pkl`

常见字段为：

```text
fps         标量
root_pos    (T, 3)
root_rot    (T, 4)
dof_pos     (T, 29)
local_body_pos    当前版本中可能为 None
link_body_list    当前版本中可能为 None
```

关键坐标约定：

- GMR 保存到 `.pkl` 的 `root_rot` 通常是 `xyzw`；
- `scipy.spatial.transform.Rotation.from_quat()` 默认接受 `xyzw`；
- MuJoCo 的 `data.qpos[3:7]` 要求 `wxyz`；
- 因此，把原始 `.pkl` 写入 MuJoCo 前必须执行 `xyzw -> wxyz`：

  ```python
  root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]
  ```

项目内部建议保留明确命名，禁止使用含义模糊的 `root_rot` 在不同模块间传递：

```python
MotionRecord(
    fps: float,
    root_pos: np.ndarray,          # (T, 3), m
    root_quat_xyzw: np.ndarray,    # (T, 4)
    joint_pos: np.ndarray,         # (T, 29), rad
    source_path: Path,
)
```

### 6.2 输入验证

加载后至少检查：

- `fps` 是有限正数；
- `T` 足够计算差分，建议至少 4 帧；
- 三个数组的第一维一致；
- `root_pos.shape == (T, 3)`；
- `root_quat_xyzw.shape == (T, 4)`；
- `joint_pos.shape == (T, 29)`，或与机器人模型实际 DoF 数一致；
- 不存在 NaN 或 Inf；
- 四元数模长接近 1；
- 关节顺序能够与 MuJoCo `qpos[7:]` 对齐；
- 数据单位确认为米、弧度和秒。

建议输出：

```text
nan_count
inf_count
quat_norm_mean_error
quat_norm_max_error
shape_valid
data_valid
```

发现 NaN、Inf、严重维度错误或不可恢复的四元数错误时，应标记 `hard_fail`，并跳过依赖该数据的后续计算，不能静默修正后继续评分。

## 7. 总体处理流程

```text
发现输入文件
    ↓
读取并验证 GMR pkl
    ↓
构建统一 MotionRecord
    ↓
读取 G1 MuJoCo 模型和关节限制
    ↓
计算纯数组轨迹指标
    ↓
逐帧执行 MuJoCo mj_forward
    ↓
提取身体、脚底、接触和碰撞信息
    ↓
计算逐帧标志和运动级聚合值
    ↓
可选：读取人体源动作并计算 fidelity
    ↓
计算维度分数、暂定总分和 hard_fail
    ↓
写 summary、frame metrics、events、manifest
```

## 8. 指标一：轨迹平滑性

### 8.1 关节单步变化与 Joint Jump

令第 `t` 帧第 `j` 个关节角为 `q[t,j]`：

```text
joint_step[t,j] = abs(q[t,j] - q[t-1,j])
```

Make Tracking Easy 使用以下定义：若某帧任意关节单步变化超过 `0.5 rad`，则该帧为 joint jump：

```text
jump_frame[t] = max_j(joint_step[t,j]) > 0.5 rad
joint_jump_rate = jump_frame_count / (T - 1)
```

输出：

```text
joint_jump_rate
joint_jump_count
joint_step_p95_rad
joint_step_max_rad
worst_step_joint
worst_step_frame
```

该指标必须基于未平滑的原始输出计算。不能先低通滤波再检测，否则会掩盖待评价的异常。

### 8.2 关节速度、加速度和 jerk

使用真实帧率 `fps`，令 `dt = 1 / fps`：

```python
joint_vel = np.diff(joint_pos, axis=0) / dt
joint_acc = np.diff(joint_vel, axis=0) / dt
joint_jerk = np.diff(joint_acc, axis=0) / dt
```

输出至少包括：

```text
joint_vel_p95_rad_s
joint_vel_max_rad_s
joint_acc_p95_rad_s2
joint_acc_max_rad_s2
joint_jerk_p95_rad_s3
joint_jerk_max_rad_s3
worst_velocity_joint
worst_velocity_frame
worst_jerk_joint
worst_jerk_frame
```

对所有关节和所有有效帧计算全局统计，同时保留逐帧最大值。最大值用于定位尖峰，P95 用于描述整体质量。

### 8.3 根节点平移速度、加速度和 jerk

```python
root_vel = np.diff(root_pos, axis=0) / dt
root_acc = np.diff(root_vel, axis=0) / dt
root_jerk = np.diff(root_acc, axis=0) / dt
root_jerk_norm = np.linalg.norm(root_jerk, axis=1)
```

输出：

```text
root_speed_p95_m_s
root_speed_max_m_s
root_acc_p95_m_s2
root_jerk_p95_m_s3
root_jerk_max_m_s3
```

### 8.4 根节点角速度

禁止直接对四元数四个分量做有限差分。应计算相邻帧的相对旋转：

```python
rot = Rotation.from_quat(root_quat_xyzw)
relative_rot = rot[:-1].inv() * rot[1:]
root_ang_vel = relative_rot.as_rotvec() / dt
```

输出：

```text
root_ang_vel_p95_rad_s
root_ang_vel_max_rad_s
```

### 8.5 差分数组与原始帧的对齐

一阶、二阶、三阶差分长度分别为 `T-1`、`T-2`、`T-3`。逐帧 CSV 应统一使用原始帧编号，可对开头无法定义的值填 `NaN`，并在文档中说明对齐方式。不能因为差分长度不同而错误比较不同时间点。

## 9. 指标二：关节可行性

### 9.1 从 MuJoCo 模型读取关节限制

不要手工维护 29 个上下限。应从实际使用的 G1 XML 读取关节名称、`qpos` 地址和 `jnt_range`。

建议：

1. 遍历 `model.njnt`；
2. 选择 hinge 关节；
3. 按 `model.jnt_qposadr[joint_id]` 排序；
4. 确认其地址与 `qpos[7:]` 的顺序一致；
5. 读取 `model.jnt_range[joint_id]`；
6. 保存关节名，用于报告最差关节。

也可以使用 GMR 的 `KinematicsModel.get_dof_limits()`，但仍须验证返回顺序与 `.pkl` 的 `dof_pos` 一致。

### 9.2 硬限位违反

```text
position_violation[t,j] = q[t,j] < lower[j] or q[t,j] > upper[j]
```

输出：

```text
joint_limit_violation_rate
joint_limit_violation_count
joint_limit_violation_element_rate
max_limit_excess_rad
worst_limit_joint
worst_limit_frame
```

其中：

- frame rate：存在任意关节越界的帧比例；
- element rate：所有“帧×关节”元素中越界的比例。

两者含义不同，应避免只写一个模糊的 violation rate。

### 9.3 接近限位

同时实现两种定义。

绝对余量：

```text
margin_rad[t,j] = min(q[t,j] - lower[j], upper[j] - q[t,j])
near_limit_0p05rad = margin_rad < 0.05 rad
```

归一化余量：

```text
range[j] = upper[j] - lower[j]
margin_norm[t,j] = min(
    (q[t,j] - lower[j]) / range[j],
    (upper[j] - q[t,j]) / range[j]
)
near_limit_2pct = margin_norm < 0.02
```

输出：

```text
near_limit_0p05rad_frame_rate
near_limit_0p05rad_element_rate
near_limit_2pct_frame_rate
near_limit_2pct_element_rate
min_limit_margin_rad
min_limit_margin_norm
```

### 9.4 速度阈值

第一版使用可配置的统一阈值：

```text
velocity_limit_rad_s = 3π
```

它是 GMR 中使用的统一参考阈值，不应描述为所有 G1 电机的真实硬件极限。后续如获得每关节硬件速度限制，应支持用长度为 29 的数组替代统一阈值。

输出：

```text
velocity_violation_frame_rate
velocity_violation_element_rate
max_velocity_excess_rad_s
```

## 10. 指标三：MuJoCo 几何、碰撞与地面接触

### 10.1 正确的逐帧处理方式

```python
model = mujoco.MjModel.from_xml_path(str(robot_xml))
data = mujoco.MjData(model)

for t in range(T):
    data.qpos[:3] = root_pos[t]
    data.qpos[3:7] = root_quat_xyzw[t][[3, 0, 1, 2]]
    data.qpos[7:] = joint_pos[t]
    mujoco.mj_forward(model, data)
```

使用 `mj_forward()`，不要使用 `mj_step()`。本阶段的目标是检查参考姿态本身。`mj_step()` 会在重力、接触和动力学作用下推进系统，使评价对象发生变化。

### 10.2 自碰撞

遍历：

```python
for contact in data.contact[:data.ncon]:
    geom1 = int(contact.geom1)
    geom2 = int(contact.geom2)
```

如果两个 geom 都不是地面，且属于不同机器人 body，则记为候选自碰撞。MuJoCo 模型自身的碰撞过滤规则应继续生效。

至少输出两种口径：

```text
self_collision_any_frame_rate
self_collision_nonhand_frame_rate
self_collision_event_count
max_self_contacts_per_frame
first_self_collision_frame
self_collision_body_pairs
```

其中 `nonhand` 口径应排除被明确允许的手部接触，便于复现 Make Tracking Easy 的非手部自碰撞统计。允许列表必须写入配置，不能散落在代码中。

对于连续多帧发生的同一身体对碰撞，应聚合为一个事件区间，记录：

```text
motion_name,start_frame,end_frame,start_time_s,end_time_s,event_type,entities,max_severity
```

### 10.3 地面穿透

优先使用 XML 中名为 `floor` 的 geom。若实际模型名称不同，应通过配置指定。

对于机器人—地面 contact：

```text
penetration_depth = max(0, -contact.dist)
```

PHUMA 的参考标准为接触期间脚底位于地面上下 `1 cm` 范围内。因此第一版使用：

```text
penetration_threshold_m = 0.01
```

输出：

```text
ground_penetration_frame_rate
ground_penetration_event_count
mean_ground_penetration_m
max_ground_penetration_m
worst_penetration_body
worst_penetration_frame
```

地面 contact 和脚底几何高度可互相校验。若 contact 信息与脚底最低点矛盾，应在日志中提示模型碰撞设置或脚底选择可能有问题。

### 10.4 非足部触地

将与地面接触的 body 分为：

- 左脚允许 body；
- 右脚允许 body；
- 其他 body。

输出：

```text
undesired_ground_contact_frame_rate
undesired_ground_contact_event_count
undesired_ground_contact_bodies
```

该指标必须保留动作类别语义。翻滚、爬行、侧手翻等动作可能合理地使用手、膝或躯干触地，不能对所有 LAFAN1 动作统一判定为失败。第一版可以在普通站立、走、跑和舞蹈动作上启用硬惩罚，对 floor motion 只做描述。

### 10.5 脚底几何点

G1 MuJoCo 模型通常在左右 `ankle_roll_link` 下放置若干小球形 geom 作为脚底接触点。应从模型中自动查找，而不是硬编码 geom 数字 ID。

示意：

```python
def find_sole_geoms(model, side):
    body_id = model.body(f"{side}_ankle_roll_link").id
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] == body_id
        and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
        and model.geom_size[geom_id, 0] <= 0.01
    ]
```

球形 geom 的真实最低点是：

```text
sole_surface_z = data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0]
```

每只脚建议保存：

- 四个脚底点的世界坐标；
- 脚底最低高度；
- 脚底点质心；
- 质心水平速度；
- 质心竖直速度。

启动时应断言左右脚都找到了合理数量的 sole geoms。如果未找到，应失败并给出清楚的配置提示，不能退化成使用踝关节中心而不告知用户。

### 10.6 接触代理

只有机器人参考轨迹、没有人体接触标签时，无法严格判断某一帧是否“应该接触”。第一版可使用启发式接触代理：

```text
near_ground = abs(foot_height) <= 0.01 m
low_vertical_speed = abs(foot_vertical_speed) <= 0.2 m/s
contact_proxy = near_ground and low_vertical_speed
```

其中：

- `0.01 m` 来自 PHUMA 的接地误差标准；
- `0.2 m/s` 是项目初始启发式参数，必须放在配置文件中，并在报告中标注为自定义参数。

如果 MuJoCo 明确检测到脚—地面 contact，可以同时输出 `mujoco_contact` 与 `contact_proxy`，并记录二者不一致的帧。

### 10.7 脚滑

接触期间的脚底水平速度：

```text
foot_horizontal_speed[t] = norm(
    foot_center_xy[t] - foot_center_xy[t-1]
) / dt
```

PHUMA 使用 `0.1 m/s` 作为 Non-Skating 的参考标准：

```text
foot_slide[t] = contact_proxy[t] and foot_horizontal_speed[t] > 0.1 m/s
```

输出：

```text
left_contact_proxy_rate
right_contact_proxy_rate
foot_slide_frame_rate
left_foot_slide_frame_rate
right_foot_slide_frame_rate
foot_slide_speed_p95_m_s
foot_slide_speed_max_m_s
foot_slide_event_count
```

### 10.8 腾空比例

可以输出：

```text
airborne_frame_rate
max_airborne_duration_s
```

但不能默认将其计入负向质量分。跳跃和跑步存在合理腾空期。若未来获得源人体接触标签，应改为评价接触时序一致性，而不是惩罚所有腾空。

## 11. 指标四：人体—机器人动作相似性（可选扩展）

仅靠当前 GMR `.pkl` 通常不能计算动作相似性，因为其中没有保存对应人体关键身体目标。可以采用以下两种方案。

### 11.1 方案 A：记录 GMR IK 残差

在重定向每帧调用 `retarget()` 后记录：

```python
ik_error_stage1 = retargeter.error1()
ik_error_stage2 = retargeter.error2()
```

输出：

```text
ik_error_stage1_mean
ik_error_stage1_p95
ik_error_stage2_mean
ik_error_stage2_p95
ik_error_stage2_max
```

该误差可能混合位置、旋转和不同任务权重，不能解释成米或度，也不应与其他方法的未归一化误差直接比较。它只适合用于同一配置下发现异常帧或比较参数变化。

### 11.2 方案 B：比较映射关键身体

重新读取对应 BVH，使用与 GMR 相同的尺度和 offset，并读取 `bvh_lafan1_to_g1.json` 的人体—机器人身体映射。对每帧比较：

- 左右脚；
- 左右手；
- 骨盆；
- 躯干；
- 其他实际参与 IK 的关键身体。

位置误差：

```text
position_error = ||p_human_target - p_robot||_2
```

旋转误差使用 SO(3) 测地角：

```text
orientation_error = acos(
    clip((trace(R_human^T R_robot) - 1) / 2, -1, 1)
)
```

PHUMA 使用位置误差小于 `0.10 m`、姿态误差小于 `10°` 作为 Motion Fidelity 的参考判据。

输出：

```text
keybody_pos_error_mean_m
keybody_pos_error_p95_m
keybody_pos_error_max_m
end_effector_pos_error_mean_m
end_effector_pos_error_p95_m
link_orientation_error_mean_deg
link_orientation_error_p95_deg
fidelity_pass_frame_rate
worst_fidelity_body
worst_fidelity_frame
```

必须确保人体目标和机器人身体处于同一坐标系，并使用与 GMR 重定向完全相同的缩放和 offset。否则误差没有意义。

## 12. 暂定评分方法

### 12.1 原则

当前阶段必须优先保留原始指标和异常原因。单一总分只能作为汇报、排序和后续实验的暂定工具，不得替代各维度指标。

原因包括：

- 动态动作天然比慢走具有更高速度和 jerk；
- 地面接触模式取决于动作类别；
- 不同指标单位和数值分布不同；
- 最终权重应通过与策略成功率和训练效率的相关性校准。

### 12.2 帧级合规率

在有明确阈值的项目上定义：

```text
smooth_valid[t] = not joint_jump[t]
                  and not velocity_violation[t]

joint_valid[t] = not position_limit_violation[t]
                 and not near_limit_2pct[t]

contact_valid[t] = not self_collision_nonhand[t]
                   and not ground_penetration[t]
                   and not foot_slide[t]

fidelity_valid[t] = keybody_position_error[t] < 0.10 m
                    and link_orientation_error[t] < 10 deg
```

各维度分数：

```text
smoothness_score = 100 * mean(smooth_valid)
joint_feasibility_score = 100 * mean(joint_valid)
contact_score = 100 * mean(contact_valid)
fidelity_score = 100 * mean(fidelity_valid)
```

jerk、加速度和最大速度等暂无可靠统一阈值的指标先作为描述量，不强行加入 v0 分数。后续可按动作类别建立经验分布，再使用中位数和 P95 做稳健归一化。

### 12.3 总分

如果 fidelity 可用：

```text
quality_score_v0 =
    0.30 * smoothness_score
  + 0.25 * joint_feasibility_score
  + 0.30 * contact_score
  + 0.15 * fidelity_score
```

如果 fidelity 不可用，应将前三项权重重新归一化，并将 `fidelity_score` 输出为 `NaN`。不允许把缺失维度自动当作满分。

CSV 中应额外包含：

```text
score_version = v0
score_available_dimensions
score_is_calibrated = false
```

### 12.4 硬失败

以下条件建议默认标记 `hard_fail`，所有阈值均应配置化：

- 数据包含 NaN 或 Inf；
- 数据形状或 DoF 数不匹配；
- 四元数严重无效且无法可靠恢复；
- 存在明确的关节硬限位违反；
- 最大地面穿透超过 `0.05 m`；
- 非手部自碰撞连续超过指定帧数或持续时间；
- 其他被配置为严重错误的条件。

输出：

```text
hard_fail
failure_reasons
warning_reasons
```

`failure_reasons` 建议使用稳定的代码并附带可读文本，例如：

```text
DATA_NAN
JOINT_LIMIT_EXCEEDED:left_knee_joint@frame_381
SEVERE_PENETRATION:right_ankle_roll_link@frame_202
SUSTAINED_SELF_COLLISION:left_knee_link|right_knee_link@frames_510-529
```

## 13. 输出文件设计

建议每次评价生成独立输出目录：

```text
outputs/<run_name>/
├── quality_summary.csv
├── quality_events.csv
├── frame_metrics/
│   ├── walk1_subject1.csv
│   └── ...
├── plots/
│   ├── walk1_subject1_joint_velocity.png
│   ├── walk1_subject1_foot_contact.png
│   └── ...
├── run_manifest.json
└── resolved_config.yaml
```

### 13.1 `quality_summary.csv`

每条动作一行，至少包含：

```text
motion_name
source_bvh
pkl_path
robot_name
fps
num_frames
duration_s

nan_count
inf_count
quat_norm_max_error
data_valid

joint_jump_rate
joint_jump_count
joint_step_max_rad
joint_vel_p95_rad_s
joint_vel_max_rad_s
joint_acc_p95_rad_s2
joint_jerk_p95_rad_s3
joint_jerk_max_rad_s3
root_speed_p95_m_s
root_jerk_p95_m_s3
root_ang_vel_p95_rad_s

joint_limit_violation_frame_rate
joint_limit_violation_element_rate
near_limit_0p05rad_frame_rate
near_limit_2pct_frame_rate
min_limit_margin_rad
min_limit_margin_norm
velocity_violation_frame_rate

self_collision_any_frame_rate
self_collision_nonhand_frame_rate
self_collision_event_count
ground_penetration_frame_rate
max_ground_penetration_m
undesired_ground_contact_frame_rate
left_contact_proxy_rate
right_contact_proxy_rate
foot_slide_frame_rate
foot_slide_speed_p95_m_s
airborne_frame_rate

ik_error_stage2_mean
keybody_pos_error_mean_m
end_effector_pos_error_mean_m
link_orientation_error_mean_deg
fidelity_pass_frame_rate

smoothness_score
joint_feasibility_score
contact_score
fidelity_score
quality_score_v0
score_version
score_available_dimensions
score_is_calibrated
hard_fail
failure_reasons
warning_reasons
```

所有列名使用 `snake_case`；带单位的数值必须在列名中写明单位，例如 `_m`、`_rad`、`_m_s`、`_rad_s2`。比例统一使用 `[0,1]`，分数统一使用 `[0,100]`，禁止有的列写百分数、有的列写小数而不注明。

### 13.2 `frame_metrics/<motion>.csv`

每帧一行，至少包含：

```text
motion_name
frame
time_s
max_joint_step_rad
max_joint_step_joint
max_joint_vel_rad_s
max_joint_vel_joint
max_joint_acc_rad_s2
max_joint_jerk_rad_s3
min_limit_margin_rad
min_limit_margin_norm
nearest_limit_joint
joint_jump_flag
position_limit_violation_flag
near_limit_0p05rad_flag
near_limit_2pct_flag
velocity_violation_flag
self_collision_any_flag
self_collision_nonhand_flag
self_collision_pairs
penetration_flag
penetration_depth_m
penetration_bodies
undesired_ground_contact_flag
undesired_ground_contact_bodies
left_foot_height_m
right_foot_height_m
left_contact_proxy_flag
right_contact_proxy_flag
left_slide_speed_m_s
right_slide_speed_m_s
left_foot_slide_flag
right_foot_slide_flag
airborne_flag
ik_error_stage2
keybody_pos_error_mean_m
link_orientation_error_mean_deg
```

### 13.3 `quality_events.csv`

将连续异常帧合并为事件：

```text
motion_name
event_type
start_frame
end_frame
start_time_s
end_time_s
duration_s
entities
max_severity
details
```

### 13.4 `run_manifest.json`

至少记录：

- 评价工具版本或 Git commit；
- GMR 仓库路径和 Git commit；
- Python 版本；
- MuJoCo 版本；
- 机器人 XML 的绝对路径和文件哈希；
- 阈值配置；
- 输入文件列表；
- 执行命令；
- 是否启用 fidelity；
- 运行成功、失败和跳过的文件数。

## 14. 配置文件建议

所有阈值和模型特定名称都放入配置文件，例如：

```yaml
robot:
  name: unitree_g1
  xml_path: /actual/path/to/g1_mocap_29dof.xml
  floor_geom_name: floor
  left_foot_body: left_ankle_roll_link
  right_foot_body: right_ankle_roll_link
  allowed_hand_contact_patterns:
    - left_wrist
    - right_wrist
    - left_rubber_hand
    - right_rubber_hand

thresholds:
  joint_jump_step_rad: 0.5
  velocity_limit_rad_s: 9.42477796076938
  near_limit_margin_rad: 0.05
  near_limit_fraction: 0.02
  penetration_m: 0.01
  severe_penetration_m: 0.05
  foot_near_ground_m: 0.01
  foot_vertical_speed_contact_m_s: 0.2
  foot_slide_speed_m_s: 0.1
  fidelity_position_m: 0.10
  fidelity_orientation_deg: 10.0
  sustained_collision_frames: 5

scoring:
  version: v0
  weights:
    smoothness: 0.30
    joint_feasibility: 0.25
    contact: 0.30
    fidelity: 0.15
```

解析后把最终使用的完整配置保存为 `resolved_config.yaml`。

## 15. 推荐代码项目结构

```text
gmr_quality_eval/
├── README.md
├── pyproject.toml
├── configs/
│   └── unitree_g1.yaml
├── src/
│   └── gmr_quality/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── types.py
│       ├── io.py
│       ├── model_info.py
│       ├── derivatives.py
│       ├── kinematic_metrics.py
│       ├── mujoco_metrics.py
│       ├── contact.py
│       ├── fidelity.py
│       ├── scoring.py
│       ├── events.py
│       ├── reporting.py
│       └── plotting.py
├── tests/
│   ├── test_io.py
│   ├── test_derivatives.py
│   ├── test_joint_limits.py
│   ├── test_quaternion_order.py
│   ├── test_mujoco_metrics.py
│   ├── test_synthetic_corruptions.py
│   └── test_csv_schema.py
└── scripts/
    └── create_corrupted_samples.py
```

如果现有环境不适合安装项目，可先采用简单脚本结构，但指标函数、I/O、MuJoCo 处理和评分仍应分离，避免把所有逻辑塞进一个文件。

## 16. 命令行接口要求

建议至少支持：

```bash
# 单文件
python -m gmr_quality evaluate \
  --input /path/to/motion.pkl \
  --config configs/unitree_g1.yaml \
  --output-dir outputs/single_motion

# 文件夹批量处理
python -m gmr_quality evaluate \
  --input-dir /path/to/gmr_pkls \
  --pattern "*.pkl" \
  --config configs/unitree_g1.yaml \
  --output-dir outputs/lafan1_gmr

# 可选人体源动作
python -m gmr_quality evaluate \
  --input /path/to/motion.pkl \
  --source-bvh /path/to/motion.bvh \
  --config configs/unitree_g1.yaml \
  --output-dir outputs/with_fidelity

# 运行合成异常验收
python -m gmr_quality validate \
  --input /path/to/known_good_motion.pkl \
  --config configs/unitree_g1.yaml \
  --output-dir outputs/validation
```

命令失败时应返回非零退出码，并给出具体问题，例如模型路径不存在、DoF 数不匹配、找不到脚底 geom，而不是只输出 Python 堆栈。

## 17. 正确性验证与测试

### 17.1 数学单元测试

至少覆盖：

- 常量关节角：速度、加速度、jerk 均为 0；
- 线性关节角：速度恒定，加速度和 jerk 为 0；
- 二次轨迹：加速度恒定，jerk 为 0；
- 四元数 `q` 与 `-q` 表示相同姿态，不应产生异常角速度；
- 已知相对旋转应得到正确角速度；
- 关节恰好在上下限内、等于限位、超出限位时标志正确；
- `xyzw -> wxyz` 转换后，MuJoCo 中的姿态与 SciPy 解释一致；
- 差分值的帧号对齐正确。

### 17.2 MuJoCo 冒烟测试

对一条已知 GMR 动作：

- 能加载模型；
- 能设置所有帧 qpos；
- `mj_forward()` 不报错；
- 左右脚底 geom 均能找到；
- 输出脚底高度为合理米制数值；
- 能区分地面 contact 和机器人内部 contact。

### 17.3 人工异常注入

从一条可正常播放的 GMR 轨迹复制出若干受控异常版本。不要覆盖原始数据。

#### 关节跳变

在中间帧对一个远离限位的关节增加 `0.6 rad`：

```text
预期：joint_jump_count 增加，异常帧和关节名正确。
```

#### 根节点下移

将根节点 z 坐标整体减小 `0.05 m`：

```text
预期：ground_penetration_frame_rate 和 max_ground_penetration_m 增加。
```

#### 关节贴近限位

把一个关节连续若干帧设置为 `upper_limit - 0.01 rad`：

```text
预期：near_limit_0p05rad 和 near_limit_2pct 正确触发。
```

#### 脚底水平漂移

在一段已检测为接触的区间给根节点或脚部参考加入足够的水平漂移：

```text
预期：foot_slide_frame_rate 和对应滑动事件增加。
```

#### 非法数据

复制轨迹并插入 NaN 或零模长四元数：

```text
预期：data_valid=false，hard_fail=true，并包含明确 failure reason。
```

### 17.4 CSV 验证

- 所有要求的列存在；
- 每个输入动作恰好对应一行 summary；
- frame CSV 行数等于原始帧数；
- `time_s == frame / fps`；
- 比例均处于 `[0,1]` 或为 `NaN`；
- 分数均处于 `[0,100]` 或为 `NaN`；
- 同一输入和配置重复运行结果一致；
- 写 CSV 前保留全精度，只在展示或绘图时控制小数位。

## 18. 结果解释注意事项

### 18.1 不同动作类别不能简单横向比较

走路、跳跃、舞蹈、摔倒起身的正常速度、jerk 和接触模式不同。summary 中应预留：

```text
motion_category
contact_policy
notes
```

当动作类别未知时，不应因为双脚腾空、手触地或较大 jerk 直接判定任务无效。

### 18.2 不要通过滤波掩盖原始质量

可以额外输出平滑后的可视化或分析结果，但主质量指标必须针对原始 GMR 输出。若未来比较“原始轨迹”和“后处理轨迹”，两套结果应分别记录。

### 18.3 速度、加速度和 jerk 必须使用真实 fps

同样的相邻帧变化，在 30 Hz 和 120 Hz 下代表完全不同的物理速度。所有时间导数均必须通过输入轨迹的 `fps` 计算，不能写死 30。

### 18.4 机器人模型决定评价结果

关节限制、碰撞几何、脚底位置和地面设置全部来自机器人 XML。更换 XML 后结果可能改变，因此 manifest 必须保存 XML 路径和哈希。

### 18.5 自碰撞和非足部触地需要允许列表

相邻连杆、手部接触、翻滚和地面动作可能产生合理 contact。评价器应输出原始 contact，再通过配置决定哪些计入质量惩罚，避免在代码中使用不可见的硬编码规则。

### 18.6 当前总分未校准

`quality_score_v0` 仅用于初步排序和展示。后续应通过以下实验校准：

- 离线指标与策略成功率的相关性；
- 离线指标与 MPJPE、W-MPJPE、跌倒率的相关性；
- Top-K、Random-K、Bottom-K 数据在相同数量和训练预算下的策略表现；
- 不同动作类别内的分位数归一化；
- 多个随机种子下的稳定性。

## 19. 验收标准

项目达到可用状态时，应满足：

1. 能对单个 GMR `.pkl` 运行并输出完整 summary 和 frame CSV；
2. 能批量处理目录，不因单个坏文件导致整个批次停止；
3. 正确处理 `xyzw` 与 `wxyz`；
4. 关节限制顺序与 `dof_pos` 顺序经过自动验证；
5. 能发现并定位 joint jump；
6. 能发现硬限位和接近限位；
7. 能通过 MuJoCo 发现自碰撞、地面穿透和非足部触地；
8. 能计算脚底高度、接触代理和脚滑；
9. 人工异常测试均按预期触发；
10. summary、frame metrics 和 events 能互相对应；
11. 所有阈值均来自配置，并保存 resolved config；
12. README 能让另一位使用者在同一环境复现实验；
13. 测试套件通过；
14. 明确区分离线运动学质量、动力学可执行性和下游任务有效性。

## 20. 执行代理的推荐工作方式

执行本文任务的 Codex 应：

1. 先检查现有 GMR 环境和真实数据格式；
2. 根据检查结果制定分步骤计划；
3. 先实现 loader、数据验证和纯数组指标；
4. 再实现模型信息提取和 MuJoCo 指标；
5. 再实现事件聚合、CSV 和评分；
6. 最后实现可选 fidelity 和绘图；
7. 每完成一类指标都用人工可预测的输入验证；
8. 不修改或删除原始 BVH、GMR `.pkl` 和机器人模型；
9. 不把生成的输出写入原始数据目录；
10. 保留清晰的运行说明、设计说明和局限性说明。

若服务器环境中的 GMR 输出格式、模型结构或字段命名与本文不同，应以实际环境为准，并在 README 和 manifest 中记录差异，不应为了符合本文而错误解释数据。

## 21. 主要参考资料

- GMR 代码仓库：<https://github.com/YanjieZe/GMR>
- GMR 论文《Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking》：<https://www.jiajunwu.com/papers/gmr_icra.pdf>
- GMR LAFAN1 保存脚本：<https://raw.githubusercontent.com/YanjieZe/GMR/master/scripts/bvh_to_robot.py>
- GMR 运动重定向实现：<https://raw.githubusercontent.com/YanjieZe/GMR/master/general_motion_retargeting/motion_retarget.py>
- GMR 运动学模型：<https://raw.githubusercontent.com/YanjieZe/GMR/master/general_motion_retargeting/kinematics_model.py>
- GMR Unitree G1 MuJoCo 模型：<https://raw.githubusercontent.com/YanjieZe/GMR/master/assets/unitree_g1/g1_mocap_29dof.xml>
- Make Tracking Easy：<https://arxiv.org/html/2603.22201>
- PHUMA：<https://arxiv.org/html/2510.26236>
- KDMR：<https://arxiv.org/html/2603.09956>

