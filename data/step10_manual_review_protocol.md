# Step-10 manual review protocol

本协议只用于评价离线质量指标是否与可视化异常大致一致，不用于宣称动力学可执行、控制器可跟踪或任务成功。

## 冻结抽样

- 每个层级按 v1 `quality_score` 选择最低分和最高分各 1 条，共 10 条；
- 在看到姿态可视化之前由程序固定候选，不允许看结果后换样本；
- 每条查看程序预先冻结的 6 个确定性帧：首尾，以及从均匀覆盖、最差 step、穿透和 fidelity 等异常候选中选出的帧；精确帧号保存在候选 CSV，不能复核后替换；
- 同时查看覆盖完整序列的异常时间线，以及关节速度、足地接触和 fidelity 三张正式诊断图；异常时间线只把正式 frame CSV 的布尔标志可视化，不引入新的判据。

## 记录字段

- `semantic_plausibility`：`plausible`、`uncertain` 或 `implausible`；只判断静态姿态序列是否与来源主题明显矛盾；
- `visible_artifact_severity`：`none`、`mild`、`moderate` 或 `severe`；
- `metric_agreement`：`agree`、`partial` 或 `disagree`；比较可见问题与指标/事件是否同向；
- `review_notes`：必须写出观察依据，不能只填结论；
- `reviewer`：记录实际执行复核的主体；
- `reviewed_at`：固定日期，避免把后续修改混入冻结结果。

## 解释限制

骨架 montage 是锁定 G1 XML 的 MuJoCo FK 静态快照，不是视频或动力学 rollout。异常时间线能定位持续区间，但不能显示接触力或控制稳定性；接触与 fidelity 图仍是主要的连续量证据。若静态复核与指标不一致，记录为 `partial/disagree`，不得为迎合总分而修改观察。
