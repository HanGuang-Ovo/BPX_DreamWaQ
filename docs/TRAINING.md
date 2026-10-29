# 训练设计与当前默认配置

按 2026-10-29 工作区核对。环境默认值来自 `src/bpx_dreamwaq/tasks/rough_env_cfg.py`，
算法默认值来自 `src/bpx_dreamwaq/tasks/agents/rsl_rl_ppo_cfg.py`。
命令行覆盖优先；运行快照保存在 `params/env.yaml` 与 `params/agent.yaml`。
完整操作顺序见 [README](../README.md)。

## 网络与优化数据流

```text
原始历史 450 → 网络内固定缩放 → CENet 共享编码器 [256,128]
                                  ├→ 速度头 [128] → vel_hat (3)
                                  └→ VAE 头 [128] → μ、logvar (16)
最新本体帧 45 + 速度 3 + μ 16 → Actor [512,256,128] → 12 维动作
采样隐变量 z (16) → Decoder [128,128] → 下一帧本体观测预测 (45)
特权观测 235 → Critic [512,256,128] → 价值
```

默认 `cenet_decoder_use_velocity=False`：Decoder 仅输入 z；设为 True 才使用
`[vel_hat,z]` 的 19 维输入。导出/验收使用确定性 μ 和动作均值，完全不需要 Decoder 或 Critic。
Actor 单帧输入与下一帧预测目标由历史末帧提取，与历史共享同一次噪声采样。

PPO rollout 保存实际用于动作采样的 CENet 特征及 AdaBoot 速度；默认更新时复用，
避免概率比因特征变化而失配。`detach_cenet_from_actor=True`、
`ppo_grad_to_cenet=False`，CENet 默认只由辅助损失训练。
开启 `--ppo_grad_to_cenet` 时更新阶段重算特征并保留原 AdaBoot 分支，要求确定性隐变量均值。
Actor/Critic 和 CENet 各用参数互不重叠的 Adam，依次反传、分别裁剪梯度。

辅助损失为速度监督 MSE、下一帧预测 MSE 与 KL 的加权和。
终止步的预测/KL 用 done 掩码屏蔽，避免把 reset 后的下一回合观测当作预测目标。
数值异常片段还会从 PPO、CENet、归一化和课程统计中排除，见 [数值隔离](NUMERICAL_QUARANTINE.md)。

| 配置 | 当前默认 |
| --- | --- |
| 历史 / latent / 速度维数 | 10 / 16 / 3 |
| Actor/Critic 隐藏层、激活 | [512,256,128]、ELU |
| PPO 学习率 / 调度 / KL 目标 | 1e-3 / adaptive / 0.01 |
| γ / GAE λ / clip / entropy | 0.99 / 0.95 / 0.2 / 0.01 |
| rollout 步数 / epoch / minibatch | 24 / 5 / 4 |
| CENet 学习率 / 梯度裁剪 | 1e-3 / 1.0，不受 PPO KL 调度影响 |
| λ_vel / λ_pred / β_KL | 1.0 / 1.0 / 1e-3 |
| 最大迭代 / checkpoint 间隔 | 1500 / 50 |
| 固定缩放 / 经验归一化 | True / False，二者互斥 |
| AdaBoot 开关 / 模式 | True / error |

固定缩放作用于网络内部，速度监督与 CENet 速度输出均为 `2 × m/s`；日志速度误差
转回物理单位。配置、buffer 随 checkpoint 保存并进入导出模型，见 [固定缩放](FIXED_SCALING.md)。
启用经验归一化需先关闭固定缩放；统计在一轮优化结束后更新，rollout 和 PPO 更新期间冻结。

## AdaBoot、恢复与微调

AdaBoot 的 p 是 Actor 使用估计速度的概率；其他样本使用真实速度。当前从头训练默认
开启 `error` 模式：根据速度 MAE 的 EMA 调整 p，阈值 0.5 m/s、EMA 系数 0.9、p 下限 0。
另支持 `reward_cv`（p = 1 − tanh(CV(R))）与 `linear`（默认 500 轮渐变）。
`--disable_adaboot` 关闭后始终使用估计速度。验收和部署始终使用估计速度，不受此开关影响。

`--resume` 恢复优化器和迭代；`--finetune` 只加载网络及缩放/归一化状态，使用新优化器，
默认 PPO 学习率 1e-4、关闭 AdaBoot。二者都要求当前网络结构与缩放模式匹配源 checkpoint。
Decoder 输入可以是 16 或 19，不能仅按“新/旧模型”推断；以源 checkpoint 为准。
旧联合优化器 checkpoint 不能直接恢复成当前独立优化器训练。
`--max_iterations` 指本次调用追加的迭代数，不是恢复后停止的绝对轮次。

## 环境与随机化

**场景**（`BPXSceneCfg`）

| 组件 | 配置 |
| --- | --- |
| 地形 | 生成器网格：`8 m × 8 m` 地块，`num_rows=num_terrain_levels`（默认 10），`num_cols=10`，边界 20 m |
| 机器人 | BPX USD（缺失时首次训练由 URDF 自动转换） |
| 高度扫描 | `GridPatternCfg(resolution=0.1, size=[1.6, 1.0])`，机体上方 20 m 向下，yaw 对齐 |
| 接触传感器 | `.*` 刚体，`history_length=3`，记录滞空时间 |
| 光照 | DomeLight 强度 750 |
| 物理 | `dt=0.005`（200 Hz），`decimation=4`（50 Hz），回合 20 s |
| 并行环境 | 配置默认 1024；示例命令用 `--num_envs 512` 覆盖，按显存调整 |

**地形类型**（按比例随机组合）

| 子地形 | 比例 | 参数 | 作用 |
| --- | --- | --- | --- |
| `random_rough` | 20% | 噪声 0.01–0.06 m，步长 0.01 | 连续崎岖路面 |
| `boxes` | 20% | 网格 0.45 m，高度 0.03–0.12 m，平台 2.0 m | 离散方块 |
| `pyramid_stairs` | 20% | 级高 0.03–0.15 m，踏面 0.15 m，平台 3.0 m | 上台阶 |
| `pyramid_stairs_inv` | 20% | 同上 | 下台阶 |
| `hf_pyramid_slope` | 10% | 坡度 0–0.4 | 上坡 |
| `hf_pyramid_slope_inv` | 10% | 坡度 0–0.4 | 下坡 |

`horizontal_scale=0.1`、`vertical_scale=0.005`、`slope_threshold=0.75`；
`use_cache=False`，每次启动重新生成。

**指令**（`CommandsCfg` / `GridAdaptiveVelocityCommandCfg`）

| 项目 | 值 |
| --- | --- |
| 任务量程 | vx ∈ [−1.0, 1.0] m/s，vy ∈ [−0.5, 0.5] m/s，wz ∈ [−1.0, 1.0] rad/s |
| 重采样 | 每 10 s（episode 内），站立指令保持到回合结束 |
| 站立比例 | 每次抽样 20% 精确全零指令 |
| 航向 | `heading_command=False`，直接采样角速度 |
| 课程关闭时 | `TerrainAwareUniformVelocityCommand` 均匀采样（仍记录地形统计） |

**域随机化**（`EventCfg`）

| 事件 | 模式 | 参数 |
| --- | --- | --- |
| 物理材质 | startup | 静摩擦 [0.5, 1.4]、动摩擦 [0.4, 1.0]；恢复系数 0、64 桶随机，与 BPX_test 相同 |
| 初始位姿 | reset | x/y ∈ ±0.1 m，yaw ∈ ±π，初速度为零，与 BPX_test 粗糙地形任务相同 |
| 关节复位 | reset | 默认关节角加 ±0.2 rad，关节速度 ±0.1 rad/s |
| PD 增益 | startup | 刚度、阻尼各乘 [0.9, 1.1]，均匀分布 |
| 本体测量噪声 | 每次观测 | 均匀噪声：角速度 ±0.2 rad/s、重力 ±0.05、关节角 ±0.01 rad、关节速度 ±1.5 rad/s |
| 外力推挤 | interval 10–15 s | x/y 速度 ±0.5 m/s |

**终止条件**（`TerminationsCfg`）

| 终止 | 条件 |
| --- | --- |
| `time_out` | 20 s 回合超时（正常结束） |
| `base_contact` | torso 接触力 > 1 N |
| `base_tilt` | 机身倾角 > 0.5 rad |
| `numerical_guard` | 非有限状态/力矩或超出诊断速度阈值，隔离异常环境与样本 |


训练从默认站姿复位，站姿为 roll=0、pitch=0.7、knee=−1.4 rad，机身初始高度 0.42 m。
不启用质量、质心、电机强度或延迟随机化。PLAY 关闭推扰、材质/增益随机化和观测噪声；
独立验收脚本还替换为平地、固定指令并关闭初始位姿/关节随机偏移。
训练与 PLAY 复用显式 PD，策略 50 Hz，物理/力矩更新 200 Hz。

## 奖励与门控

下表权重按当前 `RewardsCfg` 列出。RewardManager 将原始项乘权重和策略步长 0.02 s，
环境再将每步总奖励下限设为 −10；这不是逐项限幅。异常环境另按隔离机制处理。

| 奖励项 | 权重 |
| --- | ---: |
| `track_lin_vel_xy_exp` | 2 |
| `track_ang_vel_z_exp` | 1.5 |
| `track_lin_vel_xy_stand_exp` | 2.5 |
| `track_ang_vel_z_stand_exp` | 2 |
| `feet_slip_stand` | -1 |
| `feet_contact_stand` | -1 |
| `joint_pos_stand_l2` | -1.5 |
| `lin_vel_z_l2` | -2 |
| `ang_vel_xy_l2` | -0.05 |
| `base_height_l2` | -0.2 |
| `joint_power` | -2e-05 |
| `power_distribution` | -5e-08 |
| `dof_torques_l2` | -0.0001 |
| `dof_acc_l2` | -2.5e-07 |
| `action_rate_l2` | -0.01 |
| `action_smoothness_l2` | -0.01 |
| `feet_clearance` | -0.03 |
| `feet_lateral_distance` | -1 |
| `feet_longitudinal_distance` | -1 |
| `feet_air_time` | 0.75 |
| `feet_air_time_excess` | -1 |
| `feet_contact_balance` | -3 |
| `undesired_contacts` | -1 |
| `flat_orientation_l2` | -0.75 |
| `joint_pos_limits` | -0.2 |

站立相关项的门控不完全相同：

- `track_*_stand_exp`：三轴指令绝对值均严格小于 0.1 时启用，核宽 std=0.1。
  常规速度跟踪项核宽 std=0.4，持续生效。
- `feet_contact_stand`：同样用三轴 <0.1 门控，原始代价为未触地足数，从离地首步开始计费。
  因而门限内的非零慢走/转向也会受到四足接触约束。
- `feet_slip_stand`：仅精确全零指令，累加触地足的世界系水平速度模长。
- `joint_pos_stand_l2`：仅精确全零指令，惩罚默认站姿偏差平方和。
  当前权重 −1.5，12 个关节各偏离 0.1 rad 时，奖励率 −0.18、每步 −0.0036。
- `feet_air_time`：落脚时评价腾空时间相对 0.3 s 的差；平移速度模长 >0.1 m/s
  或 |wz| >0.1 rad/s 时启用。`feet_air_time_excess` 对超过 0.6 s 的悬空持续惩罚，每脚上限 1。
- `feet_contact_balance`：长期触地占空比偏差平方，EMA 时间常数 0.7 s，允许交替摆腿。

高度项以扫描地面均值为参考，目标 0.42 m，误差按 0.10 m 归一化；
净空项以 toe_link 原点相对局部地形 0.10 m 为目标，并乘足端水平速度。
横向足距范围 0.26–0.42 m，同侧前后足距至少 0.40 m，均在消除 yaw 后计算。
`joint_power` 为 `sum(abs(τ*q̇))`；`power_distribution` 为有符号功率总体方差的平方
`Var(τ*q̇)²`，原始代价上限 1e8。`applied_torque` 是限幅后的电机力矩指令，不包含接触约束力矩。
`dof_acc_l2` 原始代价也限为 1e8。二阶动作项使用 `a_t−2a_(t−1)+a_(t−2)`。
详细公式与无效射线处理见 `tasks/mdp/rewards.py`。

## 两套课程

| 开关 | 位置 | 默认 | 说明 |
| --- | --- | --- | --- |
| `enable_terrain_curriculum` | `BPXRoughEnvCfg` / `--disable_terrain_curriculum` | 开 | 整回合表现与空间覆盖地形课程 |
| `num_terrain_levels` | 同上 / `--num_terrain_levels` | 10 | 地形难度等级数 |
| `terrain_curriculum_mode` | 同上 | `standard` | `standard` 升降级 / `upgrade_only` 只升不降 |
| `terrain_curriculum_debug` | 同上 | 关 | 额外课程调试日志 |
| `enable_grid_adaptive_command_curriculum` | 同上 / `--enable_command_curriculum` / `--disable_command_curriculum` | 开 | Grid-Adaptive 指令课程 |

**模块 A：地形课程**（`tasks/mdp/curriculums.py`）。每个 episode 结束时按
环境独立判定，整回合数据由指令模块逐步累计，不在 10 s 指令重采样时清零：

- 升级：正常超时完成回合、无失败终止，累计平移指令路程 ≥ 1 m、移动时长
  ≥ 2 s，且满足实际空间覆盖条件。不检查速度跟踪达标比例或误差容限。
- 降级：未满足升级条件，本回合至少 50% 的时间有明确平移指令（平面指令速度
  > 0.1 m/s），并发生失败终止，或在满足 1 m / 2 s 评价门槛后至少 50% 的
  移动采样跟踪不良。明确移动任务中提前摔倒可直接降级；等级 0 不再下降。
- 保持：零速、原地旋转、评价数据不足，或正常低速/转弯且跟踪良好但覆盖不足。

降级统计中的跟踪良好定义为同一步平面速度误差 ≤ 0.1 m/s 且偏航误差
≤ 0.5 rad/s。误差容限仅用于降级和日志；满足升级条件时优先升级。
空间覆盖采用整回合的轴向最大距离：相对真实出生点、相对地形中心均须达到
地形边长 × 0.25（8 m 地形为 2 m），避免往返被净位移误判，也避免只在中央
方形平台绕小圈或沿对角线移动就升级。这是空间覆盖代理指标，并非精确障碍
接触判定。`upgrade_only` 保留上述升级条件并禁用降级。

终止步在机器人 reset 前补采，回合 reset 后重新记录出生点，各环境独立清零；
指令课程开关关闭时，均匀指令采样也使用同一套地形统计。升降级互斥，达到
最高级后随机分配等级。逐等级回合奖励/长度、实际升降级次数、保持次数写入
TensorBoard（`curriculum/terrain/*`、`Curriculum/terrain_levels/*`）；
开启 debug 可查看指令路程、移动时间、空间覆盖和跟踪达标占比。关闭课程时
等级随机初始化且不再变化。

**模块 B：Grid-Adaptive 指令课程**（`tasks/mdp/command_grid.py` + `commands.py`）。
在 (vx, wz) 联合网格（默认 0.5 m/s × 0.5 rad/s 分辨率，中心在零速）上维护
每个指令组合的采样概率与跟踪统计；初始激活 [−0.5, 0.5]²，最大扩展到任务
指令量程（[−1, 1] × [−1, 1]）。单元在「区间平均 |Δvx| < 0.25 m/s 且
|Δwz| < 0.5 rad/s」的 EMA 成功率达标且采样 ≥ `min_samples` 后激活 4 邻域。

非站立样本中，默认 70% 指令使用原分布（90% 激活单元均匀 + 10% 全合法单元
均匀探索），另外 30% 从一个避开零点的低速带采样；低速区尚未激活时退回原分布。
低速分布按格与低速区域的交叠面积加权，不绕过课程激活门控。vx 与 wz 先联合
选同一单元，再在格内连续采样；两端格截断到任务量程。常规分支 vy 均匀采样于
任务量程，低速分支同步收窄到 ±0.1 m/s。低速带默认取 |vx| ∈ [0.05, 0.25] m/s
（`command_low_speed_vx_range` 按无符号幅值解释，`command_low_speed_random_sign`
在格内正负两侧按区间长度加权），使低速分支避开零点；常规网格分支仍可采到 [0, 0.05) 内的非零速度。

每次重采样独立以 20% 概率发送精确全零站立指令（`command_rel_standing_envs`），
不参与课程成功统计，且在**回合结束前不再重采样**——重采样周期只对运动指令
生效。回合开始即抽到站立时，可连续站到回合结束（正常回合默认 20 s）；
中途抽到站立则保持到本回合结束，提前终止也会缩短站立时间。
因此在低速分支可用时，按一次抽样分支计算的比例约为 20% 精确零速、24% 低速带、56% 常规网格课程与探索；
比例指重采样次数，实际步数占比受回合提前结束影响。

统计按指令执行区间（episode 内 10 s 重采样周期）结算，不混入整个 episode。
关闭时恢复 `UniformVelocityCommand` 均匀采样；两套课程同时关闭后仍为 CENet + PPO，只取消课程调度。网格数据（概率/采样数/成功率）随 checkpoint 导出
`curriculum_<iter>.npz`，可用任意工具绘制热力图。

## 日志阅读与实现边界

| 指标 | 用途 |
| --- | --- |
| `policy/mean_reward`、`policy/episode_length` | 回合表现；20 s 回合为 1000 策略步 |
| `Episode_Reward/*`、`Episode_Termination/*` | 奖励组成、摔倒/过倾/超时原因 |
| `cenet/velocity_mse`、`velocity_error_x/y/z`、`velocity_mae` | 物理单位速度估计误差 |
| `cenet/prediction_loss`、`kl_loss`、`total_auxiliary_loss` | 辅助学习是否有限、是否失衡 |
| `cenet/adaboot_p_boot`、`adaboot_est_fraction` | 估计速度采样概率与实际比例 |
| `gait/<leg>/contact_fraction`、`air_time_max_s`、`long_air_fraction` | 四脚是否长期悬空/偏载 |
| `Metrics/base_velocity/stand_drift_vxy`、`stand_drift_wz` | 站立期间漂移 |
| `curriculum/terrain/*`、`curriculum/command/*` | 地形升降级和指令网格扩展 |
| `numerical/excluded_transition_fraction`、`Numerical/*` | 异常环境与排除样本比例 |

日志曲线只能辅助判断；最终需以纯 CENet 确定性验收与部署反馈评价策略。
本工程是 DreamWaQ 风格实现，历史长度取 10，Actor 默认使用 μ，辅助损失权重和奖励为工程配置。
Critic 不含外力特权输入；地形课程使用整回合空间覆盖规则，指令课程使用网格 EMA 门控，
不能把它们描述为原论文地形课程的逐项复现。关闭课程不会移除 CENet，也不会自动切换到纯 PPO 网络。
