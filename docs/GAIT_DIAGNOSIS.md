# 左后腿长期悬空的诊断（2026-10-21）

本文保留 2026-10-21 的模型、参数和验证结果，不代表当前模型表现。
当前奖励已继续调整：`feet_contact_balance` 权重 −3.0、时间常数 0.7 s，
并增加站立足端滑动与接触项；完整默认值见 [训练设计](TRAINING.md)。

## 复现结果

权重为用户从服务器训练后拷回的 `policies/policy.onnx` / `policies/policy.pt`，
文件在 12:52 更新；与本机留存的 checkpoint 参数不同，不能标为本机第 3100 轮。
文件 SHA-256 保存在 `outputs/analysis/gait_diagnosis/policy_identity.json`。
在平地使用固定 `vx=0.2 m/s`，统计仿真第 6～18 秒，得到：

| 足端 | MuJoCo 触地率 | Isaac Lab 触地率 |
| --- | ---: | ---: |
| FL 左前 | 96.8% | 100% |
| FR 右前 | 86.8% | 100% |
| HL 左后 | **1.8%** | **0%** |
| HR 右后 | 91.7% | 100% |

MuJoCo 左后脚最长连续悬空约 5 s，平均 toe 原点高度约 0.10 m；平均承载
约 0.26 N，另外三脚约 33～59 N。零速指令下四脚都能落地，说明问题集中在行走。
`vx=0.4 m/s` 时左后脚触地率也只有 6.7%。

Isaac Lab 对照使用同一导出策略和真实物理仿真，关闭外力、质量/材质随机化，
保留模型与部署的 PD 参数。直接读取 ActionManager 解析的关节名称，确认与
MuJoCo 的 12 维策略顺序逐项一致：roll / pitch / knee，每组 FL / FR / HL / HR。
默认角也一致。Isaac Lab 同样复现三足步态，说明当前策略在两种本机环境下都失衡。
服务器训练环境快照尚未获取，仍需核对它与本机的观测、关节顺序和控制契约。
两边接触判据不同，因此百分比用于判断明显失衡，不用于精确的跨引擎对拍。

原始报告保存在：

- `outputs/analysis/gait_diagnosis/mujoco_current_policy.json`
- `outputs/analysis/gait_diagnosis/isaac_current_policy.json`
- `outputs/analysis/gait_diagnosis/isaac_probe_gpu.log`

补做同一服务器策略的速度输入对照：两个 Isaac Lab 环境分别输入 CENet 估计速度
和真实速度，左后脚触地率为 **0% / 2%**。仅替换估计速度未消除长期抬腿。
对照前将当前 TorchScript 的完整参数加载到同结构 Python 模型，随机输入下
两者输出最大误差为零；没有将本机旧 checkpoint 当作服务器策略。
对照数据为 `outputs/analysis/gait_diagnosis/isaac_current_velocity_reference.json`。

## 训练目标缺口与修改

原奖励允许机器人靠三脚跟踪速度；没有约束每只脚都周期性参与支撑。
`feet_air_time` 只在首次触地时计算，始终悬空的脚不会持续受罚；
`feet_clearance` 又乘足端水平速度，静止抬腿时代价很小。
这些目标缺口与观察到的三足步态相符，单独提高关节 PD 不会修改策略目标。

在 `tasks/rough_env_cfg.py` 中加入两项：

| 项目 | 参数 | 作用 |
| --- | --- | --- |
| `feet_air_time_excess` | 权重 −1.0，允许 0.6 s，每脚上限 1 | 持续惩罚过长悬空；不受指令或足端速度关闭 |
| `feet_contact_balance` | 权重 −0.5，时间常数 1 s | 惩罚四脚长期触地占空比差异，允许正常的交替摆动 |

第二项使用指数滑动平均，各环境回合重置为中性占空比 0.5；不固定左右腿相位，
也不要求每个仿真步四足同时着地。新增参数是针对当前失衡的训练起点，仍需用
训练后步态、速度跟踪与楼梯表现共同评估，不能以奖励代码通过测试代替步态验收。

训练动作、三个观测组的关节位置/速度均改为显式的公共关节列表，保留顺序。
该列表与本次真实环境读到的旧顺序完全相同，既有权重的维度语义保持一致。

## 训练与验收

该次修复涉及环境、奖励、Runner 和共用关节契约。复现或开始新训练时应同步
同一版本的完整源码与配置，避免仅拷贝当时的三个文件遗漏后续依赖。
建议独立开始一轮训练，避免把旧权重已有的三足步态作为初始解：

```bash
bash scripts/training_python.sh scripts/train.py --headless \
    --num_envs 256 --seed 42 --max_iterations 3000 --run_name four_foot_v2
```

TensorBoard 新增逐脚指标：

- `gait/<leg>/contact_fraction`：四脚应在长时间统计上接近，HL 不应持续接近零。
- `gait/<leg>/air_time_max_s`：观察最长悬空时间是否持续增大。
- `gait/<leg>/long_air_fraction`：有多少环境的该脚悬空超过配置阈值。

训练进程在启动时加载奖励配置；已经运行的训练需要重新启动才使用这些修改。
旧 `policies/policy.onnx` 和 `policies/policy.pt` 仍是原策略，需要从新 checkpoint
重新导出后再验收。评估时至少分别测试平地零速、前进、侧移和转向，记录四脚
触地率、速度误差和最长悬空时间；再检查崎岖与楼梯，避免仅在平地接触均衡。

已通过 96 项训练离线测试，包括正常摆腿免罚、长期悬空持续计费、惩罚上限、
四足站立/交替对角步态/三足支撑的区分、策略频率变化和异步回合重置。
另使用 16 环境、3 轮 GPU 训练冒烟验证真实奖励接线、反向更新和逐脚 TensorBoard 日志。
短训用于验证工程接线，尚未验证重新训练后的步态是否收敛。
