"""BPX 四足机器人崎岖地形速度跟踪环境（Manager-Based）。

整体结构参照 Isaac Lab ``LocomotionVelocityRoughEnvCfg`` 模板，但观测契约
对齐本工程部署端（``bpx_dreamwaq.runtime``）：

- ``policy`` 组：45 维单帧观测 = [角速度 3, 投影重力 3, 速度指令 3,
  相对关节角 12, 关节速度 12, 上一步动作 12]，**不含真实机体线速度**。
- ``prop_history`` 组：同样 6 项观测，组级 ``history_length=10``、
  ``flatten_history_dim=True``，输出 450 维 term-major（项内从最旧到最新）
  历史观测，供 CENet 编码；与部署端 ``ObservationHistory`` 布局一致。
- ``critic`` 组（特权）：[真实机体线速度 3, 45 维单帧观测, 高度扫描 187]，
  共 235 维。真实速度位于前 3 维（``true_vel_slice=(0, 3)``）。

动作与控制契约与部署一致：12 维动作，``q_target = q_default + 0.5 × a``，
隐式 PD（stiffness 40，damping 1），200 Hz 物理步（dt=0.005）× 4 分频 =
50 Hz 策略频率。

课程开关（模块 A / 模块 B，可独立启停）：

- ``enable_terrain_curriculum``: 整回合表现与空间覆盖地形课程（默认开）。
- ``enable_grid_adaptive_command_curriculum``: Grid-Adaptive 指令课程
  （默认开；可通过配置或 ``--disable_command_curriculum`` 关闭）。

两套课程同时关闭时，本环境即为普通的崎岖地形 PPO 基线任务。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise

import isaaclab.envs.mdp as lab_mdp

from .. import assets as bpx_assets
from ..runtime.joints import POLICY_JOINT_NAMES
from .mdp import commands as bpx_commands
from .mdp import curriculums as bpx_curriculums
from .mdp import rewards as bpx_rewards
from .mdp import terminations as bpx_terminations
from .obs_contract import DEFAULT_HISTORY_LENGTH


##
# 机器人：URDF → USD 转换 + 关节驱动配置
##

_BPX_JOINT_PATTERNS = {
    ".*_hip_roll_joint": 0.0,
    ".*_hip_pitch_joint": 0.7,
    ".*_knee_joint": -1.4,
}


def _ensure_bpx_usd(force: bool | None = None) -> str:
    """确保存在可供 Isaac Lab 使用的 BPX USD；必要时从 URDF 转换。

    转换参数沿用工程内 ``assets/robots/bpx/usd/config.yaml``（该文件由
    UrdfConverter 生成）。默认缓存已有 USD，可通过环境变量
    ``BPX_FORCE_USD_CONVERSION=1`` 强制重转。
    """
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    usd_path = bpx_assets.USD_PATH
    if force is None:
        force = os.environ.get("BPX_FORCE_USD_CONVERSION", "0") == "1"
    if usd_path.exists() and not force:
        return str(usd_path)
    converter_cfg = UrdfConverterCfg(
        asset_path=str(bpx_assets.URDF_PATH),
        usd_dir=str(usd_path.parent),
        usd_file_name=usd_path.name,
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        collider_type="convex_hull",
        self_collision=True,
    )
    converter = UrdfConverter(converter_cfg)
    return converter.usd_path


BPX_USD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=_ensure_bpx_usd(), activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos=dict(_BPX_JOINT_PATTERNS),
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim=30.0,
            velocity_limit_sim=20.0,
            stiffness=40.0,
            damping=1.0,
        ),
    },
)


##
# 场景
##


@configclass
class BPXSceneCfg(InteractiveSceneCfg):
    """崎岖地形 + BPX 四足机器人场景。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,  # 在环境配置 __post_init__ 中按课程设置
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = BPX_USD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0)
    )


def _make_bpx_terrain_generator(num_levels: int):
    """DreamWaQ 风格的地形生成器（按 BPX 尺寸缩放）。

    包含崎岖（random_rough）、离散方块（boxes）、上下台阶（pyramid stairs）
    与坡面（slope，最大约 22°），行数为课程难度等级数。
    """
    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains import TerrainGeneratorCfg

    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=num_levels,
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2, grid_width=0.45, grid_height_range=(0.03, 0.12), platform_width=2.0
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.03, 0.15),
                step_width=0.10,  # 踏面深度 10 cm。
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.03, 0.15),
                step_width=0.10,  # 踏面深度 10 cm。
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
        },
    )


##
# MDP 配置
##


@configclass
class CommandsCfg:
    """速度指令：Grid-Adaptive 课程关闭时使用均匀采样。"""

    base_velocity = lab_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=lab_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """关节位置目标动作：q = q_default + 0.5 × a（与部署一致）。"""

    joint_pos = lab_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True,
        scale=0.5, use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """policy / prop_history / critic 三个观测组。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor 观测：45 维单帧，不含真实线速度。"""

        base_ang_vel = ObsTerm(func=lab_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=lab_mdp.projected_gravity)
        velocity_commands = ObsTerm(func=lab_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=lab_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True)})
        joint_vel = ObsTerm(func=lab_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True)})
        actions = ObsTerm(func=lab_mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class PropHistoryCfg(ObsGroup):
        """CENet 观测：与 policy 相同的观测项，组级历史长度 10。

        输出为 term-major、项内从最旧到最新的展平历史（450 维），
        与部署端 ObservationHistory 完全一致；reset 后首帧填满窗口。
        """

        base_ang_vel = ObsTerm(func=lab_mdp.base_ang_vel, noise=UniformNoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=lab_mdp.projected_gravity, noise=UniformNoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=lab_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=lab_mdp.joint_pos_rel, noise=UniformNoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True)})
        joint_vel = ObsTerm(func=lab_mdp.joint_vel_rel, noise=UniformNoise(n_min=-1.5, n_max=1.5), params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True)})
        actions = ObsTerm(func=lab_mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True
            self.history_length = DEFAULT_HISTORY_LENGTH
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 特权观测：真实线速度位于前 3 维（true_vel_slice=(0,3)）。"""

        base_lin_vel = ObsTerm(func=lab_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=lab_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=lab_mdp.projected_gravity)
        velocity_commands = ObsTerm(func=lab_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=lab_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True)})
        joint_vel = ObsTerm(func=lab_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True)})
        actions = ObsTerm(func=lab_mdp.last_action)
        height_scan = ObsTerm(
            func=lab_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    prop_history: PropHistoryCfg = PropHistoryCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """域随机化与重置事件。"""

    physics_material = EventTerm(
        func=lab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.4),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    actuator_gains = EventTerm(
        func=lab_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    reset_base = EventTerm(
        func=lab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=lab_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.2, 0.2), "velocity_range": (-0.1, 0.1)},
    )
    push_robot = EventTerm(
        func=lab_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Isaac Lab 速度跟踪基线，补充高度、净空、足距与动作平滑项。"""

    # -- 任务
    ''' 速度跟踪奖励 '''
    track_lin_vel_xy_exp = RewTerm(
        func=lab_mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    '''角速度跟踪奖励'''
    track_ang_vel_z_exp = RewTerm(
        func=lab_mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    '''零指令站立：只在三轴指令近零时生效的窄核速度跟踪。
    std 从 0.5 收紧到 0.1，使 0.01 m/s 残速与 0.07 rad/s 漂移产生实际梯度
    （默认宽度下两者几乎不损失奖励）。非站立环境为 0，不改变行走段。'''
    track_lin_vel_xy_stand_exp = RewTerm(
        func=bpx_rewards.track_lin_vel_xy_stand_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.1, "command_threshold": 0.05},
    )
    track_ang_vel_z_stand_exp = RewTerm(
        func=bpx_rewards.track_ang_vel_z_stand_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.1, "command_threshold": 0.05},
    )
    # -- 正则
    '''z轴抖动惩罚'''
    lin_vel_z_l2 = RewTerm(
        func=lab_mdp.lin_vel_z_l2,
        weight=-2.0
    )
    '''roll/pitch 抖动惩罚'''
    ang_vel_xy_l2 = RewTerm(
        func=lab_mdp.ang_vel_xy_l2,
        weight=-0.05
    )
    '''高度偏离目标惩罚'''
    base_height_l2 = RewTerm(
        func=bpx_rewards.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.42,
            "height_scale": 0.10,  # 10 cm 偏差对应单位代价，再乘权重和 dt。
        },
    )
    # 机械功率总量与电机功率分布（DreamWaQ Table I）。
    joint_power = RewTerm(
        func=bpx_rewards.joint_power, weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True)},
    )
    power_distribution = RewTerm(
        func=bpx_rewards.power_distribution, weight=-1.0e-8,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True),
            "penalty_cap": 1.0e8,
        },
    )
    # 保留原有力矩平方正则。
    dof_torques_l2 = RewTerm(
        func=lab_mdp.joint_torques_l2,
        weight=-1.0e-4
    )
    '''关节加速度惩罚'''
    dof_acc_l2 = RewTerm(
        func=bpx_rewards.joint_acc_l2,
        weight=-2.5e-7,
        # 原始平方和上限；乘权重和 0.02 s 后，每策略步该项最低为 -0.5。
        params={"asset_cfg": SceneEntityCfg("robot"), "penalty_cap": 1.0e8},
    )
    '''动作变化率惩罚'''
    action_rate_l2 = RewTerm(
        func=lab_mdp.action_rate_l2,
        weight=-0.01
    )
    '''二阶动作差分；与一阶项一样使用未乘 scale 的策略动作。'''
    action_smoothness_l2 = RewTerm(
        func=bpx_rewards.action_smoothness_l2,
        weight=-0.01
    )
    '''toe_link 原点相对脚下局部地形的目标高度，按足端水平速度加权。'''
    feet_clearance = RewTerm(
        func=bpx_rewards.feet_clearance,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*toe_link"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.10,
            "max_ray_distance": 0.15,
        },
    )
    '''脚间距惩罚：前、后左右足横向间距为 0.26–0.42 m，同侧前后纵向间距不小于 0.30 m。'''
    feet_lateral_distance = RewTerm(
        func=bpx_rewards.feet_lateral_distance,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link"],
                preserve_order=True,
            ),
            "minimum_distance": 0.26,
            "maximum_distance": 0.32,
        },
    )
    feet_longitudinal_distance = RewTerm(
        func=bpx_rewards.feet_longitudinal_distance,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link"],
                preserve_order=True,
            ),
            "minimum_distance": 0.30,
        },
    )
    '''脚离地时间奖励'''
    feet_air_time = RewTerm(
        func=bpx_rewards.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*toe_link"),
            "command_name": "base_velocity",
            "threshold": 0.3,
            "command_threshold": 0.05,
            "yaw_command_threshold": 0.1,
        },
    )
    '''脚离地时间惩罚'''
    feet_air_time_excess = RewTerm(
        func=bpx_rewards.feet_air_time_excess,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*toe_link"),
            "maximum_air_time": 0.6,
            "penalty_cap": 1.0,
        },
    )
    '''脚部接触平衡奖励'''
    feet_contact_balance = RewTerm(
        func=bpx_rewards.feet_contact_balance,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*toe_link"),
            "time_constant": 1.0,
        },
    )
    '''不希望的接触惩罚'''
    undesired_contacts = RewTerm(
        func=lab_mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso"), "threshold": 1.0},
    )
    '''姿态水平惩罚（惩罚 roll/pitch 偏离水平，基于投影重力）'''
    flat_orientation_l2 = RewTerm(
        func=lab_mdp.flat_orientation_l2,
        weight=-0.3
    )
    '''关节位置越界惩罚'''
    joint_pos_limits = RewTerm(
        func=lab_mdp.joint_pos_limits,
        weight=-0.2
    )


@configclass
class TerminationsCfg:
    """终止条件：超时、机身接触、过倾（1.0 rad）。"""

    time_out = DoneTerm(func=lab_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=lab_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso"), "threshold": 1.0},
    )
    base_tilt = DoneTerm(func=bpx_terminations.base_tilt_above_maximum, params={"maximum_tilt_rad": 1.0})


@configclass
class CurriculumCfg:
    """课程项：地形课程（模块 A）；指令课程在指令生成器内部实现（模块 B）。"""

    terrain_levels: CurrTerm | None = CurrTerm(func=bpx_curriculums.terrain_levels_vel)


##
# 环境
##


@configclass
class BPXRoughEnvCfg(ManagerBasedRLEnvCfg):
    """BPX 崎岖地形速度跟踪环境。"""

    minimum_step_reward: float = -10.0
    """每个环境每个策略步的总奖励下限；不缩放、不限制正奖励。"""

    # ---- 课程开关（模块 A / B 独立控制）----
    enable_terrain_curriculum: bool = True
    """是否启用地形难度课程。关闭时地形等级随机初始化且不再变化。"""
    num_terrain_levels: int = 10
    """地形难度等级数（地形网格行数）。"""
    terrain_curriculum_mode: str = "standard"
    """地形课程模式：``standard``（升级+降级）或 ``upgrade_only``（只升不降）。"""
    terrain_curriculum_debug: bool = False
    """记录额外的地形课程调试信息。"""
    terrain_curriculum_min_command_speed: float = 0.1
    """平移指令超过此速度才参与地形评价（m/s），零速和原地旋转保持等级。"""
    terrain_curriculum_min_command_distance: float = 1.0
    """评价跟踪质量至少需要的累计指令路程（m），不使用最后一条指令外推。"""
    terrain_curriculum_min_tracking_duration_s: float = 2.0
    """升级或判定持续跟踪失败所需的最小平移指令时长（s）。"""
    terrain_curriculum_min_moving_fraction: float = 0.5
    """降级所需的平移指令时间占比；明确移动任务中摔倒不受最小路程限制。"""
    terrain_curriculum_lin_vel_tolerance: float = 0.1
    terrain_curriculum_yaw_tolerance: float = 0.5
    """仅用于降级和日志：平面误差 ≤ 0.1 m/s 且偏航误差 ≤ 0.5 rad/s 算跟踪良好。"""
    terrain_curriculum_failure_ratio: float = 0.5
    """未满足升级条件时，至少 50% 的平移采样跟踪不良才视为持续跟踪失败。"""
    terrain_curriculum_coverage_ratio: float = 0.25
    """升级时轴向最大距离须达到地形边长的此比例；分别相对出生点和地形中心。"""
    enable_grid_adaptive_command_curriculum: bool = True
    """是否启用 Grid-Adaptive 指令课程。关闭时恢复均匀指令采样。"""

    # 任务速度指令量程（Grid-Adaptive 课程的最大扩展范围从此读取）。
    command_lin_vel_x_range: tuple = (-1.0, 1.0)
    command_lin_vel_y_range: tuple = (-0.5, 0.5)
    command_ang_vel_z_range: tuple = (-1.0, 1.0)
    command_rel_standing_envs: float = 0.2
    """每次抽取指令时以此概率发送精确全零指令；独立于低速带采样。
    Grid-Adaptive 下站立指令在回合结束前保持为零（重采样周期只对运动指令
    生效），使策略面对与验收同量级的连续站定时长。"""
    # 指令课程初始范围（以零为中心逐步向外扩展）。
    command_curriculum_initial_vx_range: tuple = (-0.5, 0.5)
    command_curriculum_initial_yaw_range: tuple = (-0.5, 0.5)
    command_grid_resolution_vx: float = 0.5
    command_grid_resolution_yaw: float = 0.5
    command_grid_continuous_sampling: bool = True
    """保留网格课程，在所选格内连续采样 vx/wz，并裁到任务量程。"""
    command_low_speed_sampling_prob: float = 0.3
    """额外低速分支占全部非站立指令采样的比例，其余保留原网格分布。"""
    command_low_speed_vx_range: tuple = (0.05, 0.25)
    """低速分支的 vx 幅值区间；下限避开零点，使"慢走"与"站定"成为不同的
    训练状态，而不是积分到同一个低速吸引子。"""
    command_low_speed_random_sign: bool = True
    """把 command_low_speed_vx_range 当作无符号幅值，正负两侧对称采样，
    避免低速指令只剩单向而引入前向偏移。"""
    command_low_speed_vy_range: tuple = (-0.1, 0.1)
    command_low_speed_yaw_range: tuple = (-0.25, 0.25)
    """低速分支的三轴范围；仅从已激活单元与这些范围的交集采样。"""

    # 场景与 MDP
    scene: BPXSceneCfg = BPXSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        if not 0.0 <= self.command_rel_standing_envs <= 1.0:
            raise ValueError("command_rel_standing_envs 必须在 [0, 1] 内")
        # 通用设置：200 Hz 物理 × decimation 4 = 50 Hz 策略。
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # 传感器周期与策略周期对齐。
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt

        # ---- 地形课程接线（模块 A）----
        self.scene.terrain.terrain_generator = _make_bpx_terrain_generator(self.num_terrain_levels)
        if self.enable_terrain_curriculum:
            self.scene.terrain.max_init_terrain_level = 0
            self.scene.terrain.terrain_generator.curriculum = True
            if self.terrain_curriculum_mode not in ("standard", "upgrade_only"):
                raise ValueError(f"未知地形课程模式: {self.terrain_curriculum_mode}")
            curriculum_params = {
                "debug": self.terrain_curriculum_debug,
                "coverage_ratio": self.terrain_curriculum_coverage_ratio,
                "min_command_distance": self.terrain_curriculum_min_command_distance,
                "min_tracking_duration_s": self.terrain_curriculum_min_tracking_duration_s,
                "min_moving_fraction": self.terrain_curriculum_min_moving_fraction,
                "failure_ratio": self.terrain_curriculum_failure_ratio,
                "allow_demotion": self.terrain_curriculum_mode != "upgrade_only",
            }
            self.curriculum.terrain_levels = CurrTerm(
                func=bpx_curriculums.terrain_levels_vel, params=curriculum_params
            )
        else:
            # 关闭课程：等级随机初始化且不再更新，作为基线。
            self.curriculum.terrain_levels = None
            self.scene.terrain.max_init_terrain_level = None
            self.scene.terrain.terrain_generator.curriculum = False

        # ---- 指令课程接线（模块 B）----
        if self.enable_grid_adaptive_command_curriculum:
            self.commands.base_velocity = bpx_commands.GridAdaptiveVelocityCommandCfg(
                resampling_time_range=(10.0, 10.0),
                debug_vis=False,
                asset_name="robot",
                rel_standing_envs=self.command_rel_standing_envs,
                lin_vel_y_range=tuple(self.command_lin_vel_y_range),
                vx_resolution=self.command_grid_resolution_vx,
                yaw_resolution=self.command_grid_resolution_yaw,
                initial_vx_range=tuple(self.command_curriculum_initial_vx_range),
                initial_yaw_range=tuple(self.command_curriculum_initial_yaw_range),
                max_vx_range=tuple(self.command_lin_vel_x_range),
                max_yaw_range=tuple(self.command_ang_vel_z_range),
                continuous_sampling=self.command_grid_continuous_sampling,
                low_speed_sampling_prob=self.command_low_speed_sampling_prob,
                low_speed_vx_range=tuple(self.command_low_speed_vx_range),
                low_speed_vy_range=tuple(self.command_low_speed_vy_range),
                low_speed_yaw_range=tuple(self.command_low_speed_yaw_range),
                low_speed_random_sign=self.command_low_speed_random_sign,
            )
        else:
            # __post_init__ 可被命令行覆盖后再次调用，需要完整恢复均匀指令配置。
            self.commands.base_velocity = lab_mdp.UniformVelocityCommandCfg(
                class_type=bpx_commands.TerrainAwareUniformVelocityCommand,
                asset_name="robot",
                resampling_time_range=(10.0, 10.0),
                rel_standing_envs=self.command_rel_standing_envs,
                rel_heading_envs=1.0,
                heading_command=False,
                debug_vis=False,
                ranges=lab_mdp.UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=tuple(self.command_lin_vel_x_range),
                    lin_vel_y=tuple(self.command_lin_vel_y_range),
                    ang_vel_z=tuple(self.command_ang_vel_z_range),
                ),
            )


@configclass
class BPXRoughEnvCfg_PLAY(BPXRoughEnvCfg):
    """小规模可视化配置（关闭随机化与课程更新）。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.actuator_gains = None
        self.observations.prop_history.enable_corruption = False
