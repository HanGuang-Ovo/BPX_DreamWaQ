"""离线检查工作区必需资源与配置；权重和 USD 缺失是本阶段的已知状态。"""

import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .assets import MJCF_PATH, ROBOT_DIR, URDF_PATH, USD_PATH
from .paths import CONFIGS_DIR, PROJECT_ROOT
from .runtime.joints import POLICY_JOINT_NAMES
from .sim2real.config import load_config as load_real_config
from .sim2sim.config import load_config as load_sim_config


def check_assets() -> dict:
    """检查资源引用以及 URDF/MJCF 中的受控关节集合。"""
    missing = []
    if not (ROBOT_DIR / "LICENSE").is_file():
        missing.append(str(ROBOT_DIR / "LICENSE"))
    joint_sets = {}
    for label, path in (("urdf", URDF_PATH), ("mjcf", MJCF_PATH)):
        root = ET.parse(path).getroot()
        if label == "urdf":
            references = [path.parent / mesh.attrib["filename"] for mesh in root.iter("mesh")]
            names = {j.attrib["name"] for j in root.findall("joint") if j.get("type") != "fixed"}
        else:
            compiler = root.find("compiler")
            meshdir = compiler.get("meshdir", ".") if compiler is not None else "."
            references = [path.parent / meshdir / m.attrib["file"] for m in root.findall("asset/mesh")]
            names = {j.attrib["name"] for j in root.iter("joint") if "name" in j.attrib}
        missing.extend(str(p.resolve()) for p in references if not p.is_file())
        joint_sets[label] = names == set(POLICY_JOINT_NAMES)
    return {"missing_resources": missing, "joint_contract": joint_sets, "usd_available": USD_PATH.is_file()}


def check_project() -> dict:
    errors = []
    assets = check_assets()
    if assets["missing_resources"] or not all(assets["joint_contract"].values()):
        errors.append("机器人资源引用或关节集合不完整")
    configs = []
    for path in sorted((CONFIGS_DIR / "sim2sim").glob("bpx_*.toml")):
        config = load_sim_config(path)
        if config.joint_names != POLICY_JOINT_NAMES or not config.paths.mjcf.is_file():
            errors.append(f"仿真配置的模型或关节顺序错误：{path.name}")
        configs.append({"name": path.name, "observation_dimension": config.observation.dimension})
    real = load_real_config(CONFIGS_DIR / "sim2real/runtime.toml")
    # 训练阶段已实现：检查训练模块的关键定义存在（不做 isaaclab 导入）。
    expected_symbols = {
        "src/bpx_dreamwaq/tasks/mdp/curriculums.py": "terrain_levels_vel",
        "src/bpx_dreamwaq/tasks/mdp/commands.py": "GridAdaptiveVelocityCommand",
        "src/bpx_dreamwaq/tasks/agents/cenet.py": "class CENet",
        "src/bpx_dreamwaq/tasks/agents/dreamwaq_actor_critic.py": "class DreamWaQActorCritic",
        "src/bpx_dreamwaq/tasks/agents/dreamwaq_ppo.py": "class DreamWaQPPO",
        "src/bpx_dreamwaq/tasks/rough_env_cfg.py": "class BPXRoughEnvCfg",
    }
    training_modules = []
    for rel_path, symbol in expected_symbols.items():
        path = PROJECT_ROOT / rel_path
        if not path.is_file() or symbol not in path.read_text(encoding="utf-8"):
            errors.append(f"训练模块缺失或缺少 {symbol}：{rel_path}")
        else:
            training_modules.append(Path(rel_path).stem)
    if not (PROJECT_ROOT / "scripts/train.py").is_file():
        errors.append("训练入口 scripts/train.py 缺失")
    return {
        "project_root": str(PROJECT_ROOT),
        "errors": errors,
        "assets": assets,
        "sim2sim_configs": configs,
        "training_modules": training_modules,
        "optional_dependencies": {
            name: importlib.util.find_spec(name) is not None
            for name in ("mujoco", "onnxruntime", "torch", "bpx_sdk", "tkinter")
        },
        "pending": {
            "usd": None if assets["usd_available"] else "源工程未提供 bpx.usd；训练首次启动时会从 URDF 自动转换",
            "deploy_resources": real.blockers(),
        },
    }


def main() -> int:
    try:
        report = check_project()
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        print(json.dumps({"errors": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
