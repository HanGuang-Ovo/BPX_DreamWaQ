"""机器人模型入口；只声明资源位置，不导入仿真器或自动转换模型。"""

from .paths import ASSETS_DIR

ROBOT_DIR = ASSETS_DIR / "robots" / "bpx"
URDF_PATH = ROBOT_DIR / "bpx" / "urdf" / "bpx.urdf"
MJCF_PATH = ROBOT_DIR / "mjcf" / "bpx.xml"
USD_PATH = ROBOT_DIR / "usd" / "bpx.usd"
USD_CONVERSION_CONFIG = ROBOT_DIR / "usd" / "config.yaml"
