"""源码工作区的统一路径；所有入口与当前工作目录无关。

此工程按源码工作区使用，请执行可编辑安装（pip install -e .）。
assets、configs、policies 和 outputs 均属于工作区，不打包为独立 wheel 数据。
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIGS_DIR = PROJECT_ROOT / "configs"
POLICIES_DIR = PROJECT_ROOT / "policies"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
