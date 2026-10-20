#!/usr/bin/env python3
"""从源码工作区启动 MuJoCo，不要求先执行可编辑安装。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpx_dreamwaq.sim2sim.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
