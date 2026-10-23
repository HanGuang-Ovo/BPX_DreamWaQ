#!/usr/bin/env python3
"""BPX_test visualize.py 对应入口：只读启动前检查、姿态与策略预览。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpx_dreamwaq.sim2real.monitor import main

if __name__ == "__main__":
    raise SystemExit(main())
