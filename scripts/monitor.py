#!/usr/bin/env python3
"""启动只读反馈、策略预览和姿态窗口。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpx_dreamwaq.sim2real.monitor import main

if __name__ == "__main__":
    raise SystemExit(main())
