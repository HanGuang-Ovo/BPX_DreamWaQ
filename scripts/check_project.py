#!/usr/bin/env python3
"""离线检查资源、配置和本阶段范围，不导入或连接机器人 SDK。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpx_dreamwaq.validation import main

if __name__ == "__main__":
    raise SystemExit(main())
