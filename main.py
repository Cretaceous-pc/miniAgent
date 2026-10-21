# -*- coding: utf-8 -*-
"""唯一入口：`python main.py cli|web`。

同一份 Runtime，两种前端；差别只在参数，不在逻辑。
（实现放在 miniagent/dispatch.py，这样 console script 与 main.py 共用一份 argparse）
"""
from __future__ import annotations

from miniagent.dispatch import main

if __name__ == "__main__":
    raise SystemExit(main())
