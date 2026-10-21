# -*- coding: utf-8 -*-
"""允许 `python -m miniagent cli|web` 启动，等价于 `python main.py`。"""
from __future__ import annotations

from .dispatch import main

if __name__ == "__main__":
    raise SystemExit(main())
