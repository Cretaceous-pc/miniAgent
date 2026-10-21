# -*- coding: utf-8 -*-
"""miniAgent — 从零实现的最小可用 Agent Runtime（文档问答）。

设计约束（来自笔试题）：
  * 不依赖任何 agent 框架（langgraph / openhands / openclaw），核心 Runtime 自己写
  * 零第三方依赖：LLM 走 urllib 直连 OpenAI-compatible HTTP，存储用 sqlite3，
    Web 用 http.server，测试用 unittest
  * 同一份 Runtime，CLI 与 Web 只是两个入口（`python main.py cli|web`）
"""

__version__ = "0.1.0"
