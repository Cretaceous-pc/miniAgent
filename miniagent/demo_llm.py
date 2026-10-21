# -*- coding: utf-8 -*-
"""DemoLLM：无 API key 时用来演示完整循环的“规则假模型”。

它不是玩具：它走的是和真模型完全相同的 LLMReply 协议，
所以在没有 key 的环境里也能验证「装配上下文 → 模型决策 → 调工具 → 回填 → 出答案」。
接真实模型后这个类完全不参与运行。
"""
from __future__ import annotations

import json

from .llm import LLMReply, ToolCall

_LIST_HINTS = ("哪些", "列出", "文件列表", "有什么文件", "目录里", "ls")


class DemoLLM:
    is_real = False

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.3) -> LLMReply:
        self.calls.append([dict(m) for m in messages])
        last = messages[-1] if messages else {"role": "user", "content": ""}

        # 第二步：已经有工具结果了 → 把结果整理成回答（模拟模型的“总结”行为）
        if last.get("role") == "tool":
            body = (last.get("content") or "").strip()
            preview = body if len(body) <= 700 else body[:700] + "…"
            return LLMReply(content=(
                f"【演示模式·非模型回答】我调用了工具，拿到这些内容：\n\n{preview}\n\n"
                "—— 以上由 DemoLLM 模板拼接。配置 .env 的真实 API key 后，这一步由模型生成。"))

        user = next((m.get("content") or "" for m in reversed(messages)
                     if m.get("role") == "user"), "")
        tool_names = {t["function"]["name"] for t in (tools or [])}

        if any(h in user for h in _LIST_HINTS) and "list_docs" in tool_names:
            return LLMReply(tool_calls=[ToolCall("demo-list", "list_docs", '{"path": "."}')])

        if "search_docs" in tool_names:
            query = user.strip()[:40] or "说明"
            return LLMReply(tool_calls=[ToolCall(
                "demo-search", "search_docs",
                json.dumps({"query": query, "top_k": 3}, ensure_ascii=False))])

        return LLMReply(content="【演示模式】没有可用工具，先配置真实 API key 吧。")
