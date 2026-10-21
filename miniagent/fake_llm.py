# -*- coding: utf-8 -*-
"""FakeLLM：按脚本返回预设回复，用于测试与无 key 演示。

关键点：它实现的是和 LLMClient 相同的 `chat()` 协议，因此 Runtime 完全不知道
对面是真模型还是脚本——这也是我把 provider 抽成接口的唯一原因。
"""
from __future__ import annotations

import json
from typing import Iterable

from .llm import LLMReply, ToolCall


def say(text: str) -> dict:
    """脚本项：直接回一句话（循环结束）。"""
    return {"content": text}


def call(name: str, args: dict, call_id: str = "", content: str = "") -> dict:
    """脚本项：发起一次工具调用（循环继续）。"""
    return {"content": content, "tool_calls": [{"name": name, "args": args, "id": call_id}]}


def call_many(items: Iterable[tuple[str, dict]], content: str = "") -> dict:
    """脚本项：一轮内并发请求多个工具。"""
    return {"content": content,
            "tool_calls": [{"name": n, "args": a, "id": ""} for n, a in items]}


class FakeLLM:
    is_real = False

    def __init__(self, script: list[dict] | None = None, default: str = "（FakeLLM 脚本已用完）"):
        self.script = list(script or [])
        self.default = default
        # 测试断言用：每次调用收到的 messages 快照
        self.calls: list[list[dict]] = []
        self.tool_specs_seen: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.3) -> LLMReply:
        self.calls.append([dict(m) for m in messages])
        self.tool_specs_seen.append(list(tools or []))

        item = self.script.pop(0) if self.script else {"content": self.default}
        if isinstance(item, str):
            item = {"content": item}

        calls: list[ToolCall] = []
        for i, raw in enumerate(item.get("tool_calls") or []):
            calls.append(ToolCall(
                id=raw.get("id") or f"call_{len(self.calls)}_{i}",
                name=raw["name"],
                arguments=json.dumps(raw.get("args") or {}, ensure_ascii=False),
            ))
        return LLMReply(content=item.get("content") or "",
                         reasoning=item.get("reasoning") or "",
                         tool_calls=calls)

    # ——— 测试辅助 ———
    @property
    def last_messages(self) -> list[dict]:
        return self.calls[-1] if self.calls else []

    def text_of(self, index: int = -1) -> str:
        return "\n".join(
            str(m.get("content") or "") for m in self.calls[index]
        ) if self.calls else ""
