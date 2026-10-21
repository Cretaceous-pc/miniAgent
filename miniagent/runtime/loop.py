# -*- coding: utf-8 -*-
"""Agent 主循环：整个项目的心脏，约 120 行，四个出口全部可枚举。

    user → 装配上下文 → 调模型 ─┬─ 没有 tool_calls → 出口①：返回答案
                                └─ 有 tool_calls  → 执行 → 结果回填 → 下一轮
                                                          ↳ 轮次用尽 → 出口②
                                                          ↳ 同参重复 → 出口③（打断）
                                                          ↳ 模型报错 → 出口④
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..context.manager import ContextManager, est_tokens
from ..llm import LLMError
from ..store.session import SessionStore
from ..tools.base import ToolRegistry


@dataclass
class Event:
    kind: str
    data: dict
    ms: float = 0.0

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ms": round(self.ms, 1), "data": self.data}


@dataclass
class Answer:
    reply: str
    events: list[Event] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "final"          # final | max_rounds | repeat_guard | llm_error

    def events_as_dicts(self) -> list[dict]:
        return [e.to_dict() for e in self.events]


class AgentLoop:
    def __init__(self, llm, store: SessionStore, ctx: ContextManager,
                 registry: ToolRegistry, session_id: str, root: Path,
                 trace=None, max_rounds: int = 8, max_repeat: int = 3,
                 tool_char_limit: int = 6000):
        self.llm = llm
        self.store = store
        self.ctx = ctx
        self.registry = registry
        self.sid = session_id
        self.root = root
        self.trace = trace
        self.max_rounds = max_rounds
        self.max_repeat = max_repeat
        self.tool_char_limit = tool_char_limit

    # ---------------------------------------------------------------- #
    def ask(self, user_text: str) -> Answer:
        """一轮完整交互：可能包含多次工具往返，最终一定返回一条可读回复。"""
        events: list[Event] = []
        self.store.append_message(self.sid, "user", user_text)
        # 工具 schema 每次请求都随 body 发出，属于上下文成本，必须算进预算
        tool_tokens = est_tokens(json.dumps(self.registry.specs(), ensure_ascii=False))

        for round_no in range(1, self.max_rounds + 1):
            built = self.ctx.build(self.sid, str(self.root), extra_tokens=tool_tokens)
            if built.compressed:
                event = Event("compress", {"dropped": built.dropped_messages,
                                           "tokens": built.tokens})
                events.append(event)
                self._trace("compress", event.data)

            # ---- 1. 调模型 ----
            started = time.perf_counter()
            try:
                reply = self.llm.chat(built.messages, tools=self.registry.specs())
            except LLMError as exc:
                text = (f"模型调用失败：{exc}\n"
                        "检查 .env 里的 MINIAGENT_API_KEY / MINIAGENT_BASE_URL / MINIAGENT_MODEL")
                self.store.append_message(self.sid, "assistant", text)
                event = Event("error", {"error": str(exc)}, (time.perf_counter() - started) * 1000)
                events.append(event)
                self._trace("error", event.data)
                return Answer(text, events, round_no, "llm_error")
            cost_ms = (time.perf_counter() - started) * 1000

            event = Event("llm", {
                "round": round_no,
                "content": reply.content,
                # 思考路径：只进事件与 trace，不回填 messages（reasoner 会拒绝）
                "reasoning": reply.reasoning,
                "reasoning_chars": len(reply.reasoning),
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls],
                "usage": reply.usage,
                "context_tokens": built.tokens,
                "recalled": built.recalled,
            }, cost_ms)
            events.append(event)
            self._trace("llm", event.data, cost_ms)

            # ---- 2. 出口①：没有工具调用 = 最终答案 ----
            if not reply.tool_calls:
                content = reply.content or (
                    "（模型没有返回正文：可能输出被截断或触发内容过滤，原始响应见 trace）")
                self.store.append_message(self.sid, "assistant", content)
                self._trace("final", {"rounds": round_no, "chars": len(content),
                                      "reasoning_chars": len(reply.reasoning)})
                return Answer(content, events, round_no, "final")

            # ---- 3. 记录 assistant 的 tool_calls（协议要求与 tool 消息成对） ----
            self.store.append_message(self.sid, "assistant", reply.content,
                                      tool_calls=self._protocol_tool_calls(reply.tool_calls))
            for call in reply.tool_calls:
                args = call.parsed_args()
                repeat = self.registry.note_call(call.name, args)
                started = time.perf_counter()
                if repeat > self.max_repeat:
                    # 出口③：防死循环（模型反复用同参数打同一个工具）
                    result = (f"ERROR: 工具 {call.name} 用同一参数已连续调用 {repeat} 次，"
                              "运行时已打断。请换一种策略，或直接基于现有信息回答")
                    events.append(Event("guard", {"tool": call.name, "args": args, "count": repeat}))
                else:
                    result = self.registry.invoke(call.name, call.arguments)
                cost_ms = (time.perf_counter() - started) * 1000
                if len(result) > self.tool_char_limit:
                    result = result[: self.tool_char_limit] + "\n…（工具结果过长已截断）"

                self.store.append_message(self.sid, "tool", result,
                                          tool_call_id=call.id, name=call.name)
                event = Event("tool", {"name": call.name, "args": args, "result": result,
                                       "repeat": repeat}, cost_ms)
                events.append(event)
                self._trace("tool", {"name": call.name, "args": args,
                                     "result": result[:500], "repeat": repeat}, cost_ms)

        # ---- 出口②：轮次用尽 ----
        text = (f"已达最大工具轮次 {self.max_rounds}，先停在这里。\n"
                "现有结论见上；如果需要继续，请缩小问题范围再问一次。")
        self.store.append_message(self.sid, "assistant", text)
        self._trace("final", {"stopped": "max_rounds", "rounds": self.max_rounds})
        return Answer(text, events, self.max_rounds, "max_rounds")

    # ---------------------------------------------------------------- #
    @staticmethod
    def _protocol_tool_calls(calls) -> list[dict]:
        return [{"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.arguments}} for c in calls]

    def _trace(self, kind: str, payload: dict, ms: float = 0.0) -> None:
        if self.trace is not None:
            self.trace.write(self.sid, kind, payload, ms)
