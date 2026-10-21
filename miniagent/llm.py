# -*- coding: utf-8 -*-
"""LLM 客户端：urllib 直连 OpenAI-compatible `/chat/completions`。

只做三件事：拼请求、处理重试、把响应规范化成 LLMReply。
不是 agent 框架的一部分——它等价于 curl，替换供应商只改 base_url。
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(Exception):
    """模型侧不可恢复的失败（鉴权、参数、重试耗尽）。"""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str                 # 原始 JSON 字符串，交给工具层解析与校验

    def parsed_args(self) -> dict:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


@dataclass
class LLMReply:
    content: str = ""
    # 思考路径（思考型模型的 reasoning_content / reasoning）。
    # 只用于观测与展示，**绝不回填进下一次请求的 messages**：
    # DeepSeek 的 reasoner 明确要求 reasoning_content 不得出现在输入里，否则 400。
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    model: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """一个 method 的客户端：chat(messages, tools) -> LLMReply。"""

    is_real = True          # False = FakeLLM，用来决定是否需要 LLM 摘要器

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 60.0, max_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_usage: dict = {}

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.3) -> LLMReply:
        payload: dict = {"model": self.model, "messages": messages,
                         "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return self._parse(data)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                # 4xx 里只有 429 值得重试，其余（401/400/404）重试也没用
                if exc.code == 429 or exc.code >= 500:
                    last_error = LLMError(f"HTTP {exc.code}: {detail}")
                else:
                    raise LLMError(f"HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                last_error = LLMError(f"网络不可达/超时: {exc}")
            except json.JSONDecodeError as exc:
                raise LLMError(f"响应不是合法 JSON: {exc}") from exc

            if attempt < self.max_retries:
                time.sleep(1.0 * (2 ** attempt))       # 1s, 2s
        raise last_error or LLMError("未知错误")

    @staticmethod
    def _parse(data: dict) -> LLMReply:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"响应缺少 choices: {str(data)[:300]}")
        msg = choices[0].get("message") or {}
        calls = []
        for raw in msg.get("tool_calls") or []:
            fn = raw.get("function") or {}
            calls.append(ToolCall(
                id=raw.get("id") or f"call_{len(calls)}",
                name=fn.get("name") or "",
                arguments=fn.get("arguments") or "{}",
            ))
        return LLMReply(
            content=_as_text(msg.get("content")),
            reasoning=_as_text(msg.get("reasoning_content") or msg.get("reasoning")),
            tool_calls=calls,
            usage=data.get("usage") or {},
            model=data.get("model") or "",
        )


def _as_text(value) -> str:
    """兼容三种 content 形态：字符串 / 空 / 块数组（部分供应商返回 list of blocks）。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "".join(parts).strip()
    return ""


def message(role: str, content: str, **extra) -> dict:
    """构造 OpenAI 协议消息（多余字段只在该 role 需要时出现）。"""
    msg = {"role": role, "content": content}
    msg.update({k: v for k, v in extra.items() if v is not None})
    return msg
