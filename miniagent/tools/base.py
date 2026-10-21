# -*- coding: utf-8 -*-
"""工具层骨架：Tool 描述、注册中心、参数校验、错误回填。

三个刻意的设计：
  1. **Schema 就是给模型的说明书**：description 写“什么时候用/什么时候别用”，
     比参数说明更影响调用质量
  2. **校验与执行错误一律变成 tool 消息回给模型**，不抛出循环外——
     模型看到 "ERROR: ..." 才有机会自纠，这是我们唯一的纠错通道
  3. **工具不直接返回 Markdown**，返回纯文本/紧凑 JSON，避免把格式错误传染给下游
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .docs import DocReader


class ToolError(Exception):
    """可回填给模型的错误（参数不合法、路径越界、文件不存在…）。"""


@dataclass
class ToolContext:
    """一次会话里工具能看到的全部外部状态。"""
    root: Path
    reader: DocReader
    store: Any = None            # SessionStore，todo / remember 需要
    session_id: str = ""

    def __post_init__(self) -> None:
        if self.reader is None:
            self.reader = DocReader(self.root)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --------------------------------------------------------------------------- #
# 极简 JSON-Schema 校验（只支持我们实际用到的子集：object/string/integer/enum/required）
# --------------------------------------------------------------------------- #
_TYPES = {"string": str, "integer": int, "number": (int, float),
          "boolean": bool, "object": dict, "array": list}


def validate(schema: dict, args: dict) -> dict:
    if not isinstance(args, dict):
        raise ToolError(f"参数必须是 JSON 对象，收到 {type(args).__name__}")

    props = schema.get("properties") or {}
    required = schema.get("required") or []
    # 先查"多余参数"：模型拼错键名时，这个提示比"缺少必填"更能让它自己改对
    for key in args:
        if key not in props:
            raise ToolError(f"未知参数 {key}；可用参数：{', '.join(props) or '（无）'}")
    for key in required:
        if key not in args or args[key] in (None, ""):
            raise ToolError(f"缺少必填参数 {key}；需要 {', '.join(required)}")

    cleaned: dict = {}
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            raise ToolError(f"未知参数 {key}；可用参数：{', '.join(props)}")
        expected = _TYPES.get(spec.get("type", ""))
        if expected and not isinstance(value, expected):
            # 模型常把整数写成 "3"，这里做一次宽容转换，转不了才报错
            if spec.get("type") == "integer" and isinstance(value, str) and value.strip().isdigit():
                value = int(value)
            elif spec.get("type") == "number" and isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    raise ToolError(f"参数 {key} 需要 {spec['type']}，收到 {value!r}")
            else:
                raise ToolError(f"参数 {key} 需要 {spec['type']}，收到 {type(value).__name__}")
        if spec.get("enum") and value not in spec["enum"]:
            raise ToolError(f"参数 {key} 只能是 {spec['enum']}，收到 {value!r}")
        cleaned[key] = value

    for key, spec in props.items():
        if key not in cleaned and "default" in spec:
            cleaned[key] = spec["default"]
    return cleaned


class ToolRegistry:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self._tools: dict[str, Tool] = {}
        self.call_counts: dict[str, int] = {}      # (name+参数) 计数，用于防重复调用

    # ---- 注册 ---- #
    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名：{tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    # ---- 调用 ---- #
    def invoke(self, name: str, raw_args: dict | str) -> str:
        """统一入口：校验 → 执行 → 把任何失败转成给模型看的文本。"""
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: 没有名为 {name} 的工具。可用工具：{', '.join(self.names)}"

        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args or "{}")
            except json.JSONDecodeError as exc:
                return f"ERROR: 参数不是合法 JSON（{exc}）。请重新给出符合 schema 的参数"
        try:
            args = validate(tool.parameters, raw_args)
            result = tool.fn(self.ctx, **args)
        except ToolError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:                    # 工具内部 bug 也不能掀翻整个对话
            return f"ERROR: {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else str(result)

    def note_call(self, name: str, raw_args: dict) -> int:
        key = name + "|" + json.dumps(raw_args, ensure_ascii=False, sort_keys=True)
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        return self.call_counts[key]


def build_registry(ctx: ToolContext) -> ToolRegistry:
    """装配全部内置工具（在 builtin 里 import 反向会造成循环，所以这里延时导入）。"""
    from . import builtin

    registry = ToolRegistry(ctx)
    for tool in builtin.ALL_TOOLS:
        registry.add(tool)
    return registry
