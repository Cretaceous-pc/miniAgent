# -*- coding: utf-8 -*-
"""内置工具集：文档问答用的 7 个工具。

思路是**渐进式披露**——不给模型灌全文，而是：
    list_docs  （有什么）→ outline（看目录）→ search_docs（跨文件定位）
    → read_doc（按行精读）→ todo（记阅读计划）→ calculator（算数）→ remember（长期记忆）
这样 context 里永远只有"被需要的那几段"，压缩压力也小。
"""
from __future__ import annotations

import ast
import json
import operator
from pathlib import Path

from .base import Tool, ToolContext, ToolError
from .docs import (SUPPORTED_EXTS, display_path, iter_files, outline, resolve,
                   search_file, tokenize)

MAX_TOOL_CHARS = 6000        # 单个工具结果上限，超出截断并提示（保护 context）


def _clip(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…（结果被截断，共 {len(text)} 字符；请缩小范围重试）"


# --------------------------------------------------------------------------- #
# 1. list_docs
# --------------------------------------------------------------------------- #
def list_docs(ctx: ToolContext, path: str = ".", max_files: int = 200) -> str:
    base = resolve(ctx.root, path)
    files = iter_files(base, max_files=max_files) if base.is_dir() else [base]
    if not files:
        return f"{display_path(ctx.root, base)} 下没有可读文档（支持：{' '.join(sorted(SUPPORTED_EXTS))}）"
    rows = []
    for f in files:
        try:
            size = f.stat().st_size / 1024
        except OSError:
            continue
        rows.append(f"{display_path(ctx.root, f)}\t{size:.1f}KB")
    head = f"共 {len(rows)} 个文档（根目录：{ctx.root}）："
    return _clip("\n".join([head] + rows))


# --------------------------------------------------------------------------- #
# 2. outline
# --------------------------------------------------------------------------- #
def outline_tool(ctx: ToolContext, path: str) -> str:
    target = resolve(ctx.root, path)
    lines = ctx.reader.lines(target)
    items = outline(lines, target.suffix)
    if not items:
        return (f"{display_path(ctx.root, target)}（{len(lines)} 行）没有可识别的结构，"
                f"建议直接用 search_docs 或 read_doc 前 60 行浏览")
    return _clip(f"{display_path(ctx.root, target)}（共 {len(lines)} 行）结构：\n"
                 + "\n".join(items))


# --------------------------------------------------------------------------- #
# 3. search_docs
# --------------------------------------------------------------------------- #
def search_docs(ctx: ToolContext, query: str, path: str = ".", top_k: int = 5) -> str:
    base = resolve(ctx.root, path)
    if not query.strip():
        raise ToolError("query 不能为空")
    tokens = tokenize(query)
    if not tokens:
        raise ToolError("query 里没有可用于检索的词")
    files = iter_files(base) if base.is_dir() else [base]
    if not files:
        raise ToolError(f"{display_path(ctx.root, base)} 下没有可读文档")

    scored: list[tuple[float, str, int, str]] = []
    for f in files:
        try:
            lines = ctx.reader.lines(f)
        except ToolError:
            continue
        for score, lineno, text in search_file(lines, tokens, query):
            scored.append((score, display_path(ctx.root, f), lineno, text))
    if not scored:
        return f"没有匹配「{query}」的内容（检索了 {len(files)} 个文件）。可换关键词或先用 outline 看结构"
    scored.sort(key=lambda x: -x[0])
    top = scored[:top_k]
    body = "\n".join(f"{path}:{lineno}\t{text}" for _, path, lineno, text in top)
    return _clip(f"命中 {len(scored)} 处，显示前 {len(top)} 处：\n{body}")


# --------------------------------------------------------------------------- #
# 4. read_doc
# --------------------------------------------------------------------------- #
def read_doc(ctx: ToolContext, path: str, start: int = 1, end: int = 80) -> str:
    target = resolve(ctx.root, path)
    lines = ctx.reader.lines(target)
    total = len(lines)
    if total == 0:
        return f"{display_path(ctx.root, target)} 是空文件"
    start = max(1, min(start, total))
    end = max(start, min(end, total))
    if end - start + 1 > 400:
        end = start + 399
    numbered = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
    tail = ""
    if end < total:
        tail = f"\n…（还有 {total - end} 行；用 start={end + 1} 继续读）"
    return _clip(f"{display_path(ctx.root, target)} 第 {start}-{end} 行 / 共 {total} 行：\n"
                 f"{numbered}{tail}")


# --------------------------------------------------------------------------- #
# 5. todo（题目硬要求）
# --------------------------------------------------------------------------- #
def todo(ctx: ToolContext, action: str, item: str = "", index: int = 0) -> str:
    if ctx.store is None or not ctx.session_id:
        raise ToolError("当前会话没有绑定存储，todo 不可用")
    store = ctx.store

    if action == "add":
        if not item.strip():
            raise ToolError("action=add 时 item 不能为空")
        idx = store.add_todo(ctx.session_id, item.strip())
        return f"已加入待办 #{idx}：{item.strip()}"
    if action == "list":
        rows = store.list_todos(ctx.session_id)
        if not rows:
            return "待办为空"
        return "待办清单：\n" + "\n".join(
            f"[{'x' if done else ' '}] #{i} {text}" for i, text, done in rows)
    if action == "done":
        ok = store.done_todo(ctx.session_id, index)
        return f"已勾选 #{index}" if ok else f"没有 #{index} 这条待办"
    raise ToolError(f"action 只能是 add/list/done，收到 {action!r}")


# --------------------------------------------------------------------------- #
# 6. calculator（AST 白名单，绝不 eval）
# --------------------------------------------------------------------------- #
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sum": lambda xs: sum(xs), "min": min, "max": max, "len": len,
    "avg": lambda xs: sum(xs) / len(xs) if xs else 0,
}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.List):
        return [_eval_node(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        args = [_eval_node(a) for a in node.args]
        return _FUNCS[node.func.id](*args)
    raise ToolError(f"不支持的表达式成分：{type(node).__name__}（只允许四则运算/幂/括号/sum,min,max,len,avg）")


def calculator(ctx: ToolContext, expression: str) -> str:
    expr = expression.strip()
    if not expr:
        raise ToolError("expression 不能为空")
    if len(expr) > 200:
        raise ToolError("表达式过长（>200 字符）")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"表达式语法错误：{exc.msg}")
    value = _eval_node(tree)
    if isinstance(value, float) and value == int(value):
        value = int(value)
    return f"{expr} = {value}"


# --------------------------------------------------------------------------- #
# 7. remember（长期记忆写入；召回在 context 装配时做）
# --------------------------------------------------------------------------- #
def remember(ctx: ToolContext, content: str, tags: str = "") -> str:
    if ctx.store is None or not ctx.session_id:
        raise ToolError("当前会话没有绑定存储，remember 不可用")
    if not content.strip():
        raise ToolError("content 不能为空")
    ctx.store.add_memory(ctx.session_id, content.strip(), tags.strip())
    return f"已记住：{content.strip()[:80]}（后续对话会自动召回）"


# --------------------------------------------------------------------------- #
# 工具清单（description 直接进 prompt，措辞影响调用质量）
# --------------------------------------------------------------------------- #
ALL_TOOLS = [
    Tool(
        name="list_docs",
        description=("列出根目录下的可读文档（相对路径 + 大小）。"
                     "第一次接触一个目录、或不确定有哪些文件时先用它。"
                     "不要用它反复确认同一目录。"),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": ".", "description": "相对根目录的路径，默认当前根目录"},
                "max_files": {"type": "integer", "default": 200, "description": "最多列出多少个文件"},
            },
            "required": [],
        },
        fn=list_docs,
    ),
    Tool(
        name="outline",
        description=("给出单个文档的结构目录（Markdown 标题 / 代码的 def-class / 文本段落首行），"
                     "带行号。用于在精读前快速定位，比直接读完整个文件省很多 token。"),
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "相对根目录的文件路径"}},
            "required": ["path"],
        },
        fn=outline_tool,
    ),
    Tool(
        name="search_docs",
        description=("在目录或单个文件里按关键词检索，返回 `路径:行号` 和命中片段。"
                     "适合“哪个文件提到了 X”。多个关键词用空格分隔；中文按 2 字片段匹配。"),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词，可用空格分隔多个"},
                "path": {"type": "string", "default": ".", "description": "检索范围：目录或单文件"},
                "top_k": {"type": "integer", "default": 5, "description": "返回前多少条命中"},
            },
            "required": ["query"],
        },
        fn=search_docs,
    ),
    Tool(
        name="read_doc",
        description=("按行区间读取文档片段，输出带行号。"
                     "先用 outline 或 search_docs 定位行号再来读，每次读 60-120 行为宜；"
                     "需要下一个区间时传 start=上一段结束行+1。"),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对根目录的文件路径"},
                "start": {"type": "integer", "default": 1, "description": "起始行（1 起）"},
                "end": {"type": "integer", "default": 80, "description": "结束行（含）"},
            },
            "required": ["path"],
        },
        fn=read_doc,
    ),
    Tool(
        name="todo",
        description=("会话级待办的增查改。多步阅读任务（“先看 A 再看 B”）建议先 add 列出计划，"
                     "做完一步 done 掉，避免长任务中忘记目标。"),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "done"], "description": "操作类型"},
                "item": {"type": "string", "description": "action=add 时的待办内容"},
                "index": {"type": "integer", "default": 0, "description": "action=done 时的待办编号"},
            },
            "required": ["action"],
        },
        fn=todo,
    ),
    Tool(
        name="calculator",
        description=("计算数学表达式（四则、幂、括号，以及 sum/min/max/len/avg 作用于列表）。"
                     "文档里的计数、求和、比例必须用它算，不要心算。"),
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "例如 (12+8)*3 或 avg([3,5,7])"}},
            "required": ["expression"],
        },
        fn=calculator,
    ),
    Tool(
        name="remember",
        description=("把用户明确要求记住的事实/偏好写入长期记忆（会话级持久化）。"
                     "只记稳定事实（如“我们只看 docs/ 下的规范”），不要记一次性的中间结果。"),
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要记住的一句话事实"},
                "tags": {"type": "string", "default": "", "description": "可选标签，逗号分隔"},
            },
            "required": ["content"],
        },
        fn=remember,
    ),
]
