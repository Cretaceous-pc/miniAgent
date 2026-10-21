# -*- coding: utf-8 -*-
"""CLI 入口：交互式对话 + 会话管理命令。

刻意保持“薄”：只做参数解析、打印、循环读取；所有决策都发生在 App/AgentLoop 里。
"""
from __future__ import annotations

import sys
from pathlib import Path

from .app import App
from .config import Config, ConfigError

BANNER = """miniAgent · 文档问答（从零实现的最小 Agent Runtime）
命令：/new [目录] 新会话  /list 会话列表  /switch <id> 切换  /trace 调用链
      /del [id] 删除会话  /todo 待办  /files 文档列表
      /help 帮助  /exit 退出
"""


def _print_events(answer) -> None:
    for ev in answer.events:
        if ev.kind == "llm" and ev.data.get("reasoning"):
            # 只展示思考路径的开头，完整内容在 /trace 与 traces/*.jsonl 里
            head = " ".join(str(ev.data["reasoning"]).split())[:110]
            print(f"   💭 思考 {ev.data['reasoning_chars']} 字：{head}…")
        elif ev.kind == "tool":
            args = ev.data.get("args") or {}
            preview = str(ev.data.get("result", "")).splitlines()[:1]
            head = preview[0][:120] if preview else ""
            print(f"   ⚙ {ev.data['name']}({args}) → {head}"
                  + (f"  [{ev.ms:.0f}ms]" if ev.ms else ""))
        elif ev.kind == "compress":
            print(f"   ✂ 上下文压缩：丢弃 {ev.data.get('dropped')} 条，"
                  f"压缩后约 {ev.data.get('tokens')} tokens")
        elif ev.kind == "guard":
            print(f"   ⛔ 重复调用保护：{ev.data.get('tool')} 已调用 {ev.data.get('count')} 次")
        elif ev.kind == "error":
            print(f"   ✗ {ev.data.get('error')}")


def run_cli(root: str, fake: bool = False, once: str | None = None,
            session_id: str | None = None) -> int:
    cfg = Config.from_env()
    root_path = Path(root)
    if not root_path.exists():
        print(f"根目录不存在：{root_path}", file=sys.stderr)
        return 2

    llm = None
    if fake:
        from .demo_llm import DemoLLM
        llm = DemoLLM()
        print("（演示模式：使用 DemoLLM，不消耗 API）\n")

    try:
        app = App(cfg, root_path, llm=llm)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 3

    if session_id:
        current = app.session(session_id)
        if not current:
            print(f"会话不存在：{session_id}", file=sys.stderr)
            return 2
    else:
        current = app.new_session(root_path)
    print(BANNER if once is None else "")
    print(f"会话 {current['id']} · 根目录 {current['root']}\n")

    if once:
        answer = app.chat(current["id"], once)
        _print_events(answer)
        print("\n" + answer.reply)
        return 0 if answer.stopped == "final" else 1

    while True:
        try:
            text = input("你 › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/exit", "/quit", "exit", "quit"):
            break
        if text == "/help":
            print(BANNER)
            continue
        if text.startswith("/new"):
            parts = text.split(maxsplit=1)
            target = Path(parts[1]) if len(parts) > 1 else Path(current["root"])
            if not target.exists():
                print(f"  目录不存在：{target}")
                continue
            current = app.new_session(target)
            print(f"  新会话 {current['id']} · {current['root']}")
            continue
        if text == "/list":
            for s in app.sessions():
                mark = "→" if s["id"] == current["id"] else " "
                print(f"  {mark} {s['id']}  {s['title'][:20]:<22} {s['msg_count']} 条  {s['root']}")
            continue
        if text.startswith("/switch"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                print("  用法：/switch <会话id>")
                continue
            target = app.session(parts[1].strip())
            if not target:
                print("  没有这个会话")
                continue
            current = target
            print(f"  已切到 {current['id']} · {current['root']}")
            continue
        if text.startswith("/del"):
            parts = text.split(maxsplit=1)
            target = parts[1].strip() if len(parts) > 1 else current["id"]
            session = app.session(target)
            if not session:
                print("  没有这个会话")
                continue
            if session.get("state") == "busy":
                print("  该会话正在处理消息，稍后再删")
                continue
            app.delete_session(target)
            print(f"  已删除会话 {target}（{session['title']}）")
            if target == current["id"]:
                remaining = app.sessions()
                current = remaining[0] if remaining else app.new_session(Path(current["root"]))
                print(f"  已切到 {current['id']} · {current['root']}")
            continue
        if text == "/files":
            from .tools.base import ToolContext, build_registry
            from .tools.docs import DocReader
            reg = build_registry(ToolContext(Path(current["root"]), DocReader(Path(current["root"])),
                                             app.store, current["id"]))
            print(reg.invoke("list_docs", {}))
            continue
        if text == "/todo":
            print("\n".join(
                f"  [{'x' if done else ' '}] #{i} {item}"
                for i, item, done in app.store.list_todos(current["id"])) or "  （空）")
            continue
        if text == "/trace":
            events = app.trace_tail(current["id"], 12)
            if not events:
                print("  （本会话还没有事件）")
            for ev in events:
                print(f"  {ev['kind']:<9} {ev['ms']:>7.1f}ms  {str(ev['data'])[:110]}")
            continue

        try:
            answer = app.chat(current["id"], text)
        except ConfigError as exc:
            print(f"  错误：{exc}")
            continue
        print()
        _print_events(answer)
        print(f"助手 › {answer.reply}\n")
    print("再见。")
    return 0
