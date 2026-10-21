# -*- coding: utf-8 -*-
"""入口分发：`python main.py cli|web` 与 console script 共用同一份 argparse。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="miniagent",
        description="miniAgent：从零实现的最小 Agent Runtime（文档问答），CLI / Web 双入口")
    sub = ap.add_subparsers(dest="mode", required=True)

    cli = sub.add_parser("cli", help="命令行交互模式")
    cli.add_argument("--root", default="examples/docs", help="文档根目录（默认 examples/docs）")
    cli.add_argument("--fake", action="store_true", help="用 DemoLLM 演示循环，不需要 API key")
    cli.add_argument("--once", help="只跑一次就退出（便于脚本/验收）")
    cli.add_argument("--session", help="复用已有会话 id")

    web = sub.add_parser("web", help="Web 交互模式")
    web.add_argument("--root", default="examples/docs", help="默认根目录")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--fake", action="store_true", help="用 DemoLLM 演示循环，不需要 API key")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    if not root.exists():
        print(f"根目录不存在：{root.resolve()}", file=sys.stderr)
        return 2

    if args.mode == "cli":
        from .cli import run_cli
        return run_cli(args.root, fake=args.fake, once=args.once, session_id=args.session)

    from .web.server import serve
    return serve(args.root, host=args.host, port=args.port, fake=args.fake)
