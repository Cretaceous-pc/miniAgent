# -*- coding: utf-8 -*-
"""Web 入口：标准库 http.server，零依赖。

刻意不用 FastAPI/Flask：这不是“能不能装”，而是让 Web 层和 Runtime 一样可被完整阅读。
接口只有 5 个，全部返回 JSON，前端是一个静态页。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..app import App
from ..config import ConfigError

STATIC = Path(__file__).parent


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "miniAgent/0.1"

        # ---------------- 基础工具 ---------------- #
        def _send(self, status: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict | list):
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}

        def log_message(self, fmt, *args):        # 静音，避免灌满控制台
            return

        # ---------------- GET ---------------- #
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                html = (STATIC / "index.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
                return
            if path == "/api/sessions":
                self._json(200, {"sessions": app.sessions()})
                return
            if path == "/api/roots":
                self._json(200, {"roots": app.recent_roots()})
                return
            if path == "/api/browse":
                query = parse_qs(urlparse(self.path).query)
                target = (query.get("path") or [""])[0]
                try:
                    self._json(200, app.browse(target or None))
                except ConfigError as exc:
                    self._json(400, {"error": str(exc)})
                return
            if path.startswith("/api/sessions/"):
                parts = path.strip("/").split("/")
                sid = parts[2] if len(parts) > 2 else ""
                if len(parts) == 4 and parts[3] == "messages":
                    self._json(200, {"messages": app.messages(sid),
                                     "session": app.session(sid)})
                    return
                if len(parts) == 4 and parts[3] == "trace":
                    self._json(200, {"events": app.trace_tail(sid, 40)})
                    return
                if len(parts) == 3:
                    session = app.session(sid)
                    if not session:
                        self._json(404, {"error": "会话不存在"})
                        return
                    self._json(200, {"session": session})
                    return
            self._json(404, {"error": "not found"})

        def do_DELETE(self):
            path = urlparse(self.path).path
            parts = path.strip("/").split("/")
            # DELETE /api/sessions/<id>
            if len(parts) == 3 and parts[:2] == ["api", "sessions"]:
                sid = parts[2]
                session = app.session(sid)
                if session is None:
                    self._json(404, {"error": "会话不存在"})
                    return
                if session.get("state") == "busy":
                    # 正在跑对话时删会留下孤儿消息，直接拒绝
                    self._json(409, {"error": "该会话正在处理消息，稍后再删"})
                    return
                try:
                    app.delete_session(sid)
                except ConfigError as exc:
                    self._json(404, {"error": str(exc)})
                    return
                self._json(200, {"ok": True, "deleted": sid})
                return
            self._json(404, {"error": "not found"})

        # ---------------- POST ---------------- #
        def do_POST(self):
            path = urlparse(self.path).path
            data = self._body()
            try:
                if path == "/api/sessions":
                    root = (data.get("root") or "").strip()
                    session = app.new_session(root or None, data.get("title") or "")
                    self._json(200, {"session": session})
                    return
                # POST /api/sessions/<id>/root —— 在 Web 里切换工作目录
                if path.startswith("/api/sessions/") and path.endswith("/root"):
                    parts = path.strip("/").split("/")
                    sid = parts[2] if len(parts) > 2 else ""
                    root = (data.get("root") or "").strip()
                    if not root:
                        self._json(400, {"error": "需要 root 字段"})
                        return
                    session = app.set_root(sid, root)
                    self._json(200, {"session": session})
                    return
                if path == "/api/chat":
                    sid = (data.get("session_id") or "").strip()
                    text = (data.get("message") or "").strip()
                    if not sid or not text:
                        self._json(400, {"error": "需要 session_id 和 message"})
                        return
                    answer = app.chat(sid, text)
                    self._json(200, {
                        "reply": answer.reply,
                        "stopped": answer.stopped,
                        "rounds": answer.rounds,
                        "events": answer.events_as_dicts(),
                    })
                    return
            except ConfigError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:                  # 兜底，避免浏览器一直转圈
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(404, {"error": "not found"})

    return Handler


def serve(root: str, host: str = "127.0.0.1", port: int = 8000, fake: bool = False) -> int:
    from ..config import Config

    cfg = Config.from_env()
    llm = None
    if fake:
        from ..demo_llm import DemoLLM
        llm = DemoLLM()
    app = App(cfg, Path(root), llm=llm)
    session = app.new_session(Path(root))
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}/"
    print(f"miniAgent Web 已启动：{url}")
    print(f"  默认会话 {session['id']} · 根目录 {session['root']}")
    if fake:
        print("  （演示模式：DemoLLM，不消耗 API）")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\n正在停止…")
        httpd.shutdown()
    return 0
