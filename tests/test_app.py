# -*- coding: utf-8 -*-
"""应用层测试：会话隔离、配置错误、trace 落盘、Web 接口冒烟。"""
from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from support import TempCase, say

from miniagent.config import Config, ConfigError
from miniagent.web.server import make_handler


class AppTests(TempCase):
    def test_sessions_are_isolated(self):
        app = self.make_app([say("一"), say("二"), say("三")])
        a = app.new_session()["id"]
        b = app.new_session()["id"]
        self.assertNotEqual(a, b)

        app.chat(a, "会话 A 的问题")
        self.assertEqual(len(app.messages(b)), 0, "B 会话不应看到 A 的消息")

        app.chat(b, "会话 B 的问题")
        b_msgs = app.messages(b)
        self.assertEqual(len(b_msgs), 2)
        self.assertIn("会话 B", b_msgs[0]["content"])
        self.assertNotIn("会话 A", json.dumps(b_msgs, ensure_ascii=False))

    def test_trace_is_written_to_disk(self):
        app = self.make_app([say("好")])
        sid = app.new_session()["id"]
        app.chat(sid, "你好")
        trace_file = self.cfg.data_dir / "traces" / f"{sid}.jsonl"
        self.assertTrue(trace_file.is_file(), "trace 文件应落盘")
        kinds = [json.loads(line)["kind"] for line in
                 trace_file.read_text(encoding="utf-8").splitlines()]
        self.assertIn("llm", kinds)
        self.assertIn("final", kinds)

    def test_missing_root_raises_config_error(self):
        app = self.make_app([say("x")])
        with self.assertRaises(ConfigError):
            app.new_session(self.tmp / "not-exists")

    def test_missing_api_key_message_is_actionable(self):
        """不传 llm 且没有 key 时，报错要告诉用户怎么修。"""
        from miniagent.app import App
        with self.assertRaises(ConfigError) as ctx:
            App(Config(api_key="", data_dir=self.tmp / "d2"), self.docs)
        self.assertIn("MINIAGENT_API_KEY", str(ctx.exception))

    def test_app_delete_session_cascades(self):
        app = self.make_app([say("好")])
        sid = app.new_session()["id"]
        app.chat(sid, "你好")
        app.store.add_todo(sid, "记一条待办")
        app.store.add_memory(sid, "记一条长期事实")

        self.assertTrue(app.delete_session(sid))
        self.assertIsNone(app.session(sid))
        self.assertEqual(app.store.list_todos(sid), [])
        self.assertEqual(app.store.list_memories(sid), [])

    def test_app_delete_missing_session_raises(self):
        app = self.make_app([say("好")])
        with self.assertRaises(ConfigError):
            app.delete_session("ghost123456")

    def test_trace_tail_returns_recent_events(self):
        app = self.make_app([say("好")])
        sid = app.new_session()["id"]
        app.chat(sid, "你好")
        tail = app.trace_tail(sid, 5)
        self.assertTrue(tail)
        self.assertEqual(tail[-1]["kind"], "final")


class WebTests(TempCase):
    def setUp(self) -> None:
        super().setUp()
        self.app = self.make_app([say("这是回答")])
        self.sid = self.app.new_session()["id"]
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        super().tearDown()

    def _get(self, path: str) -> tuple[int, str]:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
            return r.status, r.read().decode("utf-8")

    def _delete(self, path: str) -> tuple[int, dict]:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method="DELETE")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def test_index_page_served(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("miniAgent", body)
        self.assertIn("文档问答", body)

    def test_sessions_api(self):
        status, body = self._get("/api/sessions")
        self.assertEqual(status, 200)
        self.assertIn(self.sid, body)

    def test_chat_endpoint_runs_the_loop(self):
        status, payload = self._post("/api/chat", {"session_id": self.sid, "message": "你好"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["reply"], "这是回答")
        self.assertEqual(payload["stopped"], "final")
        self.assertTrue(any(e["kind"] == "llm" for e in payload["events"]))

    def test_chat_without_message_is_400(self):
        try:
            self._post("/api/chat", {"session_id": self.sid, "message": ""})
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            return
        self.fail("空消息应当返回 400")

    # ---- 工作目录切换（Web 目录选择器） ---- #
    def test_roots_api_lists_recent_roots(self):
        status, body = self._get("/api/roots")
        self.assertEqual(status, 200)
        self.assertIn(str(self.docs.resolve()).replace("\\", "\\\\"), body)

    def test_browse_lists_subdirs_and_doc_count(self):
        (self.docs / "sub").mkdir()
        (self.docs / "sub" / "x.md").write_text("# x", encoding="utf-8")
        status, body = self._get("/api/browse?path=" + urllib.parse.quote(str(self.docs)))
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(any(d["name"] == "sub" for d in data["dirs"]))
        self.assertGreaterEqual(data["doc_count"], 4)
        self.assertGreaterEqual(data["local_docs"], 3)

    def test_browse_missing_path_is_400(self):
        try:
            self._get("/api/browse?path=" + urllib.parse.quote(str(self.docs / "nope")))
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            return
        self.fail("不存在的目录应当返回 400")

    def test_switch_root_updates_session(self):
        other = self.tmp / "other"
        other.mkdir()
        (other / "a.md").write_text("# a", encoding="utf-8")
        status, payload = self._post(f"/api/sessions/{self.sid}/root", {"root": str(other)})
        self.assertEqual(status, 200)
        self.assertEqual(payload["session"]["root"], str(other.resolve()))
        self.assertEqual(self.app.session(self.sid)["root"], str(other.resolve()))

    def test_switch_root_rejects_missing_dir(self):
        try:
            self._post(f"/api/sessions/{self.sid}/root", {"root": str(self.tmp / "ghost")})
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            return
        self.fail("不存在的目录应当返回 400")

    # ---- 删除会话 ---- #
    def test_delete_session_removes_messages(self):
        self._post("/api/chat", {"session_id": self.sid, "message": "你好"})
        self.assertGreater(len(self.app.messages(self.sid)), 0)

        status, payload = self._delete(f"/api/sessions/{self.sid}")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertIsNone(self.app.session(self.sid))
        self.assertEqual(self.app.messages(self.sid), [], "消息必须级联删除")
        _, body = self._get("/api/sessions")
        self.assertNotIn(self.sid, body)

    def test_delete_missing_session_is_404(self):
        try:
            self._delete("/api/sessions/nope123456")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)
            return
        self.fail("删除不存在的会话应当 404")

    def test_delete_busy_session_is_409(self):
        self.store.set_state(self.sid, "busy")
        try:
            self._delete(f"/api/sessions/{self.sid}")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 409)
            self.assertIsNotNone(self.app.session(self.sid), "被拒绝删除后会话应还在")
            return
        self.fail("busy 会话应当拒绝删除（409）")

    def test_new_session_with_root(self):
        other = self.tmp / "other2"
        other.mkdir()
        status, payload = self._post("/api/sessions", {"root": str(other)})
        self.assertEqual(status, 200)
        self.assertEqual(payload["session"]["root"], str(other.resolve()))
        self.assertNotEqual(payload["session"]["id"], self.sid)


if __name__ == "__main__":
    unittest.main()
