# -*- coding: utf-8 -*-
"""上下文层测试：轮组切分、超预算压缩、记忆召回。"""
from __future__ import annotations

import unittest

from support import TempCase

from miniagent.context.manager import (ContextManager, est_tokens, group_turns,
                                       mechanical_summary)
from miniagent.store.session import SessionStore


class ContextTests(TempCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = SessionStore(self.cfg.data_dir / "ctx.db")

    def test_group_turns_keeps_tool_pairs_together(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "list_docs", "arguments": "{}"}}]},
            {"role": "tool", "content": "a.md", "tool_call_id": "c1"},
            {"role": "assistant", "content": "ok"},
        ]
        groups = group_turns(messages)
        self.assertEqual(len(groups), 3)
        self.assertEqual(len(groups[1]), 2, "assistant(tool_calls) 与 tool 必须同组")

    def test_compression_triggers_and_keeps_recent(self):
        sid = self.store.create_session(str(self.docs))["id"]
        for i in range(24):
            self.store.append_message(sid, "user", f"第{i}个问题 " + "细节" * 30)
            self.store.append_message(sid, "assistant", f"第{i}个回答 " + "结论" * 30)
        self.store.append_message(sid, "user", "最后的问题：报销标准是什么？")

        ctx = ContextManager(self.store, budget_tokens=300, keep_recent=6)
        built = ctx.build(sid, str(self.docs))

        self.assertTrue(built.compressed, "预算 300 tokens 应触发压缩")
        self.assertGreater(built.dropped_messages, 0)
        self.assertTrue(self.store.get_session(sid)["summary"], "压缩后必须落摘要")
        joined = "\n".join(str(m.get("content")) for m in built.messages)
        self.assertIn("最后的问题：报销标准是什么？", joined, "最近一轮必须保留原文")

    def test_no_compression_when_within_budget(self):
        sid = self.store.create_session(str(self.docs))["id"]
        self.store.append_message(sid, "user", "短问题")
        self.store.append_message(sid, "assistant", "短回答")
        ctx = ContextManager(self.store, budget_tokens=10_000, keep_recent=10)
        built = ctx.build(sid, str(self.docs))
        self.assertFalse(built.compressed)
        self.assertEqual(built.dropped_messages, 0)

    def test_recall_injects_matched_memory(self):
        sid = self.store.create_session(str(self.docs))["id"]
        self.store.add_memory(sid, "报销标准看 handbook.md 第三节")
        self.store.add_memory(sid, "接口错误码查 api.md 的错误码表")
        self.store.append_message(sid, "user", "报销住宿标准是多少？")
        ctx = ContextManager(self.store, budget_tokens=10_000, keep_recent=10)
        built = ctx.build(sid, str(self.docs))
        recalled_text = "\n".join(m["content"] for m in built.messages
                                  if m["role"] == "system")
        self.assertIn("已记住的长期事实", recalled_text)
        self.assertIn("handbook.md", recalled_text)
        self.assertEqual(len(built.recalled), 1, "只召回相关的那一条")

    def test_mechanical_summary_is_deterministic(self):
        text = "## 结论\n- 报销 600\n- 年假 5 天\n"
        self.assertEqual(mechanical_summary(text), mechanical_summary(text))
        self.assertIn("结论", mechanical_summary(text))

    def test_est_tokens_grows_with_length(self):
        self.assertLess(est_tokens("短"), est_tokens("很长" * 100))
        self.assertGreater(est_tokens("中文中文中文"), est_tokens(""))

    def test_tool_schema_tokens_count_toward_budget(self):
        """工具 schema 每轮都随请求发出，占的是真金白银的上下文。

        实测：7 个工具 schema ≈ 1k tokens；预算里不算它，小窗口下会系统性低估，
        表现为"估算 300 / 实际 1354"。
        """
        sid = self.store.create_session(str(self.docs))["id"]
        self.store.append_message(sid, "user", "短问题")
        self.store.append_message(sid, "assistant", "短回答")
        ctx = ContextManager(self.store, budget_tokens=1_000, keep_recent=10)
        without = ctx.build(sid, str(self.docs))
        with_tools = ctx.build(sid, str(self.docs), extra_tokens=1_200)
        self.assertFalse(without.compressed, "不算工具 schema 时不该压缩")
        self.assertGreater(with_tools.tokens, without.tokens, "extra_tokens 必须计入预算")
        self.assertTrue(with_tools.compressed, "算上工具 schema 后应触发压缩")


if __name__ == "__main__":
    unittest.main()
