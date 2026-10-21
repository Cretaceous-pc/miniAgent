# -*- coding: utf-8 -*-
"""主循环测试：四个出口 + 工具结果回填 + 协议配对。

全部用 FakeLLM 驱动，不联网、不花钱、结果确定。
"""
from __future__ import annotations

import json
import unittest

from support import TempCase, call, call_many, say


class LoopTests(TempCase):
    def test_direct_answer_needs_one_round(self):
        app = self.make_app([say("文档里有 handbook.md 和 api.md")])
        sid = app.new_session()["id"]
        answer = app.chat(sid, "有哪些文档？")
        self.assertEqual(answer.rounds, 1)
        self.assertEqual(answer.stopped, "final")
        self.assertIn("handbook.md", answer.reply)
        self.assertEqual([e.kind for e in answer.events], ["llm"])

    def test_single_tool_call_then_answer(self):
        app = self.make_app([
            call("list_docs", {"path": "."}),
            say("一共 3 个文档"),
        ])
        sid = app.new_session()["id"]
        answer = app.chat(sid, "目录里有什么？")
        self.assertEqual(answer.stopped, "final")
        self.assertEqual(answer.rounds, 2)
        kinds = [e.kind for e in answer.events]
        self.assertEqual(kinds, ["llm", "tool", "llm"])

        # 协议配对：assistant.tool_calls 里的 id 必须出现在随后的 tool 消息里
        messages = self.store.messages(sid)
        assistant = [m for m in messages if m.get("tool_calls")][0]
        call_id = assistant["tool_calls"][0]["id"]
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], call_id)
        self.assertIn("handbook.md", tool_msgs[0]["content"])

    def test_multi_tool_in_one_round(self):
        app = self.make_app([
            call_many([("calculator", {"expression": "3*4"}),
                       ("todo", {"action": "add", "item": "算完再写结论"})]),
            say("12"),
        ])
        sid = app.new_session()["id"]
        answer = app.chat(sid, "3 乘 4 是多少？")
        tool_events = [e for e in answer.events if e.kind == "tool"]
        self.assertEqual(len(tool_events), 2)
        self.assertIn("12", tool_events[0].data["result"])
        self.assertEqual(len(self.store.list_todos(sid)), 1)

    def test_tool_error_is_fed_back_not_raised(self):
        app = self.make_app([
            call("read_doc", {"path": "../../etc/passwd"}),
            say("那个路径读不了，我换 handbook.md"),
        ])
        sid = app.new_session()["id"]
        answer = app.chat(sid, "读一下 ../../etc/passwd")
        tool_msg = [m for m in self.store.messages(sid) if m["role"] == "tool"][0]
        self.assertTrue(tool_msg["content"].startswith("ERROR"), tool_msg["content"])
        self.assertEqual(answer.stopped, "final")
        self.assertIn("换 handbook.md", answer.reply)

    def test_max_rounds_is_a_hard_stop(self):
        app = self.make_app([call("list_docs", {})] * 10, max_rounds=2)
        sid = app.new_session()["id"]
        answer = app.chat(sid, "一直列目录")
        self.assertEqual(answer.stopped, "max_rounds")
        self.assertIn("最大工具轮次", answer.reply)

    def test_repeat_guard_blocks_same_call(self):
        same = call("search_docs", {"query": "报销", "top_k": 3})
        app = self.make_app([same, same, same, same, same, say("算了，直接回答")],
                            max_rounds=5)
        sid = app.new_session()["id"]
        answer = app.chat(sid, "重复调用")
        guards = [e for e in answer.events if e.kind == "guard"]
        self.assertTrue(guards, "第 4 次同参调用应触发 guard")
        self.assertEqual(guards[0].data["tool"], "search_docs")
        self.assertGreaterEqual(guards[0].data["count"], 4)

    def test_context_rebuild_between_rounds(self):
        """第二轮装配时必须能看到第一轮的工具结果——否则模型无法基于结果作答。"""
        app = self.make_app([call("list_docs", {}), say("看到了")])
        sid = app.new_session()["id"]
        app.chat(sid, "看目录")
        second_call = self.llm.calls[-1]
        self.assertTrue(any(m["role"] == "tool" for m in second_call),
                        "第二轮上下文里缺少工具结果")
        self.assertEqual(second_call[0]["role"], "system")

    def test_busy_session_rejects_new_message(self):
        app = self.make_app([say("ok")])
        sid = app.new_session()["id"]
        self.store.set_state(sid, "busy")
        answer = app.chat(sid, "现在问")
        self.assertEqual(answer.stopped, "busy")
        self.assertIn("还在处理中", answer.reply)

    def test_reasoning_is_captured_but_never_echoed_back(self):
        """思考路径要能取到（事件 + trace），但绝不能回填进下一次请求。

        DeepSeek reasoner 明确要求 reasoning_content 不出现在输入的 messages 里，否则 400。
        """
        app = self.make_app([
            {"content": "", "reasoning": "先列目录，再读 handbook",
             "tool_calls": [{"name": "list_docs", "args": {}, "id": "c1"}]},
            {"content": "看完了", "reasoning": "直接回答即可"},
        ])
        sid = app.new_session()["id"]
        answer = app.chat(sid, "看看目录")

        # 1) 事件里能拿到思考路径
        reasonings = [e.data.get("reasoning") for e in answer.events if e.kind == "llm"]
        self.assertEqual(reasonings, ["先列目录，再读 handbook", "直接回答即可"])

        # 2) 落库的 assistant 消息里不含 reasoning 内容
        for msg in self.store.messages(sid):
            self.assertNotIn("reasoning", msg)
            self.assertNotIn("先列目录", msg.get("content") or "")

        # 3) 第二轮请求体里也不能出现 reasoning
        for msg in self.llm.calls[-1]:
            self.assertNotIn("reasoning_content", msg)
            self.assertNotIn("reasoning", msg)


if __name__ == "__main__":
    unittest.main()
