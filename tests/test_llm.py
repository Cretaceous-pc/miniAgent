# -*- coding: utf-8 -*-
"""LLM 响应解析测试：思考路径、工具调用、最终答案、异常响应形态。

题目要求「提取思考过程、工具调用或最终答案」，这三条各对应一个断言。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniagent.llm import LLMClient, LLMError  # noqa: E402


def payload(message: dict, model: str = "deepseek-reasoner") -> dict:
    return {"id": "x", "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class ParseTests(unittest.TestCase):
    def test_extracts_reasoning_content(self):
        reply = LLMClient._parse(payload({
            "role": "assistant",
            "content": "结论：一线城市 600 元/晚。",
            "reasoning_content": "先看问题问的是住宿标准，需要检索 handbook.md 的报销一节。",
        }))
        self.assertEqual(reply.content, "结论：一线城市 600 元/晚。")
        self.assertIn("handbook.md", reply.reasoning)
        self.assertFalse(reply.has_tool_calls)

    def test_extracts_reasoning_alias(self):
        """部分供应商用 reasoning 而不是 reasoning_content。"""
        reply = LLMClient._parse(payload({"role": "assistant", "content": "答案",
                                          "reasoning": "推理一下"}))
        self.assertEqual(reply.reasoning, "推理一下")

    def test_extracts_tool_calls(self):
        reply = LLMClient._parse(payload({
            "role": "assistant", "content": "",
            "reasoning_content": "该先看目录",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "list_docs", "arguments": '{"path": "."}'}}],
        }))
        self.assertTrue(reply.has_tool_calls)
        self.assertEqual(reply.tool_calls[0].name, "list_docs")
        self.assertEqual(reply.tool_calls[0].parsed_args(), {"path": "."})
        self.assertEqual(reply.tool_calls[0].id, "call_1")

    def test_content_as_block_list(self):
        reply = LLMClient._parse(payload({
            "role": "assistant",
            "content": [{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}],
        }))
        self.assertEqual(reply.content, "第一段第二段")

    def test_missing_reasoning_is_empty_string(self):
        reply = LLMClient._parse(payload({"role": "assistant", "content": "普通回答"}))
        self.assertEqual(reply.reasoning, "")

    def test_usage_is_kept(self):
        reply = LLMClient._parse(payload({"role": "assistant", "content": "x"}))
        self.assertEqual(reply.usage["total_tokens"], 15)

    def test_malformed_response_raises(self):
        with self.assertRaises(LLMError):
            LLMClient._parse({"choices": []})

    def test_bad_json_arguments_fall_back_to_empty(self):
        reply = LLMClient._parse(payload({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c", "function": {"name": "read_doc", "arguments": "{oops"}}]}))
        self.assertEqual(reply.tool_calls[0].parsed_args(), {})


if __name__ == "__main__":
    unittest.main()
