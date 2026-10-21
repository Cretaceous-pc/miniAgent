# -*- coding: utf-8 -*-
"""工具层测试：路径沙箱、检索、按行读、计算器白名单、待办与记忆。"""
from __future__ import annotations

import unittest
import zipfile

from support import TempCase


class ToolTests(TempCase):
    def setUp(self) -> None:
        super().setUp()
        self.reg = self.registry("s-tools")

    def test_list_docs_lists_sample_files(self):
        out = self.reg.invoke("list_docs", {"path": "."})
        self.assertIn("handbook.md", out)
        self.assertIn("api.md", out)
        self.assertIn("共 3 个文档", out)

    def test_outline_gives_headings_with_line_numbers(self):
        out = self.reg.invoke("outline", {"path": "handbook.md"})
        self.assertIn("L1", out)
        self.assertIn("考勤", out)
        self.assertIn("二、休假", out)

    def test_search_finds_cross_file_hits(self):
        out = self.reg.invoke("search_docs", {"query": "报销", "top_k": 3})
        self.assertIn("handbook.md:", out)
        self.assertIn("命中", out)

    def test_read_doc_has_line_numbers_and_clamps(self):
        out = self.reg.invoke("read_doc", {"path": "handbook.md", "start": 1, "end": 5})
        self.assertIn("第 1-5 行", out)
        self.assertIn("| # 员工手册", out)
        huge = self.reg.invoke("read_doc", {"path": "handbook.md", "start": 9999, "end": 99999})
        self.assertIn("共", huge)
        self.assertNotIn("ERROR", huge)

    def test_path_escape_is_blocked(self):
        out = self.reg.invoke("read_doc", {"path": "../../secret.txt"})
        self.assertTrue(out.startswith("ERROR"), out)
        self.assertIn("越界", out)

    def test_missing_file_and_unsupported_ext(self):
        out = self.reg.invoke("read_doc", {"path": "nope.md"})
        self.assertIn("不存在", out)
        (self.docs / "x.pdf").write_bytes(b"%PDF-1.4 fake")
        out = self.reg.invoke("read_doc", {"path": "x.pdf"})
        self.assertIn("不支持", out)

    def test_calculator_arithmetic(self):
        self.assertIn("14", self.reg.invoke("calculator", {"expression": "2+3*4"}))
        self.assertIn("2", self.reg.invoke("calculator", {"expression": "avg([1,2,3])"}))
        self.assertIn("6", self.reg.invoke("calculator", {"expression": "len([1,2,3,4,5,6])"}))

    def test_calculator_rejects_code_injection(self):
        out = self.reg.invoke("calculator", {"expression": "__import__('os').system('echo hi')"})
        self.assertTrue(out.startswith("ERROR"), out)
        out = self.reg.invoke("calculator", {"expression": "os.system('dir')"})
        self.assertTrue(out.startswith("ERROR"), out)

    def test_todo_lifecycle(self):
        self.assertIn("#1", self.reg.invoke("todo", {"action": "add", "item": "读 handbook"}))
        listing = self.reg.invoke("todo", {"action": "list"})
        self.assertIn("读 handbook", listing)
        self.assertIn("[ ]", listing)
        self.assertIn("已勾选", self.reg.invoke("todo", {"action": "done", "index": 1}))
        self.assertIn("[x]", self.reg.invoke("todo", {"action": "list"}))

    def test_remember_persists(self):
        self.assertIn("已记住", self.reg.invoke("remember", {"content": "只回答 handbook 范围内的制度问题"}))
        memories = self.store.list_memories("s-tools")
        self.assertEqual(len(memories), 1)
        self.assertIn("handbook", memories[0]["content"])

    def test_schema_validation_reports_readable_error(self):
        out = self.reg.invoke("read_doc", {"pth": "handbook.md"})
        self.assertIn("未知参数", out)
        out = self.reg.invoke("todo", {"action": "wat"})
        self.assertIn("只能是", out)
        out = self.reg.invoke("read_doc", "{bad json")
        self.assertIn("合法 JSON", out)

    def test_unknown_tool_lists_available(self):
        out = self.reg.invoke("no_such_tool", {})
        self.assertIn("没有名为 no_such_tool 的工具", out)
        self.assertIn("list_docs", out)

    def test_docx_read_via_stdlib(self):
        """docx = zip + word/document.xml，不依赖 python-docx 也能读。"""
        path = self.docs / "spec.docx"
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body>'
            '<w:p><w:r><w:t>第一段：接口超时设置</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>第二段：默认 30 秒</w:t></w:r></w:p>'
            '</w:body></w:document>')
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("word/document.xml", xml)
        out = self.reg.invoke("read_doc", {"path": "spec.docx"})
        self.assertIn("第一段", out)
        self.assertIn("默认 30 秒", out)


if __name__ == "__main__":
    unittest.main()
