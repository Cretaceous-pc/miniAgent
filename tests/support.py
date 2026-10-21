# -*- coding: utf-8 -*-
"""公共测试夹具：临时文档目录 + 临时数据库 + FakeLLM。"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniagent.app import App                      # noqa: E402
from miniagent.config import Config                # noqa: E402
from miniagent.context.manager import ContextManager, mechanical_summary  # noqa: E402
from miniagent.fake_llm import FakeLLM, call, call_many, say  # noqa: E402
from miniagent.store.session import SessionStore   # noqa: E402
from miniagent.tools.base import ToolContext, build_registry  # noqa: E402
from miniagent.tools.docs import DocReader         # noqa: E402

SAMPLE = ROOT / "examples" / "docs"


class TempCase(unittest.TestCase):
    """每个用例一份独立目录 + 独立 sqlite，避免互相污染。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="miniagent-test-"))
        self.docs = self.tmp / "docs"
        shutil.copytree(SAMPLE, self.docs)
        self.cfg = Config(api_key="", data_dir=self.tmp / "data", context_window=8000,
                          budget_ratio=0.6, max_rounds=6, keep_recent=10)

    def tearDown(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- 便捷构造 ---- #
    def make_app(self, script=None, **cfg_overrides) -> App:
        cfg = Config(api_key="", data_dir=self.tmp / "data",
                     context_window=self.cfg.context_window,
                     budget_ratio=self.cfg.budget_ratio,
                     max_rounds=self.cfg.max_rounds,
                     keep_recent=self.cfg.keep_recent)
        for key, value in cfg_overrides.items():
            setattr(cfg, key, value)
        self.llm = FakeLLM(script or [])
        self.store = SessionStore(cfg.data_dir / "miniagent.db")
        return App(cfg, self.docs, llm=self.llm, store=self.store)

    def registry(self, session_id: str = "s1"):
        self.store = getattr(self, "store", None) or SessionStore(self.cfg.data_dir / "miniagent.db")
        ctx = ToolContext(self.docs, DocReader(self.docs), self.store, session_id)
        return build_registry(ctx)
