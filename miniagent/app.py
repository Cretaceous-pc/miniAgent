# -*- coding: utf-8 -*-
"""应用门面：把 LLM / 工具 / 上下文 / 会话串起来，供 CLI 与 Web 共用。

两个入口只做“输入输出”，业务全在这里——所以 CLI 和 Web 的行为天然一致。
"""
from __future__ import annotations

from pathlib import Path

from .config import Config, ConfigError
from .context.manager import ContextManager, LLMSummarizer
from .llm import LLMClient
from .runtime.loop import AgentLoop, Answer
from .store.session import SessionStore
from .tools.base import ToolContext, build_registry
from .tools.docs import DocReader
from .trace import TraceWriter


class App:
    def __init__(self, cfg: Config, root: Path, llm=None, store: SessionStore | None = None,
                 summarizer=None):
        self.cfg = cfg
        self.cfg.ensure_dirs()
        self.root = Path(root).resolve()
        # 先建 LLM（缺 key 时立刻失败，别白建数据库）
        self.llm = llm if llm is not None else self._build_llm()
        self.store = store or SessionStore(self.cfg.data_dir / "miniagent.db")
        self.trace = TraceWriter(self.cfg.data_dir / "traces")
        if summarizer is None and getattr(self.llm, "is_real", True):
            summarizer = LLMSummarizer(self.llm)
        self.ctx = ContextManager(
            self.store,
            budget_tokens=cfg.budget_tokens,
            keep_recent=cfg.keep_recent,
            recall_top_k=cfg.recall_top_k,
            summarizer=summarizer,
        )
        self._registry_cache: dict[str, object] = {}

    # ------------------------------------------------------------ #
    def _build_llm(self):
        if not self.cfg.api_key:
            raise ConfigError(
                "没有 API key。请在项目根目录建 .env：\n"
                "  MINIAGENT_API_KEY=sk-xxx\n"
                "  MINIAGENT_BASE_URL=https://api.deepseek.com/v1\n"
                "  MINIAGENT_MODEL=deepseek-chat\n"
                "（或加 --fake 用脚本模式跑通链路）")
        return LLMClient(self.cfg.base_url, self.cfg.api_key, self.cfg.model,
                         timeout=self.cfg.request_timeout,
                         max_retries=self.cfg.max_retries)

    # ------------------------------------------------------------ #
    def new_session(self, root: Path | str | None = None, title: str = "") -> dict:
        target = Path(root).resolve() if root else self.root
        if not target.exists():
            raise ConfigError(f"根目录不存在：{target}")
        return self.store.create_session(str(target), title)

    def sessions(self) -> list[dict]:
        return self.store.list_sessions()

    def session(self, sid: str) -> dict | None:
        return self.store.get_session(sid)

    def messages(self, sid: str) -> list[dict]:
        return self.store.messages(sid)

    # ------------------------------------------------------------ #
    def delete_session(self, sid: str) -> bool:
        """删除会话及其消息/待办/记忆（外键在 store 里手动级联）。"""
        if self.store.get_session(sid) is None:
            raise ConfigError(f"会话不存在：{sid}")
        self.store.delete_session(sid)
        return True

    def recent_roots(self) -> list[str]:
        roots = self.store.distinct_roots()
        if str(self.root) not in roots:
            roots.insert(0, str(self.root))
        return roots

    def set_root(self, sid: str, root: Path | str) -> dict:
        """切换会话的工作目录。

        取舍：允许在会话中途切换（历史消息保留），因为“换一批文档接着问”
        是这个工具最自然的用法；旧回答里引用的仍是旧目录的文件，故不重写历史。
        """
        target = Path(root).expanduser()
        if not target.exists():
            raise ConfigError(f"目录不存在：{target}")
        if not target.is_dir() and not target.is_file():
            raise ConfigError(f"既不是目录也不是文件：{target}")
        self.store.set_root(sid, str(target.resolve()))
        return self.store.get_session(sid) or {}

    def browse(self, path: Path | str | None) -> dict:
        """给 Web 的目录选择器：列出子目录 + 当前目录的文档数。"""
        from .tools.docs import SUPPORTED_EXTS, iter_files

        base = Path(path).expanduser() if path else self.root
        try:
            base = base.resolve()
        except OSError as exc:
            raise ConfigError(f"路径无法解析：{exc}") from exc
        if not base.exists():
            raise ConfigError(f"目录不存在：{base}")
        if not base.is_dir():
            raise ConfigError(f"不是目录：{base}")

        dirs: list[dict] = []
        try:
            children = sorted(base.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            raise ConfigError(f"没有权限读取：{base}")
        for child in children:
            if not child.is_dir() or child.name.startswith("."):
                continue
            if child.name in {".git", "node_modules", "__pycache__", ".venv", "venv"}:
                continue
            dirs.append({"name": child.name, "path": str(child)})

        files = iter_files(base)
        doc_count = len([f for f in files if f.suffix.lower() in SUPPORTED_EXTS])
        at_this_level = len([f for f in base.iterdir()
                             if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]) \
            if files else 0
        return {
            "path": str(base),
            "parent": str(base.parent) if base.parent != base else None,
            "dirs": dirs,
            "doc_count": doc_count,          # 递归（限深 4 / 上限 200）
            "local_docs": at_this_level,     # 本层
        }

    # ------------------------------------------------------------ #
    def _loop_for(self, sid: str) -> AgentLoop:
        session = self.store.get_session(sid)
        if session is None:
            raise ConfigError(f"会话不存在：{sid}")
        root = Path(session["root"])
        registry = build_registry(ToolContext(
            root=root,
            reader=DocReader(root),
            store=self.store,
            session_id=sid,
        ))
        return AgentLoop(self.llm, self.store, self.ctx, registry, sid, root,
                         trace=self.trace, max_rounds=self.cfg.max_rounds)

    def chat(self, sid: str, text: str) -> Answer:
        """一次完整交互。会话状态机：busy 期间不接受新请求（Web 并发保护）。"""
        session = self.store.get_session(sid)
        if session is None:
            raise ConfigError(f"会话不存在：{sid}")
        if not text.strip():
            raise ConfigError("消息不能为空")
        if session["state"] == "busy":
            return Answer("上一条消息还在处理中，请稍等。", stopped="busy")
        self.store.set_state(sid, "busy")
        try:
            return self._loop_for(sid).ask(text.strip())
        finally:
            self.store.set_state(sid, "idle")

    def trace_tail(self, sid: str, limit: int = 30) -> list[dict]:
        return self.trace.tail(sid, limit)
