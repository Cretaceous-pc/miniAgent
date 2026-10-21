# -*- coding: utf-8 -*-
"""运行配置：环境变量 / .env / 命令行参数，优先级从低到高。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 默认指向 DeepSeek（OpenAI-compatible）；换 GLM / Qwen / Ollama 只改 base_url + model
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class ConfigError(Exception):
    """配置缺失/非法（例如没有 API key），属于用户可修复的错误。"""


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 读取：KEY=VALUE，# 注释，不覆盖已存在的环境变量。"""
    path = path or Path(".env")
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Config:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    # 上下文预算：模型窗口 × ratio，超出就走压缩
    context_window: int = 32_768
    budget_ratio: float = 0.6
    # 每轮最多几次模型往返（含最终作答），防死循环的硬闸
    max_rounds: int = 8
    # 组装上下文时保留多少条最近消息的原文
    keep_recent: int = 10
    # 长期记忆召回条数
    recall_top_k: int = 3
    request_timeout: float = 60.0
    max_retries: int = 2
    data_dir: Path = field(default_factory=lambda: Path(".miniagent"))

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        env = os.environ.get
        return cls(
            base_url=env("MINIAGENT_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            api_key=env("MINIAGENT_API_KEY", "") or env("DEEPSEEK_API_KEY", ""),
            model=env("MINIAGENT_MODEL", DEFAULT_MODEL),
            context_window=int(env("MINIAGENT_CONTEXT_WINDOW", "32768")),
            budget_ratio=float(env("MINIAGENT_BUDGET_RATIO", "0.6")),
            max_rounds=int(env("MINIAGENT_MAX_ROUNDS", "8")),
            keep_recent=int(env("MINIAGENT_KEEP_RECENT", "10")),
            data_dir=Path(env("MINIAGENT_DATA_DIR", ".miniagent")),
        )

    @property
    def budget_tokens(self) -> int:
        return int(self.context_window * self.budget_ratio)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "traces").mkdir(exist_ok=True)
