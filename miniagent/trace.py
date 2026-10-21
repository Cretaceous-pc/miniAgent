# -*- coding: utf-8 -*-
"""调用链追踪：每条事件一行 JSONL，落盘到 `.miniagent/traces/<session>.jsonl`。

为什么值得单独做：面试时“你怎么知道模型为什么这么答”比“能答”更能说明工程能力。
事件类型：llm / tool / compress / error / final。
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path


class TraceWriter:
    def __init__(self, directory: Path, memory_size: int = 200):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.memory: deque[dict] = deque(maxlen=memory_size)

    def write(self, session_id: str, kind: str, payload: dict, ms: float = 0.0) -> dict:
        event = {
            "ts": round(time.time(), 3),
            "session": session_id,
            "kind": kind,
            "ms": round(ms, 1),
            "data": payload,
        }
        self.memory.append(event)
        path = self.dir / f"{session_id}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass                       # trace 写失败不该影响主流程
        return event

    def tail(self, session_id: str, limit: int = 30) -> list[dict]:
        path = self.dir / f"{session_id}.jsonl"
        if not path.is_file():
            return [e for e in self.memory if e["session"] == session_id][-limit:]
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
