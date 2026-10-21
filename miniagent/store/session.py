# -*- coding: utf-8 -*-
"""会话持久化：sqlite3（标准库）。

为什么不是 JSON 文件：一次对话里消息、待办、记忆、摘要四张表要一起改，
sqlite 的原子性省掉了“写一半崩了”的恢复逻辑。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    root        TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'idle',   -- idle | busy
    summary     TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL DEFAULT '',
    tool_calls   TEXT,                          -- OpenAI 协议的 tool_calls JSON
    tool_call_id TEXT,
    name         TEXT,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS todos (
    session_id TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    item       TEXT NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, idx)
);
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    content    TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(session_id, id);
"""


class SessionStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """清理历史残留：早期版本给 sessions 加过 effort 列，已废弃，直接删掉。

        SQLite 3.35+ 才支持 DROP COLUMN；低版本忽略即可（多一列不影响任何逻辑）。
        """
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(sessions)")}
        if "effort" in cols:
            try:
                self.conn.execute("ALTER TABLE sessions DROP COLUMN effort")
            except sqlite3.OperationalError:
                pass

    def close(self) -> None:
        self.conn.close()

    # ---------------- 会话 ---------------- #
    def create_session(self, root: str, title: str = "") -> dict:
        now = time.time()
        sid = uuid.uuid4().hex[:12]
        # 标题默认取路径末两段：一堆同名 docs 目录在侧栏里根本分不清
        if not title:
            parts = [p for p in Path(root).parts if p not in ("\\", "/")]
            title = "/".join(parts[-2:]) if len(parts) >= 2 else (Path(root).name or "session")
        self.conn.execute(
            "INSERT INTO sessions(id,title,root,state,summary,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (sid, title, root, "idle", "", now, now))
        self.conn.commit()
        return self.get_session(sid) or {}

    def list_sessions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id,title,root,state,updated_at,"
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS msg_count "
            "FROM sessions s ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_session(self, sid: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return dict(row) if row else None

    def touch(self, sid: str) -> None:
        self.conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), sid))
        self.conn.commit()

    def set_state(self, sid: str, state: str) -> None:
        self.conn.execute("UPDATE sessions SET state=?, updated_at=? WHERE id=?",
                          (state, time.time(), sid))
        self.conn.commit()

    def set_root(self, sid: str, root: str) -> None:
        self.conn.execute("UPDATE sessions SET root=?, updated_at=? WHERE id=?",
                          (root, time.time(), sid))
        self.conn.commit()

    def distinct_roots(self, limit: int = 8) -> list[str]:
        """最近用过的根目录，给 Web 的"最近"列表用。"""
        rows = self.conn.execute(
            "SELECT root, MAX(updated_at) AS t FROM sessions GROUP BY root"
            " ORDER BY t DESC LIMIT ?", (limit,)).fetchall()
        return [r["root"] for r in rows]

    def set_summary(self, sid: str, summary: str) -> None:
        self.conn.execute("UPDATE sessions SET summary=?, updated_at=? WHERE id=?",
                          (summary, time.time(), sid))
        self.conn.commit()

    def delete_session(self, sid: str) -> None:
        for table in ("messages", "todos", "memories", "sessions"):
            col = "id" if table == "sessions" else "session_id"
            self.conn.execute(f"DELETE FROM {table} WHERE {col}=?", (sid,))
        self.conn.commit()

    # ---------------- 消息 ---------------- #
    def append_message(self, sid: str, role: str, content: str = "",
                       tool_calls: list[dict] | None = None,
                       tool_call_id: str | None = None,
                       name: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages(session_id,role,content,tool_calls,tool_call_id,name,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (sid, role, content, json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
             tool_call_id, name, time.time()))
        self.conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), sid))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def messages(self, sid: str) -> list[dict]:
        """返回 OpenAI 协议格式的消息列表（含工具调用对）。"""
        rows = self.conn.execute(
            "SELECT role,content,tool_calls,tool_call_id,name FROM messages"
            " WHERE session_id=? ORDER BY id", (sid,)).fetchall()
        out: list[dict] = []
        for r in rows:
            msg: dict = {"role": r["role"], "content": r["content"]}
            if r["tool_calls"]:
                msg["tool_calls"] = json.loads(r["tool_calls"])
            if r["tool_call_id"]:
                msg["tool_call_id"] = r["tool_call_id"]
            if r["name"]:
                msg["name"] = r["name"]
            out.append(msg)
        return out

    def message_count(self, sid: str) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?",
                                (sid,)).fetchone()
        return int(row[0]) if row else 0

    # ---------------- 待办 ---------------- #
    def add_todo(self, sid: str, item: str) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(idx),0) FROM todos WHERE session_id=?",
                                (sid,)).fetchone()
        idx = int(row[0]) + 1
        self.conn.execute("INSERT INTO todos(session_id,idx,item,done) VALUES(?,?,?,0)",
                          (sid, idx, item))
        self.conn.commit()
        return idx

    def list_todos(self, sid: str) -> list[tuple[int, str, bool]]:
        rows = self.conn.execute("SELECT idx,item,done FROM todos WHERE session_id=? ORDER BY idx",
                                 (sid,)).fetchall()
        return [(int(r["idx"]), r["item"], bool(r["done"])) for r in rows]

    def done_todo(self, sid: str, idx: int) -> bool:
        cur = self.conn.execute("UPDATE todos SET done=1 WHERE session_id=? AND idx=?",
                                (sid, idx))
        self.conn.commit()
        return cur.rowcount > 0

    # ---------------- 长期记忆 ---------------- #
    def add_memory(self, sid: str, content: str, tags: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO memories(session_id,content,tags,created_at) VALUES(?,?,?,?)",
            (sid, content, tags, time.time()))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def list_memories(self, sid: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id,content,tags,created_at FROM memories WHERE session_id=? ORDER BY id",
            (sid,)).fetchall()
        return [dict(r) for r in rows]
