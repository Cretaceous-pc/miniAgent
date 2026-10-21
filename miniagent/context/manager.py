# -*- coding: utf-8 -*-
"""上下文装配：预算估算 + 按“轮组”裁剪 + 滚动摘要 + 记忆召回。

这是整个项目里最该被面试追问的一层，所以策略写得直白：

  装配顺序（固定，便于解释）
    [system]   角色 + 规则 + 根目录
    [system]   召回到的长期记忆（不会参与压缩）
    [system]   滚动摘要（更早对话压成一条）
    [..]       最近 N 条消息原文（按轮组保留）
    [user]     本轮输入（已在最近消息里）

  超出预算时，从最老的轮组开始丢：
    1) 先丢“工具结果”，保留“模型决策”；
    2) 丢掉的原文交给 summarizer 压成一条摘要；
    3) 摘要只增补事实（结论 / 待办 / 未完成动作），不保留语气。

  为什么按“轮组”删而不是按条删：OpenAI 协议里 assistant(tool_calls) 必须紧跟着
  对应的 role=tool 消息，单独删一条会让请求 400。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SYSTEM_PROMPT = """你是一个文档问答助手，工作方式必须遵守以下规则：

1. 你只回答根目录内文档能支撑的内容。先定位、再精读，不要凭空猜测。
2. 可用工具（按需组合，别一次把文档全读完）：
   - list_docs：看有哪些文档
   - outline：看单个文档的结构目录（带行号）
   - search_docs：跨文件按关键词定位，返回 路径:行号
   - read_doc：按行区间读片段
   - todo：多步任务先列计划，做完一步勾一步
   - calculator：计数/求和/比例一律用它算
   - remember：用户明确要求记住的稳定事实
3. 回答时给出依据：`文件名:行号`。找不到依据就直说“文档里没有”，不要编。
4. 工具返回 ERROR 时，读懂错误再决定下一步；同一工具同一参数连续失败 3 次会被硬性打断。
5. 回答用中文，简洁，先结论后证据。
"""

SUMMARY_PROMPT = """把下面的对话压缩成一段不超过 200 字的事实摘要，供后续对话继续使用。
必须保留：用户的目标与约束、已确认的结论（带文件:行号）、未完成的动作。
必须丢弃：寒暄、重复的中间过程、工具调用的原始输出。
已有摘要（可能为空）：
{prev}

待压缩对话：
{chunk}
"""


def est_tokens(text: str) -> int:
    """粗估 token：中文按 1 字 ≈ 1 token，英文按 3.5 字符 ≈ 1 token。

    不引 tiktoken：估偏 20% 也不影响策略（预算是软阈值），但省一个重依赖。
    """
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return int(cjk + other / 3.5) + 4


def _keywords(text: str) -> list[str]:
    """召回用的轻量分词：ASCII 按词，中文按 2-gram（不引分词库，可解释、可测）。"""
    ascii_words = re.findall(r"[A-Za-z0-9_]{2,}", text.lower())
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    grams = [cjk[i:i + 2] for i in range(len(cjk) - 1)]
    if len(cjk) == 1:
        grams = [cjk]
    return list(dict.fromkeys(ascii_words + grams))


def mechanical_summary(chunk: str, prev: str = "") -> str:
    """兜底摘要：不调模型，纯截断。保证测试与离线场景也能跑通压缩链路。"""
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    picked = [ln for ln in lines if ln.startswith(("##", "-", "结论", "用户"))][:8]
    base = "；".join((picked or lines[:6]))[:400]
    return (prev + " / " + base).strip(" /")[:600]


@dataclass
class BuildResult:
    messages: list[dict]
    compressed: bool = False
    dropped_messages: int = 0
    recalled: list[str] = field(default_factory=list)
    tokens: int = 0


def group_turns(messages: list[dict]) -> list[list[dict]]:
    """把消息切成“不可拆分的轮组”：assistant(tool_calls) + 其后的 tool 结果必须同组。"""
    groups: list[list[dict]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        group = [msg]
        i += 1
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = {c.get("id") for c in msg["tool_calls"]}
            while i < len(messages) and messages[i].get("role") == "tool" \
                    and messages[i].get("tool_call_id") in ids:
                group.append(messages[i])
                i += 1
        groups.append(group)
    return groups


class ContextManager:
    def __init__(self, store, budget_tokens: int = 20_000, keep_recent: int = 10,
                 recall_top_k: int = 3, summarizer=None):
        self.store = store
        self.budget = budget_tokens
        self.keep_recent = keep_recent
        self.recall_top_k = recall_top_k
        # summarizer(chunk, prev) -> str；默认机械摘要，真实运行时注入 LLM 版
        self.summarizer = summarizer or mechanical_summary

    # ---------------- 记忆召回 ---------------- #
    def recall(self, session_id: str, query: str) -> list[str]:
        """召回时机：每轮装配上下文时，以“最近一轮用户输入”为查询做一次。"""
        memories = self.store.list_memories(session_id)
        if not memories:
            return []
        tokens = _keywords(query)
        scored: list[tuple[int, str]] = []
        for mem in memories:
            text = mem["content"].lower()
            score = sum(1 for t in tokens if t in text)
            scored.append((score, mem["content"]))
        scored.sort(key=lambda x: -x[0])
        picked = [c for s, c in scored if s > 0][: self.recall_top_k]
        if not picked:                       # 没有关键词命中时，召回最新两条当背景
            picked = [m["content"] for m in memories[-2:]]
        return picked

    # ---------------- 装配 ---------------- #
    def build(self, session_id: str, root: str = "", extra_tokens: int = 0) -> BuildResult:
        """装配上下文。

        extra_tokens：不在 messages 里、但每次请求都要发的开销（主要是工具 schema）。
        实测 7 个工具的 schema ≈ 1k tokens；不算进来的话预算会系统性低估，
        真实窗口越小越容易在"明明没到预算"时被服务端截断。
        """
        raw = self.store.messages(session_id)
        last_user = next((m["content"] for m in reversed(raw)
                          if m.get("role") == "user"), "")
        recalled = self.recall(session_id, last_user)

        head: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if root:
            head.append({"role": "system", "content": f"会话根目录：{root}"})
        if recalled:
            head.append({"role": "system",
                         "content": "已记住的长期事实（可信，优先遵守）：\n"
                                    + "\n".join(f"- {r}" for r in recalled)})

        session = self.store.get_session(session_id) or {}
        summary = (session.get("summary") or "").strip()
        summary_block = ([{"role": "system", "content": f"更早对话的摘要：\n{summary}"}]
                         if summary else [])

        groups = group_turns(raw)
        kept: list[list[dict]] = []
        count = 0
        for group in reversed(groups):
            if count + len(group) > self.keep_recent and kept:
                break
            kept.insert(0, group)
            count += len(group)

        def assemble() -> list[dict]:
            flat: list[dict] = []
            for g in kept:
                flat.extend(g)
            return head + summary_block + flat

        messages = assemble()
        tokens = sum(est_tokens(str(m.get("content") or "")) for m in messages) + extra_tokens
        dropped_total = len(raw) - sum(len(g) for g in kept)
        compressed = False

        # 超预算：从最老轮组往外丢，丢完就用摘要补记忆
        while tokens > self.budget and len(kept) > 1:
            dropped = kept.pop(0)
            dropped_total += len(dropped)
            compressed = True
            chunk = "\n".join(f"[{m.get('role')}] {m.get('content') or ''}" for m in dropped)
            summary = self.summarizer(chunk, summary)
            summary_block = [{"role": "system", "content": f"更早对话的摘要：\n{summary}"}]
            messages = assemble()
            tokens = sum(est_tokens(str(m.get("content") or "")) for m in messages) + extra_tokens

        if compressed:
            self.store.set_summary(session_id, summary)

        return BuildResult(messages=messages, compressed=compressed,
                           dropped_messages=dropped_total, recalled=recalled,
                           tokens=tokens)


class LLMSummarizer:
    """真实运行时用模型做摘要。单独一个类，是为了让 ContextManager 保持可测。"""

    def __init__(self, llm):
        self.llm = llm
        self.calls = 0

    def __call__(self, chunk: str, prev: str = "") -> str:
        self.calls += 1
        prompt = SUMMARY_PROMPT.format(prev=prev or "（无）", chunk=chunk[:6000])
        try:
            reply = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
            text = (reply.content or "").strip()
            return text[:800] if text else mechanical_summary(chunk, prev)
        except Exception:
            return mechanical_summary(chunk, prev)     # 摘要失败也要降级，不能让对话挂掉
