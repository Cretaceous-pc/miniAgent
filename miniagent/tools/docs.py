# -*- coding: utf-8 -*-
"""文档读取层：路径沙箱 + 多格式读行 + 结构提取。

设计要点：
  * 所有路径都要过 `resolve()`，只允许落在会话根目录内——工具是模型调用的，
    不能让它 `../../` 摸到别处
  * 不同格式统一成“行列表”，上层工具（outline/search/read）只面对 lines
  * docx 用标准库 zipfile + ElementTree 解，不引第三方解析器；pdf 明确不支持
"""
from __future__ import annotations

import csv
import html as html_mod
import io
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

SUPPORTED_EXTS = {
    ".md", ".markdown", ".txt", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".py", ".js", ".ts", ".java", ".sql", ".ini", ".toml", ".conf", ".sh",
    ".html", ".htm", ".docx",
}
# 目录里常见的噪声，列文件时跳过
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode"}
MAX_FILE_BYTES = 2 * 1024 * 1024          # 单文件读入上限，防超大文件打爆内存
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocError(Exception):
    """路径/格式层面的错误，会被工具层转成给模型看的文本。"""


def resolve(root: Path, rel: str | None) -> Path:
    """把模型给的相对路径解析成绝对路径，并保证不越出根目录。"""
    root = root.resolve()
    if root.is_file():                     # 用户传的是单文件 → 以父目录为根
        root = root.parent
    if not rel or rel in (".", "./", ""):
        return root
    raw = Path(rel)
    target = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if target != root and root not in target.parents:
        raise DocError(f"路径越界：{rel} 不在根目录 {root} 内")
    return target


def display_path(root: Path, target: Path) -> str:
    root = root.resolve()
    if root.is_file():
        root = root.parent
    try:
        return target.resolve().relative_to(root).as_posix() or "."
    except ValueError:
        return target.name


def iter_files(root: Path, max_files: int = 200, max_depth: int = 4) -> list[Path]:
    base = root.resolve()
    if base.is_file():
        return [base]
    out: list[Path] = []
    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack and len(out) < max_files:
        cur, depth = stack.pop(0)
        try:
            entries = sorted(cur.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(".") or entry.name in SKIP_DIRS:
                continue
            if entry.is_dir():
                if depth + 1 <= max_depth:
                    stack.append((entry, depth + 1))
            elif entry.suffix.lower() in SUPPORTED_EXTS:
                out.append(entry)
                if len(out) >= max_files:
                    break
    return out


# --------------------------------------------------------------------------- #
# 读行
# --------------------------------------------------------------------------- #
def _read_text_file(path: Path) -> list[str]:
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise DocError(f"{path.name} 有 {size // 1024} KB，超过单文件读取上限，请改用 search_docs 定位")
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace").splitlines()


def _read_docx(path: Path) -> list[str]:
    """docx = zip + document.xml，标准库足够，不需要 python-docx。"""
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise DocError(f"{path.name} 不是合法的 docx") from exc
    root = ET.fromstring(xml)
    lines: list[str] = []
    for para in root.iter(f"{_WORD_NS}p"):
        text = "".join(t.text or "" for t in para.iter(f"{_WORD_NS}t")).strip()
        if text:
            lines.append(text)
    return lines


def _read_html(path: Path) -> list[str]:
    raw = _read_text_file(path)
    text = "\n".join(raw)
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_mod.unescape(text)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _read_csv(path: Path) -> list[str]:
    raw = path.read_bytes().decode("utf-8-sig", "replace")
    rows = list(csv.reader(io.StringIO(raw)))
    return ["\t".join(cell.strip() for cell in row) for row in rows if any(row)]


def read_lines(path: Path) -> list[str]:
    """按扩展名分派；返回“行列表”，这是所有工具的统一输入。"""
    if not path.exists():
        raise DocError(f"文件不存在：{path.name}")
    if path.is_dir():
        raise DocError(f"{path.name} 是目录，请用 list_docs / search_docs")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise DocError(f"不支持的格式 {ext or '(无扩展名)'}；支持："
                       + " ".join(sorted(SUPPORTED_EXTS)))
    if ext == ".docx":
        return _read_docx(path)
    if ext in (".html", ".htm"):
        return _read_html(path)
    if ext in (".csv", ".tsv"):
        return _read_csv(path)
    return _read_text_file(path)


class DocReader:
    """带缓存的读行器：同一文件在一次会话里只解析一次。"""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._cache: dict[tuple[str, float], list[str]] = {}

    def lines(self, path: Path) -> list[str]:
        key = (str(path), path.stat().st_mtime if path.exists() else 0.0)
        if key not in self._cache:
            self._cache[key] = read_lines(path)
        return self._cache[key]


# --------------------------------------------------------------------------- #
# 结构提取
# --------------------------------------------------------------------------- #
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_CODE_DEF = re.compile(r"^\s*(?:def|class|func|interface|public|private|protected)\s+[\w$]+"
                       r"|^\s*(?:async\s+)?function\s+\w+")
_JSON_TOP = re.compile(r'^\s{0,2}"([^"]+)"\s*:')
_SQL_STMT = re.compile(r"^\s*(CREATE|ALTER|INSERT|SELECT|UPDATE|DELETE)\b", re.I)


def outline(lines: list[str], ext: str, limit: int = 60) -> list[str]:
    """给模型一份“目录”，让它能先定位再精读。"""
    ext = ext.lower()
    items: list[str] = []
    if ext in (".md", ".markdown"):
        for i, line in enumerate(lines, 1):
            m = _MD_HEADING.match(line)
            if m:
                items.append(f"L{i}\t{'  ' * (len(m.group(1)) - 1)}{m.group(2)}")
    elif ext == ".json":
        for i, line in enumerate(lines, 1):
            m = _JSON_TOP.match(line)
            if m:
                items.append(f"L{i}\t{m.group(1)}")
    elif ext in (".sql",):
        for i, line in enumerate(lines, 1):
            if _SQL_STMT.match(line):
                items.append(f"L{i}\t{line.strip()[:90]}")
    elif ext in (".py", ".js", ".ts", ".java", ".sh"):
        for i, line in enumerate(lines, 1):
            if _CODE_DEF.match(line):
                items.append(f"L{i}\t{line.strip()[:90]}")
    else:
        # 纯文本 / docx 段落：每段首行
        for i, line in enumerate(lines, 1):
            if line.strip() and (i == 1 or not lines[i - 2].strip()):
                items.append(f"L{i}\t{line.strip()[:90]}")
    if len(items) > limit:
        return items[:limit] + [f"…（共 {len(items)} 条，只显示前 {limit} 条）"]
    return items


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(query: str) -> list[str]:
    """检索分词：ASCII 按词、中文按 2-gram（不引分词库，够用且可解释）。"""
    tokens = [w.lower() for w in _ASCII_WORD.findall(query)]
    cjk = "".join(_CJK.findall(query))
    tokens += [cjk[i:i + 2] for i in range(len(cjk) - 1)]
    if len(cjk) == 1:
        tokens.append(cjk)
    return [t for t in dict.fromkeys(tokens) if t]


def search_file(lines: list[str], tokens: list[str], phrase: str) -> list[tuple[float, int, str]]:
    """对单文件逐行打分：命中词数 + 整句命中的加权。"""
    hits: list[tuple[float, int, str]] = []
    phrase_l = phrase.lower().strip()
    for idx, line in enumerate(lines, 1):
        low = line.lower()
        score = 0.0
        for tok in tokens:
            if tok in low:
                score += 1.0 + low.count(tok) * 0.2
        if phrase_l and phrase_l in low:
            score += 3.0
        if score > 0:
            hits.append((score, idx, line.strip()[:200]))
    return hits
