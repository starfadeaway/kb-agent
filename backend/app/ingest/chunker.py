"""中文友好的分块器。

设计要点（对应实验 B：分块策略消融）：
1. 先按标题层级切"逻辑段"，避免把角色技能描述和它的数值表劈开；
2. 再按句边界贪心打包到 chunk_size，overlap 以整句为单位回退，避免截断半句话；
3. 超长单句（少见）才做硬切，并记录 truncated 标记，方便评测时区分。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# MediaWiki 标题：== 角色定位 == / === 技能 === / ==== 精通 加成 ====
HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])")


@dataclass
class Chunk:
    chunk_id: str
    game: str
    page_id: int
    page_title: str
    section: str          # 标题路径，如 "刻晴 > 天赋 > 星魂"
    chunk_index: int
    text: str
    last_revision: int
    last_revision_ts: str
    source_url: str
    truncated: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def char_len(self) -> int:
        return len(self.text)


def _split_sections(raw: str) -> list[tuple[str, str]]:
    """把 wikitext 清洗后的纯文本按标题切成 [(标题路径, 正文)]。

    标题行在清洗阶段已保留为独立行，这里逐行扫描并维护标题栈。
    """
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            path = " > ".join(t for _, t in stack) or "正文"
            sections.append((path, text))
        buf.clear()

    for line in raw.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _pack(text: str, chunk_size: int, overlap: int) -> list[tuple[str, bool]]:
    """把一段正文打包成若干 (文本, 是否硬截断)，overlap 按整句回退。"""
    sentences = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return []

    out: list[tuple[str, bool]] = []
    cur: list[str] = []
    cur_len = 0

    for sent in sentences:
        # 单句超长：硬切
        if len(sent) > chunk_size:
            if cur:
                out.append(("".join(cur).strip(), False))
                cur, cur_len = [], 0
            for i in range(0, len(sent), chunk_size):
                piece = sent[i : i + chunk_size]
                out.append((piece, i + chunk_size < len(sent)))
            continue

        if cur_len + len(sent) > chunk_size and cur:
            out.append(("".join(cur).strip(), False))
            # 回退出 overlap 长度的句子
            carry: list[str] = []
            carry_len = 0
            for s in reversed(cur):
                if carry_len + len(s) > overlap:
                    break
                carry.insert(0, s)
                carry_len += len(s)
            cur, cur_len = carry, carry_len

        cur.append(sent)
        cur_len += len(sent)

    if cur:
        out.append(("".join(cur).strip(), False))
    return [(t, tr) for t, tr in out if t]


def chunk_page(
    *,
    page_id: int,
    page_title: str,
    text: str,
    game: str,
    last_revision: int,
    last_revision_ts: str,
    source_url: str,
    chunk_size: int = 500,
    overlap: int = 50,
    min_len: int = 40,
) -> list[Chunk]:
    """把一个清洗后的 wiki 页面切成带元数据的 chunk 列表。"""
    chunks: list[Chunk] = []
    idx = 0
    for section, body in _split_sections(text):
        for piece, truncated in _pack(body, chunk_size, overlap):
            if len(piece) < min_len:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{game}-{page_id}-{idx:04d}",
                    game=game,
                    page_id=page_id,
                    page_title=page_title,
                    section=section,
                    chunk_index=idx,
                    text=piece,
                    last_revision=last_revision,
                    last_revision_ts=last_revision_ts,
                    source_url=source_url,
                    truncated=truncated,
                )
            )
            idx += 1
    return chunks


def chunk_to_index_text(chunk: Chunk, embed_mode: str = "text") -> str:
    """构造送进 embedding 的文本。

    embed_mode 是实验变量：
    - "text"   仅正文（基线）
    - "title"  正文 + 页面标题 + 小节标题（让专有名词进入向量空间）
    """
    if embed_mode == "title":
        return f"{chunk.page_title}（{chunk.section}）\n{chunk.text}"
    return chunk.text