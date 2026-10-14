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
    """一个可入库的检索单元：向量库一条记录 + MySQL `chunk_metas` 一行。

    `chunk_id` 是唯一的跨表契约，格式 `{game}-{page_id}-{chunk_index:04d}`。
    `persist.backfill_corpus` 的剪枝阶段拿它当"本轮应该存在的全集"
    （`keep_chunk_ids`），所以它**必须只由本页的切分顺序决定**：
    一旦掺进时间戳或随机数，重跑一次回填就会把向量库里真实存在的分片
    全部当成残留删掉 —— 而且删完照样返回 `ok=True`。
    """

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
        """本块正文字符数，落库为 `ChunkMeta.n_chars`。

        下游把它用在两个地方，含义不同：`ChunkMeta.n_chars` 是单块长度，
        而 `Page.n_chars = sum(c.char_len for c in chunks)` 是**求和**，
        因为块之间有 overlap，求和会大于页面真实长度 —— 它衡量的是
        "入库了多少文本量/embedding 成本"，不是"这个页面有多长"。
        拿它当页面字数去和原文长度对账，必然对不上。
        """
        return len(self.text)


def _split_sections(raw: str) -> list[tuple[str, str]]:
    """把 wikitext 清洗后的纯文本按标题切成 [(标题路径, 正文)]。

    标题行在清洗阶段已保留为独立行，这里逐行扫描并维护标题栈。

    **切边界选标题而不是选字符**，是因为这个 wiki 的事实是挂在小节上的：
    "刻晴 > 天赋 > 星魂"里的数值表和它上面的描述文字必须待在同一个 section，
    按字符硬切会把数值劈到相邻块，召回时就只剩一半信息。

    前提是入参已经过 `template_parser.clean_wikitext` 展平 ——
    本函数不认 `{{模板}}` 语法，若拿到未展平的原文，切出来的段里全是模板噪声。
    返回值只被 `chunk_page` 消费，作为外层循环的迭代源。
    """
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        """把缓冲区里攒的行落成一个 section，然后清空缓冲。

        两处调用时机不同，这是刻意的：遇到标题时 flush 一次（切段），
        循环结束后再 flush 一次（收尾）。**收尾那次不能省** ——
        页面最后一个标题下的正文后面没有标题行来触发 flush，
        省掉它等于静默丢掉整页的最后一段，页面上却不报错。
        """
        text = "\n".join(buf).strip()
        if text:
            # 标题栈可能为空（正文出现在第一个标题之前），回落到"正文"，
            # 这样调用方拿到的 section 路径永远非空，不用再判空。
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
    """把一段正文打包成若干 (文本, 是否硬截断)，overlap 按整句回退。

    **切边界只认句号**，不认字符位置：一个 chunk 的尾句和下一块的头句
    要能各自独立读懂，否则向量召回到半句话，LLM 拿到的证据就是残缺的。
    代价是块大小会波动 —— 宁可超出 chunk_size 一点，也不在句中下刀。

    三种出口，调用方都要认：
    - 正常打包：靠句边界攒满即落，`truncated=False`；
    - 单句超长：只能硬切，`truncated=True`，评测时应把这类块单列，
      它们的召回率天然偏低，混在一起会低估分块策略本身的效果；
    - 空文本：直接返回 `[]`（由末尾的 `if t` 过滤空串兜底）。

    overlap 是**回退**出来的而非切出来的：从已落的块尾部倒着取整句，
    直到再取一句就超 overlap 为止。所以实际重叠量 ≤ overlap，
    且至少是一整句 —— 不保证恰好等于配置值，别拿它做精确核算。
    """
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
    """把一个清洗后的 wiki 页面切成带元数据的 chunk 列表。

    两级串联：`_split_sections` 切逻辑段 → 每段各自 `_pack`。
    **段与段之间不做 overlap** —— 跨小节重复文本会让相邻块的向量互相竞争召回，
    而小节本身已经是天然的语义边界，段首自带标题路径足以定位。

    返回值被三处消费，改动前先确认影响面：
    - `pipeline.page_to_chunks` → `Ingestor.ingest_pages` 统计 `len(chunks)`，
      并逐块送去 embedding；
    - `persist.backfill_corpus` 用它填 `ChunkMeta`，**并把 chunk_id 收进
      keep_chunk_ids 当剪枝依据** —— 所以切块结果必须是确定性可复现的，
      同样输入必须切出同样的 id 和同样的条数；
    - `min_len` 过滤掉的短块**不占用 chunk_index**（`idx` 在 append 之后才自增），
      所以编号连续无空洞，剪枝时才对得上。

    `min_len=40` 是去噪下限：展平后残留的 `字段：` 单行、表格残片长度普遍低于它，
    留着只会占召回位。注意它砍掉的是块数而不是字符数，`Page.n_chars` 随之偏小。
    """
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