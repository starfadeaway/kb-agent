"""内容指纹：判断一个页面「实质上变了没有」。

在这条链路里它**不是通用工具**，只服务一件事：增量更新。
判断"变没变"的地方只有两处 —— `delta_sync.detect_changes` 的第 3 级
（本地指纹 vs 刚取回的正文指纹），以及 `pipeline.cache_raw` 写 sidecar 时
落下的 `content_hash`。除此之外全仓没有第三个用途。

## 误判和漏判，哪个更危险

指纹唯一会犯的错是**碰撞**（不同内容算出同一个值），两种后果严重程度不同：

- **把"没变"算成"变了"**：白重建一次页，多花 embedding 和 WAF 配额，
  代价可见、可回收。
- **把"变了"算成"没变"**：旧内容留在索引里，检索返回已作废的数值，
  而且**静默且不自愈** —— 索引看起来完全正常，没有任何地方会报错。

所以整体取向是"宁可多算一次变更"：`normalize_text` 宁可保守，
也不接受看起来等价的激进归一化。这正是下面那张表存在的理由。

为什么需要它（这是版本更新链路的第 1 步，其余步骤都依赖它）：

- `crawl.py` 的 checkpoint 只记 `done_titles`，已抓页面永不重抓；
- `cache_raw()` 虽然做了一次 `read_text() != page.wikitext` 的全等比较，
  但结果只当场丢弃，**没有落成任何可查询的字段**；
- 于是"哪些页面变了"这个问题，只能靠把全站重抓一遍来回答 ——
  在 WAF 限流下这是要付配额的，而实测配额并不便宜。

有了指纹，增量检测就退化成一句 SQL：比 `content_hash` 和 `revision_ts`。

## 归一化不是可有可无的修饰

直接对原文取哈希会有一个致命问题：**wiki 上改个错别字、
动一下换行，整页就会算成"变了"**，进而触发全页重新 embedding。
在实测"一周变更页 ∩ 已收录 = 20.9%"的规模下，这种误判会直接吃掉增量的全部收益。

所以哈希前必须归一化，去掉**不携带语义**的差异：

| 变动 | 是否应算变更 | 处理 |
|---|---|---|
| CRLF ↔ LF | 否 | 统一换行 |
| 行尾多余空格 | 否 | 去行尾空白 |
| 连续空行 3+ → 2 | 否 | 折叠空行 |
| UTF-8 BOM | 否 | 去 BOM |
| 正文改一个字 | **是** | 保留 |
| infobox 数值调整 | **是** | 保留 |

反过来，**内部空格必须保留**：wikitext 里表格与模板的对齐空格有结构含义，
折叠掉会让"排版调整"被误判成"内容变化"或反之。
"""
from __future__ import annotations

import hashlib
import re

# 行尾空白（含全角空格 —— 手工编辑 wiki 时很容易带进来）
_TRAIL_WS = re.compile(r"[ \t\u3000\u00a0]+(?=\n)")
_BLANK_RUN = re.compile(r"\n{3,}")
_LEAD_BOM = "\ufeff"
# 零宽字符：U+200B 零宽空格 / U+200C ZWNJ / U+200D ZWJ / U+FEFF BOM
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def normalize_text(text: str) -> str:
    """归一化到"语义等价"形式。只动不携带语义的差异。

    刻意**不做**的事：
    - 不折叠内部空格（表格对齐空格有结构含义）
    - 不做 Unicode 归一化 NFC —— 全角字母与半角字母在 wiki 里可能是不同字符，
      统一它们会把真实的内容改动抹掉

    零宽字符转成**空格**而不是删除：删除会把 `ignore|all` 粘成一个词，
    反而绕过按词匹配的规则；而对指纹来说，空格会被 `_TRAIL_WS` 与
    `_BLANK_RUN` 一起收掉，等价于"看不见"。
    """
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    if s.startswith(_LEAD_BOM):
        s = s[len(_LEAD_BOM):]
    s = _ZERO_WIDTH.sub(" ", s)
    s = _TRAIL_WS.sub("", s)
    s = _BLANK_RUN.sub("\n\n", s)
    return s.strip()


def _sha(text: str) -> str:
    """算 sha256 并返回十六进制串（**不截断**，截断由调用方做）。

    两个调用方都在归一化之后才调它，所以这个函数假定入参已经语义等价；
    直接喂原文进来会得到一个对排版改动敏感的指纹，那不是它该回答的问题。

    选 sha256 而不是 md5/CRC，不是为了防攻击 —— 这里没有攻击面，
    指纹存在 sidecar JSON 和 MySQL 的 `content_hash` 列里，是可信侧写入。
    真正的原因是**跨版本稳定**：指纹要能和历史数据比对，
    中途换算法等于全站指纹同时失效、一次性触发全量重建，
    这种"重算一遍"的代价远高于多存 32 个十六进制字符。

    返回 64 位而不是直接给最终答案，是为了让"截多长"成为调用方的策略：
    page_hash 与 chunk_hash 都取前 16 位，但它们对碰撞的容忍度不同
    （块级指纹基数大得多），将来若要分开调整，改这里一行即可。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def page_hash(wikitext: str) -> str:
    """页面级指纹（基于原始 wikitext）。

    取 sha256 前 16 位：64 位足够让"两页撞指纹"的概率低到可以忽略，
    而 16 位让 MySQL 的 `VARCHAR(16)` 与 sidecar 的体积都可接受。
    全长 sha256 在这里没有额外价值 —— 它不是安全哈希，是变更检测。

    消费方有两处，返回值必须和它们保持同一口径：
    - `pipeline.cache_raw` 把它写进 `<page_id>.meta.json` 的 `content_hash`，
      之后 `cached_page_hash` **只读 sidecar、不读正文**来回答"变没变"；
    - `delta_sync._local_pages` 从 MySQL `Page.content_hash` 列取同一口径的值，
      与第 3 级现算的 `page_hash` 直接比对。
    两边若有一处改了归一化规则，指纹就再也对不上，且表现为"全都变了"。
    """
    return _sha(normalize_text(wikitext))[:16]


def chunk_hash(text: str) -> str:
    """块级指纹（基于清洗后的正文）。

    与 page_hash 的区别很重要：
    - page_hash 回答「这个页面要不要重抓正文」（要走 WAF）
    - chunk_hash 回答「这个块要不要重新 embedding」（只花钱不走路）

    实测增量相对全量的主要收益来自这里：省掉约 79% 的 embedding，
    而抓取时间的节省只有一半左右。分两级是为了分别对应这两种成本。

    唯一消费方是 `persist.backfill_corpus` 写 `ChunkMeta.content_hash`。
    注意它算的是**清洗后的块正文**，而 `ChunkMeta` 的唯一键是 chunk_id：
    所以只要分块参数（chunk_size/overlap）一变，块的边界就变，
    chunk_hash 必然全变 —— 这是预期行为，但也意味着**改分块参数后
    不能靠 content_hash 判断"哪些块没变"**，那套增量只对固定参数有效。
    """
    return _sha(normalize_text(text))[:16]


def is_stale(local_hash: str, remote_hash: str) -> bool:
    """本地指纹与新算出的指纹不一致 -> 需要重建。

    两个空值**一律判为变更**（返回 True，见下方 if）。

    ⚠️ 这一句原先写的是「不判为变更」，与紧随其后的 `return True` 正好相反。
    已改正：代码、行内注释、`delta_sync` 的模块 docstring 三处都一致地认定
    空值判变更，错的只有这句描述。老数据没有指纹时确实该重建一次，
    但不能把它当成「没变」—— 那样缺指纹的页会永远停在旧内容上。
    """

    # 顺序有讲究：先挡空值，再比不等。
    # 空指纹（老数据/解析失败 sidecar）一律当"变了"—— 无法证明它没变，
    # 就不能当作没变。这是保守方向的选择，代价是一次重建，可接受。
    # 反过来把空值当"没变"，那会让缺指纹的页永远停留在旧内容上。
    if not local_hash or not remote_hash:
        return True
    # 调用方 `delta_sync` 靠这个返回值分两类：False 进 `revision_only`
    # （编辑过但内容没动，只改错别字/排版），True 进 `changed` 等待重建。
    # 所以判据放宽一点就会让大量无关编辑混进"需要重建"。
    return local_hash != remote_hash
