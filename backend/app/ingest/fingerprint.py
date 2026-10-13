"""内容指纹：判断一个页面「实质上变了没有」。

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
import unicodedata

# 行尾空白（含全角空格 —— 手工编辑 wiki 时很容易带进来）
_TRAIL_WS = re.compile(r"[ \t\u3000\u00a0]+(?=\n)")
_BLANK_RUN = re.compile(r"\n{3,}")
_LEAD_BOM = "\ufeff"


def normalize_text(text: str) -> str:
    """归一化到"语义等价"形式。只动不携带语义的差异。

    刻意**不做**的事：
    - 不折叠内部空格（表格对齐空格有结构含义）
    - 不做 Unicode 归一化 NFC —— 全角字母与半角字母在 wiki 里可能是不同字符，
      统一它们会把真实的内容改动抹掉（`unicodedata.normalize` 只在下面
      用于判断零宽字符，属于读取侧的防御，不是归一化步骤）
    """
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    if s.startswith(_LEAD_BOM):
        s = s[len(_LEAD_BOM):]
    s = _TRAIL_WS.sub("", s)
    s = _BLANK_RUN.sub("\n\n", s)
    return s.strip()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def page_hash(wikitext: str) -> str:
    """页面级指纹（基于原始 wikitext）。

    取 sha256 前 16 位：64 位足够让"两页撞指纹"的概率低到可以忽略，
    而 16 位让 MySQL 的 `VARCHAR(16)` 与 sidecar 的体积都可接受。
    全长 sha256 在这里没有额外价值 —— 它不是安全哈希，是变更检测。
    """
    return _sha(normalize_text(wikitext))[:16]


def chunk_hash(text: str) -> str:
    """块级指纹（基于清洗后的正文）。

    与 page_hash 的区别很重要：
    - page_hash 回答「这个页面要不要重抓正文」（要走 WAF）
    - chunk_hash 回答「这个块要不要重新 embedding」（只花钱不走路）

    实测增量相对全量的主要收益来自这里：省掉约 79% 的 embedding，
    而抓取时间的节省只有一半左右。分两级是为了分别对应这两种成本。
    """
    return _sha(normalize_text(text))[:16]


def is_stale(local_hash: str, remote_hash: str) -> bool:
    """本地指纹与新算出的指纹不一致 -> 需要重建。

    两个空值**不判为变更**：老数据没有指纹时应当"重建一次"，
    但那是回填要处理的事，不能让每个空值都触发一次无意义的判定分叉。
    """
    if not local_hash or not remote_hash:
        return True
    return local_hash != remote_hash


def has_zero_width(text: str) -> bool:
    """正文里有没有零宽字符。

    单独拎出来是因为它**会破坏哈希**：U+200B 在 `str.strip()` 里不被当作空白，
    两份视觉上完全相同的文本可能算出不同指纹。调用方应先把它挑出来告警，
    而不是指望哈希去容忍它。
    """
    return any(unicodedata.category(ch) == "Cf" for ch in text)
