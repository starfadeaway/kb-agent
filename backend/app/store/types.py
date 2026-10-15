"""向量库两个后端共用的**叶子类型与纯函数**。

为什么单独有这么一个模块：`VectorStore`（NumPy 暴力扫描）与
`LanceVectorStore`（LanceDB）是**有意的双实现** —— 要能互换、要能独立降级，
该分叉的是它们的**存储机制**。但下面三样不该分叉：

* `RetrievedChunk` —— 两边此前各写一份逐字相同的 dataclass。后果可证：
  `hybrid.py` 从 NumPy 模块 import 它，而 `get_vector_store` 默认返回
  Lance 后端，于是 `RetrievalHit.chunk` 在生产默认路径上装的其实是 lance 的类，
  **注解与实际类型不符**。项目没配 mypy，这个错永远不会报。
* `model_slug()` —— 同一个正则，两处各写一遍。
* `build_where_clause()` —— 同一段条件拼装。此前两份逐字相同，靠
  `test_store_parity.py` 与 `test_version_filter_api.py` 两个测试盯着不让它们漂：
  **10 行代码需要 2 个测试来维持一致，这本身就是该合并的判据**。

**不放进这里的**：`Collection` / `LanceCollection` 的存储实现、各自多出来的
方法、以及 `_match_where` 与 SQL 转义 —— 那些是机制，必须分叉。
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def model_slug(model: str) -> str:
    """BAAI/bge-m3 -> baai-bge-m3，embo-01 -> embo-01。"""
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]

    @property
    def page_title(self) -> str:
        return str(self.metadata.get("page_title", ""))

    @property
    def source_url(self) -> str:
        return str(self.metadata.get("source_url", ""))


def build_where_clause(
    *,
    game: str | None = None,
    page_titles: Iterable[str] | None = None,
    min_revision_ts: str | None = None,
) -> dict | None:
    """过滤条件。`min_revision_ts` 是版本时效性过滤的入口。

    返回值可直接喂给 `Collection.get` / `query`，两个后端的形状一致。
    返回 None 表示"无过滤"而不是"空条件" —— 调用方要原样透传，
    在外面再包一层 `{"$and": [None]}` 会让匹配函数走进另一条分支。
    单条件时返回该条件本身、多条件才包 `$and`：
    这样单条件路径不必经过 `$and` 递归，少一层出错可能。

    字段名（`game` / `page_title` / `last_revision_ts`）是**与两后端写入口的
    约定** —— 改名必须同时改 `Collection.upsert` 实际写入的 metadata，
    否则过滤恒为空。
    """
    conds: list[dict] = []
    if game:
        conds.append({"game": game})
    if page_titles:
        conds.append({"page_title": {"$in": list(page_titles)}})
    if min_revision_ts:
        conds.append({"last_revision_ts": {"$gte": min_revision_ts}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}
