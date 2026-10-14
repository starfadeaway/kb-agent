"""向量索引：NumPy 精确余弦检索 + 磁盘持久化。

**为什么不用 ChromaDB（这是实测结论，不是偏好）**：
在本机（Windows + Python 3.13）上，chromadb 只有 1.5.x 提供 cp39-abi3 预编译轮子，
但它写入后磁盘上只生成 `index_metadata.pickle`，HNSW 段目录始终不落盘，
换进程读取必然报 `Error loading hnsw index`。同版本 0.5/0.6 又没有 cp313 轮子、
本机无 MSVC 无法编译。所以 ChromaDB 在当前环境不可用。

**为什么 NumPy 精确检索够用**：
本项目语料规模是千级 chunk（星铁全站约 1.7 万页，取核心分类后 500+ 页 → 2000+ chunk）。
N=2500、D=1024 的 float32 矩阵做一次批量点积不到 1ms，比任何近似索引都快，
而且是**精确**检索 —— 评测时不存在 ANN 召回损失，指标差异只来自检索策略本身，
这正是单变量消融需要的。

规模真的到十万级以上再换 Faiss / Milvus，届时只需替换本文件。
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings


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


def _match_where(meta: dict, where: dict | None) -> bool:
    """支持 $and / $in / $gte 三种条件，够用且不会静默漏条件。

    调用方必须知道的两条语义：
    1. `where=None`/空 → **全量通过**，不是"没条件所以匹配失败"；
    2. 字段在 metadata 里缺失时一律判不匹配（`$gte`/`$lte` 显式挡了 None，
       等值比较则因 `None != cond` 落空）。所以入库时字段名拼错，
       表现是过滤后**结果为空**而不是报错 —— 查不到东西时先怀疑字段名。
    大小写、类型都不做归一：比较走 Python 原生语义（`$gte` 更是显式转字符串比），
    这也是「两个后端必须给出同样结果」的判据（见 lance_store.where_to_sql）。
    """
    if not where:
        return True
    if "$and" in where:
        return all(_match_where(meta, c) for c in where["$and"])
    for key, cond in where.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            if "$in" in cond and val not in cond["$in"]:
                return False
            if "$gte" in cond and not (val is not None and str(val) >= str(cond["$gte"])):
                return False
            if "$lte" in cond and not (val is not None and str(val) <= str(cond["$lte"])):
                return False
        elif val != cond:
            return False
    return True


class Collection:
    """单个 collection 的内存 + 磁盘表示。"""

    def __init__(self, root: Path, name: str, metadata: dict[str, Any] | None = None) -> None:
        self.root = root
        self.name = name
        self.meta = metadata or {}
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self._emb: np.ndarray | None = None
        self._index: dict[str, int] = {}
        self._lock = threading.Lock()
        self._load()

    # ---------------- 持久化 ----------------
    @property
    def dir(self) -> Path:
        return self.root / self.name

    def _load(self) -> None:
        """从磁盘恢复索引。由 `__init__` 调用，失败不抛异常（库不存在即空库）。

        进入这段的前提是 `embeddings.npy` 与 `ids.json` **成对存在**，
        少一个就只恢复一半状态：`_emb` 保持 None 而 `ids` 非空，
        后续 `upsert` 会走"空库"分支直接覆盖 ids（数据丢失）或 `query` 直接返回空。
        之所以还能容忍，是因为 `_flush` 每次写入都把四个文件一起落盘，
        真正的半截状态只会出现在外部手改目录时。
        """
        d = self.dir
        if not d.is_dir():
            return
        emb_p, ids_p = d / "embeddings.npy", d / "ids.json"
        if emb_p.is_file() and ids_p.is_file():
            self._emb = np.load(emb_p)
        if (d / "documents.json").is_file():
            self.documents = json.loads((d / "documents.json").read_text(encoding="utf-8"))
        if (d / "metadatas.json").is_file():
            self.metadatas = json.loads((d / "metadatas.json").read_text(encoding="utf-8"))
        if (d / "meta.json").is_file():
            self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.ids = json.loads(ids_p.read_text(encoding="utf-8")) if ids_p.is_file() else []
        self._index = {cid: i for i, cid in enumerate(self.ids)}

    def _flush(self) -> None:
        """把当前内存状态整份写盘。**返回即代表已落盘** —— 调用方无需额外 commit。

        没有增量写：每次 upsert 都重写全部四个文件（实测 2000+ chunk 量级仍是毫秒级，
        因为瓶颈在建索引不在写盘）。真要换成增量，得先解决
        "documents/metadatas 是平行数组、append 后还要重排 _index" 这件事。
        `embeddings.npy` 仅在 `_emb` 非空时写，避免把空库写成一个 0 字节占位。
        """
        d = self.dir
        d.mkdir(parents=True, exist_ok=True)
        if self._emb is not None:
            np.save(d / "embeddings.npy", self._emb)
        (d / "ids.json").write_text(json.dumps(self.ids, ensure_ascii=False), encoding="utf-8")
        (d / "documents.json").write_text(json.dumps(self.documents, ensure_ascii=False), encoding="utf-8")
        (d / "metadatas.json").write_text(json.dumps(self.metadatas, ensure_ascii=False), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 写入 ----------------
    def count(self) -> int:
        return len(self.ids)

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        """写入（upsert）四个平行数组：`ids` / `documents` / `metadatas` / `_emb`。

        返回 None，唯一的后作用是**已经落盘**（末尾 `_flush()`）——
        入库管线把它当作事务边界用，返回后可以立刻去查。
        两个硬前提由调用方保证，违反时表现多为静默错乱：
        1. 四条输入**等长**（这里不做校验，长度不一致会在下面
           `self._emb[pos] = new[i]` 处抛下标错，或在 fresh 分支里整体错位）；
        2. embeddings 已由写入方 L2 归一化 —— `query()` 直接把点积当余弦用，
           没归一化的话分数会随向量模长漂移，而评测只看排名，很容易被忽略。
        """
        new = np.asarray(embeddings, dtype=np.float32)
        # 整段在锁内：`_emb` 的原地写与 fresh 分支的 vstack/extend 不是原子的，
        # 并发 upsert 会在 `_index` 与四个数组之间撕出不一致。
        # 锁粒度粗（覆盖了全量落盘）是可接受的：实测写入是毫秒级，不构成瓶颈。
        with self._lock:
            if self._emb is None or self._emb.shape[0] == 0:
                self._emb = new
                self.ids = list(ids)
                self.documents = list(documents)
                self.metadatas = [dict(m) for m in metadatas]
            else:
                # 已存在的 id：**三个字段都要刷新**。
                # 原来这里只覆盖 embedding，正文和元数据走 fresh 分支永不更新 ——
                # 于是"改了 chunk_size / embed_mode 后重跑入库"在这个后端下
                # 变成"只换了向量、没换内容"，检索返回的是**新向量配旧正文**。
                # 实验 B / D 的单变量前提直接被破坏，而且不报任何错。
                # Lance 后端一直是 delete+add 整体刷新，这里对齐它。
                fresh = []
                for i, cid in enumerate(ids):
                    pos = self._index.get(cid)
                    if pos is None:
                        fresh.append(i)
                    else:
                        self._emb[pos] = new[i]
                        self.documents[pos] = documents[i]
                        self.metadatas[pos] = dict(metadatas[i])
                # 累积量 keep 住的后果：这三个集合里"本轮出现过"的下标必须与
                # `new`（embeddings）严格对齐，一个都不能少也不能重 —— 否则后面
                # `documents[i]` / `metadatas[i]` 会取到别人的内容。
                if fresh:
                    self._emb = np.vstack([self._emb, new[fresh]])
                    self.documents.extend(documents[i] for i in fresh)
                    self.metadatas.extend(dict(metadatas[i]) for i in fresh)
                    self.ids.extend(ids[i] for i in fresh)
            # `_index` 是每次写后**全量重建**的（而不是增量维护）：
            # 因为 fresh 分支会把 ids 追加到末尾、重排下标，
            # 增量更新需要同时修正所有被挪动的位置。2000 量级重建耗时可忽略。
            self._index = {cid: i for i, cid in enumerate(self.ids)}
            self._flush()   # 落盘即本方法的完成语义

    # ---------------- 读取 ----------------
    def get(
        self, ids: Sequence[str] | None = None, where: dict | None = None, include: Sequence[str] = ()
    ) -> dict:
        """按 id 取或按 where 扫，返回三个**平行列表**（不是 dict of list）。

        两个陷阱，消费方必须知道：
        - `ids` 传了就**忽略** where（二选一，不叠加）；
        - 传 `ids` 时不存在的 id 被**静默丢弃**且不打乱顺序 —— 返回列表长度可能
          小于入参，靠下标对齐会错位。
        - `documents` / `metadatas` 只在 `include` 点名时才有内容，否则是**空列表**
          （不是 None、也不是省略该键）。调用方一律用 `get(...) or []` 兜底。
        """
        if ids is not None:
            rows = [self._index[i] for i in ids if i in self._index]
        elif where:
            rows = [i for i, m in enumerate(self.metadatas) if _match_where(m, where)]
        else:
            rows = list(range(len(self.ids)))
        return {
            "ids": [self.ids[i] for i in rows],
            "documents": [self.documents[i] for i in rows] if "documents" in include else [],
            "metadatas": [self.metadatas[i] for i in rows] if "metadatas" in include else [],
        }

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 5,
        where: dict | None = None,
        include: Sequence[str] = ("documents", "metadatas", "distances"),
    ) -> dict:
        """暴力精确 top-k。返回**外层套一层列表**（[[...]]）以对齐 ChromaDB 的形状。

        两个空结果出口，返回结构与正常路径一致（下游直接 `[0]` 不会 IndexError）：
        1. 库是空的（`_emb is None` 或 ids 为空）—— 通常意味着 collection 名写错了；
        2. where 过滤后一条不剩 —— `mask.any()` 先判，就是为了避免
           `argpartition(k-1)` 在空池上抛错。
        """
        if self._emb is None or len(self.ids) == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.asarray(query_embeddings, dtype=np.float32)
        # 写入时已 L2 归一化，点积即余弦相似度
        sims = self._emb @ q.T                      # (N, Q)
        # 单 query 取第 0 列；多 query 时对每条 chunk 取**最相似的那一列**，
        # 也就是"多 query 召回的并集里各自的最优命中"，不是逐 query 的结果列表。
        sims = sims[:, 0] if sims.shape[1] == 1 else sims.max(axis=1)

        if where:
            mask = np.array([_match_where(m, where) for m in self.metadatas], dtype=bool)
            if not mask.any():
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            idx_pool = np.nonzero(mask)[0]
        else:
            idx_pool = np.arange(len(self.ids))

        pool_scores = sims[idx_pool]
        # k 要夹到池子大小：n_results 大于库容量时 argpartition 会越界。
        k = min(n_results, len(idx_pool))
        # argpartition 只做部分排序，比全排序快，且 top-k 结果完全一致
        top = np.argpartition(-pool_scores, k - 1)[:k]
        order = top[np.argsort(-pool_scores[top])]
        rows = idx_pool[order]

        # **`distances` 这一项是 1 - cos，不是距离**，名字沿用 ChromaDB 的字段名。
        # 上层 `VectorStore.query` 会再用 `1 - d` 换回余弦当 score，
        # 这一对换算必须成对存在：任何一边单独改都会让 score 变成 -cos 或 (cos-1)/2。
        # distances 里装的是 `order` 的下标顺序，不是 `rows`，别混用。
        return {
            "ids": [[self.ids[i] for i in rows]],
            "documents": [[self.documents[i] for i in rows]] if "documents" in include else [[]],
            "metadatas": [[self.metadatas[i] for i in rows]] if "metadatas" in include else [[]],
            "distances": [[1.0 - float(pool_scores[i]) for i in order]] if "distances" in include else [[]],
        }


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.root = self.s.data_root_abs / "vectors"
        self.root.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, Collection] = {}

    # ---------------- collection 管理 ----------------
    def collection_name(self, game: str, model: str | None = None, variant: str = "base") -> str:
        model = model or self.s.embed_model_main
        # variant 用来隔离分块/embedding 实验：不同参数绝不写进同一个索引
        return f"{self.s.collection_prefix}_{game}_{model_slug(model)}_{variant}"

    def get_collection(self, game: str, model: str | None = None, variant: str = "base") -> Collection:
        name = self.collection_name(game, model, variant)
        if name not in self._collections:
            self._collections[name] = Collection(
                self.root, name, {"game": game, "embed_model": model or self.s.embed_model_main, "variant": variant}
            )
        return self._collections[name]

    def upsert(
        self,
        *,
        game: str,
        model: str | None,
        variant: str,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not ids:
            return
        self.get_collection(game, model, variant).upsert(ids, embeddings, documents, metadatas)

    def count(self, game: str, model: str | None = None, variant: str = "base") -> int:
        return self.get_collection(game, model, variant).count()

    def list_collections(self) -> list[dict]:
        """列出磁盘上所有 collection 及各自条数。**只读，不进 `_collections` 缓存**。

        代价是每个目录都真的 `new` 一个 Collection 把 embeddings.npy 读进内存，
        调用完即回收；所以它适合低频的 `/api/health` 展示，不适合放进请求热路径。
        """
        out = []
        if not self.root.is_dir():
            return out
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            col = Collection(self.root, d.name)
            out.append({"name": col.name, "count": col.count(), **col.meta})
        return out

    def drop(self, game: str, model: str | None = None, variant: str = "base") -> bool:
        """删掉内存里的缓存条目 + 磁盘目录。返回是否真的删掉了目录。

        两个动作的顺序不能反：先从 `_collections` 弹出，
        否则已持有该 Collection 对象的调用方还能往内存里写，
        而目录已被删掉，下一次 `_flush` 会把索引"复活"到磁盘上。
        """
        name = self.collection_name(game, model, variant)
        self._collections.pop(name, None)
        d = self.root / name
        if not d.is_dir():
            return False
        import shutil

        shutil.rmtree(d)
        return True

    # ---------------- 查询 ----------------
    def build_where(
        self,
        *,
        game: str | None = None,
        page_titles: Iterable[str] | None = None,
        min_revision_ts: str | None = None,
    ) -> dict | None:
        """过滤条件。min_revision_ts 是版本时效性过滤的入口。

        返回值可直接喂给 `Collection.get` / `query`，两个后端的形状一致。
        返回 None 表示"无过滤"而不是"空条件" —— 调用方要原样透传，
        在外面再包一层 `{"$and": [None]}` 会让 `_match_where` 走进另一条分支。
        单条件时返回该条件本身、多条件才包 `$and`：
        这样单条件路径不必经过 `$and` 递归，少一层出错可能。
        字段名（`game` / `page_title` / `last_revision_ts`）是**与两后端写入口的约定**，
        改名必须同时改 `Collection.upsert` 实际写入的 metadata，否则过滤恒为空。
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

    def query(
        self,
        *,
        query_embedding: Sequence[float],
        game: str,
        model: str | None = None,
        variant: str = "base",
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """向量检索，返回按相似度降序的 `RetrievedChunk` 列表（可能为空）。

        返回的 `score` 是**余弦相似度**，不是距离 —— 它由 `distances`（1-cos）
        再取一次 `1 -` 换回来，量纲 [−1, 1]。
        唯一的消费者是 `hybrid._fuse`：只用它的**名次**参与 RRF，
        `score` 则被原样写进 `RetrievalHit.score` 供前端展示与溯源。
        注意 `where=None` 时这里会**兜底补一个 game 过滤**，而不是不加过滤 ——
        跨游戏共库时不补这一条会把另一个游戏的 chunk 一起召回。
        """
        col = self.get_collection(game, model, variant)
        res = col.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=top_k,
            where=where if where is not None else self.build_where(game=game),
            include=("documents", "metadatas", "distances"),
        )
        ids = res["ids"][0]
        if not ids:
            # 空结果与"库里没有"是同一种返回，调用方无法区分 —— 这是保守设计：
            # 检索拿不到候选就返回空列表，而不是造一个占位对象污染融合。
            return []
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        # `strict=False` 兜住 include 被改小导致长度不齐的情况：宁可截短，
        # 也不要因为某个字段缺失就整次检索抛错。
        return [
            RetrievedChunk(
                chunk_id=str(cid),
                text=str(doc or ""),
                score=round(1.0 - float(d), 6),
                metadata=dict(meta or {}),
            )
            for cid, doc, meta, d in zip(ids, docs, metas, dists, strict=False)
        ]

    def get_by_ids(
        self, *, game: str, model: str | None, variant: str, ids: Sequence[str]
    ) -> dict[str, RetrievedChunk]:
        """按 id 精确取回正文与元数据，供溯源/引用还原用。

        **不要拿这里的 `score` 排序**：它是硬编码的 1.0 占位，
        因为按 id 取根本不存在相似度可言。需求若要"给引用标相关度"，
        应该回头用 `query()` 重新算，否则会给用户一个假的置信度。
        返回的是 dict（key=chunk_id），查不到的 id 直接不在结果里。
        """
        col = self.get_collection(game, model, variant)
        got = col.get(ids=list(ids), include=("documents", "metadatas"))
        return {
            str(cid): RetrievedChunk(chunk_id=str(cid), text=str(doc or ""), score=1.0, metadata=dict(m or {}))
            for cid, doc, m in zip(got["ids"], got["documents"], got["metadatas"], strict=False)
        }


_default: VectorStore | Any | None = None


def get_vector_store(settings: Settings | None = None):
    """按 VECTOR_BACKEND 返回 LanceDB 或 NumPy 实现，接口完全一致。

    - lancedb（默认）：真实向量库，跨进程持久化已实测可用；暴力精确扫描。
    - numpy：零依赖兜底，LanceDB 不可用时（未装 / 平台无轮子）自动降级。

    **只有这两个后端。** 早先这里还列过一个 sqlite 后端，但 `sqlite3` /
    `SQLiteStore` 在 `app/` 下从来没有任何实现 —— 那是一句写在注释里的
    虚构实现。宣传一个不存在的东西比不宣传更糟：面试官追问"第三个后端怎么
    实现的"会当场露底。现在由 `tests/test_store_parity.py` 的
    `test_no_phantom_backend_declared` 守住：docstring 与 config 注释里
    出现的每个后端名都必须有对应实现。

    两者共用 Collection 接口，换后端不需要动上层任何代码。
    """
    global _default
    if _default is None:
        backend = (settings or get_settings()).vector_backend
        if backend == "lancedb":
            try:
                from app.store.lance_store import LanceVectorStore

                _default = LanceVectorStore(settings or get_settings())
                logger_backends("lancedb")
                return _default
            except Exception as exc:  # noqa: BLE001
                # 这里 catch 的是**一切**异常，不只是 ImportError：LanceDB 在缺轮子的平台上
                # 会在 import 内部抛，也可能连到目录权限问题。
                # 代价是"代码 bug"也会被当成"环境不支持"吞掉 —— 所以必须 warning 而非静默，
                # 且降级只影响性能/磁盘格式，不影响接口（两个后端形状一致）。
                import logging

                logging.getLogger(__name__).warning(
                    "LanceDB 不可用（%s: %s），降级为 NumPy 索引", type(exc).__name__, exc
                )
        # 降级后的**副作用要提前知道**：两个后端的索引落在不同目录
        # （data/lance vs data/vectors），不会互相看到。
        # 也就是说从 lancedb 降级到 numpy 后，检索会命中一个空库而不是"旧索引还在"——
        # 看到空结果先确认后端，而不是以为数据丢了。
        _default = VectorStore(settings)
        logger_backends("numpy")
        return _default
    return _default


def logger_backends(name: str) -> None:
    import logging

    logging.getLogger(__name__).info("向量后端：%s", name)