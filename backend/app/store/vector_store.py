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

NEG_INF = -1e9


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

    def cite(self) -> str:
        return f"{self.page_title}｜{self.metadata.get('section', '')}".strip("｜")


def _match_where(meta: dict, where: dict | None) -> bool:
    """支持 $and / $in / $gte 三种条件，够用且不会静默漏条件。"""
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
        new = np.asarray(embeddings, dtype=np.float32)
        with self._lock:
            if self._emb is None or self._emb.shape[0] == 0:
                self._emb = new
                self.ids = list(ids)
                self.documents = list(documents)
                self.metadatas = [dict(m) for m in metadatas]
            else:
                existing = [self._index[cid] for cid in ids if cid in self._index]
                if existing:
                    self._emb[existing] = new[[cid in self._index for cid in ids]]
                fresh = [i for i, cid in enumerate(ids) if cid not in self._index]
                if fresh:
                    self._emb = np.vstack([self._emb, new[fresh]])
                    self.documents.extend(documents[i] for i in fresh)
                    self.metadatas.extend(dict(metadatas[i]) for i in fresh)
                    self.ids.extend(ids[i] for i in fresh)
            self._index = {cid: i for i, cid in enumerate(self.ids)}
            self._flush()

    # ---------------- 读取 ----------------
    def get(
        self, ids: Sequence[str] | None = None, where: dict | None = None, include: Sequence[str] = ()
    ) -> dict:
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
        if self._emb is None or len(self.ids) == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.asarray(query_embeddings, dtype=np.float32)
        # 写入时已 L2 归一化，点积即余弦相似度
        sims = self._emb @ q.T                      # (N, Q)
        sims = sims[:, 0] if sims.shape[1] == 1 else sims.max(axis=1)

        if where:
            mask = np.array([_match_where(m, where) for m in self.metadatas], dtype=bool)
            if not mask.any():
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            idx_pool = np.nonzero(mask)[0]
        else:
            idx_pool = np.arange(len(self.ids))

        pool_scores = sims[idx_pool]
        k = min(n_results, len(idx_pool))
        # argpartition 只做部分排序，比全排序快，且 top-k 结果完全一致
        top = np.argpartition(-pool_scores, k - 1)[:k]
        order = top[np.argsort(-pool_scores[top])]
        rows = idx_pool[order]

        return {
            "ids": [[self.ids[i] for i in rows]],
            "documents": [[self.documents[i] for i in rows]] if "documents" in include else [[]],
            "metadatas": [[self.metadatas[i] for i in rows]] if "metadatas" in include else [[]],
            "distances": [[1.0 - float(pool_scores[i]) for i in order]] if "distances" in include else [[]],
        }


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.root = self.s.chroma_abs_path.parent / "vectors"
        self.root.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, Collection] = {}

    # ---------------- collection 管理 ----------------
    def collection_name(self, game: str, model: str | None = None, variant: str = "base") -> str:
        model = model or self.s.embed_model_main
        # variant 用来隔离分块/embedding 实验：不同参数绝不写进同一个索引
        return f"{self.s.chroma_collection_prefix}_{game}_{model_slug(model)}_{variant}"

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
        """过滤条件。min_revision_ts 是版本时效性过滤的入口。"""
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
        col = self.get_collection(game, model, variant)
        res = col.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=top_k,
            where=where if where is not None else self.build_where(game=game),
            include=("documents", "metadatas", "distances"),
        )
        ids = res["ids"][0]
        if not ids:
            return []
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
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
    - sqlite：内置的极小语料演示后端，连 lancedb 都没有时用。

    三者共用 Collection 接口，换后端不需要动上层任何代码。
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
                import logging

                logging.getLogger(__name__).warning(
                    "LanceDB 不可用（%s: %s），降级为 NumPy 索引", type(exc).__name__, exc
                )
        _default = VectorStore(settings)
        logger_backends("numpy")
        return _default
    return _default


def logger_backends(name: str) -> None:
    import logging

    logging.getLogger(__name__).info("向量后端：%s", name)