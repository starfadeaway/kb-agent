"""LanceDB 后端（生产默认）——与 NumPy 后端共用同一套 Collection 接口。

## 为什么换成 LanceDB

ChromaDB 在本机不可用（1.5.9 索引不落盘，换进程必崩），
NumPy 精确索引能跑通但简历上"自己写了个向量库"始终是个减分项。
实测 LanceDB 0.40 在 Python 3.13 上：
- cp313 预编译轮子齐全，`pip install lancedb` 直接成功；
- **跨进程读回正常**（2314 行，另一个进程能查能过滤）；
- 磁盘 10.8MB，与 NumPy 的 9.5MB 同量级。

## 一个必须记住的坑：默认走的是近似索引

`table.create_index(metric="cosine")` 在 0.40 里已经废弃，
它建的是 **IVF-PQ 近似索引**，会把向量量化。
实测同一句查询：
- 暴力扫描 top3：太空封印站 / 泛银河商业公司 / 沉陆海域露莎卡
- 建 IVF-PQ 之后 top3：太空封印站 / 泛银河商业公司 / **出云显世与高天神国**（第三名变了）

评测里悄悄改变排名是不可接受的 —— 那样指标差异就归因不到检索策略头上了。
所以默认 **ann=False（暴力精确扫描）**，要开 ANN 必须显式配置。

## 分数换算

暴力扫描返回的是**平方 L2 距离**。对已 L2 归一化的向量：

    d = ‖a - b‖² = 2 - 2·cos  ⇒  cos = 1 - d / 2

实测校验：cos=0.6978 ⇒ d = 2 - 2×0.6978 = 0.6044，与返回的 _distance 完全一致。
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _sql_escape(v: str) -> str:
    return str(v).replace("'", "''")


def where_to_sql(where: dict | None) -> str | None:
    """把内部过滤字典翻译成 LanceDB 的 SQL where 子句。

    与 NumPy 后端的 _match_where 保持语义一致：
    {"game": "sr"} | {"page_title": {"$in": [...]}} | {"last_revision_ts": {"$gte": "..."}} | {"$and": [...]}
    """
    if not where:
        return None
    if "$and" in where:
        parts = [where_to_sql(c) for c in where["$and"]]
        return "(" + " AND ".join(p for p in parts if p) + ")" if any(parts) else None

    clauses: list[str] = []
    for key, cond in where.items():
        if isinstance(cond, dict):
            if "$in" in cond:
                vals = ", ".join(f"'{_sql_escape(v)}'" for v in cond["$in"])
                clauses.append(f"{key} IN ({vals})")
            if "$gte" in cond:
                clauses.append(f"{key} >= '{_sql_escape(cond['$gte'])}'")
            if "$lte" in cond:
                clauses.append(f"{key} <= '{_sql_escape(cond['$lte'])}'")
        else:
            clauses.append(f"{key} = '{_sql_escape(cond)}'")
    return " AND ".join(c for c in clauses if c) or None


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


class LanceCollection:
    def __init__(self, root: Path, name: str, metadata: dict[str, Any] | None = None, ann: bool = False) -> None:
        self.root = root
        self.name = name
        self.meta = metadata or {}
        self.ann = ann
        self._db = None
        self._tbl = None

    @classmethod
    def open(cls, root: Path, name: str) -> LanceCollection:
        """打开一个**已存在**的表，并把 `_save_meta` 写下的元数据读回来。

        `__init__` 只能接住调用方手头的 meta；直接 new 一个出来枚举目录时
        拿不到，于是 game / embed_model / variant 全是空 —— 元数据存了但没人读，
        `list_collections()` 于是退化成"只有名字和条数"。
        """
        saved_meta, saved_ann = {}, False
        sidecar = root / f"{name}.meta.json"
        if sidecar.is_file():
            try:
                blob = json.loads(sidecar.read_text(encoding="utf-8"))
                saved_meta = blob.get("meta") or {}
                saved_ann = bool(blob.get("ann", False))
            except (OSError, ValueError, AttributeError):
                # 元数据丢了不该让整个枚举挂掉，退化成空 meta 即可
                saved_meta, saved_ann = {}, False
        return cls(root, name, saved_meta, saved_ann)

    # ---------------- 表管理 ----------------
    @property
    def path(self) -> Path:
        return self.root / f"{self.name}.lance"

    def _table(self):
        if self._tbl is None:
            import lancedb

            self._db = lancedb.connect(str(self.root))
            self._tbl = self._db.open_table(self.name)
        return self._tbl

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return self._table().count_rows()

    def drop(self) -> bool:
        if self.path.exists():
            shutil.rmtree(self.path)
        self._tbl = None
        self._db = None
        return True

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        import lancedb

        rows = [
            {
                "id": cid,
                "vector": list(map(float, e)),
                "text": doc,
                **{k: v for k, v in (m or {}).items() if isinstance(v, str | int | float | bool) or v is None},
            }
            for cid, e, doc, m in zip(ids, embeddings, documents, metadatas, strict=False)
        ]

        if not self.path.exists():
            self._db = lancedb.connect(str(self.root))
            self._db.create_table(self.name, data=rows, mode="overwrite")
            self._tbl = None
            self._save_meta()
            return

        tbl = self._table()
        existing = tbl.search(rows[0]["vector"]).limit(1_000_000).to_list()
        existing_ids = {r["id"] for r in existing}
        stale = [cid for cid in ids if cid in existing_ids]
        if stale:
            quoted = ", ".join(f"'{_sql_escape(c)}'" for c in stale)
            tbl.delete(f"id IN ({quoted})")
        tbl.add(rows)

    def _save_meta(self) -> None:
        p = self.root / f"{self.name}.meta.json"
        p.write_text(json.dumps({"meta": self.meta, "ann": self.ann}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 读取 ----------------
    def get(
        self, ids: Sequence[str] | None = None, where: dict | None = None, include: Sequence[str] = ()
    ) -> dict:
        n = self.count()
        if n == 0:
            return {"ids": [], "documents": [], "metadatas": []}

        clauses: list[str] = []
        if ids is not None:
            quoted = ", ".join(f"'{_sql_escape(i)}'" for i in ids)
            clauses.append(f"id IN ({quoted})")
        if where:
            w = where_to_sql(where)
            if w:
                clauses.append(w)
        sql = " AND ".join(c for c in clauses if c) or None

        rows = self._table().search().limit(n).where(sql).to_list() if sql else self._table().search().limit(n).to_list()
        out = {"ids": [r["id"] for r in rows]}
        out["documents"] = [r.get("text", "") for r in rows] if "documents" in include else []
        out["metadatas"] = [
            {k: v for k, v in r.items() if k not in ("vector", "text", "_distance")} for r in rows
        ] if "metadatas" in include else []
        return out

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 5,
        where: dict | None = None,
        include: Sequence[str] = ("documents", "metadatas", "distances"),
    ) -> dict:
        if self.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = query_embeddings[0] if len(query_embeddings) == 1 else list(query_embeddings)[0]
        tbl = self._table()
        sql = where_to_sql(where)
        qb = tbl.search(list(map(float, q))).limit(n_results)
        if sql:
            qb = qb.where(sql)
        rows = qb.to_list()
        if not rows:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        ids = [r["id"] for r in rows]
        docs = [r.get("text", "") for r in rows]
        metas = [{k: v for k, v in r.items() if k not in ("vector", "text", "_distance")} for r in rows]
        # 平方 L2 → 余弦（向量已归一化）：cos = 1 - d / 2
        dists = [round(float(r.get("_distance", 0.0)) / 2.0, 6) for r in rows]
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}

    # ---------------- 与 NumPy 后端对齐的辅助 ----------------
    def all_rows(self) -> list[dict]:
        """导出全部行（供自检/迁移使用）。"""
        n = self.count()
        if n == 0:
            return []
        rows = self._table().search().limit(n).to_list()
        return [{k: v for k, v in r.items() if k != "_distance"} for r in rows]


class LanceVectorStore:
    """与 VectorStore 同名的 LanceDB 实现。"""

    def __init__(self, settings) -> None:
        self.s = settings
        self.root = self.s.chroma_abs_path.parent / "lance"
        self.root.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, LanceCollection] = {}

    @staticmethod
    def _slug(model: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")

    def collection_name(self, game: str, model: str | None = None, variant: str = "base") -> str:
        model = model or self.s.embed_model_main
        return f"{self.s.chroma_collection_prefix}_{game}_{self._slug(model)}_{variant}"

    def get_collection(self, game: str, model: str | None = None, variant: str = "base") -> LanceCollection:
        name = self.collection_name(game, model, variant)
        if name not in self._collections:
            ann = str(getattr(self.s, "vector_ann", "false")).lower() in {"1", "true", "yes", "on"}
            self._collections[name] = LanceCollection(
                self.root, name, {"game": game, "embed_model": model or self.s.embed_model_main, "variant": variant}, ann
            )
        return self._collections[name]

    def upsert(self, *, game: str, model: str | None, variant: str, ids, embeddings, documents, metadatas) -> None:
        if len(ids):
            self.get_collection(game, model, variant).upsert(ids, embeddings, documents, metadatas)

    def count(self, game: str, model: str | None = None, variant: str = "base") -> int:
        return self.get_collection(game, model, variant).count()

    def drop(self, game: str, model: str | None = None, variant: str = "base") -> bool:
        return self.get_collection(game, model, variant).drop()

    def list_collections(self) -> list[dict]:
        out = []
        for d in sorted(self.root.glob("*.lance")) if self.root.is_dir() else []:
            col = LanceCollection.open(self.root, d.name[:-6])
            out.append({"name": col.name, "count": col.count(), **col.meta})
        return out

    def build_where(self, *, game: str | None = None, page_titles: Iterable[str] | None = None,
                    min_revision_ts: str | None = None) -> dict | None:
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

    def query(self, *, query_embedding, game: str, model: str | None = None, variant: str = "base",
              top_k: int = 5, where: dict | None = None) -> list[RetrievedChunk]:
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
            RetrievedChunk(chunk_id=str(c), text=str(d or ""), score=round(1.0 - float(dist), 6), metadata=dict(m or {}))
            for c, d, m, dist in zip(ids, docs, metas, dists, strict=False)
        ]

    def get_by_ids(self, *, game: str, model: str | None, variant: str, ids) -> dict[str, RetrievedChunk]:
        col = self.get_collection(game, model, variant)
        got = col.get(ids=list(ids), include=("documents", "metadatas"))
        return {
            str(c): RetrievedChunk(chunk_id=str(c), text=str(t or ""), score=1.0, metadata=dict(m or {}))
            for c, t, m in zip(got["ids"], got["documents"], got["metadatas"], strict=False)
        }
