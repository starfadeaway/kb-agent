"""LanceDB 后端（生产默认）——与 NumPy 后端共用同一套 Collection 接口。

## 为什么换成 LanceDB

ChromaDB 在本机不可用（1.5.9 索引不落盘，换进程必崩），
NumPy 精确索引能跑通但简历上"自己写了个向量库"始终是个减分项。
实测 LanceDB 0.40 在 Python 3.13 上：
- cp313 预编译轮子齐全，`pip install lancedb` 直接成功；
- **跨进程读回正常**（2314 行，另一个进程能查能过滤）；
- 磁盘 10.8MB，与 NumPy 的 9.5MB 同量级。

## 一个必须记住的坑：ANN 会改变排名，所以这里**只有精确扫描这一条路**

`table.create_index(metric="cosine")` 在 0.40 里已经废弃，
它建的是 **IVF-PQ 近似索引**，会把向量量化。
实测同一句查询：
- 暴力扫描 top3：太空封印站 / 泛银河商业公司 / 沉陆海域露莎卡
- 建 IVF-PQ 之后 top3：太空封印站 / 泛银河商业公司 / **出云显世与高天神国**（第三名变了）

评测里悄悄改变排名是不可接受的 —— 那样指标差异就归因不到检索策略头上了。
所以本后端**不提供 ANN 路径**：`query()` 只做暴力精确扫描，全模块没有任何
`create_index` 调用。

这不是「还没实现的待办」，是一个**已测量的决策**：当前语料 10843 chunk
（sr 3932 / ys 5413 / zzz 1498），暴力扫描已是毫秒级，ANN 的速度收益换不来
排名不确定性。

曾经有过 `VECTOR_ANN` 配置项，它被读进 Settings、传进 `LanceCollection`、
写进侧车 meta、再读回来，**但查询路径从不读它，全模块也没有 `create_index`** ——
配成 true 什么都不会发生。那是一个假开关，已删除（见 `config.py` 与
`get_collection` 的注释）。真正的规模压力出现时，应该在这里新写一条
ANN 路径并重跑一遍评测，而不是先留一个不生效的旋钮。

## 分数换算

暴力扫描返回的是**平方 L2 距离**。对已 L2 归一化的向量：

    d = ‖a - b‖² = 2 - 2·cos  ⇒  cos = 1 - d / 2

实测校验：cos=0.6978 ⇒ d = 2 - 2×0.6978 = 0.6044，与返回的 _distance 完全一致。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from app.store.types import RetrievedChunk, build_where_clause, model_slug


def _sql_escape(v: str) -> str:
    """把值里的单引号翻倍。仅转义值，**key 不转义**。

    前提：传进来的 key 全部是本项目入库时写死的字段名（game / page_title /
    last_revision_ts 等），值才可能含用户或语料内容。
    任何把 metadata 的键名直接透传进 where 的改动都会变成 SQL 注入入口。
    """
    return str(v).replace("'", "''")


def where_to_sql(where: dict | None) -> str | None:
    """把内部过滤字典翻译成 LanceDB 的 SQL where 子句。

    与 NumPy 后端的 _match_where 保持语义一致：
    {"game": "sr"} | {"page_title": {"$in": [...]}} | {"last_revision_ts": {"$gte": "..."}} | {"$and": [...]}

    返回 None 表示"无过滤"，与 `$and` 分支里"各子条件都翻译不出来"的情况同义 ——
    也就是说：一个不认识的条件会被**静默忽略**而不是报错。
    这是与 NumPy 后端对齐的代价（那边同样是忽略未知条件），
    换来的好处是 `tests/test_store_parity.py` 能要求两个后端对同一 where 给出同一结果集。

    ⚠️ `$in` 的值列表为空时会生成 `key IN ()` 这种非法 SQL（由数据库层报错，
    而不是这里报错）。所以任何新增 `$in` 的调用方都必须先判空 ——
    `build_where` 现在的写法（`if page_titles:` 才加条件）正好满足这一点。
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


class LanceCollection:
    def __init__(self, root: Path, name: str, metadata: dict[str, Any] | None = None) -> None:
        self.root = root
        self.name = name
        self.meta = metadata or {}
        self._db = None
        self._tbl = None

    @classmethod
    def open(cls, root: Path, name: str) -> LanceCollection:
        """打开一个**已存在**的表，并把 `_save_meta` 写下的元数据读回来。

        `__init__` 只能接住调用方手头的 meta；直接 new 一个出来枚举目录时
        拿不到，于是 game / embed_model / variant 全是空 —— 元数据存了但没人读，
        `list_collections()` 于是退化成"只有名字和条数"。

        前提：侧车文件是**建表时**写下的。也就是说改了游戏/模型的 collection 命名规则后，
        老 collection 的侧车里存的是旧命名下的 meta，`list_collections` 会照实显示旧值 ——
        这是对的（显示事实），不要在 open 里"修正"成当前配置。

        侧车里曾经还有一个 `ann` 字段，与它对应的 `vector_ann` 配置项都已删除
        （假开关，见模块 docstring）。老侧车里的残留键会被下面的
        `blob.get("meta")` 安静忽略 —— 未知键不参与任何行为，所以不必做迁移。
        """
        saved_meta = {}
        sidecar = root / f"{name}.meta.json"
        if sidecar.is_file():
            try:
                blob = json.loads(sidecar.read_text(encoding="utf-8"))
                saved_meta = blob.get("meta") or {}
            except (OSError, ValueError, AttributeError):
                # 元数据丢了不该让整个枚举挂掉，退化成空 meta 即可
                saved_meta = {}
        return cls(root, name, saved_meta)

    # ---------------- 表管理 ----------------
    @property
    def path(self) -> Path:
        return self.root / f"{self.name}.lance"

    def _table(self):
        """惰性连接 + 惰性 open_table，**返回的表句柄被本对象缓存复用**。

        前提：表必须已经存在（`upsert` 的建表分支会先把 `self._tbl` 置 None 再建，
        用完重新走这条路径）。对不存在的名字 open_table 会抛，
        所以每个调用方都先用 `count()`（它先判 `path.exists()`）挡一道。
        """
        if self._tbl is None:
            import lancedb

            self._db = lancedb.connect(str(self.root))
            self._tbl = self._db.open_table(self.name)
        return self._tbl

    def count(self) -> int:
        """行数。空库返回 0 而不是抛错，让上层能把"没数据"和"连不上"区分开。"""
        if not self.path.exists():
            return 0
        return self._table().count_rows()

    def drop(self) -> bool:
        """删表 + **必须把缓存的 `_tbl`/`_db` 置空**，否则句柄指向已删的表。"""
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
        """写入。与 NumPy 后端语义对齐：**同 id 视为替换而非追加**。

        两处必须知道的前提：
        1. LanceDB 本身不提供主键 upsert，所以要自己先删后加（见下面的 delete+add）。
           因此"刷新"的粒度是整条记录 —— 正文、metadata、vector 一起换掉，
           这正是 NumPy 后端曾经只刷 vector 造成"新向量配旧正文"时要对齐的目标行为。
        2. metadata 里非标量值（list / dict）会被**丢弃**（下面那个 isinstance 过滤）。
           所以 VectorStore 侧能存的 metadata，这个后端未必存得下，
           换后端时别往 metadata 里塞结构化字段。
        """
        import lancedb

        # 非标量字段直接不写入 —— LanceDB 的表 schema 由首行数据推断，
        # 塞进 list/dict 会让建表 schema 与后续批次不一致。
        rows = [
            {
                "id": cid,
                "vector": list(map(float, e)),
                "text": doc,
                **{k: v for k, v in (m or {}).items() if isinstance(v, str | int | float | bool) or v is None},
            }
            for cid, e, doc, m in zip(ids, embeddings, documents, metadatas, strict=False)
        ]

        # 建表分支：mode="overwrite" 保证重跑不会报"表已存在"。
        # 建完立刻把 `_tbl` 置空，下一次访问走 `_table()` 重新打开拿到的才是新表。
        if not self.path.exists():
            self._db = lancedb.connect(str(self.root))
            self._db.create_table(self.name, data=rows, mode="overwrite")
            self._tbl = None
            self._save_meta()
            return

        tbl = self._table()
        # ⚠️ 这里的 `search(rows[0]["vector"])` **不是检索，只是为了把全表行读出来**：
        # 向量库没有"列出全部 id"的轻接口，就用一次最大 limit 的扫描来收集 existing_ids。
        # 代价是每次 upsert 都 O(N) 全量搬运，N 到十万级会成为写入瓶颈 ——
        # 现在能接受只因为语料是千级、且入库是离线批处理。
        existing = tbl.search(rows[0]["vector"]).limit(1_000_000).to_list()
        existing_ids = {r["id"] for r in existing}
        stale = [cid for cid in ids if cid in existing_ids]
        if stale:
            quoted = ", ".join(f"'{_sql_escape(c)}'" for c in stale)
            tbl.delete(f"id IN ({quoted})")
        tbl.add(rows)

    def _save_meta(self) -> None:
        """写侧车 meta.json。**只在建表时调用**，后续 upsert 不刷新。

        于是 `embed_model` / `variant` 记录的是"建那一刻"的配置；
        靠改 variant 重建索引是对的做法（不会污染旧表），
        但同一个 variant 内换模型就会让 meta 与实际内容不符。
        """
        p = self.root / f"{self.name}.meta.json"
        p.write_text(json.dumps({"meta": self.meta}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 读取 ----------------
    def get(
        self, ids: Sequence[str] | None = None, where: dict | None = None, include: Sequence[str] = ()
    ) -> dict:
        """按 id / where 取行。与 NumPy 后端同样形状，同样两个前提：
        ids 与 where 二选一、不存在的 id 静默丢弃、include 没点的键返回空列表。

        代价：走的是「全表扫 + limit(N) + where 后过滤」，N 取自 `count()`，
        所以复杂度是 O(N) —— 这只用于取回归档原文，量级可控，别挪进高频路径。
        """
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

        # `search()` 不带向量在这里是**故意**的：借它拿到一个"全表、不排序"的查询，
        # where 由 LanceDB 侧下推到过滤，而不是把行拉回内存再筛。
        rows = self._table().search().limit(n).where(sql).to_list() if sql else self._table().search().limit(n).to_list()
        out = {"ids": [r["id"] for r in rows]}
        out["documents"] = [r.get("text", "") for r in rows] if "documents" in include else []
        # metadata 是从行里"减掉"三个内部列还原出来的 —— 所以 upsert 时过滤掉的
        # 非标量字段在这里也拿不回来，两处是同一个约束的两面。
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
        """暴力精确 top-k。返回结构与 NumPy 后端逐字一致（含 `distances` 的含义）。

        两个前提，缺一个结果就不对：
        1. **写入侧必须 L2 归一化** —— 下面的 `d/2` 换算只在归一化向量上成立；
           没归一化时它是错的常数变换，而排名仍然"看起来正常"，评测很难发现。
        2. **不要在此处建 ANN 索引** —— 见模块 docstring 的实测：IVF-PQ 会改变 top3 排名，
           那样指标差异就归因不到检索策略上了。本后端刻意只有精确扫描一条路径。
        两个空结果出口（空库 / 过滤后为空）与 NumPy 后端结构一致，下游 `[0]` 不会 IndexError。
        """
        if self.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = query_embeddings[0] if len(query_embeddings) == 1 else list(query_embeddings)[0]
        tbl = self._table()
        sql = where_to_sql(where)
        # where 是在查询计划里作为过滤条件下推的，与 limit 的**书写先后无关**：
        # n_results 是"过滤后的 top-k"，不是"先取 k 条再过滤"。
        # （这里写成 .limit() 在前只是可读性，LanceDB 不会因此先截断再筛。）
        qb = tbl.search(list(map(float, q))).limit(n_results)
        if sql:
            qb = qb.where(sql)
        rows = qb.to_list()
        if not rows:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        ids = [r["id"] for r in rows]
        docs = [r.get("text", "") for r in rows]
        metas = [{k: v for k, v in r.items() if k not in ("vector", "text", "_distance")} for r in rows]
        # 平方 L2 → (1 - cos)：d = 2 - 2cos ⇒ d/2 = 1 - cos。
        # 这一步刻意与 NumPy 后端 `1.0 - pool_scores` 对齐成同一种"距离"，
        # 上层 `LanceVectorStore.query` 再统一 `1 - d` 换回余弦 ——
        # 所以跨后端的 RetrievedChunk.score 可直接比较，A/B 换后端不会引入分数量纲差。
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
        self.root = self.s.data_root_abs / "lance"
        self.root.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, LanceCollection] = {}

    @staticmethod
    def _slug(model: str) -> str:
        # 与 NumPy 后端共用同一个正则（`store/types.py`）—— 两处各写一遍时，
        # collection 名一旦漂移，表现是「找不到库」而不是报错。
        return model_slug(model)

    def collection_name(self, game: str, model: str | None = None, variant: str = "base") -> str:
        model = model or self.s.embed_model_main
        return f"{self.s.collection_prefix}_{game}_{self._slug(model)}_{variant}"

    def get_collection(self, game: str, model: str | None = None, variant: str = "base") -> LanceCollection:
        """返回本 collection 的句柄，进程内按名字缓存（同名复用同一实例）。

        命名里必须带 variant —— 不同分块/embedding 参数绝不写进同一张表，
        否则实验 A–E 的结果会互相污染，且事后无法拆分。

        这里曾有 `VECTOR_ANN` 配置项与 `ann` 形参，现已删除：它们只被
        `_save_meta` / `open` 存取，查询路径从不读，全仓也没有 `create_index`，
        配成 true 既不会变快也不会建近似索引 —— 一个看起来能开、实际不生效的旋钮，
        比明说「本后端只有精确扫描」更糟。实测依据见模块 docstring。
        """
        name = self.collection_name(game, model, variant)
        if name not in self._collections:
            self._collections[name] = LanceCollection(
                self.root, name, {"game": game, "embed_model": model or self.s.embed_model_main, "variant": variant}
            )
        return self._collections[name]

    def upsert(self, *, game: str, model: str | None, variant: str, ids, embeddings, documents, metadatas) -> None:
        """空批次直接跳过：否则建表分支会用空 rows 去 create_table，抛难以定位的错。"""
        if len(ids):
            self.get_collection(game, model, variant).upsert(ids, embeddings, documents, metadatas)

    def count(self, game: str, model: str | None = None, variant: str = "base") -> int:
        return self.get_collection(game, model, variant).count()

    def drop(self, game: str, model: str | None = None, variant: str = "base") -> bool:
        return self.get_collection(game, model, variant).drop()

    def list_collections(self) -> list[dict]:
        """枚举 `*.lance` 目录并读回侧车 meta。

        `d.name[:-6]` 是去掉 `.lance` 后缀（6 个字符），改成别的前缀就得同步改长度 ——
        这里写成魔法数字很容易在换命名时被漏掉，`list_collections` 就会静默列出空名字。
        `count()` 只读元数据行数，不加载向量，代价远低于 NumPy 后端同名方法。
        """
        out = []
        for d in sorted(self.root.glob("*.lance")) if self.root.is_dir() else []:
            col = LanceCollection.open(self.root, d.name[:-6])
            out.append({"name": col.name, "count": col.count(), **col.meta})
        return out

    def build_where(self, *, game: str | None = None, page_titles: Iterable[str] | None = None,
                    min_revision_ts: str | None = None) -> dict | None:
        """与 NumPy 后端共用实现（`store/types.py` 的 `build_where_clause`）。"""
        return build_where_clause(
            game=game, page_titles=page_titles, min_revision_ts=min_revision_ts
        )

    def query(self, *, query_embedding, game: str, model: str | None = None, variant: str = "base",
              top_k: int = 5, where: dict | None = None) -> list[RetrievedChunk]:
        """与 `VectorStore.query` 完全同签名同语义，`score` 同样是换回来的余弦。

        这是两个后端能互换的**唯一保证**：`hybrid._fuse` 只依赖
        "按相似度降序的 RetrievedChunk 列表"，不关心它从哪来。
        任何一处改动若让 score 的量纲或排序与 NumPy 后端不同，
        跨后端的评测数字就不再可比 —— `tests/test_store_parity.py` 是这件事的守门测试。
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
            return []   # 空结果即"没召回到"，与 NumPy 后端同样不抛错
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        # `score = 1 - (1 - cos) = cos`：与 NumPy 后端同一行换算，
        # 也是"距离语义在两层之间流转"的终点，再往上就只有余弦了。
        return [
            RetrievedChunk(chunk_id=str(c), text=str(d or ""), score=round(1.0 - float(dist), 6), metadata=dict(m or {}))
            for c, d, m, dist in zip(ids, docs, metas, dists, strict=False)
        ]

    def get_by_ids(self, *, game: str, model: str | None, variant: str, ids) -> dict[str, RetrievedChunk]:
        """按 id 取回原文。`score=1.0` 同样是占位，不是相似度（与 NumPy 后端一致）。"""
        col = self.get_collection(game, model, variant)
        got = col.get(ids=list(ids), include=("documents", "metadatas"))
        return {
            str(c): RetrievedChunk(chunk_id=str(c), text=str(t or ""), score=1.0, metadata=dict(m or {}))
            for c, t, m in zip(got["ids"], got["documents"], got["metadatas"], strict=False)
        }
