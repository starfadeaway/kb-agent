"""混合检索：BM25 稀疏 + 向量稠密 + RRF 融合 +（可选）Cross-Encoder Rerank。

对应实验 A 的阶梯消融，每一级只打开一个开关：
  A0 = 仅 BM25
  A1 = 仅向量
  A2 = BM25 + 向量（RRF 融合）
  A3 = A2 + Rerank

为什么稀疏和稠密都要留：星铁的「位面饰品」「行迹」「命途」是低频专名，
词面匹配比语义相似度更稳；而「怎么配队」这类问题纯 BM25 一定召不回。
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from app.core.config import Settings, get_settings
from app.llm.client import LLMClient
from app.store.vector_store import RetrievedChunk, VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """检索配置快照。所有字段都会随评测结果落库，保证可复现。"""

    game: str = "sr"
    embed_model: str | None = None
    variant: str = "base"
    use_bm25: bool = True
    use_vector: bool = True
    use_rerank: bool = True
    bm25_top: int = 20
    vector_top: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    min_revision_ts: str | None = None
    page_titles: list[str] | None = None

    # 送进 Cross-Encoder 的候选上限。融合后可能有 30+ 条，全送重排要多花 ~130ms
    # （实测 31 文档 449ms vs 20 文档 321ms），而 top-20 之外的本来就进不了 top-5。
    # 设为 0 表示不裁剪（还原优化前的行为，供 A/B 对照）。
    rerank_candidates: int = 20

    @classmethod
    def from_settings(cls, *, settings=None, **over) -> RetrievalConfig:
        """**从 `.env` 派生默认值**，显式传入的 `over` 优先。

        为什么必须有这个入口 —— 这是本项目第三次栽在同一类问题上：
        字段「声明了默认值」不等于「那个默认值真的被用」。
        之前 `RETRIEVE_TOP_K` / `RETRIEVE_CANDIDATE` / `RRF_K` 三个 env var
        在 `config.py` 里定义得整整齐齐，`.env.example` 也列了，
        但 `main.py` 构造 `RetrievalConfig` 时压根没传这三个值，
        于是 `.env` 里怎么改都不生效，而检索照常跑、**不报任何错**。

        复制 `IngestConfig.from_settings()` 的既定范式，不要再各写一套推导方式。
        """
        s = settings or get_settings()
        base = {
            "embed_model": s.embed_model_main,
            # top_k 与候选池共用一个旋钮：候选池必须 >= top_k，
            # 否则用户把 top_k 调大到超过候选池就会静默拿不满结果。
            "final_top_k": s.retrieve_top_k,
            "bm25_top": s.retrieve_candidate,
            "vector_top": s.retrieve_candidate,
            "rerank_candidates": s.retrieve_candidate,
            "rrf_k": s.rrf_k,
        }
        base.update(over)   # 显式传入的实验参数必须覆盖 .env，否则调参调不动
        return cls(**base)

    def label(self) -> str:
        if self.use_bm25 and self.use_vector and self.use_rerank:
            return "A3"
        if self.use_bm25 and self.use_vector:
            return "A2"
        if self.use_vector:
            return "A1"
        return "A0"

    def as_dict(self) -> dict:
        """**必须自动派生，不要手写字段清单。**

        这个方法有两个消费者，都靠它保证可复现：
        - `main.py` 的 `/api/retrieve` 响应（前端据此显示实际生效的过滤条件）
        - `run_eval.py` 落库的 `config` 快照（实验结果的可复现凭据）

        之前这里是手写字典，加 `min_revision_ts` / `page_titles` 时忘了同步 ——
        结果是"带版本过滤跑的评测"落库后看不出带过滤，
        而类注释里那句「所有字段都会随评测结果落库」就成了假承诺。
        `dataclasses.asdict` 直接从字段声明派生，新增字段自动跟上。
        """
        return dataclasses.asdict(self)


@dataclass
class RetrievalHit:
    chunk: RetrievedChunk
    score: float
    rank_source: str            # bm25 / vector / rrf / rerank
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None
    fused_score: float = 0.0

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class BM25Index:
    """进程内稀疏索引。

    语料规模在十万 chunk 以内时，重建成本 < 1s，比引 Elasticsearch 划算；
    真到需要分布式再换，对上层接口不变。
    """

    chunk_ids: list[str] = field(default_factory=list)
    corpus: list[str] = field(default_factory=list)
    _bm25: BM25Okapi | None = None

    @classmethod
    def build(cls, chunk_ids: Sequence[str], corpus: Sequence[str], tokenizer: Callable[[str], list[str]]) -> BM25Index:
        idx = cls(chunk_ids=list(chunk_ids), corpus=list(corpus))
        idx._bm25 = BM25Okapi([tokenizer(t) for t in idx.corpus])
        return idx

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def search(self, query_tokens: list[str], top: int) -> list[tuple[str, float, int]]:
        if self._bm25 is None or not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top]
        return [
            (self.chunk_ids[i], float(scores[i]), i)
            for i in order
            if scores[i] > 0
        ]


def tokenize(text: str) -> list[str]:
    """中文用 jieba 切词 + 英文数字保留 + 中文 2-gram 兜底。

    游戏专名（位面饰品、行迹）分词后仍可能整词未命中，
    补一层 2-gram 能明显改善召回，且不引入新依赖。
    """
    import jieba

    lowered = text.lower()
    words = [w.strip() for w in jieba.lcut(lowered) if w.strip()]
    grams: list[str] = []
    cjk = "".join(ch for ch in text if "一" <= ch <= "鿿")
    for i in range(len(cjk) - 1):
        grams.append(cjk[i : i + 2])
    return words + grams


class HybridRetriever:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.s = settings or get_settings()
        self.llm = llm
        self.store = store or get_vector_store(self.s)
        self._bm25_cache: dict[tuple, BM25Index] = {}
        self._text_cache: dict[tuple, tuple] = {}
        # 这两个缓存都无上限：_cache_key 含 page_titles / min_revision_ts，
        # 理论上可以无限膨胀（一份全量语料 + 一个 BM25 索引 ≈ 几十 MB）。
        # 当前只有 sr / zzz 两个库、page_titles 实际不常用，所以够用。
        # 一旦要支持"按页面过滤检索"或"按版本时间过滤"，必须换成有界 LRU ——
        # 这里显式记下来，因为这是个会悄悄吃掉内存的地方，不是崩溃那种明显的坏。
        self._max_cached_indexes = 4

    # ---------------- 语料加载 ----------------
    def _cache_key(self, cfg: RetrievalConfig):
        return (
            cfg.game,
            cfg.embed_model or self.s.embed_model_main,
            cfg.variant,
            tuple(sorted(cfg.page_titles or [])),
            cfg.min_revision_ts or "",
        )

    def _load_corpus(self, cfg: RetrievalConfig) -> tuple[list[str], list[str], dict[str, RetrievedChunk]]:
        """从向量库拉全量文本，用于建 BM25 索引和把 chunk 映射回页面。

        实验规模（万级 chunk）下一次性拉全量是可接受的；
        所以实验报告里注明「稀疏检索未做工程优化，
        目标是隔离检索策略本身的差异」。
        """
        key = self._cache_key(cfg)
        if key in self._text_cache:
            ids, corpus, meta = self._text_cache[key]
            return ids, corpus, meta

        col = self.store.get_collection(cfg.game, cfg.embed_model, cfg.variant)
        where = self.store.build_where(
            game=cfg.game,
            page_titles=cfg.page_titles,
            min_revision_ts=cfg.min_revision_ts,
        )
        data = col.get(where=where, include=["documents", "metadatas"])
        ids = list(data.get("ids") or [])
        docs = list(data.get("documents") or [""] * len(ids))
        metas = list(data.get("metadatas") or [{}] * len(ids))
        mapping = {
            cid: RetrievedChunk(chunk_id=cid, text=doc, score=1.0, metadata=dict(m or {}))
            for cid, doc, m in zip(ids, docs, metas, strict=False)
        }
        self._text_cache[key] = (ids, docs, mapping)
        # 淘汰必须放在这里而不是只在 _get_bm25 里：
        # 关掉 BM25 的配置（A1/A2）只走 _load_corpus，
        # 放错位置的话那条路径的语料副本永远不会被回收。
        self._evict_locked()
        return ids, docs, mapping

    def _get_bm25(self, cfg: RetrievalConfig) -> BM25Index:
        key = self._cache_key(cfg)
        if key not in self._bm25_cache:
            ids, docs, _ = self._load_corpus(cfg)
            self._bm25_cache[key] = BM25Index.build(ids, docs, tokenize)
            self._evict_locked()
            logger.info("BM25 索引: %d chunks (game=%s)", len(ids), cfg.game)
        return self._bm25_cache[key]

    def _evict_locked(self) -> None:
        """超量时丢最早建的索引。

        不用 LRU 是因为建索引很贵（~1.8s），更该淘汰「最久没被用到」的而不是
        「最早建的」。这里用「插入顺序」近似 FIFO 就够 —— 实际会同时用到
        的库最多两三个，淘汰策略的细微差别测不出来。
        """
        cap = self._max_cached_indexes
        while len(self._bm25_cache) > cap:
            self._bm25_cache.pop(next(iter(self._bm25_cache)), None)
        # _text_cache 与 _bm25_cache 一一对应（同一个 key），淘汰时同步清，
        # 否则会留下没人用的语料副本白占内存。
        while len(self._text_cache) > cap:
            k = next(iter(self._text_cache))
            self._text_cache.pop(k, None)
            self._bm25_cache.pop(k, None)

    def invalidate(self) -> None:
        self._bm25_cache.clear()
        self._text_cache.clear()

    def stats(self) -> dict:
        return {"bm25_cached": len(self._bm25_cache), "corpus_cached": len(self._text_cache)}

    # ---------------- 检索 ----------------
    async def retrieve(self, query: str, cfg: RetrievalConfig) -> list[RetrievalHit]:
        if not query.strip():
            return []
        _, _, mapping = self._load_corpus(cfg)
        if not mapping:
            return []

        bm25_hits: list[tuple[str, float, int]] = []
        if cfg.use_bm25:
            bm25_hits = self._get_bm25(cfg).search(tokenize(query), cfg.bm25_top)

        vec_hits: list[RetrievedChunk] = []
        if cfg.use_vector:
            if self.llm is None:
                raise RuntimeError("启用向量检索需要 LLMClient")
            qv = await self.llm.embed([query], model=cfg.embed_model)
            where = self.store.build_where(
                game=cfg.game, page_titles=cfg.page_titles, min_revision_ts=cfg.min_revision_ts
            )
            vec_hits = self.store.query(
                query_embedding=qv[0],
                game=cfg.game,
                model=cfg.embed_model,
                variant=cfg.variant,
                top_k=cfg.vector_top,
                where=where,
            )

        hits = self._fuse(bm25_hits, vec_hits, mapping, cfg)
        if cfg.use_rerank and hits:
            hits = await self._rerank(query, hits, cfg)
        return hits[: cfg.final_top_k]

    @staticmethod
    def _fuse(
        bm25_hits: list[tuple[str, float, int]],
        vec_hits: list[RetrievedChunk],
        mapping: dict[str, RetrievedChunk],
        cfg: RetrievalConfig,
    ) -> list[RetrievalHit]:
        """Reciprocal Rank Fusion。

        用排名而非分数做融合，是因为 BM25 分数和余弦相似度不同量纲，
        直接加权相加会被其中一路的分数分布带偏。
        """
        pool: dict[str, RetrievalHit] = {}

        for rank, (cid, _score, _idx) in enumerate(bm25_hits, start=1):
            if cid not in mapping:
                continue
            pool[cid] = RetrievalHit(
                chunk=mapping[cid], score=0.0, rank_source="bm25", bm25_rank=rank,
                fused_score=1.0 / (cfg.rrf_k + rank),
            )

        for rank, ch in enumerate(vec_hits, start=1):
            if ch.chunk_id in pool:
                h = pool[ch.chunk_id]
                h.vector_rank = rank
                h.rank_source = "bm25+vector"
                h.fused_score += 1.0 / (cfg.rrf_k + rank)
                h.score = ch.score
            else:
                pool[ch.chunk_id] = RetrievalHit(
                    chunk=ch, score=ch.score, rank_source="vector", vector_rank=rank,
                    fused_score=1.0 / (cfg.rrf_k + rank),
                )

        if not cfg.use_bm25 and not cfg.use_vector:
            return []

        fused = sorted(pool.values(), key=lambda h: -h.fused_score)
        if not cfg.use_bm25:
            for h in fused:
                h.rank_source = "vector"
        elif not cfg.use_vector:
            for h in fused:
                h.rank_source = "bm25"
        return fused

    async def _rerank(self, query: str, hits: list[RetrievalHit], cfg: RetrievalConfig) -> list[RetrievalHit]:
        """Cross-Encoder 重排。只对 bge 系向量模型有效，换模型对比时必须关闭。

        重排是整条检索链里最贵的一步（实测 449ms / 31 文档），
        所以只对融合结果的前 `rerank_candidates` 条做重排，
        剩下的在按重排分数排完序后原样接在后面。
        """
        if self.llm is None or not hits:
            return hits

        limit = cfg.rerank_candidates
        head, tail = (hits, []) if (limit <= 0 or len(hits) <= limit) else (hits[:limit], hits[limit:])

        docs = [h.chunk.text for h in head]
        try:
            ranked = await self.llm.rerank(query, docs, top_n=len(docs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rerank 失败，降级为融合排序：%s", exc)
            return hits
        out: list[RetrievalHit] = []
        for r in ranked:
            hit = head[r.index]
            hit.rerank_score = r.score
            hit.rank_source = hit.rank_source + "+rerank"
            hit.score = r.score
            out.append(hit)
        # 未参与重排的候选按原融合分追加，保证排序结果对候选数不敏感
        out.extend(tail)
        return out

    def warmup(self, cfg: RetrievalConfig) -> dict:
        """预热：拉全量语料 + 建 BM25 索引。

        实测冷启动要 2.8s（拉语料 717ms + jieba 分词建索引 2132ms），
        而这个索引在整个进程生命周期里只建一次 —— 让它发生在服务启动阶段，
        而不是由第一个用户请求承担。
        返回各环节耗时，供 /api/health 展示。
        """
        import time as _time

        out: dict[str, float] = {}
        t0 = _time.perf_counter()
        ids, docs, _ = self._load_corpus(cfg)
        out["load_corpus_ms"] = round((_time.perf_counter() - t0) * 1000, 1)

        t0 = _time.perf_counter()
        self._get_bm25(cfg)
        out["build_bm25_ms"] = round((_time.perf_counter() - t0) * 1000, 1)
        out["chunks"] = len(ids)
        return out


# --------------------------------------------------------------------------- #
# 进程级单例
# --------------------------------------------------------------------------- #
# 为什么必须是单例（实测出来的，不是设计出来的）：
#   HybridRetriever 的 BM25 索引和语料缓存都是**实例级**的，而建索引要
#   拉全量语料 + jieba 分词 —— 实测 1800ms。
#   如果每次请求都新建一个实例（之前 ToolContext.build 就是这么干的），
#   那么每个 /api/chat 请求都要重新付这 1800ms。
#   实测对照：
#       同一实例连续检索  →  3744ms → 16ms → 17ms
#       每次新建实例      →  1820ms → 1778ms → 1799ms
#
# 多游戏（sr / zzz）不冲突：_cache_key() 里带了 game，
# 两个库各建各的缓存，互不覆盖。
_default: HybridRetriever | None = None


def get_retriever(settings: Settings | None = None) -> HybridRetriever:
    global _default
    if _default is None:
        # 必须复用 get_llm() 单例，不能 LLMClient() 新建 ——
        # 新建的话检索用的 embedding 缓存在另一个池子里：
        #   1. /api/health 报的缓存命中率永远是 0（看的是另一个实例）
        #   2. 同一条 query 在 /retrieve 命中过、到 /chat 又要重新 embedding 一次
        # 检索和对话本来就是同一套向量与同一个上游，拆成两个实例没有收益。
        from app.llm.client import get_llm

        _default = HybridRetriever(settings=settings, llm=get_llm(settings),
                                   store=get_vector_store(settings))
    return _default


def reset_retriever() -> None:
    """切后端 / 重建索引后调用，否则会拿着旧缓存继续服务。"""
    global _default
    if _default is not None:
        _default.invalidate()
    _default = None