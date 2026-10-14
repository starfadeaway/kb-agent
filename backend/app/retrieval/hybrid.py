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

import asyncio
import dataclasses
import logging
import threading
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
        """返回实验阶梯 A0–A3 的档位名，**唯一判据是三个开关的组合**。

        消费方（`run_eval.py` 落库 / 实验报告分组）拿它当分组键，
        所以这里返回什么就会直接变成表格里的行名。
        前提：`use_*` 三个布尔必须由调用方显式设成实验设计的那一组；
        只要有一个字段从 `.env` 或默认值漏传，这段判断就会落到另一档，
        而落库里的 label 仍然是"自洽"的 —— 看不出是配错了。
        注意 A2 与 A3 无法区分 bm25_top/vector_top 等候选池参数，
        单变量消融靠的是 `config` 快照里的字段，不是这个 label。
        """
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
        """一次性建索引。

        前提：`chunk_ids` 与 `corpus` **同序等长** —— `search()` 返回的第三分量
        就是 `corpus` 下标，调用方要靠它对齐；两者错位不会报错，只会静默返回错行。
        代价由调用方承担：`BM25Okapi` 会在这里把整个语料分词一遍（实测 1.9s），
        所以只在 `_get_bm25` 的缓存未命中路径上调用。
        """
        idx = cls(chunk_ids=list(chunk_ids), corpus=list(corpus))
        idx._bm25 = BM25Okapi([tokenizer(t) for t in idx.corpus])
        return idx

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def search(self, query_tokens: list[str], top: int) -> list[tuple[str, float, int]]:
        """返回 `[(chunk_id, bm25分数, corpus下标), ...]`，已按分数降序。

        - **分数在这里被丢弃是安全的**：融合阶段只用名次（RRF），
          BM25 原始分与余弦不同量纲，混用会把一路的结果带偏。
        - 第三分量（corpus 下标）目前唯一消费者 `_fuse` 也丢掉了（写作 `_idx`），
          留它是为了将来做「BM25 命中片段高亮」时不必重排一次索引。
        - 提前返回空列表的两种情况：索引没建成、查询切词后为空 ——
          后者对纯标点/纯空白 query 会出现，融合阶段按"稀疏路交白卷"处理。
        """
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
        # 必须 RLock 而不是 Lock：_get_bm25 的慢路径会在持锁状态下调用
        # _load_corpus（它自己也要取同一把锁），普通非重入锁在并发下必然死锁。
        self._cache_lock = threading.RLock()
        # 缓存是**有界**的：最多留 `_max_cached_indexes` 份，超出走 FIFO 淘汰
        # （见 `_evict_locked`）。
        # 这里显式写下来，是因为这段注释曾经**是反的**：它宣称「无上限、够用、
        # 将来要做按页面过滤时必须换 LRU」，而紧挨着的下一行就是上限与淘汰逻辑。
        # 一段连自己实现都不描述的注释，比没有注释更容易误导下一个人。
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

        返回三元组 `(ids, docs, mapping)`，三者**同序**，但三个消费者各拿各的：
        - `_get_bm25` 只要 `(ids, docs)` —— BM25 需要原始文本；
        - `retrieve` 只要 `mapping` —— 融合阶段要把命中的 id 换回带 metadata 的 chunk；
        - `warmup` 只要 `len(ids)` —— 用来报语料规模。
        注意 `mapping` 里的 `RetrievedChunk.score` 是**占位的 1.0**，不是相似度：
        融合时会被向量路的真实分数覆盖（见 `_fuse`），只看 BM25 一路时它保持 1.0。

        实验规模（万级 chunk）下一次性拉全量是可接受的；
        所以实验报告里注明「稀疏检索未做工程优化，
        目标是隔离检索策略本身的差异」。

        注意这是同步阻塞调用（实测 sr 全量约 800ms），会占住一条线程；
        请求路径必须经 `asyncio.to_thread` 进入，否则整个事件循环会被卡住。
        """
        key = self._cache_key(cfg)
        with self._cache_lock:
            cached = self._text_cache.get(key)
        if cached is not None:
            return cached

        # 双检：并发首次加载时只有一个线程真的去拉全量，
        # 其余线程在锁外等，避免把快路径也塞进临界区。
        with self._cache_lock:
            cached = self._text_cache.get(key)
            if cached is not None:
                return cached
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
        """返回**已建好**的索引，调用方只需 `search()`，不用管缓存与并发。

        前提：调用方已经确认要走稀疏路（A0/A2/A3）；只走向量路时调它纯属白付 1.9s。
        慢路径全程持 `_cache_lock`，所以并发首访时后来的请求是**排队**而不是并行重建；
        这换来的好处是绝不会出现两份同 key 的索引互相顶掉缓存条目。
        """
        key = self._cache_key(cfg)
        with self._cache_lock:
            idx = self._bm25_cache.get(key)
        if idx is not None:
            return idx

        with self._cache_lock:
            idx = self._bm25_cache.get(key)
            if idx is not None:
                return idx
            ids, docs, _ = self._load_corpus(cfg)  # 同一把 RLock，重入安全
            idx = BM25Index.build(ids, docs, tokenize)
            self._bm25_cache[key] = idx
            self._evict_locked()
            logger.info("BM25 索引: %d chunks (game=%s)", len(ids), cfg.game)
            return idx

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
        """丢掉全部缓存。**必须由写入侧在重建索引后显式调用**。

        缓存 key 里不含 chunk 内容，所以「向量库被换掉了」这件事这套缓存自己发现不了；
        漏调的结果是检索继续用旧语料服务，看起来完全正常。
        """
        with self._cache_lock:
            self._bm25_cache.clear()
            self._text_cache.clear()

    def stats(self) -> dict:
        """缓存命中情况的唯一观测点，供 /api/health 展示。"""
        with self._cache_lock:
            return {"bm25_cached": len(self._bm25_cache), "corpus_cached": len(self._text_cache)}

    # ---------------- 检索 ----------------
    async def retrieve(self, query: str, cfg: RetrievalConfig) -> list[RetrievalHit]:
        """检索主入口。返回已按最终排序取好 `final_top_k` 条的命中列表。

        消费方：API `/api/retrieve` 直接把它序列化返回；
        `run_eval.py` 拿 chunk_id 映射回 page_title 算 recall/MRR/NDCG；
        Agent 侧拿 `chunk.text` 进 prompt、用 `chunk.metadata` 拼引用出处。
        它**不关心排序过程**，所以下面各路的分数/名次细节只在 `_fuse`/`_rerank` 里有意义。
        """
        if not query.strip():
            return []
        # 全量语料扫描（~800ms）和 BM25 建索引（~1.9s）都是同步阻塞调用。
        # 直接在协程里调会把事件循环整个卡住 —— 这期间所有并发的
        # /api/chat、/api/retrieve 都在排队，连健康检查都答不了。
        # 所以丢进线程池；缓存加了锁之后线程之间不会重复建索引。
        _, _, mapping = await asyncio.to_thread(self._load_corpus, cfg)
        if not mapping:
            return []

        # 稀疏路。前提：语料 mapping 非空（上面已判）—— 否则 BM25 建在一个空索引上，
        # 查出来的空结果会和"库里没有相关内容"混为一谈，分不清是哪种失败。
        bm25_hits: list[tuple[str, float, int]] = []
        if cfg.use_bm25:
            bm25_hits = (await asyncio.to_thread(self._get_bm25, cfg)).search(
                tokenize(query), cfg.bm25_top
            )

        # 稠密路。比稀疏路多两个硬前提：
        #   1. 必须注入 LLMClient —— 缺了就直接抛错，因为这一路是「要 embedding 才能跑」，
        #      静默退化成只走 BM25 会让 A1 的评测结果其实是 A0 的。
        #   2. **嵌入模型必须与建库时同一个**（由 cfg.embed_model 解析到同一个 collection 名保证）。
        #      模型换了却仍读旧 collection 的话，维度对不上会在点积时报错而不是静默降级。
        # 过滤条件与稀疏路共用同一个 where，保证两路候选集合同源 —— 这是 RRF 公平的前提。
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

        # 融合：两路都空才等于"没召回到"；只有一路空则退化成那一路的排序（见 `_fuse`）。
        hits = self._fuse(bm25_hits, vec_hits, mapping, cfg)
        # 重排入口条件：用户开了 rerank 且融合后还有候选。
        # 关掉 rerank 时 hits 的顺序就是 fused_score 序（`_fuse` 已排好），直接截断即可。
        if cfg.use_rerank and hits:
            hits = await self._rerank(query, hits, cfg)
        # 唯一的「截断到 top_k」发生在这里；`rerank_candidates` 只是送重排的候选上限，
        # 不影响最终条数，所以两者调大调小互不干扰。
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

        进入这段之前要先知道四个量的量纲差别，否则很容易把 `score` 和
        `fused_score` 搞混：
        - `bm25_hits[1]`（BM25 原始分）：无上界，随语料长度漂移 → **只用来定名次，从不参与打分**；
        - `vec_hits[].score`（余弦，[-1,1]）→ 只写进 `hit.score` 供前端展示/溯源；
          **注意只走稀疏路（A0）时没有任何东西会覆盖它，`score` 恒为 0.0** ——
          所以 A0 的 `score` 是占位值，不能拿去跨档位比较"相关度"；
        - `fused_score`（RRF 分）：名次的倒数和，量级 ~1/60 ≈ 0.017，**只有相对大小有意义**，
          跨查询不可比，别拿它当绝对置信度；
        - `rrf_k=60` 是平滑常数（业界惯例），作用是压制头部名次的过度优势：
          rank1 得 1/61、rank2 得 1/62，只差 0.26%。调小它会更"头部化"，调大更平权。
        `score` 与 `fused_score` 谁决定最终排序：开 rerank 时看 `rerank_score`，
        不开时看 `fused_score` —— 两者都不会让这里返回的顺序被推翻。
        """
        pool: dict[str, RetrievalHit] = {}

        # 第一路：稀疏。命中的 id 必须能在 mapping 里找到，否则没有正文和 metadata 可用；
        # 跳过的这种说明 BM25 索引与语料缓存不同源（正常不会发生，因为两者同 key 同源构建）。
        for rank, (cid, _score, _idx) in enumerate(bm25_hits, start=1):
            if cid not in mapping:
                continue
            pool[cid] = RetrievalHit(
                chunk=mapping[cid], score=0.0, rank_source="bm25", bm25_rank=rank,
                fused_score=1.0 / (cfg.rrf_k + rank),
            )

        # 第二路：稠密。两个分支的差别只在 `chunk` 从哪来：
        # 已经在池子里 → 两路都召回了（这正是 RRF 想要的信号，fused_score 累加）；
        # 不在池子里 → 只被向量召回到，直接用向量那份（它带真实 score）。
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

        # 排序键只有 fused_score 一个，所以 A0/A1/A2 三个档位的「融合」都退化成单路名次序，
        # 指标差异只来自召回集合不同，不来自任何加权技巧。
        fused = sorted(pool.values(), key=lambda h: -h.fused_score)
        # 下面两个分支只改**标签不改顺序**：单路时 rank_source 不该写成 "bm25+vector"
        # （那会让落库的评测记录谎报用了融合）。
        # 消费方 `rank_source` 主要用于人看 /api/retrieve 的溯源与实验记录，不是排序依据。
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

        返回值仍是 `list[RetrievalHit]`，且**顺序即最终顺序**，由 `retrieve` 截断 top_k。
        进入时 `hits` 必须已经按 fused_score 排好序 —— head/tail 的切分靠的就是这个序；
        若上游换了排序方式，tail 里就会混进本该重排的高分候选。
        """
        if self.llm is None or not hits:
            return hits

        # 切分的三种落点，决定了"候选数对结果不敏感"这个承诺能不能兑现：
        #   limit<=0（不裁剪）/ 候选不足 limit → head 就是全部，tail 为空，行为等同优化前；
        #   否则 head=前 limit 条进重排，tail=其余按原融合序兜底。
        # 注意 tail 里这些 hit 的 rerank_score 保持 None，消费方要能区分"重排过"和"没重排过"。
        limit = cfg.rerank_candidates
        head, tail = (hits, []) if (limit <= 0 or len(hits) <= limit) else (hits[:limit], hits[limit:])

        docs = [h.chunk.text for h in head]
        try:
            ranked = await self.llm.rerank(query, docs, top_n=len(docs))
        except Exception as exc:  # noqa: BLE001
            # 降级必须返回**未经重排的原序**：重排只是加分项，
            # 上游 RRF 排序本身已经可用，让它异常把整次检索带崩不划算。
            logger.warning("Rerank 失败，降级为融合排序：%s", exc)
            return hits
        out: list[RetrievalHit] = []
        # 注意这里按 rerank 返回的**下标**回取，并且是原地改写 head 里的对象：
        # rerank_score 是 Cross-Encoder 的原始分（与余弦不同量纲），
        # 覆盖 score 之后 hit.score 的含义就变成了"重排分" —— 展示时别再拿它当相似度。
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
        返回各环节耗时，供 /api/health 展示：`load_corpus_ms` 与 `build_bm25_ms` 相加
        就是首个请求本来要付的代价，`chunks` 是这次被建索引的语料规模。
        注意耗时只在**冷启动**这一次有意义（缓存命中后两个值都接近 0）。
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
    """切后端 / 重建索引后调用，否则会拿着旧缓存继续服务。

    先 `invalidate()` 再置空是刻意的：`reset_retriever()` 常在测试里连着调，
    若旧实例还被别的引用持有（比如已在飞行中的请求），清空缓存能保证
    它接着服务的那几个请求不会继续读到已失效的语料。
    """
    global _default
    if _default is not None:
        _default.invalidate()
    _default = None