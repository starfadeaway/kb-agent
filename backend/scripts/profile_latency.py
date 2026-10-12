"""逐环节延迟剖析。

目的：把「感觉慢」拆成可归因的数字 —— 每个环节单独计时，
再决定优化什么。**不测就优化等于猜。**

用法：
    ..\\.venv\\Scripts\\python.exe scripts\\profile_latency.py
    ..\\.venv\\Scripts\\python.exe scripts\\profile_latency.py --repeat 3
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.llm.client import get_llm
from app.retrieval.hybrid import BM25Index, HybridRetriever, RetrievalConfig, tokenize
from app.store.vector_store import get_vector_store

QUERIES = [
    "太空封印站位面饰品的效果是什么",
    "布洛妮娅的遗器词条怎么配",
    "怎么配队才能提高输出",
]


class Timer:
    """收集 (环节, 毫秒) 的采样，最后按环节聚合。"""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {}

    def record(self, stage: str, ms: float) -> None:
        self.samples.setdefault(stage, []).append(ms)

    @staticmethod
    async def stage(name: str, coro):
        t0 = time.perf_counter()
        try:
            return await coro
        finally:
            Timer.last = (time.perf_counter() - t0) * 1000

    def report(self) -> None:
        print("\n" + "=" * 62)
        print("  环节延迟分解（中位数 / 最小 / 最大，单位 ms）")
        print("=" * 62)
        for stage, vals in self.samples.items():
            med = statistics.median(vals)
            print(f"  {stage:<26} {med:>8.1f}  {min(vals):>8.1f}  {max(vals):>8.1f}")
        print("=" * 62)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sr")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    s = get_settings()
    llm = get_llm(s)
    store = get_vector_store(s)
    r = HybridRetriever(settings=s, llm=llm, store=store)
    t = Timer()

    cfg = RetrievalConfig(game=args.game, use_bm25=True, use_vector=True, use_rerank=True)

    # ---- 1. 冷启动：全量拉文本 + 建 BM25（这是首个请求的大头）----
    t0 = time.perf_counter()
    ids, docs, _ = await asyncio.to_thread(r._load_corpus, cfg)
    ms_load = (time.perf_counter() - t0) * 1000
    t.record("① 拉全量语料文本", ms_load)
    print(f"  语料 {len(ids)} chunk / {len(docs)} 文档")

    t0 = time.perf_counter()
    await asyncio.to_thread(BM25Index.build, ids, docs, tokenize)
    ms_bm25 = (time.perf_counter() - t0) * 1000
    t.record("② 建 BM25 索引(jieba)", ms_bm25)

    # ---- 2. 各热路径 ----
    for q in QUERIES:
        for _ in range(args.repeat):
            # embedding
            t0 = time.perf_counter()
            qv = await llm.embed([q])
            t.record("③ embed(bge-m3)", (time.perf_counter() - t0) * 1000)

            # 向量检索
            t0 = time.perf_counter()
            await asyncio.to_thread(
                store.query, query_embedding=qv[0], game=args.game,
                model=None, variant="base", top_k=20, where=None,
            )
            t.record("④ 向量检索(LanceDB)", (time.perf_counter() - t0) * 1000)

            # BM25 检索
            t0 = time.perf_counter()
            await asyncio.to_thread(
                r._get_bm25(cfg).search, tokenize(q), 20
            )
            t.record("⑤ BM25 检索", (time.perf_counter() - t0) * 1000)

    # ---- 3. rerank 的输入量决定成本 ----
    hits = await r.retrieve(QUERIES[0], RetrievalConfig(game=args.game, use_rerank=False, final_top_k=40))
    t.record("⑥ 融合(不含重排)", 0)
    print(f"  送入 rerank 的候选文档数：{len(hits)}")

    for _ in range(args.repeat):
        t0 = time.perf_counter()
        await llm.rerank(QUERIES[0], [h.chunk.text for h in hits], top_n=len(hits))
        ms = (time.perf_counter() - t0) * 1000
        t.record(f"⑦ rerank({len(hits)}文档)", ms)

    # 对照：只重排 20 个
    sub = [h.chunk.text for h in hits[:20]]
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        await llm.rerank(QUERIES[0], sub, top_n=20)
        t.record("⑧ rerank(20文档) 对照", (time.perf_counter() - t0) * 1000)

    # ---- 4. Chat 首包 ----
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        await llm.chat([{"role": "user", "content": "用一句话回答：1+1等于几"}], max_tokens=32)
        t.record("⑨ chat 短问答", (time.perf_counter() - t0) * 1000)

    t.report()


if __name__ == "__main__":
    asyncio.run(main())