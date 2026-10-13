"""重测 `_load_corpus` / `_get_bm25` 在 3932 chunk 下的真实耗时。

docs/concurrency.md 里引用的 799.7ms / 1881.9ms 是 2306 chunk 时测的。
索引涨 70% 之后这两个数直接决定"第一个请求等多久"，必须重测。
"""
import asyncio
import sys
import time

sys.path.insert(0, "backend")
sys.stdout.reconfigure(encoding="utf-8")

from app.core.config import get_settings
from app.retrieval.hybrid import RetrievalConfig, get_retriever


async def main() -> None:
    s = get_settings()
    cfg = RetrievalConfig(game="sr", use_bm25=True, use_vector=False,
                          use_rerank=False, final_top_k=1)

    r = get_retriever(s)
    r.invalidate()  # 冷缓存

    t0 = time.perf_counter()
    ids, docs, mapping = r._load_corpus(cfg)
    load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r._get_bm25(cfg)
    bm25_ms = (time.perf_counter() - t0) * 1000

    print(f"语料规模      : {len(ids)} chunk / {len(mapping)} 页面")
    print(f"_load_corpus  : {load_ms:.1f} ms   （旧 2306 chunk 时 799.7 ms）")
    print(f"_get_bm25     : {bm25_ms:.1f} ms   （旧 2306 chunk 时 1881.9 ms）")
    print(f"冷启动总等待   : {load_ms + bm25_ms:.1f} ms")
    print(f"stats         : {r.stats()}")


asyncio.run(main())