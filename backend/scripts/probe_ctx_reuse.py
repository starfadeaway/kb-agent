"""验证：每次 ToolContext.build() 是否真的在重建 BM25 索引。

怀疑点：
    tools.py:38  retriever=HybridRetriever(settings=s, llm=get_llm(), store=...)
而 HybridRetriever 的 _bm25_cache / _text_cache 都是 **实例级**的。

如果 main.py 的 /chat 每次都 ToolContext.build(game)，那么每来一个请求
就要重新拉全量语料 + 重新 jieba 分词建索引 —— 实测那一步是 2849ms。

如果成立，这解释了为什么 smoke 里单次检索要 3 秒，而单独测 HybridRetriever
只要 8ms。

用法：.venv\\Scripts\\python.exe scripts\\probe_ctx_reuse.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent.tools import ToolContext, call_tool
from app.core.config import get_settings
from app.llm.client import get_llm
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig
from probe_io import save_probe


async def main() -> None:
    s = get_settings()
    out: dict[str, list[float]] = {}

    print("=" * 74)
    print("  A. 同一个 HybridRetriever 实例连续检索（缓存应当复用）")
    print("=" * 74)
    r = HybridRetriever(settings=s, llm=get_llm())
    cfg = RetrievalConfig(game="sr", final_top_k=3)
    out["A_reuse_instance"] = []
    for i in range(3):
        t0 = time.perf_counter()
        await r.retrieve("太空封印站的两件套效果", cfg)
        ms = (time.perf_counter() - t0) * 1000
        out["A_reuse_instance"].append(round(ms))
        print(f"  第 {i + 1} 次: {ms:>8.0f} ms")

    print("\n" + "=" * 74)
    print("  B. 每次新建 HybridRetriever（= ToolContext.build 的行为）")
    print("=" * 74)
    out["B_new_instance"] = []
    for i in range(3):
        t0 = time.perf_counter()
        rr = HybridRetriever(settings=s, llm=get_llm())
        await rr.retrieve("太空封印站的两件套效果", cfg)
        ms = (time.perf_counter() - t0) * 1000
        out["B_new_instance"].append(round(ms))
        print(f"  第 {i + 1} 次: {ms:>8.0f} ms")

    print("\n" + "=" * 74)
    print("  C. 走真实路径：ToolContext.build() + call_tool('search_kb')")
    print("     —— 这就是 /api/chat 每个请求干的事")
    print("=" * 74)
    out["C_tool_path"] = []
    for i in range(3):
        t0 = time.perf_counter()
        ctx = ToolContext.build("sr")
        text, ok = await call_tool("search_kb", {"query": "太空封印站的两件套效果"}, ctx, timeout=30)
        ms = (time.perf_counter() - t0) * 1000
        out["C_tool_path"].append(round(ms))
        print(f"  第 {i + 1} 次: {ms:>8.0f} ms  ok={ok}")

    print("\n" + "=" * 74)
    print("  D. 复用同一个 ToolContext 再检索（对照）")
    print("=" * 74)
    ctx = ToolContext.build("sr")
    out["D_reuse_ctx"] = []
    for i in range(3):
        t0 = time.perf_counter()
        await ctx.retriever.retrieve("太空封印站的两件套效果", cfg)
        ms = (time.perf_counter() - t0) * 1000
        out["D_reuse_ctx"].append(round(ms))
        print(f"  第 {i + 1} 次: {ms:>8.0f} ms")

    print("\n  → 若 B/C 的第 2、3 次仍然很慢，说明每次请求都在重建索引（bug）。")
    print("    若 A/C 快而 B/C 慢，那就确认了 ToolContext.build 的复用缺陷。")

    save_probe(
        "probe_ctx_reuse",
        {"timings_ms": out, "query": "太空封印站的两件套效果",
         "final_top_k": cfg.final_top_k},
        context={"game": "sr", "repeats": 3,
                 "note": "每组只有 3 次采样，方差大；用于看趋势而非精确值"},
    )


# 原本是模块顶层的 asyncio.run(main())：任何 `import probe_ctx_reuse`
# 都会立刻跑起 4 组真实检索并发出 API 请求。文件名看着像个库，
# 被测试收集器或 `python -c "import ..."` 扫到就会误触发。
if __name__ == "__main__":
    asyncio.run(main())