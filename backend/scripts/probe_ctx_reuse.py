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
import statistics
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

    # 判读必须由数据给出，不能只丢一句"若…则…"让读者自己推：
    # C 的第 1 次慢、第 2/3 次快，说明 ToolContext.build 已经在复用 retriever；
    # 只看 B（每次 new）会误判成"每个请求都在重建索引"。
    c_first, c_rest = out["C_tool_path"][0], out["C_tool_path"][1:]
    c_steady = statistics.median(c_rest)
    b_steady = statistics.median(out["B_new_instance"])
    reuse_effective = c_steady < c_first / 3

    print("\n  C 的第 1 次 {c_first:.0f}ms，第 2/3 次中位 {c_steady:.0f}ms "
          f"（{b_steady:.0f}ms 若不复用的代价）".format(
              c_first=c_first, c_steady=c_steady, b_steady=b_steady))
    if reuse_effective:
        print(f"  → 复用有效：ToolContext.build 只在首次构建 retriever，"
              f"稳态 {c_steady:.0f}ms 比不复用的 {b_steady:.0f}ms 快约 "
              f"{b_steady / max(c_steady, 1e-9):.0f} 倍。")
        print("    不存在「每个请求重建 BM25 索引」的缺陷 —— B 组慢是因为它故意不复用。")
    else:
        print("  → 复用无效：第 2/3 次仍然接近首次，"
              "说明每个请求都在重建索引，这是一个真实缺陷。")

    save_probe(
        "probe_ctx_reuse",
        {"timings_ms": out, "query": "太空封印站的两件套效果",
         "final_top_k": cfg.final_top_k,
         "verdict": {
             "ctx_reuse_effective": reuse_effective,
             "c_first_ms": round(c_first, 1),
             "c_steady_median_ms": round(c_steady, 1),
             "b_no_reuse_median_ms": round(b_steady, 1),
             "reuse_speedup_x": round(b_steady / max(c_steady, 1e-9), 1),
         }},
        context={"game": "sr", "repeats": 3,
                 "note": "每组只有 3 次采样，方差大；用于看趋势而非精确值"},
    )


# 原本是模块顶层的 asyncio.run(main())：任何 `import probe_ctx_reuse`
# 都会立刻跑起 4 组真实检索并发出 API 请求。文件名看着像个库，
# 被测试收集器或 `python -c "import ..."` 扫到就会误触发。
if __name__ == "__main__":
    asyncio.run(main())