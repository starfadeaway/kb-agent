"""并发压测：回答"这个 Agent 能不能多开、同时跑会怎样"。

设计原则（沿用单变量控制）：
  - 输入用**评测集里的真实 query**，不是 "test" 这种退化短串。
    短 query 走不到 rerank、不触发 BM25 分词，测出来的并发能力是假的。
  - 先测**串行基线**，再测并发。并发比串行慢不一定是坏事 ——
    只要"总吞吐"提高而"单请求延迟"没有线性恶化，就是健康的。
  - 同时校验**结果一致性**：并发下拿到错答案比变慢严重得多。

跑法：
    python scripts/bench_concurrency.py --mode retrieve
    python scripts/bench_concurrency.py --mode chat --levels 2,4,8
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

BASE = "http://127.0.0.1:8000"

# 兜底 query：只有 MySQL 不可用时才用。
FALLBACK_QUERIES = [
    "太空封印站的两件套效果是什么",
    "遗器副词条的强化规则是什么？每级提升多少",
    "凝滞虚影巽风之形在哪张地图，需要多少开拓力",
    "三月七的终结技可以叠加几层",
    "位面球和连结绳分别是什么，怎么获取",
    "速度达到120时攻击力额外提升多少",
    "忘却之庭的深渊怎么解锁，有哪些房间",
    "光锥「我」的诞生是几星，定位是什么",
]


async def load_queries() -> list[str]:
    """从评测集取真实问题。

    为什么要真实 query：短串（如 "test"）几乎不触发 BM25 分词差异，
    rerank 候选也可能退化，测出来的并发能力是假的。
    """
    try:
        from app.core.config import get_settings
        from app.store.db import init_db, session_scope
        from sqlalchemy import text

        s = get_settings()
        await init_db(s)
        async with session_scope(settings=s) as sess:
            rows = await sess.execute(
                text("SELECT question FROM eval_cases WHERE game='sr' ORDER BY case_id")
            )
            qs = [r[0] for r in rows.fetchall() if r[0]]
        if qs:
            print(f"  query 来源：评测集 eval_cases（{len(qs)} 条真实问题）\n")
            return qs
    except Exception as exc:  # noqa: BLE001
        print(f"  query 来源：评测集读取失败（{type(exc).__name__}），退回内置列表\n")
    return FALLBACK_QUERIES


RETRIEVE_BODY = {
    "game": "sr",
    "top_k": 5,
    "use_bm25": True,
    "use_vector": True,
    "use_rerank": True,
    "rerank_candidates": 20,
}


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((len(s) - 1) * p))))
    return s[k]


async def one_retrieve(client: httpx.AsyncClient, q: str) -> tuple[float, list[str], int]:
    body = {**RETRIEVE_BODY, "query": q}
    t0 = time.perf_counter()
    r = await client.post(f"{BASE}/api/retrieve", json=body, timeout=120)
    dt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    return dt, [h["chunk_id"] for h in data["hits"]], data["hits"][0]["page_title"] if data["hits"] else ""


async def one_chat(client: httpx.AsyncClient, q: str) -> tuple[float, dict]:
    body = {"question": q, "game": "sr", "backend": "deepseek"}
    t0 = time.perf_counter()
    r = await client.post(f"{BASE}/api/chat", json=body, timeout=180)
    dt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return dt, r.json()


async def run_level(fn, n: int, label: str, pool: list[str], offset: int = 0) -> dict:
    """发 n 个请求；n=1 走串行，其余同时发起。

    `offset` 让每一档都从 pool 里取**自己那一段**互不重叠的 query ——
    否则第二档会白捡第一档留下的 embedding/rerank 缓存，
    测出"并发反而更快"这种自欺结论（第一版就踩了这个坑）。
    """
    qs = [pool[(offset + i) % len(pool)] for i in range(n)]
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=32)
    async with httpx.AsyncClient(limits=limits) as client:
        t0 = time.perf_counter()
        if n == 1:
            results = [await fn(client, qs[0])]
        else:
            results = await asyncio.gather(*(fn(client, q) for q in qs))
        wall = (time.perf_counter() - t0) * 1000

    lat = [r[0] for r in results]
    print(
        f"  {label:<22} n={n:<3} 单请求 p50={pct(lat, .5):7.0f}ms  max={max(lat):7.0f}ms  "
        f"总墙钟={wall:7.0f}ms  吞吐={n / (wall / 1000):5.2f} req/s"
    )
    return {"n": n, "lat": lat, "wall": wall, "results": results, "queries": qs}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieve", "chat"], default="retrieve")
    ap.add_argument("--levels", default="1,2,4,8")
    args = ap.parse_args()

    fn = one_retrieve if args.mode == "retrieve" else one_chat
    levels = [int(x) for x in args.levels.split(",")]

    health = httpx.get(f"{BASE}/api/health", timeout=30).json()
    pool = await load_queries()
    need = sum(levels)
    if len(pool) < need:
        print(f"  ⚠️ 评测集只有 {len(pool)} 条，不够 {need} 条互不重叠的 query；")
        print("     将会循环复用，缓存隔离对高并发档不完全成立。\n")

    print("=" * 96)
    print(f"  并发压测 · mode={args.mode}   服务端 CONCURRENCY_LIMIT={health.get('concurrency_limit')}")
    print("=" * 96)

    out: dict[int, dict] = {}
    cursor = 0
    for n in levels:
        out[n] = await run_level(fn, n, "串行基线" if n == 1 else f"并发 {n}", pool, cursor)
        cursor += n

    # ---- 正确性：同一 query 在不同并发度下必须拿到同一批 chunk ----
    print("\n" + "=" * 96)
    print("  一致性校验")
    print("=" * 96)
    if args.mode == "retrieve":
        base = None
        ok = True
        for n in levels:
            # 复跑一次固定 query，与 1 并发下的结果比对
            _, ids, _ = await one_retrieve(
                httpx.AsyncClient(timeout=60), "太空封印站的两件套效果是什么"
            )
            if base is None:
                base = ids
            same = ids == base
            ok &= same
            print(f"  n={n:<3} top5 chunk_id 与基线一致: {'✅' if same else '❌'}  {ids[:3]}")
        print(f"\n  结论：{'并发下结果完全一致（确定性保住了）' if ok else '出现不一致 —— 存在隐藏的可变状态'}")
    else:
        for n in levels:
            r = out[n]["results"][0][1]
            print(
                f"  n={n:<3} 首个响应 steps={r['steps']} stop={r['stop_reason']} "
                f"refs={len(r['refs'])} failed={r['tool_failures']}"
            )

    # ---- 汇总 ----
    print("\n" + "=" * 96)
    print("  汇总")
    print("=" * 96)
    base = out[levels[0]]
    base_thr = base["n"] / (base["wall"] / 1000)
    print(f"  {'并发度':<8}{'p50(ms)':>10}{'max(ms)':>10}{'吞吐(req/s)':>14}{'吞吐倍数':>12}")
    for n in levels:
        r = out[n]
        thr = n / (r["wall"] / 1000)
        print(
            f"  {n:<8}{pct(r['lat'], .5):>10.0f}{max(r['lat']):>10.0f}{thr:>14.2f}"
            f"{thr / base_thr:>11.2f}x"
        )
    print(
        f"\n  注：并发度 > CONCURRENCY_LIMIT({health.get('concurrency_limit')}) 时多出的请求会排队，"
        "\n      所以吞吐应趋于饱和而 p50 线性上升 —— 这是预期行为，不是 bug。"
    )


if __name__ == "__main__":
    asyncio.run(main())
