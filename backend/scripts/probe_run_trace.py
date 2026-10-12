"""核查「Agent 轨迹落库」这件事到底能不能复现文档里的熔断数字。

背景：`EVALUATION.md` 写着不可答问题从「7 步 / 30317 token」降到
「4 步 / 6514 token」。这条数字**不是**从某个 benchmark 脚本来的，
是跑真实 `/api/chat` 看轨迹得到的。而 `/api/chat` 会把每次运行写进
MySQL 的 `runs` / `tool_calls` —— 也就是说它理论上可追溯。

但"理论上"不算数：库里可能早就没那几轮了。所以这里直接查，
把结论钉成「可复现」或「已丢失」，而不是靠推测。

跑法：`python backend/scripts/probe_run_trace.py`
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))


async def main() -> int:
    from app.store.db import close_db, db_status, init_db, session_scope
    from sqlalchemy import text

    # db_status() 在 init_db() 之前一直报 unknown —— 和 backfill_mysql.py 一样的顺序。
    if not await init_db():
        st = db_status()
        print(f"数据库不可用：{st}")
        return 1
    st = db_status()
    print(f"数据库：{st.get('state')}  {st.get('detail')}")

    async with session_scope() as s:
        total = (await s.execute(text("SELECT COUNT(*) FROM runs"))).scalar_one()
        print(f"runs 总行数：{total}\n")

        # 熔断针对的是"知识库里没有的内容"，这类问题 steps 数最高。
        # 按 steps 倒序看最长的几轮，正是文档里 7 步那类的来源。
        rows = (await s.execute(text("""
            SELECT run_id, LEFT(question, 34) AS q, steps,
                   prompt_tokens, completion_tokens, tool_calls, tool_failures,
                   stop_reason, latency_ms, created_at
            FROM runs
            ORDER BY steps DESC, prompt_tokens DESC
            LIMIT 12
        """))).mappings().all()

        if not rows:
            print("runs 表为空 —— 文档里的熔断数字无法从库里复现")
            return 1

        print(f"{'steps':>6}{'prompt tok':>11}{'工具':>6}{'失败':>6}{'耗时ms':>9}  问题")
        print("-" * 96)
        for r in rows:
            print(f"{r['steps']:>6}{r['prompt_tokens'] or 0:>11}"
                  f"{r['tool_calls'] or 0:>6}{r['tool_failures'] or 0:>6}"
                  f"{r['latency_ms'] or 0:>9}  {r['q']}")

        max_steps = max(r["steps"] or 0 for r in rows)
        max_pt = max(r["prompt_tokens"] or 0 for r in rows)
        print("-" * 96)
        print(f"库里可见的最大 steps={max_steps}，最大 prompt_tokens={max_pt}")
        print("文档声称的熔断前状态：7 步 / 30317 token")
        print("文档声称的熔断后状态：4 步 / 6514 token")

        tc = (await s.execute(text("SELECT COUNT(*) FROM tool_calls"))).scalar_one()
        print(f"\ntool_calls 总行数：{tc}（每轮的工具轨迹明细）")

    # 不显式关连接的话，解释器退出时 aiomysql 的 __del__ 会在
    # "Event loop is closed" 上刷一串 traceback，并把退出码搞成非 0 ——
    # 让人误以为核查失败。
    await close_db()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))