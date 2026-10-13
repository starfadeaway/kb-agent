"""把代码里定义的评测集 upsert 进 MySQL `eval_cases` 表。

为什么需要单独一个入口：
    原来 upsert 逻辑内嵌在 `run_eval.py` 的落库分支里，且只处理 `args.game`。
    实验 A–E 全部只跑 sr，于是 zzz / ys 的 24 条评测集从来没被写进过库 ——
    表里只有 32 条 sr，看起来像"评测集就这么大"。
    这不是数据缺失，是写入路径从来没被覆盖到。

零 API 消耗：只写库，不检索也不调模型。

用法：
    .venv\\Scripts\\python.exe backend\\scripts\\seed_eval_cases.py
    .venv\\Scripts\\python.exe backend\\scripts\\seed_eval_cases.py --game zzz
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.eval.cases import ALL_CASES, cases_for  # noqa: E402
from app.store.db import init_db, session_scope  # noqa: E402
from app.store.models import EvalCase as EvalCaseRow  # noqa: E402
from app.store.persist import save_eval_cases  # noqa: E402
from sqlalchemy import func, select  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="all", choices=["all", *ALL_CASES])
    args = ap.parse_args()

    s = get_settings()
    if not await init_db(s):
        print("MySQL 不可用，未落库")
        return

    games = list(ALL_CASES) if args.game == "all" else [args.game]
    for game in games:
        n = await save_eval_cases(game, s)
        print(f"  {game:<4} 代码 {len(cases_for(game)):>3} 条 -> upsert {n} 条")

    async with session_scope(s) as ses:
        rows = (await ses.execute(
            select(EvalCaseRow.game, func.count()).group_by(EvalCaseRow.game)
        )).all()
    total = sum(n for _, n in rows)
    print("\n落库后各库条数：")
    for g, n in rows:
        code_n = len(cases_for(g))
        flag = "" if code_n == n else f"  ⚠ 代码里 {code_n} 条"
        print(f"  {g:<4} {n:>3} 条{flag}")
    print(f"  合计 {total} 条（代码里共 {sum(len(c) for c in ALL_CASES.values())} 条）")


if __name__ == "__main__":
    asyncio.run(main())