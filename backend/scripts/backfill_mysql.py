"""把本地缓存的语料全量回填到 MySQL，并打印落库结果。

用法：
    .venv\\Scripts\\python.exe backend\\scripts\\backfill_mysql.py --game sr
    .venv\\Scripts\\python.exe backend\\scripts\\backfill_mysql.py --game all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.store.db import close_db, db_status, init_db, session_scope  # noqa: E402
from app.store.models import ChunkMeta, Fact, Page  # noqa: E402
from app.store.persist import backfill_corpus  # noqa: E402
from sqlalchemy import func, select  # noqa: E402


async def table_counts() -> dict:
    s = get_settings()
    out: dict = {}
    async with session_scope(s) as session:
        for model in (Page, ChunkMeta, Fact):
            n = await session.execute(select(func.count()).select_from(model))
            out[model.__tablename__] = int(n.scalar() or 0)
    return out


async def per_game_counts(slugs: list[str]) -> dict:
    """按游戏分组统计 —— 合计数非空不代表每个游戏都进去了。

    原神当初只进了 LanceDB、MySQL 里 ys 分组全是 0 行，
    而总表计数看起来完全正常 —— 光看总数发现不了"某个库漏了"。
    """
    out: dict[str, dict[str, int]] = {}
    async with session_scope(get_settings()) as session:
        for g in slugs:
            per: dict[str, int] = {}
            for model in (Page, ChunkMeta, Fact):
                n = await session.execute(
                    select(func.count()).select_from(model).where(model.game == g)
                )
                per[model.__tablename__] = int(n.scalar() or 0)
            out[g] = per
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="all",
                    help=f"要回填的游戏 slug，或 all。可选：{', '.join(get_settings().game_slugs)}、all")
    args = ap.parse_args()

    if not await init_db():
        print(json.dumps({"ok": False, "reason": db_status()}, ensure_ascii=False))
        raise SystemExit(1)

    s = get_settings()
    # 游戏集合只从 Settings.game_slugs 这一处取。
    # 之前这里是 `["sr", "zzz"]`，连 argparse 的 choices 都只列了这两个 ——
    # 于是加第三个库时，回填脚本"看起来支持 all"，实际 all 永远是两个游戏，
    # 原神就这么漏进了 MySQL，而表总数看起来一切正常。
    all_slugs = s.game_slugs
    if args.game == "all":
        games = list(all_slugs)
    elif args.game in all_slugs:
        games = [args.game]
    else:
        print(json.dumps({"ok": False, "reason": f"未知游戏 {args.game!r}",
                          "available": all_slugs}, ensure_ascii=False))
        raise SystemExit(1)

    t0 = time.perf_counter()
    report = {"ok": True, "games": {}}
    for g in games:
        r = await backfill_corpus(g)
        report["games"][g] = r
        print(f"[{g}] pages={r.get('pages')} chunks={r.get('chunks')} facts={r.get('facts')}")

    report["tables"] = await table_counts()
    report["per_game"] = await per_game_counts(all_slugs)
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    await close_db()

    # 漏填自检：某个游戏 0 行就明确报出来，而不是让读表的人自己发现。
    missing = {g: per for g, per in report["per_game"].items() if not per["pages"]}
    report["missing_games"] = sorted(missing)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        print(f"\n⚠ 以下游戏在 MySQL 里没有任何页面：{sorted(missing)}")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
