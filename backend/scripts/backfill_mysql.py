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


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="all", choices=["sr", "zzz", "all"])
    args = ap.parse_args()

    if not await init_db():
        print(json.dumps({"ok": False, "reason": db_status()}, ensure_ascii=False))
        raise SystemExit(1)

    games = ["sr", "zzz"] if args.game == "all" else [args.game]
    t0 = time.perf_counter()
    report = {"ok": True, "games": {}}
    for g in games:
        r = await backfill_corpus(g)
        report["games"][g] = r
        print(f"[{g}] pages={r.get('pages')} chunks={r.get('chunks')} facts={r.get('facts')}")

    report["tables"] = await table_counts()
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    await close_db()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
