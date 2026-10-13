"""给 MySQL 补上内容指纹字段（幂等迁移），然后离线回填。

为什么需要单独脚本：`Base.metadata.create_all()` 只会建**新表**，
对已存在的 `pages` / `chunks` 一列都不会加 —— 而这两个表已经有 1400+ 页数据了。
所以「改了 model 就以为生效了」是错的，必须显式 ALTER。

为什么是幂等的：迁移会被反复执行。MySQL **不支持**
`ADD COLUMN IF NOT EXISTS`（那是 MariaDB 的语法），所以只能先查
`information_schema` 再决定要不要 ALTER。照抄 MariaDB 语法会直接报语法错。

用法：
    python backend/scripts/migrate_content_hash.py            # 只迁移
    python backend/scripts/migrate_content_hash.py --backfill # 迁移并重算指纹

回填**完全离线**：指纹从 data/raw 的本地缓存算，不碰 wiki，不消耗 WAF 配额。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.store.db import get_engine, init_db  # noqa: E402
from sqlalchemy import text  # noqa: E402

# (表, 列, 列定义, 索引名)
TARGETS = [
    ("pages", "content_hash", "VARCHAR(16) NOT NULL DEFAULT ''", "ix_pages_content_hash"),
    ("chunks", "content_hash", "VARCHAR(16) NOT NULL DEFAULT ''", None),
]


async def _column_exists(conn, table: str, column: str) -> bool:
    r = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = :t AND column_name = :c"
        ),
        {"t": table, "c": column},
    )
    return bool(r.scalar())


async def _index_exists(conn, table: str, index: str) -> bool:
    r = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.statistics "
            "WHERE table_schema = DATABASE() AND table_name = :t AND index_name = :i"
        ),
        {"t": table, "i": index},
    )
    return bool(r.scalar())


async def migrate() -> dict:
    out: dict = {"added_columns": [], "added_indexes": [], "already_present": []}
    engine = get_engine()
    async with engine.begin() as conn:
        for table, column, ddl, index in TARGETS:
            # 表本身可能还不存在（首次初始化时 init_db 会建好，这里只处理已存在的表）
            r = await conn.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = DATABASE() AND table_name = :t"
                ),
                {"t": table},
            )
            if not r.scalar():
                print(f"  {table} 表不存在，跳过（先跑 init_db 建表）")
                continue

            if await _column_exists(conn, table, column):
                out["already_present"].append(f"{table}.{column}")
                print(f"  {table}.{column} 已存在")
            else:
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                out["added_columns"].append(f"{table}.{column}")
                print(f"  已添加 {table}.{column}")

            if index and not await _index_exists(conn, table, index):
                await conn.execute(text(f"ALTER TABLE {table} ADD INDEX {index} ({column})"))
                out["added_indexes"].append(index)
                print(f"  已添加索引 {index}")
            elif index:
                print(f"  索引 {index} 已存在")
    return out


async def report() -> None:
    engine = get_engine()
    async with engine.connect() as conn:
        for table in ("pages", "chunks"):
            r = await conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            )
            total = r.scalar()
            r2 = await conn.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE content_hash = '' OR content_hash IS NULL")
            )
            empty = r2.scalar()
            print(f"  {table}: 共 {total} 行，指纹为空 {empty} 行"
                  f"（已覆盖 {(total - empty) / total * 100:.1f}%）" if total else
                  f"  {table}: 共 0 行")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="迁移后重算全量指纹（离线，不消耗 WAF 配额）")
    args = ap.parse_args()

    async def run() -> int:
        if not await init_db():
            print("!! MySQL 不可用，先把 mysqld 起起来")
            return 1
        print("== 迁移 ==")
        await migrate()
        print("\n== 现状 ==")
        await report()
        if args.backfill:
            print("\n== 离线回填 ==")
            from app.store.persist import backfill_corpus

            for game in ("sr", "ys", "zzz"):
                res = await backfill_corpus(game)
                print(f"  {game}: ok={res.get('ok')} pages={res.get('pages')} "
                      f"chunks={res.get('chunks')} facts={res.get('facts')}")
            print("\n== 回填后 ==")
            await report()
        else:
            print("\n（加 --backfill 才会真正算指纹）")
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
