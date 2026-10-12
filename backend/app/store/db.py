"""MySQL 结构化存储（async SQLAlchemy + aiomysql）。

降级策略：MySQL 服务默认停止且本机无管理员权限启动，
所以连接失败不能抛到业务层——统一由 `db_status()` 暴露，
向量检索链路在 MySQL 缺席时仍可完整运行（纯向量库模式，不依赖 SQL 侧）。
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None
_status: dict = {"state": "unknown", "detail": ""}


class MySQLUnavailable(RuntimeError):
    """显式要求 MySQL 但连不上时才抛（MYSQL_REQUIRED=true）。"""


def get_engine(settings: Settings | None = None) -> AsyncEngine | None:
    global _engine
    if _engine is None:
        s = settings or get_settings()
        _engine = create_async_engine(
            s.mysql_dsn,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=5,
            echo=False,
        )
    return _engine


def get_sessionmaker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(settings), expire_on_commit=False, class_=AsyncSession
        )
    return _sessionmaker


async def ensure_database(settings: Settings | None = None) -> bool:
    """先连到服务器（不指定库）把库建出来。

    MySQL 的连接串必须带上已存在的库名，所以第一次跑时
    `mysql_dsn` 直接连会报 Unknown database。先用 aiomysql 不带库连一次，
    CREATE DATABASE IF NOT EXISTS 建好，再交给 init_db 建表。
    """
    s = settings or get_settings()
    try:
        import aiomysql

        conn = await aiomysql.connect(
            host=s.mysql_host, port=s.mysql_port, user=s.mysql_user, password=s.mysql_password,
            autocommit=True,
        )
        async with conn.cursor() as cur:
            # 先查再建：MySQL 的 `CREATE DATABASE IF NOT EXISTS` 在库已存在时
            # 仍会抛一条 note 级 warning，日志里像报错一样刺眼，先查更干净。
            await cur.execute(
                "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = %s",
                (s.mysql_db,),
            )
            exists = await cur.fetchone()
            if not exists:
                await cur.execute(
                    f"CREATE DATABASE `{s.mysql_db}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
                logger.info("已创建数据库 %s", s.mysql_db)
        conn.close()
        return True
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}".split("\n")[0][:300]
        _status.update(state="unavailable", detail=msg)
        if s.mysql_required:
            raise MySQLUnavailable(f"建库失败且 MYSQL_REQUIRED=true: {msg}") from exc
        logger.warning("MySQL 不可用，跳过建库：%s", msg)
        return False


async def init_db(settings: Settings | None = None) -> bool:
    """尝试建库建表。成功返回 True，失败记录原因并返回 False（不抛）。"""
    s = settings or get_settings()
    if not await ensure_database(s):
        return False
    engine = get_engine(s)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _status.update(state="ready", detail=s.mysql_db)
        logger.info("MySQL ready: %s/%s", s.mysql_host, s.mysql_db)
        return True
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}".split("\n")[0][:300]
        _status.update(state="unavailable", detail=msg)
        if s.mysql_required:
            raise MySQLUnavailable(f"MySQL 不可用且 MYSQL_REQUIRED=true: {msg}") from exc
        logger.warning("MySQL 不可用，降级为纯向量库模式：%s", msg)
        return False


def db_status() -> dict:
    return dict(_status)


@asynccontextmanager
async def session_scope(settings: Settings | None = None) -> AsyncIterator[AsyncSession]:
    """一个事务边界。MySQL 缺席时抛 MySQLUnavailable，由调用方决定是否致命。"""
    s = settings or get_settings()
    # 看着像没用（结果没接住），但**位置有意义**：它在 try 之外，
    # 作用是 MySQL 缺席时立刻抛 MySQLUnavailable，而不是进到 try 里
    # 才开始建 session。删掉会让错误语义变模糊。
    get_engine(s)  # noqa: F841
    try:
        async with get_sessionmaker(s)() as session:
            yield session
            await session.commit()
    except MySQLUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        _status.update(state="unavailable", detail=f"{type(exc).__name__}: {exc}".split("\n")[0][:300])
        raise MySQLUnavailable(_status["detail"]) from exc


async def close_db() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None