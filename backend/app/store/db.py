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
    """所有 ORM 模型的元数据注册点，也是 `create_all` 建表的唯一依据。

    前提：`app.store.models` 必须**先被 import**，否则 `Base.metadata` 是空的，
    `init_db` 会建出一套零张表的数据库而且不报错。
    persist.py 在模块顶部 import models 就是为了保证这一点。

    本项目没有 migration 体系（改列结构不会被自动应用），所以这里的字段定义
    等于线上 schema；加字段要同时考虑已有库里的旧表。
    """
    pass


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None
_status: dict = {"state": "unknown", "detail": ""}


class MySQLUnavailable(RuntimeError):
    """显式要求 MySQL 但连不上时才抛（MYSQL_REQUIRED=true）。"""


def get_engine(settings: Settings | None = None) -> AsyncEngine | None:
    """惰性建连接池的进程级单例，第二个参数一旦传入就被永久忽略。

    前提与后果要说清楚：
    - 单例按**首次调用**的 settings 固化。测试若先用一个假 Settings 建过池，
      后面换 DSN 拿到的是同一个旧池 —— 改 `.env` 不重载进程就不生效。
    - `create_async_engine` 本身**不建立任何连接**，连接是首次使用时才懒建的，
      所以这里返回成功不代表 MySQL 可用；真正的可用性判断在 `init_db`/`db_status`。
    - 连接池是 5+5：本机 MySQL 默认停止且无业务数据，池开大没有意义，
      而 `pool_pre_ping` 会在每次取连接前探活，避免用到已被关掉的连接。

    结果消费：`session_scope` 与 `init_db` 共用这一个引擎；
    `close_db()` 会把它置回 None（进程内重建），因此不能在 lifespan 关闭后再用旧引用。
    """
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
    """会话工厂的进程级单例，与 `_engine` 同生命周期（`close_db` 一起置空）。

    `expire_on_commit=False` 是这里最关键的一个选项：
    默认行为会在 commit 之后让 ORM 实例过期，之后再读它的属性就得重新发 SQL，
    在 session 已关闭时就直接抛 `DetachedInstanceError`。
    落库链路正是「commit 之后把对象拼成 dict 返回」，
    所以必须关掉过期 —— 否则 save_run 这类函数会在最后一步炸掉。

    结果消费：`session_scope()` 是唯一推荐用法（自带事务边界与异常转换），
    直接拿它开 session 的调用方要自己负责 commit/rollback。
    """
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
    """返回当前连接状态的**副本**（不是 `_status` 本身，外部改不到内部状态）。

    `state` 的三个取值要能区分对待：
    - `unknown`：进程起来还没跑过 `init_db`。
    - `ready`：建表成功（detail 是库名）。
    - `unavailable`：连不上/建表失败（detail 是首行错误信息，已截断）。
    MySQL 缺席时这里只是记录，**不抛** —— 纯向量检索链路不依赖 SQL 侧。

    结果消费：`/api/health` 和 `/api/stats` 直接把它塞进响应，
    `detail` 因此会出现在 HTTP 响应里 —— 它是首行异常文本，
    所以新增错误分支时要检查异常消息里有没有 DSN 或口令，别顺手带出去。
    """
    return dict(_status)


@asynccontextmanager
async def session_scope(settings: Settings | None = None) -> AsyncIterator[AsyncSession]:
    """一个事务边界。MySQL 缺席时抛 MySQLUnavailable，由调用方决定是否致命。"""
    s = settings or get_settings()
    # 位置有意义（不能挪进 try），但**理由不是注释原先写的那样**：
    # 它并不会在 MySQL 缺席时立刻抛 —— `create_async_engine` 是惰性的，
    # 不建立任何连接（见 `get_engine` 的 docstring）。
    # 把它放在 try 之外的真实作用是：让「建引擎」这一步的异常走 except Exception
    # 分支，从而把 `_status` 更新成 unavailable；而不是让它和 session 的异常混在一起。
    # 真正的 MySQLUnavailable 由下面 session 首次取连接时抛出。
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