"""MySQL 表结构：结构化数值 / 页面元数据 / 会话与 Agent 轨迹 / 评测结果。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.store.db import Base


class Page(Base):
    """wiki 页面元数据。last_revision 用于版本时效性过滤。"""

    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("game", "page_id", name="uq_pages_game_pageid"),
        Index("ix_pages_game_title", "game", "title"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game: Mapped[str] = mapped_column(String(16), index=True)
    page_id: Mapped[int] = mapped_column(BigInteger)
    title: Mapped[str] = mapped_column(String(255))
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    url: Mapped[str] = mapped_column(String(512), default="")
    last_revision: Mapped[int] = mapped_column(BigInteger, default=0)
    last_revision_ts: Mapped[str] = mapped_column(String(32), default="")
    infobox: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    n_chars: Mapped[int] = mapped_column(Integer, default=0)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChunkMeta(Base):
    """chunk 元数据（正文与向量在 ChromaDB，这里只放可过滤的结构化字段）。

    独立于 Page 存 chunk，是为了让"按版本/按章节过滤"成为一次索引扫描而不是 join。
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_chunks_chunk_id"),
        Index("ix_chunks_game_page", "game", "page_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(96), index=True)
    game: Mapped[str] = mapped_column(String(16), index=True)
    page_id: Mapped[int] = mapped_column(BigInteger)
    page_title: Mapped[str] = mapped_column(String(255))
    section: Mapped[str] = mapped_column(String(255), default="正文")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    n_chars: Mapped[int] = mapped_column(Integer, default=0)
    last_revision: Mapped[int] = mapped_column(BigInteger, default=0)
    last_revision_ts: Mapped[str] = mapped_column(String(32), default="")
    source_url: Mapped[str] = mapped_column(String(512), default="")
    truncated: Mapped[bool] = mapped_column(Integer, default=0)


class Fact(Base):
    """从 infobox 抽出的结构化数值（角色属性、光锥效果、位面饰品）。

    这类字段绝不能靠向量检索拿——必须精确匹配，所以放 MySQL。
    """

    __tablename__ = "facts"
    __table_args__ = (
        UniqueConstraint("game", "entity_type", "entity_name", "attr", name="uq_fact_key"),
        Index("ix_facts_lookup", "game", "entity_type", "entity_name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game: Mapped[str] = mapped_column(String(16), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))   # 角色 / 光锥 / 位面饰品 / ...
    entity_name: Mapped[str] = mapped_column(String(128))
    attr: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(Text)
    value_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    page_id: Mapped[int] = mapped_column(BigInteger, default=0)
    source_url: Mapped[str] = mapped_column(String(512), default="")
    last_revision_ts: Mapped[str] = mapped_column(String(32), default="")
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    game: Mapped[str] = mapped_column(String(16), default="sr")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_session", "session_key", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_key: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))          # user / assistant / tool
    content: Mapped[str] = mapped_column(Text)
    tool_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    refs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Run(Base):
    """一次 Agent 运行 = 一条可复现轨迹（面试里讲"可观测性"的抓手）。"""

    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_session", "session_key", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    session_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    game: Mapped[str] = mapped_column(String(16), default="sr")
    model: Mapped[str] = mapped_column(String(64), default="")
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)   # 检索/实验配置快照
    steps: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    tool_failures: Mapped[int] = mapped_column(Integer, default=0)
    stop_reason: Mapped[str] = mapped_column(String(32), default="")  # answer/limit/timeout/error
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    refs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ToolCallLog(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (Index("ix_toolcalls_run", "run_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    step: Mapped[int] = mapped_column(Integer, default=0)
    tool_name: Mapped[str] = mapped_column(String(64))
    args: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ok: Mapped[bool] = mapped_column(Integer, default=1)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    result_chars: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EvalCase(Base):
    """评测集。必须标注 source_page 才能算 Recall@k，否则"召回"无从判定。"""

    __tablename__ = "eval_cases"
    __table_args__ = (
        UniqueConstraint("case_id", name="uq_evalcase_id"),
        Index("ix_evalcase_game", "game", "category"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    game: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(48), default="single_hop")
    question: Mapped[str] = mapped_column(Text)
    gold_answer: Mapped[str] = mapped_column(Text)
    gold_pages: Mapped[list] = mapped_column(JSON, default=list)   # Recall@k 判定依据
    difficulty: Mapped[int] = mapped_column(Integer, default=1)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EvalResult(Base):
    """单变量实验结果：同一 case 在某组配置下的检索+生成得分。"""

    __tablename__ = "eval_results"
    __table_args__ = (
        UniqueConstraint("exp_label", "case_id", name="uq_evalresult"),
        Index("ix_evalresult_label", "exp_label"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    exp_label: Mapped[str] = mapped_column(String(64), index=True)   # 如 A2 / B-title / D-embo01
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retrieved_ids: Mapped[list] = mapped_column(JSON, default=list)
    recall_at_k: Mapped[float] = mapped_column(Float, default=0.0)
    mrr: Mapped[float] = mapped_column(Float, default=0.0)
    ndcg: Mapped[float] = mapped_column(Float, default=0.0)
    hit_rate: Mapped[float] = mapped_column(Float, default=0.0)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    judge_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    answer: Mapped[str] = mapped_column(Text, default="")
    judged_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)