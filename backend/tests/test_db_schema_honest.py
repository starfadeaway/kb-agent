"""表结构与写入路径必须对得上 —— 「定义了但从不写」的字段不许含糊。

## 背景

`Session.meta` 与 `Message.tool_name` 两个字段**在当前代码里一次都不会被写入**：
`ensure_session` 只 upsert `game`，`save_turn` 只落 user/assistant 两条且都不设
`tool_name`。它们既不是 bug 也不是死代码，而是一类更隐蔽的东西 ——
**表结构宣称能存的东西，比实际存的多**。

做数据的人看 schema 会以为「会话有元数据」「消息记了工具名」，
于是写出依赖它们的查询，然后拿到一片 NULL。

本项目已有同类的血泪教训：`LanceCollection` 的 sidecar 元数据写了不读回，
导致侧栏语料量虚高 4.4 倍。所以这类"声明 ≠ 现实"一律要变成可执行断言。

## 为什么是断言"不写入"而不是断言"写入"

断言"`tool_name` 必须被写"是错的 —— tool 消息根本不能进 messages 表
（不带 `tool_call_id` 重放给模型会 400），工具明细在 `tool_call_logs`。
正确的不变量是**当前不写入**，且一旦有人要改这个行为，
必须同时改掉这里的断言并说明理由 —— 强制留下决策痕迹。

同理 `Session.meta`：将来真要实现「跨会话记忆」，第一个落点就是它。
到那时这条断言应该**主动失败**，提醒你把它和记忆实现一起改掉。
"""
from __future__ import annotations

import inspect

from app.store import persist
from app.store.models import Message, Session


def test_save_turn_writes_only_user_and_assistant():
    """落库只写两条，且不设 tool_name —— 锁住这个事实。"""
    src = inspect.getsource(persist.save_turn)
    assert "tool_name" not in src, (
        "save_turn 开始写 tool_name 了。如果这是有意的（比如为了审计补记 "
        "role=tool 的行），请同时更新 models.Message.tool_name 的注释与 "
        "test_load_history_only_returns_replayable_roles 的前提。"
    )
    assert '"role": "user"' in src
    assert '"role": "assistant"' in src


def test_ensure_session_does_not_write_meta():
    """会话元数据当前不落库 —— 不要让人以为能查到。"""
    src = inspect.getsource(persist.ensure_session)
    assert "meta" not in src.replace("Session.meta", ""), (
        "ensure_session 开始写 Session.meta 了。若这是记忆功能的第一步，"
        "请同时更新 Session.meta 的注释 —— 它现在写的是「预留字段」。"
    )


def test_load_history_only_returns_replayable_roles():
    """历史只取 user/assistant：tool 行重放给模型会 400。

    这条与 `save_turn` 不写 tool 消息是配套的，两边必须同时成立。
    """
    src = inspect.getsource(persist.load_history)
    assert '("user", "assistant")' in src


def test_tool_call_details_live_in_their_own_table():
    """工具调用明细在 tool_call_logs，不在 messages。

    断言它存在且带齐定位字段 —— `messages.tool_name` 只是名字，
    真要排查"当时为什么检索错了"，需要的是 args / ok / error / latency。
    """
    from app.store.models import ToolCallLog

    cols = {c.name for c in ToolCallLog.__table__.columns}
    needed = {"run_id", "step", "tool_name", "args", "ok", "error", "latency_ms"}
    assert needed <= cols, f"tool_call_logs 缺字段：{sorted(needed - cols)}"


def test_session_meta_is_declared_nullable_json():
    """meta 必须能存 NULL —— 强制非空会让「不写」变成「写空串」，更难发现。"""
    col = Session.__table__.columns["meta"]
    assert col.nullable
    assert Message.__table__.columns["tool_name"].nullable