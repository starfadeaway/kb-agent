"""守卫：`mcp_server.main` 不崩 + 流式 `stop_reason` 不再假装是 answer。

这两条都是「没人走 + 一走就崩 / 一走就说谎」的路径，修好之后必须有守卫，
否则下一次改配置项就会静默退回原样。

## A 组：`main()` 的命令行入口

docstring 宣称 `python -m app.agent.mcp_server [game]` 可用，但真实入口是
`scripts/mcp_stdio_entry.py`（走 `serve_stdio`），所以 `main()` 长期没人走。
它原先调的是 `_TC.create(game)`，而 `ToolContext` 只有 `@classmethod build()`
—— 敲一次就 AttributeError。

不真的起 stdio server：`asyncio.run` 换成同步桩、`serve_stdio` 换成记录桩，
断言点落在「ctx 到底被交给了谁、game 是什么」上，而不是「服务起没起来」。

## B 组：流式路径的 `stop_reason`

`AgentState.stop_reason` 的 dataclass 默认值是 `"answer"`。`_run_stream`
命中安全阀时原先只 yield 一条 `limit` 就 break，从不写 `state.stop_reason`，
于是最终 `done` 事件把「被安全阀掐掉」报成「正常回答」—— 评测会把它计成
一次正常作答，缺陷在指标上完全隐形。非流式的 `_run_once` 没这个问题，
它走 `_finalize()`，那里第一件事就是 `state.stop_reason = reason`。

## 双向验证

见模块底部的 `assert_stream_reports_limit` / `invoke_main`：两条断言都写成
接受「待测模块」的函数，而不是写死 `app.agent.*`，
所以可以把 HEAD 上（修复前）的源码用 importlib 加载成独立模块后喂给同一套断言。
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys

import pytest
from app.agent import mcp_server, react
from app.agent import tools as T
from app.core.config import get_settings
from fakes import FakeCtx as _FakeCtx

# --------------------------------------------------------------------------- #
# A 组：`mcp_server.main` 不崩
# --------------------------------------------------------------------------- #


class _StubToolContext:
    """ToolContext 的**最小**替身：只有 `build()`，没有 `create()`。

    必须刻意不提供 `create` —— 否则这条守卫就退化成「调一个自己写的桩」，
    抓不到 `_TC.create` 这个真实缺陷。它模拟的是真实 ToolContext 的**接口形状**：
    `@classmethod build(cls, game="sr")`。
    """

    created: list[str] = []

    def __init__(self, game: str) -> None:
        self.game = game

    @classmethod
    def build(cls, game: str = "sr") -> _StubToolContext:
        cls.created.append(game)
        return cls(game)


def invoke_main(mod, argv: list[str]) -> tuple[str | None, list[str]]:
    """在给定 argv 下跑 `mod.main()`，返回 (交给 serve_stdio 的 ctx.game, build 收到的实参)。

    三处替身都是为了「不真的起 server」：
      - `sys.argv`     → 覆盖命令行参数；
      - `serve_stdio`  → 记录 ctx，不起 stdio server；
      - `asyncio.run`  → 同步桩，所以连事件循环都不开。
    两个替身都是**同步函数**：`serve_stdio(ctx)` 这一步是普通函数调用，
    在 `asyncio.run` 之前就求值了，所以 ctx 在进 asyncio.run 之前已经记下来了。
    `ToolContext` 用 `monkeypatch` 换在 `app.agent.tools` 上：`main()` 是
    函数内 import（`from app.agent.tools import ToolContext as _TC`），
    取的是调用那一刻的属性，所以替身换在模块上就能生效。
    """
    seen: dict[str, object] = {}

    def fake_serve_stdio(ctx, **_kw):
        seen["ctx"] = ctx
        return None

    def fake_run(coro):
        return None

    # 本地持有这份 list：monkeypatch.undo() 会把类属性换回原对象，
    # 事后读 `_StubToolContext.created` 读到的是那份已经被丢弃的空列表。
    created: list[str] = []
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(sys, "argv", argv)
        mp.setattr(T, "ToolContext", _StubToolContext)
        mp.setattr(_StubToolContext, "created", created)
        mp.setattr(mod, "serve_stdio", fake_serve_stdio)
        mp.setattr(asyncio, "run", fake_run)
        mod.main()  # 修复前在这里抛 AttributeError
    finally:
        mp.undo()

    ctx = seen.get("ctx")
    assert ctx is not None, "serve_stdio 没被调用 —— main() 在构造 ctx 之后就断了"
    return ctx.game, list(created)


@pytest.mark.parametrize(
    "argv,expected_game",
    [
        (["mcp_server"], "sr"),
        (["mcp_server", "zzz"], "zzz"),
    ],
    ids=["no_arg", "with_arg"],
)
def test_main_builds_ctx_and_hands_it_to_serve_stdio(argv, expected_game):
    """`main()` 两种命令行形态都要走通，且 ctx.game 必须是对的。"""
    game, built_args = invoke_main(mcp_server, argv)

    assert built_args == [expected_game], (
        f"ToolContext.build 收到的实参是 {built_args}，期望 [{expected_game!r}]"
    )
    assert game == expected_game, f"serve_stdio 拿到的 ctx.game={game!r}，期望 {expected_game!r}"


def test_main_without_arg_does_not_pass_none_to_build():
    """不给参数时 `sys.argv[1]` 根本不存在，**不能**把 None 透给 build()。

    这一条与上面那条不是同一件事：上面保证「不崩」，这条保证「崩的修法没换成
    另一种坏法」——`game = sys.argv[1] if ... else None` 同样不崩，
    但会把 None 一路带进 ToolContext.game，工具里 `args.get("game") or ctx.game`
    的兜底随之退化成「检索一个叫 None 的库」。
    """
    game, built_args = invoke_main(mcp_server, ["mcp_server"])

    assert built_args[0] is not None, "build() 收到了 None，命令行默认值没生效"
    assert game is not None and game == "sr"


# --------------------------------------------------------------------------- #
# B 组：流式 stop_reason
# --------------------------------------------------------------------------- #


class _FakeLLM:
    """只提供 ReActAgent 实际用到的那一个方法（chat 由 chat_fn 替换掉了）。"""

    def assistant_tool_message(self, content: str, tool_calls: list[dict]) -> dict:
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {"id": tc.get("id", f"call_{i}"), "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                for i, tc in enumerate(tool_calls)
            ],
        }


def stream_events(react_mod, s) -> list[dict]:
    """让 ReAct 在流式路径上必定撞安全阀，把事件流收集完。

    「必定撞」靠两个条件同时成立：
      - `agent_max_steps=1`（非默认值）；
      - `chat_fn` 永远返回带 tool_calls 的结果，模型永不主动给最终答案。
    工具本身是真的会执行的，但用的是 retriever=None 的假 ctx，
    异常被 `_exec_call` 兜成文本喂回模型 —— 这里要验的是终止语义，不是检索。
    """
    calls = {"n": 0}

    async def chat_fn(messages, **_kw):
        calls["n"] += 1
        return {
            "content": "",
            "tool_calls": [{"id": f"call_{calls['n']}", "name": "search_kb",
                            "args": {"query": f"永不收敛的问题 {calls['n']}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    agent = react_mod.ReActAgent(
        settings=s, llm=_FakeLLM(),
        tools=[{"type": "function", "function": {"name": "search_kb"}}],
        chat_fn=chat_fn,
    )

    async def drive():
        # run() 是 async def，stream=True 时它先返回一个**协程**，
        # 必须 await 一次才拿到 _run_stream 这个异步生成器。
        it = await agent.run("星穹铁道的光锥效果如何？", ctx=_FakeCtx(), stream=True)
        return [ev async for ev in it]

    events = asyncio.run(drive())
    assert calls["n"] == 1, (
        f"chat_fn 被调用 {calls['n']} 次，期望恰好 1 次（max_steps=1 应当一步就撞阀）"
    )
    return events


def assert_stream_reports_limit(events: list[dict], expected_reason: str = "max_steps") -> None:
    """流式撞安全阀时，done 事件必须报出真实 reason，而不是默认的 "answer"。"""
    names = [e["event"] for e in events]
    assert "limit" in names, (
        f"事件流里没有 limit（实际 {names}）—— 测试根本没进安全阀分支，"
        f"这条断言等于没验"
    )
    assert events[-1]["event"] == "done", f"最后一条事件不是 done，实际 {names[-1:]}"
    stop_reason = events[-1]["data"]["stop_reason"]
    assert stop_reason != "answer", (
        f"done 事件把被安全阀掐掉的运行报成了 stop_reason={stop_reason!r}（"
        f"AgentState 的 dataclass 默认值）—— 修复前就是这条失败"
    )
    assert stop_reason == expected_reason, (
        f"stop_reason={stop_reason!r}，期望 {expected_reason!r}"
    )


def test_stream_limit_is_not_reported_as_answer():
    """主守卫：流式撞安全阀 → done.stop_reason 必须是 max_steps。"""
    s = dataclasses.replace(get_settings(), agent_max_steps=1)
    assert s.agent_max_steps == 1, "守卫必须跑在非默认步数上（Settings 是 frozen dataclass）"

    assert_stream_reports_limit(stream_events(react, s))


def test_stream_last_state_agrees_with_done_event():
    """落库路径（`last_state`）与 SSE done 事件必须说同一件事。

    两者都读 `state.stop_reason`，但消费方不同：前者进 DB / 评测产物，
    后者进前端。只断言 done 事件的话，字段在 done 之前被重置成 "answer"
    这种改法仍然会漏过。
    """
    s = dataclasses.replace(get_settings(), agent_max_steps=1)

    async def drive():
        chat_fn_calls = {"n": 0}

        async def chat_fn(messages, **_kw):
            chat_fn_calls["n"] += 1
            return {"content": "",
                    "tool_calls": [{"id": "call_1", "name": "search_kb",
                                    "args": {"query": "永不收敛的问题"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        agent = react.ReActAgent(
            settings=s, llm=_FakeLLM(),
            tools=[{"type": "function", "function": {"name": "search_kb"}}],
            chat_fn=chat_fn,
        )
        # run() 是 async def，stream=True 要先 await 才拿到异步生成器。
        it = await agent.run("星穹铁道的光锥效果如何？", ctx=_FakeCtx(), stream=True)
        events = [ev async for ev in it]
        return events, agent.last_state

    events, state = asyncio.run(drive())

    assert_stream_reports_limit(events)
    assert state is not None, "last_state 没写 —— 流式路径无法落库"
    assert state.stop_reason == events[-1]["data"]["stop_reason"], (
        f"last_state.stop_reason={state.stop_reason!r} 与 done 事件里的 "
        f"{events[-1]['data']['stop_reason']!r} 不一致"
    )
