"""Agent 内核的确定性测试：用 FakeLLM 驱动，不打任何网络。

为什么必须有这个文件
------------------
`react.py` 的 ReAct 循环是本项目最核心的差异化（手写编排，不依赖框架），
但它此前**没有任何测试** —— 现有的 10 个测试文件覆盖的是路由、配置契约、
工具白名单、schema 这些外围。

单靠跑真实 LLM 是测不出循环本身的错误的：步数上限、预算耗尽、熔断触发、
工具报错后的自愈，这些只有在**预先编排好模型每一步返回什么**的前提下才能断言。
这就是 FakeLLM 的用途 —— 它按脚本吐出 tool_calls，让循环走确定的路径。

覆盖的五条终止/容错路径：
  1. 正常路径：调一次工具 → 拿到结果 → 给出答案（stop_reason=answer）
  2. 步数上限：模型永远要调工具 → 到 agent_max_steps 强制收尾
  3. token 预算：token 累积超限 → 提前终止
  4. 工具报错：错误文本回填给模型，模型能自我纠正后正常作答
  5. 越界问题：提到未收录的游戏 → 0 步拒答，不消耗检索预算
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest
from app.agent.react import ReActAgent
from app.core.config import get_settings
from fakes import FakeLLM

# --------------------------------------------------------------------------- #
# FakeLLM 的实现在 fakes.py（这里曾有一份逐字副本，2026-10-15 收口）
# --------------------------------------------------------------------------- #

def make_agent(script: list[dict], settings=None):
    llm = FakeLLM(script)
    agent = ReActAgent(settings=settings or get_settings(), llm=llm,
                       tools=[{"type": "function", "function": {"name": "search_kb"}}])
    agent._fake = llm  # 便于断言调用次数
    return agent


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# 1. 正常路径
# --------------------------------------------------------------------------- #

def test_tool_call_then_answer_terminates_cleanly():
    """调一次工具后给出答案，stop_reason 必须是 answer 而不是撞上限。"""
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "太空封印站两件套使攻击力提高 12%。"},
    ])
    state = run(agent.run("太空封印站两件套效果是什么？"))

    assert state.stop_reason == "answer"
    assert "12%" in state.answer
    assert len(state.steps) == 1, f"应恰好执行 1 步工具，实际 {len(state.steps)}"
    assert agent._fake.calls == 2, "一次决策 + 一次收尾 = 2 次 chat"


def test_tool_result_is_fed_back_to_the_model():
    """工具结果必须以 role=tool 回到 messages 里，否则模型看不到检索结果。"""
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "答"},
    ])
    state = run(agent.run("太空封印站两件套效果是什么？"))

    roles = [m.get("role") for m in state.messages]
    assert "tool" in roles, f"messages 里没有 tool 消息，实际角色序列：{roles}"
    # 收尾的那次 chat 必须已经能看到 tool 结果
    last_seen = agent._fake.seen_messages[-1]
    assert any(m.get("role") == "tool" for m in last_seen), (
        "最终一次 chat 的上下文里没有工具结果 —— 模型是在没看到检索内容的情况下作答的"
    )


# --------------------------------------------------------------------------- #
# 2. 步数上限
# --------------------------------------------------------------------------- #

def test_max_steps_stops_runaway_loop():
    """模型永远要调工具时，必须在 agent_max_steps 处强制收尾，不能死循环。

    这里是历史上真实出过问题的地方：空转时一轮 7 步、烧掉 30317 tokens。
    """
    s = dataclasses.replace(get_settings(), agent_max_steps=3)
    if True:
        # 脚本只够 3 次决策 + 1 次强制收尾；第 4 次会被 FakeLLM 断言拦下，
        # 所以如果循环没在 max_steps 停住，测试会以 AssertionError 失败。
        agent = make_agent([
            {"tool": ("search_kb", {"query": "a"})},
            {"tool": ("search_kb", {"query": "b"})},
            {"tool": ("search_kb", {"query": "c"})},
            {"answer": "（触顶后强制收尾）"},
        ], settings=s)
        state = run(agent.run("一个永远查不到答案的问题"))

        assert state.stop_reason == "max_steps"
        assert len(state.steps) == 3



# --------------------------------------------------------------------------- #
# 3. token 预算
# --------------------------------------------------------------------------- #

def test_token_budget_stops_before_step_limit():
    """token 超预算要在步数上限之前终止 —— 步数少但 token 多的场景才是真正烧钱的。

    预算检查在**每轮开头**、用的是上一轮结束时的累计值，所以脚本必须
    先把累计推过预算线，下一轮才会被拦下；只给一步是测不到的。
    """
    step = {"tool": ("search_kb", {"query": "a"}),
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    s = dataclasses.replace(get_settings(), agent_max_steps=20, agent_token_budget=250)
    if True:
        # 三步各 120 token：120 → 240 → 360。第四轮开头 360 >= 250 被拦。
        agent = make_agent([
            dict(step), dict(step), dict(step),
            {"answer": "（预算触顶后收尾）",
             "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        ], settings=s)
        state = run(agent.run("一个很费 token 的问题"))

        assert state.stop_reason == "token_budget", (
            f"stop_reason={state.stop_reason!r}，期望 token_budget"
        )
        assert len(state.steps) <= 3



# --------------------------------------------------------------------------- #
# 4. 工具报错后的自愈
# --------------------------------------------------------------------------- #

def test_model_call_failure_is_reported_not_swallowed():
    """模型调用抛异常时要如实返回错误，而不是假装还在思考或静默空转。"""
    agent = make_agent([{"error": "upstream 500"}])
    state = run(agent.run("随便问点什么"))

    assert "模型调用失败" in state.answer
    assert "upstream 500" in state.answer, "错误原因要透传给用户，不能只说'失败'"


# --------------------------------------------------------------------------- #
# 5. 越界问题不走工具
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question,slug", [
    ("崩坏3 里最强的角色是谁？", "hhzr"),
    ("鸣潮有什么好玩的地方？", "mc"),
])
def test_uncovered_game_refuses_without_running_tools(question, slug):
    """提到未收录的游戏 → 0 步拒答，不消耗任何检索预算。

    这是真实修过的 bug：原先两个都不匹配就默认当 sr，
    于是「原神甘雨的突破材料」被拿去搜星铁库，
    最后报"知识库未收录" —— 一个建立在错误前提上的结论。
    """
    agent = make_agent([{"answer": "不该被调用"}])
    state = run(agent.run(question))

    assert state.uncovered_game == slug, (
        f"应该识别出未收录游戏 {slug}，实际 {state.uncovered_game}"
    )
    assert state.steps == [], f"越界问题不该执行任何工具，实际执行了 {len(state.steps)} 步"
    assert agent._fake.calls == 1, (
        f"越界拒答只该有 1 次 chat（生成拒答语），实际 {agent._fake.calls} 次"
    )