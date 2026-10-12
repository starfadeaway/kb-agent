"""Agent 实际挂载的工具集 —— 盯住"文档结论有没有落到代码上"。

背景是一处已经踩过的坑：EVALUATION.md 写着「按实验 C 的数据，
`/chat` 默认只给 `search_kb`」，但 `main.py` 构造 Agent 时**没传 tools**，
而 `ReActAgent.__init__` 的兜底是 `get_schemas()`（全量）。
于是线上跑的是 C2 形态，而文档声称收敛到了 C1 ——
这种"结论和实现不一致"面试被翻代码就穿了，所以必须有断言锁住。
"""

from __future__ import annotations

import pytest
from app.agent.react import ReActAgent
from app.agent.tools import REGISTRY, get_schemas


def test_production_default_is_search_kb_only(settings):
    """实验 C：多挂工具没有可测收益（C1 4.60 ≥ C2 4.47，p=0.47 未显著），
    却要多付 22.6% token —— 所以生产默认必须收敛到 search_kb。"""
    assert settings.agent_tools == "search_kb"
    assert settings.agent_tool_names == ["search_kb"]
    active = [s["function"]["name"] for s in get_schemas(settings.agent_tool_names)]
    assert active == ["search_kb"], f"生产默认工具集变成了 {active}"


def test_agent_without_explicit_tools_uses_configured_subset(settings):
    """**回归**：不传 tools 时不能回退到全量注册表。"""
    agent = ReActAgent()
    assert [s["function"]["name"] for s in agent.tools] == ["search_kb"]


def test_full_toolset_still_reachable_for_experiment_c(settings):
    """C2 档必须还能跑，否则实验 C 就不可复现了。"""
    names = list(REGISTRY)
    active = [s["function"]["name"] for s in get_schemas(names)]
    assert active == ["search_kb", "get_page", "calculator", "list_games"]


def test_get_schemas_with_none_means_all():
    assert len(get_schemas(None)) == len(REGISTRY)


def test_get_schemas_rejects_unknown_name():
    with pytest.raises(ValueError, match="未知工具"):
        get_schemas(["definitely_not_a_tool"])


def test_agent_accepts_explicit_tool_override(settings):
    """`/chat` 传 tools 参数能复现 C2 —— 文档承诺了这个开关，它必须存在。"""
    agent = ReActAgent(tools=get_schemas(list(REGISTRY)))
    assert len(agent.tools) == len(REGISTRY)


def test_empty_tools_degrades_to_single_turn(settings):
    """空列表 = 不给任何工具（退化成单轮问答），这是 C0 档。"""
    agent = ReActAgent(tools=[])
    assert agent.tools == []
    assert agent._tool_schemas is None, "空工具集应当退化成不给任何声明"


def test_env_example_documents_agent_tools():
    from tests.conftest import PROJECT_ROOT

    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8-sig")
    assert "AGENT_TOOLS=search_kb" in text, ".env.example 没写默认工具集，改了不会生效"