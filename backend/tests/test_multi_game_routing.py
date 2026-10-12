"""多游戏路由：三态（库内 / 越界 / 未提及）。

这组断言的由来是一个真实事故：最初 `route_game` 在"两个游戏都不匹配"时
默认返回 `sr`，于是「原神甘雨的突破材料」被判成星铁问题去搜星铁库，
搜不到后报"知识库未收录"——库里其实压根没有原神这个库，
问题不是"没收录这条"，而是"没有这个游戏"。一个建立在错误前提上的结论。
"""

from __future__ import annotations

import pytest
from app.agent.react import (
    UNCOVERED_GAMES,
    build_system_prompt,
    game_display_name,
    route_game,
)


def test_covered_is_n_games_not_hardcoded_two(covered):
    assert len(covered) >= 3, f"只配了 {list(covered)}，多游戏改造没落地"
    assert set(covered) == {"sr", "zzz", "ys"}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("原神里圣遗物套装有哪些效果", "ys"),
        ("绝区零的邦布等级怎么提升", "zzz"),
        ("星穹铁道的位面饰品怎么获取", "sr"),
    ],
)
def test_in_scope_game_is_routable(question, expected, covered):
    assert route_game(question, covered) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("崩坏3的雷电女王怎么配装", "hhzr"),
        ("明日方舟的罗德岛是什么", "bzhy"),
        ("塞尔达王国之泪怎么通关", "jsr"),
    ],
)
def test_uncovered_game_is_detected(question, expected, covered):
    got = route_game(question, covered)
    assert got == expected
    assert got not in covered


@pytest.mark.parametrize(
    "question",
    [
        "请计算 (0.8+0.2)*1000",
        "这段代码怎么写",
        "今天天气如何",
    ],
)
def test_unmentioned_returns_none_not_default(covered, question):
    """关键回归：没有提到任何游戏时必须返回 None，交给调用方兜底。

    曾经这里"都不匹配就默认 sr"，把没提及游戏的问题也强行按星铁处理。
    """
    assert route_game(question, covered) is None


def test_display_name_resolves_for_both_tables(covered):
    assert game_display_name("ys", covered) == "原神"
    assert game_display_name("hhzr", covered) == "崩坏3"
    assert game_display_name("nonexistent", covered) == "nonexistent"
    assert game_display_name(None, covered) == "未知游戏"


def test_uncovered_table_never_overlaps_covered(covered):
    assert not (set(UNCOVERED_GAMES) & set(covered))


def test_system_prompt_lists_every_covered_game(covered, settings):
    """另一个真实事故：SYSTEM_PROMPT 是写死"星铁与绝区零"的常量。

    加了第三个库之后，检索层能召回原神，Agent 却拿着"只覆盖两款游戏"的
    提示词把原神问题自己拒了（steps=0）。改成从 settings.games 注入。
    """
    prompt = build_system_prompt(covered, settings.game_terms)
    missing = [name for name in covered.values() if name not in prompt]
    assert not missing, f"系统提示词漏了 {missing}"


def test_system_prompt_declares_all_games_in_one_clause(settings):
    """覆盖列表必须在一句话里列全，不能散落在各处 —— 模型据此判断问题是否越界。

    （游戏名在术语行还会再出现一次，那是刻意的：既要说"覆盖哪些"，
    也要说"各自叫什么"。所以这里只断言覆盖那一行完整。）
    """
    covered = settings.games
    prompt = build_system_prompt(covered, settings.game_terms)
    first_line = prompt.splitlines()[0]
    assert "、".join(covered.values()) in first_line, f"首行没列全覆盖范围：{first_line}"


def test_game_terms_cover_every_configured_game(covered, settings):
    terms = settings.game_terms
    assert set(terms) == set(covered), f"术语表和游戏集合不一致：{set(terms)} vs {set(covered)}"


def test_out_of_scope_prompt_lists_all_covered_games(covered):
    from app.agent.react import ReActAgent, ToolContext

    agent = ReActAgent()
    _, _, state = agent._prepare("崩坏3的雷电女王怎么配装", None, None, ToolContext.build("sr"))
    assert state.uncovered_game == "hhzr"
    prompt = state.messages[0]["content"]
    missing = [n for n in covered.values() if n not in prompt]
    assert not missing, f"越界提示词没列全已收录游戏，漏了 {missing}"


def test_in_scope_question_is_not_flagged_out_of_scope(covered):
    from app.agent.react import ReActAgent, ToolContext

    agent = ReActAgent()
    _, _, state = agent._prepare("太空封印站的两件套效果是什么", None, None, ToolContext.build("sr"))
    assert state.uncovered_game is None