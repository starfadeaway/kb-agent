"""守卫：基础设施失败必须计入熔断计数。

这个 bug 的形状
--------------
`SearchGuard.observe()` 原本只认两种结果形态：
`status=empty`（检索为空）和 `top_score=`（召回了但分数低）。

`search_kb` 在上游抖动、429、超时的时候返回的是
`[检索失败] …` / `[超时] …` / `[异常] …` —— 这三种都不含上面两个标记，
于是全部落到函数末尾的兜底分支 `consecutive_misses = 0`。

后果正是 SearchGuard 当初要防的那件事：**连续拿不到信息时熔断失效，
Agent 一路空转到 agent_max_steps**（历史上真实发生过：一轮 7 步、
30317 tokens）。而且它从统计上看不出来 —— guard_stats 里的
consecutive_misses 一直是 0，看起来"熔断工作正常"。

而 `[错误]`（比如 query 为空）不该计入：那是模型自己传错参数，
改一下就能重试，计入熔断等于惩罚一次可自愈的错误。
"""
from __future__ import annotations

from app.agent.react import SearchGuard
from app.agent.tools import _ERROR_PREFIXES, _INFRA_FAILURE_PREFIXES


def test_infra_failure_increments_misses():
    """超时 / 检索失败 / 异常 都要累加 miss，否则熔断永远不触发。"""
    for prefix in ("[超时]", "[检索失败]", "[异常]", "[抓取失败]"):
        g = SearchGuard(miss_limit=3)
        g.observe("search_kb", f"{prefix} 工具 search_kb 超过 30s 未返回，请换用更简单的查询。")
        assert g.consecutive_misses == 1, (
            f"{prefix} 之后 consecutive_misses 应为 1，实际 {g.consecutive_misses} —— "
            f"熔断被失败结果悄悄解除"
        )


def test_repeated_infra_failure_trips_the_breaker():
    """连续基础设施失败必须达到熔断阈值并真的拦截下一次检索。"""
    g = SearchGuard(miss_limit=3)
    for _ in range(3):
        g.observe("search_kb", "[检索失败] OperationalError: connection reset")

    blocked = g.check("search_kb", {"query": "再查一次试试"})
    assert blocked is not None, "连续 3 次检索失败后仍未熔断"
    assert blocked.startswith("[熔断]")
    assert g.forced_finishes == 1


def test_param_error_does_not_trip_the_breaker():
    """`[错误]` 是模型自己传错参数，可自愈，不该计入熔断。"""
    g = SearchGuard(miss_limit=2)
    for _ in range(4):
        g.observe("search_kb", "[错误] query 不能为空。")
    assert g.consecutive_misses == 0, "参数错误被误计为检索 miss —— 会让 Agent 因自己的笔误被熔断"
    assert g.check("search_kb", {"query": "正常查询"}) is None


def test_empty_result_still_counts():
    """原有语义不能被破坏：检索为空仍然计 miss。"""
    g = SearchGuard(miss_limit=2)
    g.observe("search_kb", "[search_kb] status=empty hits=0 top_score=0.000 12ms")
    assert g.consecutive_misses == 1


def test_successful_hit_resets_counter():
    """成功命中后计数归零 —— 否则一次成功之后又会很快熔断。"""
    g = SearchGuard(miss_limit=3, low_score=0.5)
    g.observe("search_kb", "[检索失败] timeout")
    g.observe("search_kb", "[search_kb] status=ok hits=5 top_score=0.800 300ms top_page=X")
    assert g.consecutive_misses == 0


def test_breaker_messages_themselves_are_not_counted():
    """熔断/去重文案不能计入 —— 否则计数会被自己产生的文本推高。"""
    g = SearchGuard(miss_limit=2)
    g.observe("search_kb", "[熔断] 此前已连续 2 次检索未找到可信内容")
    g.observe("search_kb", "[重复查询已拦截] 「x」与本轮之前的检索等价")
    assert g.consecutive_misses == 0


# --------------------------------------------------------------------------- #
# 前缀集合自身的契约
# --------------------------------------------------------------------------- #

def test_infra_set_is_strict_subset_of_error_set():
    """基础设施失败是工具错误的一部分，反之不然 —— 否则就是两个独立维护的真相。"""
    assert set(_INFRA_FAILURE_PREFIXES) < set(_ERROR_PREFIXES)


def test_param_error_is_excluded_from_infra_set():
    """这条是上面那个语义区分的锚点：改错方向会直接改变熔断行为。"""
    assert "[错误]" not in _INFRA_FAILURE_PREFIXES


def test_every_infra_prefix_is_produced_somewhere_in_tools():
    """每个前缀都必须在 tools.py 里真的被产出，否则就是死约定。"""
    import inspect

    from app.agent import tools

    src = inspect.getsource(tools)
    missing = [p for p in _INFRA_FAILURE_PREFIXES if f'"{p}' not in src and f"'{p}" not in src]
    assert not missing, f"这些前缀在 tools.py 里找不到产出点：{missing}"