"""守卫：链路失败不能混进能力指标，且均值与配对检验必须同一批样本。

缺陷（2026-10-13 实验 E 实测暴露）：
    E1 档 sr-008 的 judge_score=0.0，读起来像「DeepSeek 这题彻底答崩了」，
    实际是 Judge 侧一次 APITimeoutError —— `judge_reason` 写着
    "judge_error:APITimeoutError"，而 `error` 字段是空的。

    为什么 `error` 拦不住：它只记 **Agent 侧** 异常。Judge 自己失败时
    Agent 其实好好跑完了，所以 `error=None`，但 `score()` 已经把这个 0 分
    兜底写进了 judge_score。于是 `is_kept` 放行，0 分进均值：
    E1 从 4.60 掉到 4.30 —— 一眼看过去像是 DeepSeek 能力不行，
    实际是评测链路抖了一下。**方向完全相反的误判。**

    更隐蔽的是 bug #8：`paired_scores` 当时硬编码 `not r.get("error")`，
    绕过了 `is_kept`。于是 aggregate 按 n=14 算、配对检验按 n=15 算，
    两份数字来自不同样本集，Δ 和 p 却被当成同一份数据的结论写进文档 ——
    这种不自洽事后极难发现，因为每个数字单独看都「正常」。

本组测试锁住三条不变量：
    1. judge_error 行不进均值、不进配对；
    2. 均值与配对检验的样本集必须一致；
    3. 正常低分与空回答不被误伤（回归保护）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import exp_common  # noqa: E402
from exp_common import aggregate_rows, is_kept, paired_scores  # noqa: E402

# 注意：这里刻意**不** `from exp_common import judge_failed`。
# 双向验证时要把测试跑在修复前的快照上，而 judge_failed 是修复后才有的。
# 顶层就 import 它的话，旧代码在 collection 阶段就 ImportError ——
# 那只证明「新函数不存在」，跑不到任何行为断言，等于没验证。
# 现在只依赖 is_kept / aggregate_rows / paired_scores 三个修复前就有的 API，
# 旧代码能正常收集并逐条执行，断言失败才是真的失败。


def _row(case_id, score, *, reason="答得不错", error=None, tokens=1000):
    return {
        "case_id": case_id, "category": "c", "judge_score": score,
        "correctness": score, "faithfulness": score, "completeness": score,
        "judge_reason": reason, "error": error, "prompt_tokens": tokens,
        "completion_tokens": 100, "steps": 1, "tool_failures": 0,
        "has_refs": True, "latency_ms": 1000,
    }


def _good(cid, score=5.0):
    return _row(cid, score)


def _judge_broken(cid):
    """Agent 跑完了，是 Judge 超时 —— score() 兜成了 0 分。"""
    return _row(cid, 0.0, reason="judge_error:APITimeoutError")


# ---------- 判定层 ----------

def test_judge_failed_detects_error_prefix():
    assert exp_common.judge_failed(_judge_broken("x")) is True


def test_judge_failed_spares_normal_and_empty_answer():
    """回归保护：正常低分与空回答都不该被当成链路故障。"""
    assert exp_common.judge_failed(_row("x", 2.0, reason="只答对一半")) is False
    assert exp_common.judge_failed(_row("x", 0.0, reason="空回答")) is False
    assert exp_common.judge_failed(_row("x", 5.0, reason="完全正确")) is False


def test_is_kept_blocks_judge_failure_but_spares_low_score():
    assert is_kept(_judge_broken("x")) is False
    assert is_kept(_row("x", 1.0, reason="基本没答对")) is True, (
        "真实低分必须保留 —— 把低分当故障剔掉就是粉饰"
    )


# ---------- 聚合层 ----------

def test_mean_excludes_judge_failure():
    """核心：0 分的链路失败不得拉低能力均值。"""
    rows = [_good("a"), _good("b"), _good("c"), _judge_broken("d")]
    agg = aggregate_rows(rows)
    assert agg["judge_score"] == 5.0, "judge_error 的 0 分被算进均值了"
    assert agg["n_ok"] == 3


def test_judge_errors_counted_separately():
    """Judge 侧失败要单独计数：它是链路健康度，不是能力。"""
    agg = aggregate_rows([_good("a"), _good("b"), _judge_broken("d")])
    assert agg["judge_errors"] == 1
    assert agg["errors"] == 0, "judge 侧失败不应混进 Agent 侧的 errors 计数"


def test_agent_error_still_excluded():
    """回归保护：原有的 Agent 侧排除不能被改坏。"""
    rows = [_good("a"), _good("b"), _row("c", 0.0, error="boom")]
    assert aggregate_rows(rows)["n_ok"] == 2


def test_all_failed_is_flagged():
    """全失败时必须显式标记，否则「评测挂了」会被读成「模型全错」。"""
    agg = aggregate_rows([_judge_broken("a"), _judge_broken("b")])
    assert agg["all_failed"] is True


def test_not_all_failed_flag_is_false():
    agg = aggregate_rows([_good("a"), _judge_broken("b")])
    assert agg["all_failed"] is False


# ---------- 样本集一致性（bug #8）----------

@pytest.mark.asyncio
async def test_paired_and_mean_use_identical_sample_set():
    """均值与配对检验必须落在同一批 case 上。

    修复前：aggregate 按 n=4 算、paired 按 n=5 算，两套样本集，
    Δ 与 p 却被当成同一份数据的结论。
    """
    shared = [_good("a"), _good("b"), _judge_broken("c"), _good("d"), _good("e")]
    r1 = {"label": "E1", "rows": [dict(x) for x in shared]}
    r2 = {"label": "E2", "rows": [_good("a", 4.0), _good("b", 4.0),
                                   _good("c", 0.0), _good("d", 4.0), _good("e", 4.0)]}
    results = [r1, r2]

    agg1 = aggregate_rows(r1["rows"])
    assert agg1["n_ok"] == 4, "均值侧应剔除 judge_error"

    res = paired_scores(results, metric="judge_score")
    assert res is not None
    assert res["n"] == agg1["n_ok"], (
        f"配对检验 n={res['n']} 与均值 n_ok={agg1['n_ok']} 不一致 —— "
        "两份数字来自不同样本集，Δ 和 p 不能一起写进结论"
    )


def test_paired_scores_accepts_exclude_channel():
    """exp_e 特有的额外排除（模型压根没调通）也要能传进去。"""
    rows_a = [_row("a", 5.0, tokens=0), _good("b"), _good("c"), _good("d")]
    rows_b = [_row("a", 0.0, tokens=0), _good("b"), _good("c"), _good("d")]
    results = [{"label": "E1", "rows": rows_a}, {"label": "E2", "rows": rows_b}]

    def only_counted(row):
        return is_kept(row) and (row.get("prompt_tokens") or 0) > 0

    agg = aggregate_rows(rows_a, exclude=only_counted)
    assert agg["n_ok"] == 3
    res = paired_scores(results, metric="judge_score", exclude=only_counted)
    assert res is not None
    assert res["n"] == agg["n_ok"]


def test_exp_e_is_counted_wires_up_the_exclude():
    """exp_e 必须真的把这个谓词传下去，而不是定义了却不用。"""
    import ast

    src = (Path(__file__).resolve().parents[1] / "scripts" / "exp_e.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in {"aggregate_rows", "paired_scores"}
    ]
    assert calls, "exp_e 里找不到聚合调用"
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "exclude" in kw and isinstance(kw["exclude"], ast.Name) \
            and kw["exclude"].id == "is_counted", (
            f"{c.func.id}() 没有传 exclude=is_counted —— "
            "定义了失败谓词却不用，等于没修"
        )