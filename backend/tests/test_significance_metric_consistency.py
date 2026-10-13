"""守卫：均值与配对检验必须用**同一个样本集**。

这个 bug 是实测撞出来的，不是想出来的：
重跑实验 C 后 `exp_c.py` 汇总报 C1 judge=4.77（n=13，剔了 2 条 Judge 失败），
`analyze_significance.py` 却报 C1 judge=4.13（n=15）—— 同一份文件，两个均值。

根因不是"算错了"，而是**脚本内部口径漂移**：
`paired_scores` 内部按 `exp_common.is_kept` 剔（Agent 侧 + Judge 侧），
而均值那段只按 `is_infra_failure` 剔（**只有模型侧**），
于是配对 n=13 是对的、均值 n=15 是错的。

为什么危险：这种漂移不会报错，两份数字都"看起来合理"，
而其中一份要写进简历。它只在某档恰好出现 Judge 失败时才显形。

双向验证要求：
1. 对修复前的代码实跑，本文件必须失败；
2. 顶层只 import 修复前就存在的符号（`exp_common` 的既有函数、
   `analyze_significance` 模块本身），不能 import 本轮才有的东西 ——
   否则旧代码在 collection 阶段就 ImportError，一条断言都跑不到。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
for p in (str(BACKEND), str(BACKEND / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import analyze_significance as mod  # noqa: E402
from exp_common import aggregate_rows, is_kept, judge_failed  # noqa: E402


def _ok_row(case_id: str, score: float) -> dict:
    return {
        "case_id": case_id, "judge_score": score, "correctness": score,
        "faithfulness": score, "judge_reason": "ok", "error": None,
        "prompt_tokens": 100, "completion_tokens": 50, "steps": 1,
        "latency_ms": 1000, "has_refs": True, "tool_failures": 0,
        "category": "single_hop",
    }


def _judge_broken_row(case_id: str) -> dict:
    """Judge 侧失败：模型侧一切正常（token 有消耗），但评分链路挂了。

    这是最难认的一类 —— `error` 是空的，token 也正常，
    只有 `judge_reason` 的 `judge_error:` 前缀能认出它，
    而 `judge_score` 已经被兜成 0 分混在均值里。
    """
    row = _ok_row(case_id, 0.0)
    row["judge_reason"] = "judge_error:APITimeoutError"
    return row


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把 EVAL_DIR 指到临时目录，并接管 argv 与 stdout。"""
    monkeypatch.setattr(mod, "EVAL_DIR", tmp_path)
    return tmp_path


def _write(sandbox: Path, payload: list[dict]) -> None:
    (sandbox / "exp_probe.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _run(monkeypatch: pytest.MonkeyPatch, capsys) -> str:
    monkeypatch.setattr(sys, "argv", ["analyze_significance.py", "exp_probe.json"])
    mod.main()
    return capsys.readouterr().out


# --------------------------------------------------------------------------
# 核心不变式：剔除 Judge 失败后，均值必须与 aggregate_rows 的口径逐位一致
# --------------------------------------------------------------------------

def test_mean_excludes_judge_failures(sandbox, monkeypatch, capsys):
    """3 条正常(5.0) + 2 条 Judge 失败(0 分) -> 均值必须是 5.00 而不是 3.00。"""
    _write(sandbox, [{"label": "C1", "rows": [
        _ok_row("a", 5.0), _ok_row("b", 5.0), _ok_row("c", 5.0),
        _judge_broken_row("d"), _judge_broken_row("e"),
    ]}])
    out = _run(monkeypatch, capsys)

    assert "judge=5.00" in out, f"均值未剔除 Judge 失败：\n{out}"
    assert "judge=3.00" not in out


def test_mean_and_paired_use_same_sample_set(sandbox, monkeypatch, capsys):
    """配对检验的 n 与均值的 n 必须同时反映剔除结果，不能一个剔一个不剔。"""
    _write(sandbox, [
        {"label": "C0", "rows": [_ok_row(x, 5.0) for x in "abc"]},
        {"label": "C1", "rows": [
            _ok_row(x, 5.0) for x in "abc"
        ] + [_judge_broken_row("d")]},
    ])
    out = _run(monkeypatch, capsys)

    # 均值侧：n 应为 3
    assert "n=3" in out, f"均值样本集未剔除 Judge 失败：\n{out}"
    # 配对侧：C0 有 3 条且 C1 剔后有 3 条 -> 交集也应是 3
    assert "n=3  →" in out, f"配对检验样本集异常：\n{out}"


def test_reported_mean_matches_aggregate_rows(sandbox, monkeypatch, capsys):
    """脚本打印的均值必须与 exp_common.aggregate_rows 算出来的一致。

    这是"同一份数据两个均值"最直接的判据：
    报告口径一旦与标准口径漂移，这里立刻红。
    """
    rows = [_ok_row(x, 4.0) for x in "abcd"] + [
        _judge_broken_row("e"), _judge_broken_row("f")]
    _write(sandbox, [{"label": "C1", "rows": rows}])
    expected = aggregate_rows([r for r in rows if is_kept(r)])["judge_score"]

    out = _run(monkeypatch, capsys)
    assert f"judge={expected:.2f}" in out, (
        f"与 aggregate_rows 口径漂移：期望 {expected:.2f}\n{out}")


# --------------------------------------------------------------------------
# 报告层：两类失败必须分开报，且都要报
# --------------------------------------------------------------------------

def test_reports_both_failure_kinds(sandbox, monkeypatch, capsys):
    """模型侧与 Judge 侧失败要分标签报出，不能混成一类或只报一类。"""
    infra = _ok_row("x", 0.0)
    infra.update({"prompt_tokens": 0, "completion_tokens": 0})
    _write(sandbox, [{"label": "C1", "rows": [
        _ok_row("a", 5.0), infra, _judge_broken_row("b"),
    ]}])
    out = _run(monkeypatch, capsys)

    assert "[模型侧]" in out, f"模型侧失败未单独报出：\n{out}"
    assert "[Judge 侧]" in out, f"Judge 侧失败未单独报出：\n{out}"
    assert "x" in out and "b" in out


# --------------------------------------------------------------------------
# 判据本身不能被改坏
# --------------------------------------------------------------------------

def test_judge_failure_not_mistaken_for_infra(sandbox, monkeypatch, capsys):
    """Judge 失败 token 正常，不能被当模型侧失败；反之亦然。"""
    _write(sandbox, [{"label": "C1", "rows": [
        _ok_row("a", 5.0), _judge_broken_row("b"),
    ]}])
    out = _run(monkeypatch, capsys)
    assert "[模型侧]" not in out, f"Judge 失败被误分类为模型侧：\n{out}"
    assert "[Judge 侧]: b" in out or "[Judge 侧]" in out


def test_is_infra_failure_only_covers_model_side():
    """is_infra_failure 只该认模型侧 —— 它若把 Judge 失败也算进去，
    两个谓词就又合并成一份实现，漂移会重新长出来。"""
    assert mod.is_infra_failure(_judge_broken_row("b")) is False
    infra = _ok_row("x", 0.0)
    infra.update({"prompt_tokens": 0, "completion_tokens": 0})
    assert mod.is_infra_failure(infra) is True
    # Judge 侧确实能被 is_kept 认出
    assert judge_failed(_judge_broken_row("b")) is True
    assert is_kept(_judge_broken_row("b")) is False


def test_empty_after_filter_does_not_crash(sandbox, monkeypatch, capsys):
    """整档全失败时不能崩，也不能把 0 分当能力均值报出去。"""
    _write(sandbox, [{"label": "C1", "rows": [
        _judge_broken_row("a"), _judge_broken_row("b"),
    ]}])
    out = _run(monkeypatch, capsys)
    assert "judge=" not in out, f"全失败档仍报出了均值：\n{out}"
