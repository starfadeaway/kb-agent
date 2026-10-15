"""守卫：`run_eval.py` 的均值与配对检验必须落在**同一批样本**上。

同一个不变式在 `analyze_significance.py` 那边已有一组守卫
（`test_significance_metric_consistency.py`），它来自实验 C 的一次实测翻车：
汇总报 C1 judge=4.77（n=13，剔了 Judge 失败），配对检验报 C1 judge=4.13（n=15）
—— 同一份产物，两个均值。

检索侧有同一个洞，且当时没修：`summarize` 用 `is_kept` 滤掉了失败行，
但 `M.paired_test(...)` 收到的是**原始 rows**，而失败行的 `mrr` 被硬写成
`0.0`（见检索异常那条分支）。后果是某一档只要多挂一条链路故障，
Δ 就被凭空拉低 —— 看起来像「这个配置检索能力更差」，实际是「它没跑起来」。

本文件三层判据：

1. **行为**：`kept_rows` 真的滤掉两类失败行；
2. **接线**：`run_eval.py` 里每一处 `paired_test(` 的实参都过了 `kept_rows`
   —— 漏一处就等于没修，而这种漏法不会报错；
3. **显现**：打印表必须带 `n_ok` / `errors`，全档失败要当场把话说死。

第 3 条与第 1、2 条同等重要：口径修对了但**看不见**，下一次仍会被误读成能力。
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "run_eval.py"

# run_eval 与 exp_common 都在 scripts/ 下，不在 backend 的包路径里，
# 不先把两个路径塞进 sys.path，加载时会是 ModuleNotFoundError。
BACKEND = SCRIPTS.parent
for _p in (str(BACKEND), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SOURCE = SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def runeval():
    spec = importlib.util.spec_from_file_location("run_eval_parity_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_eval_parity_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ok_row(case_id: str, mrr: float) -> dict:
    return {
        "case_id": case_id, "mrr": mrr, "recall@3": mrr, "hit_rate": mrr,
        "ndcg": mrr, "page_hit": mrr > 0, "latency_ms": 100, "error": None,
    }


def _infra_row(case_id: str) -> dict:
    """Agent 侧链路故障：指标被硬写成 0，并带 error 字段。

    这正是 `run_eval.py` 检索异常分支产出的行 —— 它不是「检索能力为 0」，
    而是「这条压根没跑」。
    """
    row = _ok_row(case_id, 0.0)
    row["error"] = "RuntimeError: 索引不可用"
    return row


def _judge_row(case_id: str) -> dict:
    """Judge 侧失败：error 为空，只有 judge_reason 前缀能认出它。"""
    row = _ok_row(case_id, 0.0)
    row["judge_reason"] = "judge_error:APITimeoutError"
    return row


# --------------------------------------------------------------------------- #
# 1. 行为：谓词本身
# --------------------------------------------------------------------------- #
def test_kept_rows_滤掉两类失败行(runeval):
    rows = [
        _ok_row("a", 1.0),
        _infra_row("b"),
        _judge_row("c"),
        _ok_row("d", 0.5),
    ]
    kept = runeval.kept_rows(rows)
    assert [r["case_id"] for r in kept] == ["a", "d"], (
        "kept_rows 没有同时滤掉 Agent 侧与 Judge 侧的失败行 —— "
        "两边的失败都会把 0 分带进配对差里"
    )


def test_kept_rows_与均值的谓词同源(runeval):
    """`kept_rows` 不能自己发明一套判据，必须就是 `summarize` 用的那个。

    判据：同一批 rows，`kept_rows` 的长度必须等于 `summarize` 报出的 n_ok。
    """
    rows = [_ok_row("a", 1.0), _infra_row("b"), _judge_row("c")]
    assert len(runeval.kept_rows(rows)) == runeval.summarize(rows, 3)["n_ok"]


# --------------------------------------------------------------------------- #
# 2. 接线：每一处配对检验都过了同一谓词
# --------------------------------------------------------------------------- #
def _is_kept_rows_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "kept_rows"
    )


def test_每一处配对检验都过了kept_rows():
    """接线守卫。漏一处就等于没修，而漏了不会报错 —— 只是数字悄悄变错。

    用 AST 而不是字符串匹配：`# M.paired_test(...)` 这种注释也会被
    文本匹配算进去，而 `ast.walk` 只认真实调用点。
    """
    tree = ast.parse(SOURCE)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "paired_test"
    ]
    assert calls, "run_eval.py 里找不到 paired_test 调用 —— 是扫描器坏了，不是修好了"

    for call in calls:
        assert len(call.args) >= 2, f"第 {call.lineno} 行的 paired_test 实参不足两个"
        for arg in call.args[:2]:
            assert _is_kept_rows_call(arg), (
                f"第 {call.lineno} 行的 paired_test 实参没走 kept_rows："
                f"{ast.unparse(arg)} —— 均值滤了失败行、配对没滤，"
                f"两个数字就不是同一份数据算出来的"
            )


# --------------------------------------------------------------------------- #
# 3. 双向验证：不过滤到底会不会把结论带偏
# --------------------------------------------------------------------------- #
def test_不过滤时链路故障会把Δ拉低(runeval):
    """把「修复前」的行为实跑一遍，证明这个洞有真实后果。

    四题两档，唯一的差别是当前档的第 4 题**挂了**（不是答错）。
    正确口径下 Δmrr = 0；修复前它会报成一个负的、看起来很真实的差距。
    """
    from app.eval import metrics as M

    base = [_ok_row(x, 1.0) for x in "abcd"]
    cur = [_ok_row(x, 1.0) for x in "abc"] + [_infra_row("d")]

    fixed = M.paired_test(runeval.kept_rows(cur), runeval.kept_rows(base), "mrr")
    assert fixed["n"] == 3, f"过滤后应剩 3 条配对样本，实为 {fixed.get('n')}"
    assert fixed["mean_diff"] == pytest.approx(0.0), (
        "两档在能比的样本上完全一致，Δ 必须是 0"
    )

    before = M.paired_test(cur, base, "mrr")
    assert before["n"] == 4, "修复前的口径把失败行也配进去了"
    assert before["mean_diff"] == pytest.approx(-0.25), (
        f"修复前应把 Δ 拉到 -0.25，实为 {before.get('mean_diff')}"
    )
    assert before["mean_diff"] < fixed["mean_diff"], (
        "这正是危险之处：一次链路故障伪装成「这个配置更差」"
    )


# --------------------------------------------------------------------------- #
# 4. 显现：打印表必须把失败暴露出来
# --------------------------------------------------------------------------- #
def _func_source(name: str) -> str:
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"run_eval.py 里找不到函数 {name}")


def test_打印表带n_ok与errors():
    """只打 `n` 的话，全档失败会打印成一张正常形状的全零表。"""
    line = _func_source("line")
    assert "n_ok" in line, "打印行没有 n_ok —— 读者看不出这一档成功了几条"
    assert "errors" in line, "打印行没有 errors —— 读者看不出有几条是链路故障"


def test_表头声明了这两列():
    assert "{'n_ok':>6}" in SOURCE, "表头没有 n_ok 列"
    assert "{'错误':>7}" in SOURCE, "表头没有 errors 列"


def test_全档失败时打印行显式提示():
    """`n_ok=0` 只是一列数字，扫一眼仍会读成「分数很低」，所以要说死。"""
    line = _func_source("line")
    assert "n_ok" in line and "全档失败" in line, (
        "全档失败时打印行没有给出提示 —— 一次故障会被读成检索能力为零"
    )


def test_全档失败时聚合仍如实报出n_ok与errors(runeval):
    rows = [_infra_row("a"), _infra_row("b")]
    agg = runeval.summarize(rows, 3)
    assert agg["n"] == 2
    assert agg["n_ok"] == 0
    assert agg["errors"] == 2
    # 指标键必须补齐：全档失败不能让下游读 key 时 KeyError
    assert agg["mrr"] == 0.0
