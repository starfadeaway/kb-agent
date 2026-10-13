"""守卫：跨模型比较的输出必须自带方向，不能让读者反推符号。

缺陷（修 `analyze_model_compare.py` 时自查出来的）
--------------------------------------------------
JSON 里原本只有 `significant: true` 和 `mean_diff: +37.6552`。
但延迟是**越低越好**的指标，于是这行读起来是：

    latency_ms  mean_diff=+37.6552  significant=true

而实际含义是 **Qwen 更快**。任何人（包括写文档的我自己）
都可能直接读成「bge 显著更好」—— 正负号刚好反着。

更糟的是这个实验**唯二显著的结果全是延迟**，
正好全部落在这个陷阱里。

这和 `judge.pairwise` 把解析失败静默算成 TIE 是同一类错误：
**把需要语义判断的信息，编码成要读者自己解码的裸值。**

所以输出必须给出 `winner` / `verdict`，方向判定由
`LOWER_IS_BETTER` 真正参与，而不是只拿来打一个标签。

已对修复前的快照实跑确认能抓到缺陷。
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_model_compare.py"


def _load():
    """直接按文件加载脚本（它在 scripts/ 下、靠 sys.path 找 backend 包）。"""
    sys.path.insert(0, str(SCRIPT.parents[1]))
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("amc_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _artifact():
    import json
    p = SCRIPT.parents[2] / "data" / "eval" / "model_compare.json"
    if not p.is_file():
        pytest.skip("model_compare.json 尚未生成")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------- 方向判定本身 ----------

def test_positive_diff_on_latency_means_other_is_better():
    """延迟差为正 = bge 更慢 = Qwen 更好。这一条最容易读反。"""
    mod = _load()
    assert mod._winner(37.66, higher_is_better=False) == "other"


def test_negative_diff_on_latency_means_bge_is_better():
    mod = _load()
    assert mod._winner(-37.66, higher_is_better=False) == "bge"


def test_positive_diff_on_quality_means_bge_is_better():
    mod = _load()
    assert mod._winner(0.0517, higher_is_better=True) == "bge"


def test_zero_diff_is_tie():
    mod = _load()
    assert mod._winner(0.0, higher_is_better=True) == "tie"
    assert mod._winner(0.0, higher_is_better=False) == "tie"


def test_latency_is_registered_as_lower_is_better():
    """延迟必须在表里，否则方向判定会当成「越大越好」。"""
    mod = _load()
    assert "latency_ms" in mod.LOWER_IS_BETTER


# ---------- 结构守卫：winner 必须真的被计算并写入 ----------

def test_winner_is_computed_in_main_not_just_defined():
    """定义了不算数 —— 必须真的参与输出。

    这正是原缺陷的形态：`LOWER_IS_BETTER` 只被用来打 `higher_is_better`
    这个标签，从头到尾没有任何代码用它判断谁更好。
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    called = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_winner" in called, "_winner() 定义了却没被调用：又是一个半成品"


def test_winner_field_is_written_into_output():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    keys = {
        k.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Dict)
        for k in n.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert "winner" in keys, "输出字典里没有 winner 字段"
    assert "verdict" in keys, "输出字典里没有可读的结论字段"


def test_p_underflow_is_flagged():
    """p 的浮点下溢要单独标记，否则 JSON 里的 0.0 会被当成真的 0。"""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    keys = {
        k.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Dict)
        for k in n.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert "p_is_underflow" in keys, "p=0.0 未与浮点下溢区分开"


# ---------- 真实产物校验 ----------

@pytest.mark.parametrize("label", ["A1", "A2"])
def test_real_artifact_latency_winner_is_qwen(label):
    """本实验唯二显著的结果是延迟，方向必须是 Qwen。

    如果哪天 bge 真的变快了，这里会红 —— 那正是要提醒的。
    """
    d = _artifact()
    m = d[label]["metrics"]["latency_ms"]
    assert m["significant"] is True, f"{label} 延迟不再显著，结论需复核"
    assert m["winner"] == "other", (
        f"{label} 延迟 winner 应为 other(Qwen)，实际 {m['winner']} —— "
        "方向判定或产物已过期"
    )
    assert m["mean_diff"] > 0, "bge 延迟更大即 bge 更慢"


def test_p_underflow_only_on_the_extreme_case():
    """只有 A1 的延迟 p 小到浮点下溢；A2 是 0.0163，不该被一起标成下溢。

    把两者混为一谈会让「p=0.0」和「p 极小」失去区分，
    正是 memory 里记过的那个坑：p 下溢必须写 p<1e-10，不能写 p=0。
    """
    d = _artifact()
    a1 = d["A1"]["metrics"]["latency_ms"]
    a2 = d["A2"]["metrics"]["latency_ms"]
    assert a1["p_is_underflow"] is True, "A1 p<1e-10 应标记为下溢"
    assert a2["p_is_underflow"] is False, "A2 p≈0.016 不是下溢"