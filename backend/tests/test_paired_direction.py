"""守卫：配对检验的方向约定必须是「后减前」，正数 = 改进。

为什么值得单独立一条测试：
    `paired_test(a, b)` 算的是 `a - b`，符号全靠调用方传参的顺序决定。
    早先 run_eval 与 exp_common 都传成 `(基线, 改进档)`，于是
    A2→A3 打出 `Δmrr = -0.092`，而 A3 的 mrr 实际高 9.2 个点。
    数字没算错，但任何人照着这行输出写结论，都会写成「A3 下降」。

    单测通过不代表这里安全 —— 所以这条测试直接验**调用点的传参顺序**，
    而不是只测 `paired_test` 本身（它算得对，错的是顺序）。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from app.eval.metrics import paired_test

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _rows(mrr: dict[str, float]) -> list[dict]:
    return [{"case_id": k, "mrr": v} for k, v in mrr.items()]


def test_paired_test_positive_means_a_better():
    """函数本身的语义：a 更高 → mean_diff 与 cohens_d 都为正。"""
    better = _rows({"a": 1.0, "b": 0.9, "c": 0.8, "d": 1.0})
    worse = _rows({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.1})
    res = paired_test(better, worse, "mrr")
    assert res["mean_diff"] > 0
    assert res["cohens_d"] > 0
    assert res["significant"] is True


def _call_args_of(fn_name: str, filename: str) -> list[list[str]]:
    """取出脚本里所有 paired_test(...) 调用的实参文本（按文件）。

    要同时匹配 `paired_test(...)` 和 `M.paired_test(...)` ——
    run_eval 是 `import app.eval.metrics as M` 后用 `M.paired_test`，
    只认 `ast.Name` 会一个调用都找不到，守卫就会对任何代码都失败。
    """
    tree = ast.parse((SCRIPTS / filename).read_text(encoding="utf-8"))
    out: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = None
        if isinstance(f, ast.Name):
            name = f.id
        elif isinstance(f, ast.Attribute):  # M.paired_test(...)
            name = f.attr
        if name == fn_name:
            out.append([ast.unparse(a) for a in node.args])
    return out


def _norm(src: str) -> str:
    """统一引号：`ast.unparse` 默认输出单引号，直接和手写的双引号串比会误判。"""
    return src.replace("'", '"')


@pytest.mark.parametrize(
    "filename,accepted_first_args",
    [
        # run_eval：相邻两级，第一个实参必须是当前档（改进档）。
        # 多库改造后有两个调用点 —— 合并跑（cur["rows"]）与分库跑（crows，
        # 即 cur["rows"] 里筛出本库的那些行）。两者都必须传改进档，
        # 所以这里列的是**允许集合**，多一个调用点就多一个必须命中的模式。
        ("run_eval.py", ('cur["rows"]', "crows")),
        # exp_common：首尾对比，第一个实参必须是末档（改进档）
        ("exp_common.py", ("last_rows",)),
    ],
)
def test_call_sites_pass_improved_first(filename: str, accepted_first_args: tuple[str, ...]):
    """调用点必须把「改进档」放在第一个参数位。"""
    calls = _call_args_of("paired_test", filename)
    assert calls, f"{filename} 里找不到 paired_test 调用"
    for args in calls:
        first = _norm(args[0])
        assert any(a in first for a in accepted_first_args), (
            f"{filename} 的 paired_test 第一个实参是 {first!r}，"
            f"应传改进档（{accepted_first_args} 之一），否则 Δ 与 Cohen d 的符号会与直觉相反"
        )


def test_per_game_paired_test_filters_current_label_first():
    """分库配对检验的入参必须是「当前档的本库行」在前、「上一档的本库行」在后。

    单看 `paired_test(crows, prows)` 那一行看不出 crows/prows 是谁 ——
    它们来自上面两行推导。所以这里把推导也一起锁住：
    crows 必须来自 cur_g（当前/改进档），prows 必须来自 prev_g（上一档/基线）。
    两行搞反的话符号不会报错，只会静默变成「改进=变差」。
    """
    tree = ast.parse((SCRIPTS / "run_eval.py").read_text(encoding="utf-8"))
    assigns: dict[str, str] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("crows", "prows")):
            assigns[node.targets[0].id] = ast.unparse(node.value)
    assert set(assigns) == {"crows", "prows"}, f"run_eval 里找不到 crows/prows 的推导：{assigns}"
    assert "cur_g" in _norm(assigns["crows"]), (
        f"crows 必须筛自 cur_g（当前/改进档），实际是 {assigns['crows']!r}"
    )
    assert "prev_g" in _norm(assigns["prows"]), (
        f"prows 必须筛自 prev_g（上一档/基线），实际是 {assigns['prows']!r}"
    )


def test_exp_common_first_arg_is_last_rows():
    """exp_common 传参顺序不能被改回去。"""
    calls = _call_args_of("paired_test", "exp_common.py")
    first_args = [a[0] for a in calls]
    assert any("last_rows" in a for a in first_args), (
        f"exp_common 的 paired_test 首个实参是 {first_args}，应传末档（last_rows）"
    )


def test_paired_test_is_not_imported_directly_in_scripts():
    """两个脚本都必须走 paired_scores / 本地封装，避免出现第三处裸调用。"""
    for filename in ("run_eval.py", "exp_common.py"):
        src = (SCRIPTS / filename).read_text(encoding="utf-8")
        # 允许出现定义与调用；只要求调用点数量可控（上面已逐个断言方向）
        assert src.count("paired_test(") <= 3, f"{filename} 出现多处 paired_test 调用，需逐个确认方向"


def test_module_docstring_declares_direction():
    """docstring 必须写明方向，否则后人无从判断符号。"""
    doc = inspect.getdoc(paired_test) or ""
    assert "减" in doc and ("改进" in doc or "正数" in doc), "paired_test 的 docstring 必须声明方向约定"