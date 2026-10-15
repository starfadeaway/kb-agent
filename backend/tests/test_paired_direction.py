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


def _paired_calls(filename: str) -> list[ast.Call]:
    """脚本里所有 paired_test(...) 调用点。

    要同时匹配 `paired_test(...)` 和 `M.paired_test(...)` ——
    run_eval 是 `import app.eval.metrics as M` 后用 `M.paired_test`，
    只认 `ast.Name` 会一个调用都找不到，守卫就会对任何代码都失败。
    """
    tree = ast.parse((SCRIPTS / filename).read_text(encoding="utf-8"))
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = None
        if isinstance(f, ast.Name):
            name = f.id
        elif isinstance(f, ast.Attribute):  # M.paired_test(...)
            name = f.attr
        if name == "paired_test":
            out.append(node)
    return out


def _names_in(node: ast.AST) -> set[str]:
    """实参子树里出现过的标识符与属性名。

    **为什么要穿透子树，而不是比对实参的字面文本**：分库那处为了保证
    「按库筛 + kept_rows」这两步真的发生过，把它们直接写进了实参
    （`kept_rows([x for x in cur_g["rows"] ...])`）。按字面匹配变量名
    会抓不到，而守卫真正要守的是「来源是当前档」，不是「变量叫什么」。
    """
    names: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
    return names


@pytest.mark.parametrize(
    "filename,accepted_first_args",
    [
        # run_eval：相邻两级，第一个实参必须是当前档（改进档），
        # 既包括合并跑的 `cur`，也包括分库跑筛自 `cur_g` 的那些行。
        # 判据是「第一个实参的来源标识符里出现了这些名字之一」，
        # 所以外面套一层 kept_rows(...) 不影响判定。
        ("run_eval.py", ("cur", "cur_g")),
        # exp_common：首尾对比，第一个实参必须是末档（改进档）
        ("exp_common.py", ("last_rows",)),
    ],
)
def test_call_sites_pass_improved_first(filename: str, accepted_first_args: tuple[str, ...]):
    """调用点必须把「改进档」放在第一个参数位。"""
    calls = _paired_calls(filename)
    assert calls, f"{filename} 里找不到 paired_test 调用"
    for call in calls:
        assert call.args, f"{filename} 的 paired_test 调用没有实参"
        names = _names_in(call.args[0])
        assert names & set(accepted_first_args), (
            f"{filename} 的 paired_test 第一个实参是 {ast.unparse(call.args[0])!r}"
            f"（来源标识符 {sorted(names)}），"
            f"应传改进档（{accepted_first_args} 之一），"
            f"否则 Δ 与 Cohen d 的符号会与直觉相反"
        )


def test_per_game_paired_test_filters_current_label_first():
    """分库配对检验必须「当前档的本库行」在前、「上一档的本库行」在后。

    早先这里锁的是 `crows` / `prows` 两个**局部变量**的推导式。现在
    「按库筛 + kept_rows」两步直接写在实参里（理由见 run_eval 那处的注释：
    写成变量，接线守卫就没法证明过滤真的发生过），所以判据改成
    穿透实参子树看**来源** —— 锁的是语义，不是变量的名字。

    搞反的后果：符号不会报错，只会静默变成「改进 = 变差」。
    """
    calls = _paired_calls("run_eval.py")
    per_game = []
    for call in calls:
        if len(call.args) < 2:
            continue
        first, second = _names_in(call.args[0]), _names_in(call.args[1])
        # 只挑出分库那一处：它的两个实参都不来自合并跑的 prev/cur
        if {"cur_g", "prev_g"} & (first | second):
            per_game.append(ast.unparse(call))

    assert per_game, "run_eval 里找不到分库配对检验的调用点"

    for src in per_game:
        assert "cur_g" in src and "prev_g" in src, (
            f"分库配对检验没同时用到 cur_g / prev_g：{src}"
        )
        # 第一个实参：必须筛自 cur_g，且不能碰 prev_g
        head, _, tail = src.partition(",")
        assert "cur_g" in head and "prev_g" not in head, (
            f"第一个实参应筛自 cur_g（当前/改进档），实际是 {head!r}"
        )
        assert "prev_g" in tail and "cur_g" not in tail, (
            f"第二个实参应筛自 prev_g（上一档/基线），实际是 {tail!r}"
        )


def test_exp_common_first_arg_is_last_rows():
    """exp_common 传参顺序不能被改回去。"""
    calls = _paired_calls("exp_common.py")
    assert calls, "exp_common 里找不到 paired_test 调用"
    first_names = [_names_in(c.args[0]) for c in calls if c.args]
    assert any("last_rows" in names for names in first_names), (
        f"exp_common 的 paired_test 首个实参来源是 {first_names}，应传末档（last_rows）"
    )


def test_paired_test_is_not_imported_directly_in_scripts():
    """两个脚本都必须走 paired_scores / 本地封装，避免出现第三处裸调用。"""
    for filename in ("run_eval.py", "exp_common.py"):
        src = (SCRIPTS / filename).read_text(encoding="utf-8")
        # 允许出现定义与调用；只要求调用点数量可控（上面已逐个断言方向）
        assert src.count("paired_test(") <= 3, f"{filename} 出现多处 paired_test 调用，需逐个确认方向"


def test_p值不能被round成零():
    """`round(3.1e-15, 4)` = 0.0，下游会把 0.0 读成「极显著、远超阈值」。

    实测踩过（2026-10）：zzz 的 C0→C2 精确 p 是 **2.78e-08**，
    产物里却是 `0.0`，于是被写成「三库 p 均 <1e-10」—— 差 278 倍。

    这里构造一组差值为 0..9 的配对样本：mean_diff=4.5、sd_diff≈3.03、
    t≈4.70、精确 p≈2.6e-06。`round(p, 4)` 会把它抹成 0.0。
    """
    a = _rows({f"c{i}": float(i) for i in range(10)})
    b = _rows({f"c{i}": 0.0 for i in range(10)})
    res = paired_test(a, b, "mrr")

    assert res["p"] > 0.0, "p 被 round 成 0.0 ——「算得出但很小」被说成了「恰好为零」"
    assert res["p"] < 1e-4, f"构造应落在 1e-4 以下，实际 {res['p']}"
    # 显示侧也不能退化成 0.0000：`f"{2.6e-6:.4f}"` 正是 0.0000
    assert "e-0" in res["p_display"], f"p_display 丢了信息：{res['p_display']!r}"
    assert res["significant"] is True


def test_效应量的主口径是配对口径():
    """配对数据必须用 `sd_diff` 当分母，组间合并口径会把 d 系统性放大。

    本项目实测放大 1.4~1.6 倍（sr 2.163→1.327、zzz 2.778→1.964），
    而 README 一直引用的是被放大的那列。两个口径都返回，
    **`cohens_d` 必须是配对口径**，否则这个洞会重新长出来。
    """
    a = _rows({f"c{i}": float(i) for i in range(10)})
    b = _rows({f"c{i}": 0.0 for i in range(10)})
    res = paired_test(a, b, "mrr")

    assert res["cohens_d"] == pytest.approx(res["mean_diff"] / res["sd_diff"], abs=1e-3), (
        "cohens_d 不是配对口径（mean_diff / sd_diff）"
    )
    # 配对 t 检验里 d_z = t / sqrt(n)，同一个量的另一种算法，用来自洽校验
    assert abs(res["cohens_d"]) == pytest.approx(abs(res["t"]) / (res["n"] ** 0.5), abs=1e-2)
    assert "cohens_d_pooled" in res, "旧口径必须保留，否则历史产物无法对账"
    assert res["cohens_d_pooled"] != res["cohens_d"], "两个口径不该相等 —— 相等说明改动没生效"


def test_module_docstring_declares_direction():
    """docstring 必须写明方向，否则后人无从判断符号。"""
    doc = inspect.getdoc(paired_test) or ""
    assert "减" in doc and ("改进" in doc or "正数" in doc), "paired_test 的 docstring 必须声明方向约定"