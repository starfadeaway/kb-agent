"""守卫：实验产物的每一行必须带 `guard` 与 `stop_reason`。

`docs/failure_analysis.md` 第六节承认「熔断机制被测试验证过，但在真实问题
分布上从没被观测过」。根因不是没跑题 —— `exp_c` / `exp_e` 都走完整的
ReAct 循环，`AgentState` 上明明有 `guard_stats()` 和 `stop_reason`。

问题是**跑完就把状态丢了**：行字典里一个都没记。后果是 `data/eval/` 全部
产物里 `guard` 出现 0 次，谁也没法回答「这三库的题里熔断到底触发过几次」。

行的形状在 2026-10-15 收口到 `exp_common.case_row`：两个脚本此前各写一份
15 个键的字典字面量，加这两个字段时不得不改两处。所以本守卫分三层 ——
**共享构造函数里有这两个键**、**两个脚本都调它**、
**传进去的值确实来自 AgentState**。少任何一层都挡不住一类退化。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
SCRIPTS = BACKEND / "scripts"
COMMON = SCRIPTS / "exp_common.py"

# 字段名 → 它在 AgentState 上的来源属性
SOURCES = (("guard", "guard_stats"), ("stop_reason", "stop_reason"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _case_row_return_keys() -> set[str]:
    """`exp_common.case_row` 返回的字典字面量的键名。"""
    for node in ast.walk(_tree(COMMON)):
        if isinstance(node, ast.FunctionDef) and node.name == "case_row":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    return {
                        k.value for k in sub.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    }
    raise AssertionError("exp_common.py 里找不到 case_row 的返回字典 —— 是扫描器坏了")


def _case_row_calls(filename: str) -> list[ast.Call]:
    return [
        n for n in ast.walk(_tree(SCRIPTS / filename))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "case_row"
    ]


def _names_in(node: ast.AST) -> set[str]:
    """节点子树里出现过的标识符与属性名。"""
    names: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
    return names


def _assigned_from_state(filename: str, var: str, attr: str) -> bool:
    """脚本里有没有把 `st.<attr>` 赋给 `var`。

    **元组解包也要认**：实际代码写的是
    `guard, stop_reason = st.guard_stats(), st.stop_reason`，
    只认 `var = ...` 这种最简单形式会漏判，然后报一个假的"没接上"。
    """
    for node in ast.walk(_tree(SCRIPTS / filename)):
        if not isinstance(node, ast.Assign):
            continue
        targets: set[str] = set()
        for t in node.targets:
            if isinstance(t, ast.Name):
                targets.add(t.id)
            elif isinstance(t, ast.Tuple | ast.List):
                targets |= {e.id for e in t.elts if isinstance(e, ast.Name)}
        if var in targets and attr in _names_in(node.value):
            return True
    return False


def test_共用的行构造函数带这两个键():
    keys = _case_row_return_keys()
    assert keys, "case_row 的返回字典是空的"
    for key, _attr in SOURCES:
        assert key in keys, (
            f"exp_common.case_row 的产物行里没有 {key} —— 熔断与终止原因"
            f"又不会被记录，「熔断在真实问题分布上从没被观测过」会继续成立"
        )


@pytest.mark.parametrize("filename", ["exp_c.py", "exp_e.py"])
def test_两个脚本都调用共用的行构造函数(filename: str):
    """各写一份字典字面量也能过「键存在」，所以调用点也要锁。"""
    assert _case_row_calls(filename), (
        f"{filename} 没有调用 exp_common.case_row —— 行的形状又分叉了，"
        f"下次加字段就得记得改两处（这次就是这么被绊到的）"
    )


@pytest.mark.parametrize("filename", ["exp_c.py", "exp_e.py"])
@pytest.mark.parametrize(("key", "attr"), SOURCES)
def test_字段值确实来自AgentState(filename: str, key: str, attr: str):
    """两层：传的是那个局部变量，且那个局部变量确实取自 AgentState。

    只看一层都会被绕过 —— `guard={}` 能过第一层但过不了第二层；
    而只查赋值不看传参，则改了调用点也发现不了。
    """
    calls = _case_row_calls(filename)
    assert calls, f"{filename} 里找不到 case_row 调用"

    values = [
        _names_in(kw.value) for call in calls for kw in call.keywords
        if kw.arg == key
    ]
    assert values, f"{filename} 调 case_row 时没有传 {key}="
    assert any(names == {key} for names in values), (
        f"{filename} 的 {key}= 不是一个取自 AgentState 的同名局部变量，"
        f"实际传的是 {values}"
    )
    assert _assigned_from_state(filename, key, attr), (
        f"{filename} 里没有 `{key} = ... st.{attr} ...` 这样的赋值 —— "
        f"字段存在但永远是空值，等于没记"
    )


def test_AgentState确实提供这两个状态():
    """前置断言：上游没这两个字段的话，上面几条测的就是空气。"""
    src = (BACKEND / "app" / "agent" / "react.py").read_text(encoding="utf-8")
    assert "def guard_stats(self)" in src
    assert "stop_reason: str =" in src
