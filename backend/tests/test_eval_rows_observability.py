"""守卫：实验产物的每一行必须带 `guard` 与 `stop_reason`。

`docs/failure_analysis.md` 第六节承认「熔断机制被测试验证过，但在真实问题
分布上从没被观测过」。根因不是没跑题 —— `exp_c` / `exp_e` 都走完整的
ReAct 循环，`AgentState` 上明明有 `guard_stats()` 和 `stop_reason`。

问题是**跑完就把状态丢了**：行字典里一个都没记。后果是 `data/eval/` 全部
产物里 `guard` 出现 0 次，谁也没法回答「这三库的题里熔断到底触发过几次」。

这条守卫锁「别再丢」。判据用 AST 而不是真跑一遍实验：真跑要打网络、花 token，
而这里要证明的只是「字段接上了」—— 值对不对由实验本身负责。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
SCRIPTS = BACKEND / "scripts"
REQUIRED_KEYS = ("guard", "stop_reason")


def _row_dict_keys(filename: str) -> set[str]:
    """脚本里 return 的字典字面量的键名（取并集）。"""
    tree = ast.parse((SCRIPTS / filename).read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for k in node.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    return keys


def _assigns_from_state(filename: str, attr: str) -> bool:
    """脚本里有没有把 `st.<attr>` / `st.<attr>()` 取出来赋值。"""
    tree = ast.parse((SCRIPTS / filename).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for sub in ast.walk(node.value):
            if (isinstance(sub, ast.Attribute) and sub.attr == attr
                    and isinstance(sub.value, ast.Name) and sub.value.id == "st"):
                return True
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == attr
                    and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "st"):
                return True
    return False


@pytest.mark.parametrize("filename", ["exp_c.py", "exp_e.py"])
def test_每行都带guard与stop_reason(filename: str):
    keys = _row_dict_keys(filename)
    assert keys, f"{filename} 里找不到 per-case 行字典 —— 是扫描器坏了"
    for key in REQUIRED_KEYS:
        assert key in keys, (
            f"{filename} 的产物行里没有 {key} —— 熔断与终止原因又不会被记录，"
            f"「熔断在真实问题分布上从没被观测过」会继续成立"
        )


@pytest.mark.parametrize("filename", ["exp_c.py", "exp_e.py"])
@pytest.mark.parametrize("attr", ["guard_stats", "stop_reason"])
def test_字段值必须取自AgentState(filename: str, attr: str):
    """写死成空值也能过「键存在」那条，所以来源也要锁。"""
    assert _assigns_from_state(filename, attr), (
        f"{filename} 里的 {attr} 不是从 AgentState 取的 —— "
        f"字段存在但永远是空值，等于没记"
    )


def test_AgentState确实提供这两个状态():
    """前置断言：上游没这两个字段的话，上面几条测的就是空气。"""
    src = (BACKEND / "app" / "agent" / "react.py").read_text(encoding="utf-8")
    assert "def guard_stats(self)" in src
    assert "stop_reason: str =" in src
