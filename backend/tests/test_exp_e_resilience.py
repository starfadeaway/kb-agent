"""守卫：实验脚本的单次调用不许无限期挂起，产物不许只在最后落一次盘。

背景（2026-10-13 实测事故，不是假想）：
    实验 E 跑了 65 分钟没出结果。诊断结论是**挂死**而不是慢：
      - CPU 增量 0s、读入增量 0KB、内存不变，连续 90s 完全静止
      - py-spy dump：主线程停在 select，13 个 asyncio_* 线程全部 idle
      - 磁盘上没有任何产物

    三个缺陷叠在一起：
      1. `chat_with` 每次调用新建 `AsyncOpenAI`，而 `await client.close()`
         写在 `return` 之前、不在 finally —— 任何异常（当时是 MiniMax
         额度耗尽返回 429）都泄漏一个 httpx 连接池，累积到 socket 耗尽后
         新连接建不出去，协程永久挂在 await 上。
      2. `chat_with(timeout=120.0)` 声明了超时却从不接给 SDK，
         等于单点故障可以无限期挂住实验。
      3. `save()` 只在所有档位跑完后调用一次 —— 上面那次挂死把已经跑完的
         E1（DeepSeek，有效数据）一起带走了，65 分钟 API 花费零产物。

单测全绿说明不了这里安全，所以这组测试直接读源码 AST 验结构，
并且已对「修复前的代码快照」实跑确认每条都能真抓到缺陷。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXP_E = Path(__file__).resolve().parents[1] / "scripts" / "exp_e.py"


def _tree() -> ast.Module:
    return ast.parse(EXP_E.read_text(encoding="utf-8"))


def _func(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{EXP_E.name} 里找不到函数 {name}")


def _called_names(fn) -> set[str]:
    """函数体内所有被调用的名字（含 `a.b()` 的 `b`）。"""
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def test_chat_with_never_constructs_client():
    """chat_with 里不得再新建 AsyncOpenAI —— 那是连接池泄漏的源头。

    修复前它每次调用都 `AsyncOpenAI(...)`，异常路径下 close() 到不了。
    """
    fn = _func(_tree(), "chat_with")
    assert "AsyncOpenAI" not in _called_names(fn), (
        "chat_with 内新建 AsyncOpenAI：异常时 close() 不会执行，"
        "连接池泄漏会累积到协程永久挂起（见本文件顶部事故记录）"
    )


def test_chat_with_enforces_timeout():
    """声明了 timeout 就必须真的接上去，否则单点故障可无限期挂起。"""
    fn = _func(_tree(), "chat_with")
    assert "wait_for" in _called_names(fn), (
        "chat_with 没有把 timeout 接给调用："
        "声明了 timeout 参数却不用，等于没有超时保护"
    )
    # 两个分支都要覆盖：E1 与 E2 各有一个调用点
    waits = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "wait_for"]
    assert len(waits) >= 2, f"E1/E2 两个分支都应受 timeout 约束，实际只找到 {len(waits)} 处"


def test_client_lifecycle_closes_in_finally():
    """共享 client 的 close() 必须在 finally 里，异常路径也要关。"""
    fn = _func(_tree(), "_client_for")
    trys = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    assert trys, "_client_for 缺少 try/finally，异常时 client 不会被关闭"
    finally_closes = []
    for t in trys:
        for stmt in t.finalbody:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                        and sub.func.attr == "close":
                    finally_closes.append(sub)
    assert finally_closes, "close() 不在 finally 里 —— 异常路径会泄漏 client"


def test_partial_results_are_persisted_per_label():
    """每跑完一档就要落一次盘，且中间产物不能占用正式文件名。"""
    main = _func(_tree(), "main")
    loops = [n for n in ast.walk(main) if isinstance(n, ast.For)]
    assert loops, "main 里找不到逐档循环"
    saves_in_loop = [
        n for loop in loops for n in ast.walk(loop)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "save"
    ]
    assert saves_in_loop, (
        "逐档循环里没有 save()：save 只在全部档位跑完后调用，"
        "任一档挂死就会带走前面所有已完成的有效结果"
    )

    src = EXP_E.read_text(encoding="utf-8")
    assert "partial" in src, "中间产物未与正式文件区分，半截结果会被当成完整实验读入"


def test_exp_e_module_still_imports(monkeypatch):
    """改完不能把脚本改坏 —— 至少要能被 import（语法/名字错误会在这里暴露）。

    exp_e.py 内部是 `from exp_common import ...`，而 exp_common 与它同在
    scripts/ 下，不在 sys.path 里 import 会直接 ModuleNotFoundError ——
    所以这里要把 scripts/ 临时加进 sys.path，否则测的是环境而不是代码。
    """
    import importlib.util
    monkeypatch.syspath_prepend(str(EXP_E.parent))
    spec = importlib.util.spec_from_file_location("exp_e_under_test", EXP_E)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "chat_with") and hasattr(mod, "_client_for")


@pytest.mark.parametrize("fn_name", ["chat_with", "_client_for", "run_one", "main"])
def test_expected_entrypoints_exist(fn_name):
    """重构不该把这些入口悄悄弄没了。"""
    assert _func(_tree(), fn_name) is not None