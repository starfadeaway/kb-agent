"""C 类校验在 **ReAct 主流程** 上的接线守卫。

`test_citation_audit.py` 测的是纯函数。这里测的是另一件事：
**这层校验有没有真的挂在生产路径上**。

纯函数测得再好，没接进 `react.py` 就等于零 ——
而这正是 C 类校验此前的真实状态：`docs/failure_analysis.md` 第四节
把它列为唯一结构性缺口，代码里却完全没有对应实现。
一个只测纯函数、不测接线的测试套件，会在「实现被删掉」时依然全绿。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REACT = Path(__file__).resolve().parents[1] / "app" / "agent" / "react.py"
MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _callee_name(node: ast.Call) -> str | None:
    """取被调用者的名字。

    `Name` 与 `Attribute` 两种形态都要认：源码里全是 `self._audit_citations(...)`，
    只认 Name 会让**每一个**调用点都数不到 —— 而这类"扫描器自己坏了"的失败
    表现为「断言 0 个调用点」，很容易被误读成"实现没接上"，把排查引向错误方向。
    """
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _calls(fn: ast.FunctionDef, name: str) -> list[ast.Call]:
    """函数体内对 `name` 的调用点。"""
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _callee_name(n) == name]


def _methods(tree: ast.Module, cls_name: str) -> dict[str, ast.FunctionDef]:
    """类里的方法。**async 的也要收** —— `_run_once` / `_run_stream` /
    `_finalize` 全是 AsyncFunctionDef，只认 FunctionDef 会让三条主线
    的方法一个都找不到，然后以 KeyError 报出来（而不是安静地跳过）。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return {
                n.name: n
                for n in node.body
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            }
    raise AssertionError(f"找不到类 {cls_name}")


# --------------------------------------------------------------------------- #
# 1. 方法必须存在
# --------------------------------------------------------------------------- #
def test_校验方法存在于ReActAgent上():
    tree = ast.parse(_src(REACT))
    methods = _methods(tree, "ReActAgent")
    assert "_audit_citations" in methods, "ReActAgent 上没有 _audit_citations —— 校验没接进主流程"


# --------------------------------------------------------------------------- #
# 2. 三个出口都要接上
# --------------------------------------------------------------------------- #
def test_非流式正常收尾会跑校验():
    """模型给出最终答案的那条出口（`_run_once` 返回前）。"""
    tree = ast.parse(_src(REACT))
    run_once = _methods(tree, "ReActAgent")["_run_once"]
    assert _calls(run_once, "_audit_citations"), "_run_once 的返回前没有校验"


def test_安全阀触发的强制收尾会跑校验():
    """熔断 / 超时后的 `_finalize` —— 信息不足时模型最容易补一个没查过的页面名。"""
    tree = ast.parse(_src(REACT))
    finalize = _methods(tree, "ReActAgent")["_finalize"]
    assert _calls(finalize, "_audit_citations"), "安全阀收尾路径没有校验"


def test_流式路径会跑校验():
    tree = ast.parse(_src(REACT))
    run_stream = _methods(tree, "ReActAgent")["_run_stream"]
    assert _calls(run_stream, "_audit_citations"), "流式路径没有校验"


def test_越界拒答分支不跑校验():
    """越界分支没调过工具，retrieved 是空的，硬跑只会得到全假阳性。

    这条锁的是「不该跑的地方别跑」—— 它和上面三条是一对。
    """
    tree = ast.parse(_src(REACT))
    refuse = _methods(tree, "ReActAgent")["_refuse_uncovered"]
    assert not _calls(refuse, "_audit_citations"), "越界拒答分支不该跑引用校验"


# --------------------------------------------------------------------------- #
# 3. 顺序约束：必须先收证据再校验
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["_run_once", "_finalize", "_run_stream"])
def test_校验必须排在收引用之后(method: str):
    """顺序反了 = retrieved 是空的 = 每一个引用都被判编造并剥掉。

    一次正确的回答被削成空话，而且**不报任何错** ——
    这是比不做校验更糟的失败模式，所以必须锁死。
    """
    tree = ast.parse(_src(REACT))
    fn = _methods(tree, "ReActAgent")[method]
    refs_calls = _calls(fn, "_collect_refs")
    audit_calls = _calls(fn, "_audit_citations")
    assert refs_calls and audit_calls
    assert refs_calls[0].lineno < audit_calls[0].lineno, (
        f"{method} 里 _collect_refs(第 {refs_calls[0].lineno} 行)"
        f" 必须早于 _audit_citations(第 {audit_calls[0].lineno} 行)"
    )


def test_校验内部也先收再判():
    """方法内部同样依赖 collected 状态，方法体里也要有对应说明或调用。"""
    tree = ast.parse(_src(REACT))
    fn = _methods(tree, "ReActAgent")["_audit_citations"]
    assert "retrieved_pages" in ast.unparse(fn), "校验必须读 state.retrieved_pages"


# --------------------------------------------------------------------------- #
# 4. 流式路径不改写 —— 显示与落库不许是两份文本
# --------------------------------------------------------------------------- #
def test_流式路径显式传rewrite为False():
    """流式的 token 已经推给用户了，再改 state.answer 就是两份文本。

    反过来（非流式改写）也是错的：那时用户还没看到整段答案，
    本来可以安全地剥掉。这是刻意的路径差异，不是遗漏。
    """
    tree = ast.parse(_src(REACT))
    fn = _methods(tree, "ReActAgent")["_run_stream"]
    audit_calls = _calls(fn, "_audit_citations")
    assert audit_calls
    kw = {k.arg: k.value for k in audit_calls[0].keywords}
    assert "rewrite" in kw, "流式路径没显式声明 rewrite —— 将来有人加调用点会误改写"
    assert isinstance(kw["rewrite"], ast.Constant) and kw["rewrite"].value is False


def test_非流式路径不改rewrite参数():
    """非流式默认改写；显式写出来是为了让「两条路径的差异」在代码里可见。"""
    tree = ast.parse(_src(REACT))
    for method in ("_run_once", "_finalize"):
        fn = _methods(tree, "ReActAgent")[method]
        for call in _calls(fn, "_audit_citations"):
            assert not {k.arg for k in call.keywords} & {"rewrite"}, (
                f"{method} 的非流式校验不该覆盖 rewrite 默认值"
            )


# --------------------------------------------------------------------------- #
# 5. 状态字段与 API 出口
# --------------------------------------------------------------------------- #
def test_AgentState有校验结果与证据两个字段():
    tree = ast.parse(_src(REACT))
    state = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "AgentState"
    )
    fields = {
        t.target.id
        for t in state.body
        if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
    }
    assert "citation_audit" in fields, "AgentState 上没有 citation_audit"
    assert "retrieved_pages" in fields, "AgentState 上没有 retrieved_pages（校验的判据来源）"


def test_默认是None而不是空的审计对象():
    """None 与 {"ok": true} 必须可区分，否则「没校验」会被当成「校验通过」。"""
    src = _src(REACT)
    assert "citation_audit: CitationAudit | None = None" in src


@pytest.mark.parametrize("path", [MAIN])
def test_chat响应带上校验结果(path: Path):
    src = _src(path)
    assert "citation_audit" in src, "/api/chat 响应没有 citation_audit —— 校验结果没人看得见"


def test_done事件带上校验结果():
    tree = ast.parse(_src(REACT))
    fn = _methods(tree, "ReActAgent")["_run_stream"]
    assert "citation_audit" in ast.unparse(fn), "SSE done 事件没有 citation_audit"
