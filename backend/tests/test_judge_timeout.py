"""守卫：Judge 的每一次调用都必须有超时上限。

缺陷
----
`Judge.score()` 与 `Judge.pairwise()` 原先直接 `await self.llm.judge(...)`，
超时完全依赖 OpenAI SDK 的默认值 —— **600 秒**。

为什么这不是「默认值够大就行」：评测跑在无人值守的后台里，
一次 Judge 挂住就让整轮实验停在那里；而 600 秒远长于任何一次正常打分
（实测 28.5s），等下去既拿不到结果，也看不出是哪一档出的问题。
已实测踩到两次：实验 E 的 `sr-008` 拿到 `judge_error:APITimeoutError`，
并发一致性探针第二轮出现连续 60 秒零读入。

超时后必须走既有 except 分支打上 `judge_error:*` 标记 ——
这样它会被 `exp_common.is_kept` 剔除，而不是变成一个混进能力均值的 0 分。

已对修复前的快照实跑确认能抓到缺陷。
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap

import app.eval.judge as judge_mod
import pytest
from app.eval.judge import Judge

# 注意：这里**不** `from app.eval.judge import DEFAULT_JUDGE_TIMEOUT`。
# 双向验证时要把测试跑在修复前的快照上，而那个常量是修复后才有的；
# 顶层 import 它会让旧代码在 collection 阶段就 ImportError，
# 结果一条行为断言都跑不到 —— 那只证明「新符号不存在」，等于没验证。
# 只依赖修复前就存在的 `Judge`，常量走模块属性访问。
SRC = inspect.getsourcefile(Judge)
assert SRC


class _SlowLLM:
    """永远不返回的 Judge，用于验证超时真的生效。"""

    async def judge(self, prompt, model=None):
        await asyncio.sleep(3600)


class _FastLLM:
    def __init__(self, out='{"correctness":5,"faithfulness":5,"completeness":5,"reason":"ok"}'):
        self.out = out

    async def judge(self, prompt, model=None):
        return self.out


def _tree() -> ast.Module:
    return ast.parse(textwrap.dedent(inspect.getsource(Judge)))


# ---------- 行为层：超时真的会触发 ----------

@pytest.mark.asyncio
async def test_score_times_out_and_is_marked():
    """挂死的 Judge 必须变成 judge_error，而不是无限期等待。"""
    j = Judge(_SlowLLM(), timeout=0.05)  # type: ignore[arg-type]
    res = await j.score(question="q", gold="g", answer="答点什么", contexts="c")
    # JudgeScore 上没有 judge_score —— 那是聚合时才由三项加权算出来的
    assert res.correctness == 0
    assert "judge_error" in res.reason, (
        f"超时未被标记为 judge_error，reason={res.reason!r} —— "
        "它会被混进能力均值当成「模型答崩了」"
    )


@pytest.mark.asyncio
async def test_pairwise_times_out_and_is_marked():
    j = Judge(_SlowLLM(), timeout=0.05)  # type: ignore[arg-type]
    verdict, reason = await j.pairwise(
        question="q", gold="g", answer_a="a", answer_b="b", contexts="c")
    assert verdict == "TIE"
    assert "judge_error" in reason
    assert "非真实平局" in reason


@pytest.mark.asyncio
async def test_fast_judge_unaffected():
    """回归保护：正常路径不能被超时机制误伤。"""
    j = Judge(_FastLLM())  # type: ignore[arg-type]
    res = await j.score(question="q", gold="g", answer="答点什么", contexts="c")
    assert res.correctness == 5.0
    assert "judge_error" not in res.reason


@pytest.mark.asyncio
async def test_empty_answer_short_circuits_without_calling_llm():
    """回归保护：空回答本就不该调 Judge。"""
    class _Boom:
        async def judge(self, prompt, model=None):
            raise AssertionError("不该被调用")

    j = Judge(_Boom(), timeout=0.05)  # type: ignore[arg-type]
    res = await j.score(question="q", gold="g", answer="   ", contexts="c")
    assert res.reason == "空回答"


# ---------- 结构层：两处调用都必须包上 ----------

def test_both_call_sites_use_wait_for():
    """score 与 pairwise 两处都要包，漏一处就还有一条无保护的路径。"""
    tree = _tree()
    waits = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "wait_for"
    ]
    assert len(waits) >= 2, f"只找到 {len(waits)} 处 wait_for，两处调用至少要有 2 处"


def test_no_bare_llm_judge_await_left():
    """不得再有裸 await（没有超时保护的）judge 调用。

    用 AST 找所有 await 表达式，确认它不是被 wait_for 包住的直接调用。
    """
    tree = _tree()
    bare = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
                continue
            f = node.value.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name != "judge":
                continue
            # 父节点不是 wait_for -> 裸 await
            bare.append(fn.name)
    assert not bare, f"函数 {set(bare)} 里仍有无超时保护的裸 await judge()"


def test_default_timeout_is_sane():
    """默认值必须远小于 SDK 的 600s，否则等于没设。"""
    dft = getattr(judge_mod, "DEFAULT_JUDGE_TIMEOUT", None)
    assert dft is not None, "缺少默认超时常量"
    assert 0 < dft <= 300, (
        f"默认超时 {dft}s 不合理："
        "要么太短会把正常打分误杀，要么太长失去意义"
    )


def test_timeout_is_injectable():
    """timeout 必须可注入，否则测试只能 sleep 真等。"""
    params = inspect.signature(Judge.__init__).parameters
    assert "timeout" in params, "Judge(timeout=...) 不存在，无法按场景调整上限"