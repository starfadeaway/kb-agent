"""守卫：Judge 拿不到结论时，不得把"无法判断"伪装成"两者打平"。

缺陷（代码审查 bug #7）：
    `pairwise()` 里写的是 `_parse_json(raw) or {}`，于是 `winner` 取默认值
    `"TIE"` —— Judge 输出解析失败（模型没按 JSON 输出、响应被截断、网络层
    返回了非 JSON）被静默算成"A 和 B 一样好"，而且**连一行日志都不打**。

    比"多了一次噪声"严重得多的是它会伪造结论：
    正反两次里只要有一次解析失败，r1 就是 TIE，后面 `if r1 == "TIE"`
    直接 return 平局 —— 另一次本来明确的胜负被整个盖掉。
    换句话说，**Judge 系统性故障恰好会制造出"两档没差别"的假象**，
    而"没差别"恰恰是评测最不该出现的错误结论。

    同一个文件的 `score()` 一直是对的（`if not data: return ... "judge 输出无法解析"`）。
    两处口径不一致，等于同一套 Judge 按两套标准计分。

所以这里不只验行为（解析失败要带 `parse_error` 标记），还验两条不变量：
    1. 解析失败的 reason 必须自证"非真实平局"，不能被下游读成结论；
    2. 解析失败必须留下日志，"静默"本身就是缺陷的一部分。

已对修复前的代码快照实跑确认每条都能抓到真缺陷。
"""
from __future__ import annotations

import ast
import inspect
import logging
import textwrap

import pytest
from app.eval.judge import Judge


class _StubLLM:
    """按顺序吐出预置输出；给 Exception 则抛。"""

    def __init__(self, outputs: list) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    async def judge(self, prompt, model=None):
        self.calls += 1
        out = self.outputs.pop(0) if self.outputs else '{"winner":"A","reason":"默认"}'
        if isinstance(out, Exception):
            raise out
        return out


async def _run(outputs: list):
    j = Judge(_StubLLM(outputs))  # type: ignore[arg-type]
    return await j.pairwise(
        question="奇点是什么？", gold="星核失控形成的时空异常。",
        answer_a="A 答案", answer_b="B 答案", contexts="[1] 证据",
    )


@pytest.mark.asyncio
async def test_parse_failure_marked_as_parse_error():
    """解析失败必须留下 judge_parse_error 标记。"""
    verdict, reason = await _run(["这不是 JSON，是模型跑偏了", '{"winner":"A","reason":"A 更好"}'])
    assert verdict == "TIE"
    assert "parse_error" in reason


@pytest.mark.asyncio
async def test_parse_failure_never_reads_as_a_real_tie():
    """关键不变量：reason 必须自证这不是真实平局。

    否则下游看到 "TIE" 就当成"两档没差别"—— 那正是故障想伪造的结论。
    """
    _, reason = await _run(["完全跑偏的输出", '{"winner":"A","reason":"A 更好"}'])
    assert "非真实平局" in reason
    assert "parse_error" in reason


@pytest.mark.asyncio
async def test_parse_failure_does_not_swallow_the_other_verdict():
    """一次解析失败不应吃掉另一次本可明确的胜负。

    修复前：r1=TIE 直接 return 平局，"A 更好"这个信号被静默丢弃。
    """
    verdict, reason = await _run(["跑偏", '{"winner":"A","reason":"A 明显更好"}'])
    # 结论仍是 TIE（无法比较），但必须标出故障且保留原始理由
    assert verdict == "TIE"
    assert "parse_error" in reason
    assert "A 明显更好" in reason, "第二次的判断理由被整个丢弃了，日志里也无从追溯"


@pytest.mark.asyncio
async def test_missing_winner_key_counts_as_parse_failure():
    """JSON 合法但缺 winner 字段，同样是没拿到结论，不等于平局。"""
    verdict, reason = await _run(['{"reason":"忘了写 winner"}', '{"winner":"A","reason":"ok"}'])
    assert verdict == "TIE"
    assert "parse_error" in reason


@pytest.mark.asyncio
async def test_parse_failure_is_logged(caplog):
    """解析失败必须留日志 —— 静默本身就是这个缺陷的一部分。"""
    with caplog.at_level(logging.WARNING):
        await _run(["跑偏的输出", '{"winner":"A","reason":"ok"}'])
    assert any(r.levelno == logging.WARNING for r in caplog.records), "解析失败没有告警"


@pytest.mark.asyncio
async def test_agreeing_verdicts_still_win():
    """回归保护：正常路径不能被改坏 —— 两次都指 A 仍应判 A 赢。"""
    verdict, reason = await _run(
        ['{"winner":"A","reason":"更完整"}', '{"winner":"A","reason":"更完整"}'])
    assert verdict == "A"
    assert "parse_error" not in reason


@pytest.mark.asyncio
async def test_position_sensitive_still_ties():
    """回归保护：位置不一致仍判平局，且照旧说明原因。"""
    verdict, reason = await _run(['{"winner":"A"}', '{"winner":"B"}'])
    assert verdict == "TIE"
    assert "位置敏感" in reason


@pytest.mark.asyncio
async def test_explicit_tie_is_still_a_real_tie():
    """回归保护：模型明确说平局时，不该被误标成故障。"""
    verdict, reason = await _run(['{"winner":"TIE","reason":"各有千秋"}',
                                  '{"winner":"TIE","reason":"各有千秋"}'])
    assert verdict == "TIE"
    assert "parse_error" not in reason
    assert "非真实平局" not in reason


@pytest.mark.asyncio
async def test_llm_exception_marked_as_error_not_tie():
    """调用异常也要标明"非真实平局"（与解析失败同一原则）。"""
    verdict, reason = await _run([RuntimeError("boom")])
    assert verdict == "TIE"
    assert "judge_error" in reason
    assert "非真实平局" in reason


def _pairwise_ast() -> ast.Module:
    """只取代码，不含 docstring。

    早先用 inspect.getsource 做字符串扫描，结果被自己的文档绊倒：
    docstring 里为了说明缺陷根源，恰好引用了 `_parse_json(raw) or {}` 这行原文，
    于是守卫在自己写的说明上失败。docstring 是 ast.Constant，不是代码节点 ——
    按 AST 判断结构才既准确又不受注释影响。
    """
    return ast.parse(textwrap.dedent(inspect.getsource(Judge.pairwise)))


def _is_parse_json_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_parse_json"
    )


def test_pairwise_source_has_no_default_winner_fallback():
    """结构守卫：不得再用 `or {}` 把解析失败抹成默认值。

    这行是缺陷的原始形态；只看行为测试不够 —— 它可能在别的重构里复活。
    """
    tree = _pairwise_ast()

    swallowed = False
    for parent in ast.walk(tree):
        if not isinstance(parent, ast.BoolOp) or not isinstance(parent.op, ast.Or):
            continue
        if any(_is_parse_json_call(child) for child in parent.values):
            swallowed = True
            break
    assert not swallowed, (
        "pairwise 又用 `_parse_json(...) or {}` 把解析失败抹成默认 winner，"
        "故障会重新被伪装成平局"
    )

    # 也不能对缺失的 winner 兜一个 "TIE" 默认值 —— 缺字段是没拿到结论，不是打平
    defaulted = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"):
            continue
        args = node.args
        if (len(args) >= 2 and isinstance(args[0], ast.Constant) and args[0].value == "winner"
                and isinstance(args[1], ast.Constant) and args[1].value == "TIE"):
            defaulted = True
            break
    assert not defaulted, "不能对缺失的 winner 兜 TIE 默认值 —— 缺字段不等于打平"


def test_score_and_pairwise_share_the_same_parse_policy():
    """同一文件两处口径必须一致：一个记录故障，另一个也不能静默。"""
    pair_src = inspect.getsource(Judge.pairwise)
    score_src = inspect.getsource(Judge.score)
    assert "无法解析" in score_src, "score() 应当对解析失败有显式标记"
    assert "parse_error" in pair_src, "pairwise() 与 score() 口径不一致"