"""守卫：回填在「语料为空」时必须拒绝执行，而不是把整库删光。

这个 bug 的形状特别危险：

    backfill_corpus 的剪枝逻辑是「以本轮语料为准的全量替换」——
    keep_chunk_ids / keep_fact_keys / keep_page_ids 三个集合是"应该存在的全集"，
    库里凡是不在这三个集合里的行都会被 DELETE。

    所以只要 `cached_page_ids()` 返回空（cwd 跑错、data/raw 被清、game 拼错），
    循环体一次都不跑，三个集合全空 → 这个 game 的 Page / ChunkMeta / Fact
    **被全部删除**，而函数照旧返回 `ok=True, pages=0`，不报错、不告警。

    触发它不需要什么罕见条件，脚本从别的目录跑一次就够了。
    数据不可逆，而且从返回值上完全看不出出事了。

所以这里同时守住两件事：
  1. 纯函数 `empty_corpus_refusal` 的判断逻辑；
  2. `backfill_corpus` 真的调用了它（AST 层面确认，不依赖数据库）。
"""
from __future__ import annotations

import ast
from pathlib import Path

from app.store.persist import empty_corpus_refusal

PERSIST = Path(__file__).resolve().parents[1] / "app" / "store" / "persist.py"


# --------------------------------------------------------------------------- #
# 1. 纯函数层面
# --------------------------------------------------------------------------- #

def test_refuses_when_corpus_empty_but_library_has_data():
    """语料为空 + 库里有数据 → 必须拒绝。"""
    reason = empty_corpus_refusal([], 517, "sr")
    assert reason is not None, "库里已有数据、语料为空时必须拒绝执行，否则会把整库删光"
    assert "517" in reason
    assert "sr" in reason


def test_allows_when_corpus_present():
    """语料非空 → 不该有任何拦截。"""
    assert empty_corpus_refusal([1, 2, 3], 517, "sr") is None


def test_allows_when_both_empty():
    """语料和库都为空 → 正常返回，不是错误。"""
    assert empty_corpus_refusal([], 0, "sr") is None


def test_single_page_corpus_is_not_treated_as_empty():
    """只有 1 页也算"非空" —— 否则小库会被误判成语料丢失。"""
    assert empty_corpus_refusal([7], 0, "zzz") is None


# --------------------------------------------------------------------------- #
# 2. 调用点层面（不连数据库也能验证）
# --------------------------------------------------------------------------- #

def test_backfill_corpus_calls_the_guard_before_touching_pages():
    """`backfill_corpus` 必须在进入写/删循环之前调用该守卫。

    只测纯函数是不够的：完全可能出现"守卫写对了，但主流程没调用它"的情况，
    而那种情况下 bug 依然存在。单测全绿，数据照样被删光。
    """
    tree = ast.parse(PERSIST.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "backfill_corpus"
    )
    called = {
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "empty_corpus_refusal" in called, (
        "backfill_corpus 没有调用 empty_corpus_refusal —— "
        "语料为空时会把 keep_* 三个集合清空，剪枝随即删光整个 game 的数据"
    )


def test_guard_is_consulted_before_the_delete_statements():
    """守卫调用必须早于任何 DELETE，否则拦下来时数据已经写了。"""
    src = PERSIST.read_text(encoding="utf-8")
    guard_at = src.index("empty_corpus_refusal(")
    delete_at = src.index("delete(ChunkMeta)")
    assert guard_at < delete_at, (
        "守卫调用出现在 DELETE 之后 —— "
        "那时 keep 集合已经算完、删除逻辑已经在同一次 session 里排队了"
    )


def test_cached_page_ids_result_is_reused_not_recalled():
    """page_ids 只应取一次。剪枝和写入必须基于同一份 id 列表。"""
    tree = ast.parse(PERSIST.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "backfill_corpus"
    )
    n = sum(
        1 for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "cached_page_ids"
    )
    assert n == 1, f"cached_page_ids 被调用 {n} 次；两次调用之间目录可能变化，结果会不一致"