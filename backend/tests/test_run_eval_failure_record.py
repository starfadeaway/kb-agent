"""守卫：检索失败必须留下痕迹，不能让 case 从某档悄悄消失。

缺陷
----
`run_eval.run_one()` 原先写的是

    except Exception as exc:
        logger.error("[%s] %s 检索失败: %s", label, case.case_id, exc)
        continue          # ← 没有任何行落下来

两个后果：

1. **各档样本集不同。** 若 A0 的某条 case 检索失败，A0 的 rows 就少一条，
   而 A1/A2/A3 完整。均值不再是同一批 case 上的对比，配对检验也失真 ——
   而实验 A 的全部结论正是建立在「只改一个组件」的对比上。
2. **静默丢数据。** 失败只在日志里 error，聚合结果里看不出少了 case，
   看到「n=28」也不会知道本该是 29。

这与 `exp_common.is_kept` 那批修复是同一类问题，但更隐蔽：
那里失败会变成一条 0 分记录（混进均值），这里是**整条记录消失**。

修法：失败时补一条带 error 标记的全零行，并让聚合只统计成功行，
同时报出 n / n_ok / errors。

已对修复前的快照实跑确认能抓到缺陷。
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"


def _load():
    sys.path.insert(0, str(SCRIPT.parents[1]))
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("run_eval_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # 必须先注册进 sys.modules：run_eval 用 `from __future__ import annotations`，
    # dataclass 在处理注解时会去 sys.modules 查自身，缺了这一步会直接抛错。
    sys.modules["run_eval_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- 行为层 ----------

class _Boom:
    """永远失败的检索器。"""

    async def retrieve(self, question, cfg):
        raise RuntimeError("模拟检索故障")


@pytest.mark.asyncio
async def test_failed_case_is_still_recorded():
    """失败的 case 必须留下一行，而不是消失。"""
    mod = _load()
    # 只有一条有 gold_pages 的 case，且检索必失败
    mod.cases_for = lambda game: [_Case("t-001", gold_pages=["太空封印站"])]

    res = await mod.run_one(
        label="A0", game="sr", top_k=3, rerank_candidates=20,
        embed_model=None, variant="base", retriever=_Boom(),
    )
    assert len(res["rows"]) == 1, (
        f"检索失败的 case 从 rows 里消失了（rows={len(res['rows'])}）—— "
        "各档样本集会不同，均值不可比"
    )
    assert res["rows"][0]["error"], "失败行没有 error 标记，无法与真实低分区分"


@pytest.mark.asyncio
async def test_failed_case_excluded_from_mean_but_counted():
    """失败行不进均值，但必须体现在 n / n_ok / errors 上。"""
    mod = _load()
    mod.cases_for = lambda game: [_Case("t-001", gold_pages=["太空封印站"])]

    res = await mod.run_one(
        label="A0", game="sr", top_k=3, rerank_candidates=20,
        embed_model=None, variant="base", retriever=_Boom(),
    )
    agg = res["aggregate"]
    assert agg["n"] == 1, "n 应记全部 case 数"
    assert agg["n_ok"] == 0, "n_ok 应记成功数"
    assert agg["errors"] == 1, "errors 应显式报出，否则失败只存在于日志里"
    # 全失败时均值不能是 NaN 或凭空一个数
    assert agg["mrr"] in (0.0, 0), f"全失败时 mrr 应为 0，实际 {agg['mrr']!r}"


@pytest.mark.asyncio
async def test_success_case_unaffected():
    """回归保护：正常路径的指标不能被这次改动影响。"""
    mod = _load()
    mod.cases_for = lambda game: [_Case("t-001", gold_pages=["太空封印站"])]

    class _Good:
        async def retrieve(self, question, cfg):
            return [_Hit("c1", "太空封印站")]

    res = await mod.run_one(
        label="A0", game="sr", top_k=3, rerank_candidates=20,
        embed_model=None, variant="base", retriever=_Good(),
    )
    agg = res["aggregate"]
    assert agg["mrr"] == 1.0, f"单条命中排第一，mrr 应为 1.0，实际 {agg['mrr']}"
    assert agg["errors"] == 0
    assert agg["n_ok"] == 1


def test_failed_row_uses_production_metric_keys():
    """失败行的指标键名必须由 RetrievalMetrics 生成，不能手写。

    `recall@{k}` 的 k 是动态的，手写字段名就会和 k 脱钩 ——
    聚合时 `k.startswith("recall")` 抓不到，均值会缺一项。
    """
    src = SCRIPT.read_text(encoding="utf-8")
    assert "M.RetrievalMetrics(" in src, (
        "失败行应复用 RetrievalMetrics.as_dict() 生成指标键名，"
        "手写字典会让 recall@{k} 与实际 k 脱钩"
    )


def test_bare_continue_without_recording_is_gone():
    """结构守卫：**检索循环内**的 except 必须往 rows 里记一行。

    范围限定在 run_one 内 —— persist_results 里的 except「落库失败不影响结果」
    是合理设计，不该被这条守卫误伤。

    判据是「有没有 rows.append」，而不是「是不是只有 continue」：
    早先版本写的是 `logger.error(...); continue`，
    因为 logger 那行使它「不是纯 continue」，只看 continue 的守卫会漏掉它。
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    run_one = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_one"
    )
    found_bad = []
    for node in ast.walk(run_one):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            records = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id in {"rows", "ok_rows"}
                for n in ast.walk(handler)
            )
            if not records:
                found_bad.append(handler.lineno)
    assert not found_bad, (
        f"run_one 内这些 except 没有向 rows 记录任何内容（行 {found_bad}）："
        "检索失败的 case 会从该档样本集里消失，各档均值不再可比"
    )


# ---------- 替身 ----------

class _Case:
    def __init__(self, case_id, gold_pages):
        self.case_id = case_id
        self.gold_pages = gold_pages
        self.question = "问题"


class _Hit:
    def __init__(self, chunk_id, page_title):
        self.chunk_id = chunk_id
        self.chunk = type("C", (), {"page_title": page_title})()