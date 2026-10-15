"""守卫：`search_kb` 必须真的听 `.env` 的话，而不只是 API 路径听。

缺陷
----
`tools.py` 的 `search_kb` 原先写的是

    top_k = min(int(args.get("top_k") or 5), 10)
    cfg = RetrievalConfig(game=game, final_top_k=top_k)

`rrf_k` / `bm25_top` / `vector_top` / `rerank_candidates` 全走 dataclass 默认值，
**`RETRIEVE_CANDIDATE` 与 `RRF_K` 在 Agent 这条路径上是死的** ——
改 `.env` 完全不生效，检索照常跑、不报任何错。

为什么已有的检查没抓到
----------------------
`scripts/check_config_wiring.py` 验的是 `/api/retrieve`，那条路径早就是
`from_settings()` 了。**两条路径各修各的，验收脚本只覆盖了一条。**

而且这里还叠了一层假绿：`.env` 的默认值恰好等于 dataclass 默认值
（TOP_K=5 / CANDIDATE=20 / RRF_K=60），所以「接线断了」在默认配置下
根本看不出来。本组测试因此**一律用非默认值**构造 Settings ——
用默认值写这类守卫等于没写。
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
from app.agent.tools import ToolContext, call_tool
from app.core.config import get_settings
from app.retrieval.hybrid import RetrievalConfig

# 结构守卫要读 tools.py 源码，直接按路径取，不经过被 @tool 包装过的符号
TOOLS_SRC = Path(__file__).resolve().parents[1] / "app" / "agent" / "tools.py"



async def _search(args: dict, ctx) -> str:
    """search_kb 经 @tool 装饰器包装过，没有 .func 属性；生产入口是 call_tool。"""
    out, _ok = await call_tool("search_kb", args, ctx, timeout=5)
    return out


def _settings(**over):
    return dataclasses.replace(get_settings(), **over)


class _RecordingRetriever:
    """记录传给 retrieve 的 cfg，好断言「.env 的值真的走到了检索配置」。

    这个 4 行类此前在本文件里被抄了 5 遍 —— 每抄一遍就多一处
    「改了字段只改一半」的机会，而这类失败表现为断言静默失效而非报错。
    """

    def __init__(self) -> None:
        self.seen: dict = {}

    async def retrieve(self, q, cfg):
        self.seen["cfg"] = cfg
        return []


# ---------- 关键：非默认值下的接线 ----------

@pytest.mark.asyncio
async def test_candidate_pool_follows_env(monkeypatch):
    """RETRIEVE_CANDIDATE 必须进到 Agent 的检索配置里。

    用非默认值 37（≠20），否则接线断了也测不出来。
    """
    ret = _RecordingRetriever()
    ctx = ToolContext(settings=_settings(retrieve_candidate=37), retriever=ret, store=None)  # type: ignore[arg-type]

    await _search({"query": "太空封印站"}, ctx)

    cfg = ret.seen["cfg"]
    assert cfg.bm25_top == 37, f"bm25_top 没跟 .env 走：{cfg.bm25_top}"
    assert cfg.vector_top == 37, f"vector_top 没跟 .env 走：{cfg.vector_top}"
    assert cfg.rerank_candidates == 37, f"rerank_candidates 没跟 .env 走：{cfg.rerank_candidates}"


@pytest.mark.asyncio
async def test_rrf_k_follows_env(monkeypatch):
    ret = _RecordingRetriever()
    ctx = ToolContext(settings=_settings(rrf_k=7), retriever=ret, store=None)  # type: ignore[arg-type]
    await _search({"query": "太空封印站"}, ctx)
    assert ret.seen["cfg"].rrf_k == 7, f"rrf_k 没跟 .env 走：{ret.seen['cfg'].rrf_k}"


@pytest.mark.asyncio
async def test_default_top_k_follows_env():
    """不给 top_k 时要用 .env 的值，不是写死的 5。"""
    ret = _RecordingRetriever()
    ctx = ToolContext(settings=_settings(retrieve_top_k=4), retriever=ret, store=None)  # type: ignore[arg-type]
    await _search({"query": "太空封印站"}, ctx)
    assert ret.seen["cfg"].final_top_k == 4, (
        f"默认 top_k 没跟 .env 走：{ret.seen['cfg'].final_top_k}（写死的 5）"
    )


# ---------- 回归保护：显式参数仍然优先 ----------

@pytest.mark.asyncio
async def test_explicit_top_k_still_overrides_env():
    """模型显式传 top_k 时必须压过 .env —— 否则调参调不动。"""
    ret = _RecordingRetriever()
    ctx = ToolContext(settings=_settings(retrieve_top_k=4), retriever=ret, store=None)  # type: ignore[arg-type]
    await _search({"query": "x", "top_k": 2}, ctx)
    assert ret.seen["cfg"].final_top_k == 2


@pytest.mark.asyncio
async def test_top_k_still_capped_at_10():
    """回归保护：上限保护不能被 from_settings 顺手丢掉。"""
    ret = _RecordingRetriever()
    ctx = ToolContext(settings=_settings(retrieve_top_k=4), retriever=ret, store=None)  # type: ignore[arg-type]
    await _search({"query": "x", "top_k": 99}, ctx)
    assert ret.seen["cfg"].final_top_k == 10


@pytest.mark.asyncio
async def test_unknown_game_still_rejected():
    """回归保护：越界校验不能因为改配置而失效。"""

    class _NeverCalled:
        async def retrieve(self, q, cfg):
            raise AssertionError("越界游戏不该走到检索")

    ctx = ToolContext(settings=_settings(), retriever=_NeverCalled(), store=None)  # type: ignore[arg-type]
    out = await _search({"query": "x", "game": "wmy"}, ctx)
    assert "[错误]" in out


# ---------- 结构守卫 ----------

def test_search_kb_uses_from_settings():
    """结构层：search_kb 内部必须出现 from_settings 调用。"""
    tree = ast.parse(TOOLS_SRC.read_text(encoding="utf-8"))
    calls = {
        n.func.attr for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "from_settings" in calls, (
        "tools.py 里没有 RetrievalConfig.from_settings 调用 —— "
        "Agent 路径仍在用 dataclass 默认值，.env 的候选池与 rrf_k 是死的"
    )


def test_no_hardcoded_candidate_pool():
    """不得再出现 bm25_top/vector_top/rerank_candidates 的写死常量。"""
    src = TOOLS_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "RetrievalConfig"):
            continue
        kwargs = {k.arg for k in node.keywords}
        hardcoded = {"bm25_top", "vector_top", "rerank_candidates", "rrf_k"} & kwargs
        assert not hardcoded, (
            f"RetrievalConfig(...) 里又写死了 {hardcoded}："
            "这些应当由 .env 派生，写死等于 .env 失效"
        )


def test_from_settings_semantics_unchanged():
    """回归保护：from_settings 的「显式 over 优先」语义不能被改坏。"""
    cfg = RetrievalConfig.from_settings(settings=_settings(retrieve_top_k=4), final_top_k=2)
    assert cfg.final_top_k == 2
    assert cfg.bm25_top == get_settings().retrieve_candidate