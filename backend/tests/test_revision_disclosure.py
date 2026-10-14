"""修订时间披露：工具输出 → 引用 → 前端气泡，三段都要真的接上。

## 为什么这个文件的断言长这样

**断言「配置的值」不等于断言「行为真的变了」** —— 本项目在这上面栽过
两次：`TOOL_TRANSPORT`（定义了没人读）、`judge_api_key`（名单里有、字段不存在）。

所以这里每条断言都指向**输出里真实出现的字符**，而不是某个配置值：

- 若 `_fmt_revision` 的入参 key 写错（比如读 `last_revision` 而库里是
  `last_revision_ts`），日期会是空串，1/2/3 号断言立刻红；
- 若 `_collect_refs` 的分隔符解析写错，4 号断言红；
- 若哪天有人把「页面修订」改成「数据截至」，5 号断言红 ——
  那是最隐蔽的一种错：功能正常、测试全绿、但语义在说谎。

## 双向验证不适用

`trim_history` 那次已经说明原因：新增符号在改造前的 HEAD 上不存在，
没有"修复前快照"可注入。这里的替代手段是**手工变异验证**，
步骤记在 commit message 里：把三处披露分别改回不输出日期，
本文件必须由绿转红。

## 一条容易被忽略的语义边界

`last_revision_ts` 是**页面最后一次编辑**的时间，不是数据生效日期。
一页 2020 年建的 wiki 昨天改了个错别字，它也是昨天。

所以文案只能是「页面修订」。写成「数据截至 / 数据更新于 / 截至今天」，
是用一个更隐蔽的误导替换掉原本的沉默 —— 比不说更糟，因为用户会信。
"""
from __future__ import annotations

import asyncio
import types

import pytest
from app.agent.react import AgentState, ReActAgent
from app.agent.tools import ToolContext, _fmt_revision, _revision_span, get_page, search_kb
from app.core.config import get_settings

WIKI = "https://wiki.biligame.com/sr"


def _chunk(title: str = "三月七", ts: str | None = "2026-07-28T04:48:04Z"):
    meta = {"section": "正文"}
    if ts is not None:
        meta["last_revision_ts"] = ts
    return types.SimpleNamespace(
        chunk_id="c1", page_title=title, text="正文内容", source_url=f"{WIKI}/{title}",
        metadata=meta,
    )


def _hit(title: str, ts: str | None, score: float = 0.9):
    from app.retrieval.hybrid import RetrievalHit

    return RetrievalHit(chunk=_chunk(title, ts), score=score, rank_source="rrf")


class _FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    async def retrieve(self, query, cfg):
        return self._hits


def _run_search(hits, **overrides) -> str:
    import dataclasses

    s = dataclasses.replace(get_settings(), **overrides) if overrides else get_settings()
    ctx = ToolContext(settings=s, retriever=_FakeRetriever(hits), store=None, game="sr")
    return asyncio.run(search_kb({"query": "三月七 基础生命值", "top_k": len(hits)}, ctx))


# --------------------------------------------------------------------------- #
# 1. 输出里真的有日期
# --------------------------------------------------------------------------- #
def test_search_kb_discloses_revision_per_hit():
    out = _run_search([_hit("三月七", "2026-07-28T04:48:04Z")])
    assert "页面修订 2026-07-28" in out, (
        "每条命中没带页面修订日期。若 metadata 的 key 拼错（如读成 last_revision），"
        "这里会是空串 —— 而检索照常跑、不报任何错。"
    )


def test_search_kb_discloses_span_in_header():
    out = _run_search([
        _hit("三月七", "2026-07-28T04:48:04Z", 0.9),
        _hit("芳华待灼", "2020-11-06T06:57:53Z", 0.8),
    ])
    assert "页面修订 2020-11-06~2026-07-28" in out, (
        "header 没给批次的修订区间。模型需要先知道"
        "「这批数据整体有多旧」，才能决定要不要在回答里声明时效。"
    )


def test_search_kb_puts_revision_on_the_source_line():
    """出处行也必须带 —— 前端气泡的修订时间全靠这一行。"""
    out = _run_search([_hit("三月七", "2026-07-28T04:48:04Z")])
    assert f"出处：{WIKI}/三月七 | 修订 2026-07-28" in out, (
        "出处行没有修订。没有它，_collect_refs 抽出来的引用就只有链接，"
        "前端右下角无从展示 —— 整条链路在这里断掉。"
    )


def test_search_kb_survives_chunks_without_revision():
    """老数据没有这个字段时必须优雅降级，不能编一个日期出来。"""
    out = _run_search([_hit("三月七", None)])
    assert "页面修订" not in out
    assert f"出处：{WIKI}/三月七" in out


# --------------------------------------------------------------------------- #
# 2. 格式化与区间计算
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expect", [
    ("2026-07-28T04:48:04Z", "2026-07-28"),
    ("2026-07-28", "2026-07-28"),
    ("", ""),
    (None, ""),
    ("短", ""),          # 长度不足 10，当作解析失败而不是硬切
])
def test_fmt_revision(raw, expect):
    assert _fmt_revision(raw) == expect


def test_revision_span_single_value_has_no_range():
    span = _revision_span([_hit("a", "2026-07-28T00:00:00Z")])
    assert span == " 页面修订 2026-07-28", span


def test_revision_span_sorted_ascending():
    hits = [
        _hit("a", "2026-07-28T00:00:00Z"),
        _hit("b", "2020-11-06T00:00:00Z"),
        _hit("c", "2024-05-19T00:00:00Z"),
    ]
    assert _revision_span(hits) == " 页面修订 2020-11-06~2026-07-28"


def test_revision_span_ignores_chunks_without_revision():
    """全都没有时返回空串；有的话只统计有的 —— 别让缺失把真实值稀释掉。"""
    assert _revision_span([_hit("a", None)]) == ""
    assert _revision_span([_hit("a", None), _hit("b", "2026-01-02T00:00:00Z")]) == (
        " 页面修订 2026-01-02"
    )


# --------------------------------------------------------------------------- #
# 3. 引用抽取
# --------------------------------------------------------------------------- #
def _collect(messages):
    agent = ReActAgent(settings=get_settings(), tools=[])
    state = AgentState(question="q", messages=messages)
    agent._collect_refs(state)
    return state.refs


def test_collect_refs_carries_revision():
    refs = _collect([{
        "role": "tool",
        "content": f"出处：{WIKI}/三月七 | 修订 2026-07-28\n出处：{WIKI}/光锥",
    }])
    assert refs[0]["last_revision_ts"] == "2026-07-28"
    # 没有修订时间的来源不该被塞一个空串，前端要能区分"没有"和"有但是空"
    assert "last_revision_ts" not in refs[1]


def test_collect_refs_still_dedupes():
    refs = _collect([{
        "role": "tool",
        "content": f"出处：{WIKI}/三月七 | 修订 2026-07-28\n出处：{WIKI}/三月七 | 修订 2026-07-28",
    }])
    assert len(refs) == 1


# --------------------------------------------------------------------------- #
# 4. get_page：把 revision ID 换成日期
# --------------------------------------------------------------------------- #
def test_get_page_shows_revision_date_not_raw_iso(monkeypatch):
    import app.ingest.wiki_client as wc_mod

    class _WC:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def fetch_page(self, title):
            return types.SimpleNamespace(
                title=title, wikitext="正文", last_revision="2026-07-28T04:48:04Z",
                revision_id=1700000000,
            )

    monkeypatch.setattr(wc_mod, "WikiClient", _WC)
    ctx = ToolContext(settings=get_settings(), retriever=None, store=None, game="sr")
    out = asyncio.run(get_page({"title": "三月七"}, ctx))
    assert "页面最后修订 2026-07-28" in out
    # 早先这里直接把 ISO 串拼进句子，模型既读不出时效，时分秒还吃 token
    assert "T04:48:04Z" not in out, "get_page 不该再暴露原始 ISO 串"


# --------------------------------------------------------------------------- #
# 5. 语义守卫：措辞不能说谎
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("forbidden", [
    "数据截至", "数据更新于", "截至今天", "最新版本数据", "当前版本数据",
])
def test_no_misleading_freshness_wording(forbidden):
    """`last_revision_ts` 是**页面最后编辑**时间，不是数据生效时间。

    一页 2020 年建的 wiki 昨天改了个错别字，它也是昨天 ——
    写成「数据截至 2026-09-01」会让用户以为内容是那天的，
    比不给日期更坏，因为用户会信。

    这条守卫守的是**措辞**，功能正常也可能在这里红，
    而红的时候它是对的：说明有人把「页面修订」改成了「数据截至」。
    """
    from pathlib import Path

    import app.agent.tools as tools_mod
    from app.agent.react import SYSTEM_PROMPT

    tools_src = Path(tools_mod.__file__).read_text(encoding="utf-8")
    # 注释里为了说明「为什么不能这么写」而引用这些词是允许的，
    # 所以只查真正会被模型/用户看到的字符串与提示词。
    assert forbidden not in SYSTEM_PROMPT, (
        f"系统提示词里出现了「{forbidden}」—— "
        f"页面修订 ≠ 数据生效时间，这么写是在教模型撒谎。"
    )
    assert forbidden not in tools_src or f"# {forbidden}" in tools_src or (
        f"「{forbidden}」" in tools_src
    ), (
        f"tools.py 的输出文案里出现了「{forbidden}」"
    )


def test_system_prompt_tells_model_to_disclose_stale_data():
    """提示词必须明确要求「引用旧条目时说明修订日期」，否则时间戳只是装饰。"""
    from app.agent.react import SYSTEM_PROMPT

    assert "修订" in SYSTEM_PROMPT, (
        "系统提示词没提修订日期。工具给了时间戳但没告诉模型该怎么用，"
        "等于白给 —— 模型会把它当噪音忽略。"
    )
    assert "优先采信" in SYSTEM_PROMPT or "较早" in SYSTEM_PROMPT, (
        "提示词只说了有日期，没说**该怎么用**：什么时候该声明、什么时候该优先采信新的。"
    )


# --------------------------------------------------------------------------- #
# 6. 端到端：真检索必须真的带日期（不是空转）
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_real_index_has_revision_metadata():
    """真实索引里必须有 `last_revision_ts`，否则前面所有断言都在测假数据。

    没有索引时 skip（CI 上 `data/raw` 不存在，恒红没意义），
    但**只允许因为「没有索引」而 skip**：
    这里刻意不用 `except Exception: skip` ——
    那种写法会把「方法名写错了」「索引损坏」「权限问题」全都悄悄放过，
    一条 skip 就是一条假绿（本文件第一次跑就栽在这：方法名写错，它安静地 skip 了）。
    """
    from pathlib import Path

    from app.core.config import get_settings as gs
    from app.store.vector_store import get_vector_store

    s = gs()
    root = Path(s.data_root_abs)
    if not (root / "vectors").exists() and not (root / "lance").exists():
        pytest.skip("本机没有索引")

    store = get_vector_store(s)
    col = store.get_collection("sr", s.embed_model_main, "base")
    if hasattr(col, "all_rows"):
        # LanceDB 后端：metadata 在表里，只能走 API
        metas = [r.get("metadata", r) for r in col.all_rows()]
    else:
        # NumPy 后端：metadatas.json 就是全部 metadata
        col._load()
        metas = list(col.metadatas)

    assert metas, "sr 库读到 0 个 chunk，语料没了？"
    with_ts = [m for m in metas if (m or {}).get("last_revision_ts")]
    assert len(with_ts) >= len(metas) * 0.9, (
        f"只有 {len(with_ts)}/{len(metas)} 个 chunk 带 last_revision_ts。"
        f"披露功能会大面积空转 —— 先确认入库时写进去了。"
    )