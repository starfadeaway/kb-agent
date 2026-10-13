"""守卫：增量变更检测的三级筛。

`app/ingest/delta_sync.py` 的全部价值在于**不取不该取的正文**：
WAF 配额是稀缺资源，每多取一页正文都是真金白银与被拦风险。
所以守卫的核心不是"能不能发现变更"，而是**会不会多花钱**。

分层职责：
- 本文件守三级筛的**分流正确性**（纯逻辑，用假 WikiClient，不碰网络）
- 实测（`scripts/probe_delta_sync.py`）守它**在真实 WAF 下能不能跑通**

双向验证说明：本模块是新增的，HEAD 上不存在，所以没有「修复前快照」可跑。
这是正常的 —— 双向验证适用于**修改**已有行为的改动，
不适用于**新增**模块。新模块的守卫靠这三件事成立：
分流逻辑写死成可测的纯函数、断言覆盖每条分流路径、
以及必须配一次真实网络实测（否则再绿的测试也只是在测假想）。
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.ingest import delta_sync as ds  # noqa: E402
from app.ingest.fingerprint import page_hash  # noqa: E402
from app.ingest.wiki_client import PageContent, WAFBlocked  # noqa: E402

SINCE = datetime(2026, 10, 6, tzinfo=UTC)


class FakeWiki:
    """记录每一次调用，用来断言"哪些级别被触发了"。

    `fail_at` 模拟 WAF 在某一级的第 N 次请求抛异常 —— 实测撞过：
    翻页第 6 页吃满 4 次退避后抛 WAFBlocked，前面积累的 2500 条记录全丢。
    """

    def __init__(self, changes, metas, pages, *, cursor=None, fail_at=None):
        self._changes = changes
        self._metas = metas
        self._pages = pages
        self._cursor = cursor
        self._fail_at = fail_at
        self.calls: list[str] = []

    def _maybe_fail(self, stage: str) -> None:
        if self._fail_at == stage:
            raise WAFBlocked(f"假 WAF：{stage}")

    async def recent_changes(self, since, **kw):
        self.calls.append("L1")
        self._maybe_fail("L1")
        return self._changes, self._cursor

    async def recent_changes_page(self, since, cursor, **kw):
        self.calls.append("L1+")
        self._maybe_fail("L1+")
        # 翻页时继续给数据、游标保持非空 —— 用来模拟"窗口还有更多页"
        return (self._changes, self._cursor)

    async def fetch_page_meta(self, titles, batch=20):
        self.calls.append(f"L2:{len(titles)}")
        self._maybe_fail("L2")
        return [self._metas[t] for t in titles if t in self._metas]

    async def fetch_pages(self, titles, batch=10):
        self.calls.append(f"L3:{','.join(titles)}")
        self._maybe_fail("L3")
        return [self._pages[t] for t in titles if t in self._pages]

    async def aclose(self):
        self.calls.append("close")


def _local(monkeypatch, mapping):
    async def _fake(game, settings):
        return mapping
    monkeypatch.setattr(ds, "_local_pages", _fake)


def _meta(title, ts, revid=1, page_id=1):
    return PageContent(title=title, page_id=page_id, wikitext="",
                       last_revision=ts, revision_id=revid)


def _body(title, wikitext, ts, revid=2, page_id=1):
    return PageContent(title=title, page_id=page_id, wikitext=wikitext,
                       last_revision=ts, revision_id=revid)


@pytest.fixture()
def anyio_backend():
    return "asyncio"


# --------------------------------------------------------------------------
# 分流正确性：每一级都必须挡掉它该挡的
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_revision_unchanged_never_fetches_body(monkeypatch):
    """revision 没变 -> 内容不可能变 -> 连正文都不许取。

    这是最贵的一级：取正文每页要占一个请求，且正文越大越容易撞 WAF。
    """
    local = {"三月七": {"page_id": 1, "content_hash": "abc", "revision_ts": "2026-10-10T00:00:00Z"}}
    _local(monkeypatch, local)
    wiki = FakeWiki(
        changes=[{"title": "三月七", "timestamp": "2026-10-10T00:00:00Z"}],
        metas={"三月七": _meta("三月七", "2026-10-10T00:00:00Z")},
        pages={},
    )
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)

    assert rep.changed == [], "revision 未变却报内容变化"
    assert not any(c.startswith("L3") for c in wiki.calls), \
        f"取了不该取的正文：{wiki.calls}"


@pytest.mark.asyncio
async def test_revision_changed_but_hash_same_is_not_a_change(monkeypatch):
    """被编辑过但内容没动（错别字/排版）-> 不算变更。

    这一条直接决定增量还省不省钱：wiki 上这类编辑很常见。
    """
    text = "{{Infobox\n|name=三月七\n}}\n\n命途：存护。\n"
    local = {"三月七": {"page_id": 1, "content_hash": page_hash(text),
                        "revision_ts": "2026-10-01T00:00:00Z"}}
    _local(monkeypatch, local)
    wiki = FakeWiki(
        changes=[{"title": "三月七", "timestamp": "2026-10-12T00:00:00Z"}],
        metas={"三月七": _meta("三月七", "2026-10-12T00:00:00Z", revid=99)},
        pages={"三月七": _body("三月七", text.replace("\n", "\r\n"),
                               "2026-10-12T00:00:00Z", revid=99)},
    )
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)

    assert rep.changed == [], "纯格式变动被报成内容变化"
    assert rep.revision_only == ["三月七"]
    assert any(c.startswith("L3") for c in wiki.calls), \
        "revision 变了却没取正文比对指纹 —— 那就分不出内容变没变"


@pytest.mark.asyncio
async def test_real_change_is_reported(monkeypatch):
    """内容真的变了 -> 必须报出来，且带上新旧指纹。"""
    old_text = "命途：存护。"
    new_text = "命途：存护、巡猎。终结技倍率提升。"
    local = {"三月七": {"page_id": 1494, "content_hash": page_hash(old_text),
                        "revision_ts": "2026-10-01T00:00:00Z"}}
    _local(monkeypatch, local)
    wiki = FakeWiki(
        changes=[{"title": "三月七", "timestamp": "2026-10-12T00:00:00Z"}],
        metas={"三月七": _meta("三月七", "2026-10-12T00:00:00Z", revid=99, page_id=1494)},
        pages={"三月七": _body("三月七", new_text, "2026-10-12T00:00:00Z",
                               revid=99, page_id=1494)},
    )
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)

    assert len(rep.changed) == 1
    c = rep.changed[0]
    assert c["page_id"] == 1494
    assert c["old_hash"] == page_hash(old_text)
    assert c["new_hash"] == page_hash(new_text)
    assert c["old_hash"] != c["new_hash"]


@pytest.mark.asyncio
async def test_uncollected_page_is_ignored(monkeypatch):
    """全站被编辑的页里，绝大多数不在收录范围 —— 它们不该进检测。"""
    local = {"三月七": {"page_id": 1, "content_hash": "abc", "revision_ts": "2026-10-01T00:00:00Z"}}
    _local(monkeypatch, local)
    wiki = FakeWiki(
        changes=[{"title": "活动公告", "timestamp": "2026-10-12T00:00:00Z"},
                 {"title": "三月七", "timestamp": "2026-10-12T00:00:00Z"}],
        metas={}, pages={},
    )
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)

    assert rep.unique_changed_site == 2
    assert rep.intersect == 1
    assert not any(c.startswith("L3") for c in wiki.calls), \
        f"未收录页竟然触发了取正文：{wiki.calls}"


@pytest.mark.asyncio
async def test_empty_local_corpus_short_circuits(monkeypatch):
    """本地没收录时必须直接返回，不能去打 wiki（那会白花配额）。"""
    _local(monkeypatch, {})
    wiki = FakeWiki(changes=[{"title": "x"}], metas={}, pages={})
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)
    assert rep.changed == []
    assert wiki.calls == [], f"空语料仍打了网络：{wiki.calls}"


# --------------------------------------------------------------------------
# 读数纪律：窗口没翻完就必须自报是下界
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unexhausted_window_is_flagged_lower_bound(monkeypatch):
    """WAF 截断导致窗口没翻完时，所有计数都是下界，必须显式标出。

    这不是形式主义：实测 7 天窗口确实会被截断，
    把下界当精确值就会低估需要重建的规模。
    """
    local = {"三月七": {"page_id": 1, "content_hash": "", "revision_ts": ""}}
    _local(monkeypatch, local)
    wiki = FakeWiki(
        changes=[{"title": "三月七", "timestamp": "2026-10-12T00:00:00Z"}],
        metas={"三月七": _meta("三月七", "2026-10-12T00:00:00Z")},
        pages={"三月七": _body("三月七", "新内容", "2026-10-12T00:00:00Z")},
        cursor="still-more",
    )
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)
    assert rep.window_exhausted is False
    assert rep.as_dict()["lower_bound"] is True


@pytest.mark.asyncio
async def test_missing_local_hash_counts_as_changed(monkeypatch):
    """本地指纹为空时必须判为"变了"—— 无法证明它没变。

    反过来（把空指纹当"没变"）会让漏补指纹的页面永远重建不了。
    """
    local = {"三月七": {"page_id": 1, "content_hash": "", "revision_ts": ""}}
    _local(monkeypatch, local)
    wiki = FakeWiki(
        changes=[{"title": "三月七", "timestamp": "2026-10-12T00:00:00Z"}],
        metas={"三月七": _meta("三月七", "2026-10-12T00:00:00Z")},
        pages={"三月七": _body("三月七", "内容", "2026-10-12T00:00:00Z")},
    )
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)
    assert len(rep.changed) == 1, "空指纹未被判为需重建"


# --------------------------------------------------------------------------
# 请求量必须可数：它就是这套方案的成本
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_counts_are_reported(monkeypatch):
    """请求量要能报出来，否则"省了多少"根本无从谈起。"""
    local = {
        "A": {"page_id": 1, "content_hash": "h1", "revision_ts": "2026-10-01T00:00:00Z"},
        "B": {"page_id": 2, "content_hash": page_hash("B 旧"),
              "revision_ts": "2026-10-01T00:00:00Z"},
    }
    _local(monkeypatch, local)
    wiki = FakeWiki(
        changes=[{"title": "A"}, {"title": "B"}],
        metas={"A": _meta("A", "2026-10-01T00:00:00Z"),
               "B": _meta("B", "2026-10-12T00:00:00Z")},
        pages={"B": _body("B", "B 新", "2026-10-12T00:00:00Z")},
    )
    d = (await ds.detect_changes("sr", SINCE, wiki=wiki)).as_dict()

    assert d["requests"]["level1_recentchanges"] == 1
    assert d["requests"]["level2_revision_meta"] == 1
    # 只有 1 页真变了 -> 第 3 级只该为它取正文
    assert d["requests"]["level3_content"] == 1
    assert d["changed_count"] == 1
    assert d["revision_only_count"] == 0


# --------------------------------------------------------------------------
# WAF 中断：已读到的数据不能跟着一起丢
# --------------------------------------------------------------------------
# 这组守卫来自一次真实事故：翻页第 6 页吃满 4 次退避后抛 WAFBlocked，
# 前面 5 页共 2500 条变更记录全部作废，整轮检测白跑。
# 「基础设施失败不该带走有效结果」在这份评测报告里写过好几遍，
# 这次轮到增量检测自己犯 —— 所以必须用测试钉住。

@pytest.mark.asyncio
async def test_waf_on_first_page_does_not_crash(monkeypatch):
    _local(monkeypatch, {"三月七": {"page_id": 1, "content_hash": "h",
                                   "revision_ts": "2026-10-01T00:00:00Z"}})
    wiki = FakeWiki(changes=[], metas={}, pages={}, fail_at="L1")
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)   # 不抛异常即为通过
    assert rep.waf_hits >= 1


@pytest.mark.asyncio
async def test_waf_on_paging_keeps_already_read_pages(monkeypatch):
    """翻页被拦时，第一页读到的变更页必须保留，并自报是下界。"""
    local = {f"P{i}": {"page_id": i, "content_hash": "h", "revision_ts": "t"}
             for i in range(3)}
    _local(monkeypatch, local)
    wiki = FakeWiki(
        changes=[{"title": "P0"}, {"title": "P1"}, {"title": "P2"}],
        metas={}, pages={},
        cursor="more", fail_at="L1+",
    )
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)

    assert rep.unique_changed_site == 3, "翻页被拦把已读到的页弄丢了"
    assert rep.intersect == 3
    assert rep.window_exhausted is False, "中断被当成了「窗口已翻完」"
    assert rep.as_dict()["lower_bound"] is True


@pytest.mark.asyncio
async def test_waf_on_level2_keeps_level1_result(monkeypatch):
    """第 2 级被拦不该让第 1 级的交集结果消失。"""
    _local(monkeypatch, {"三月七": {"page_id": 1, "content_hash": "h",
                                   "revision_ts": "2026-10-01T00:00:00Z"}})
    wiki = FakeWiki(changes=[{"title": "三月七"}], metas={}, pages={}, fail_at="L2")
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)

    assert rep.intersect == 1, "第 2 级失败把第 1 级的结果也带走了"
    assert rep.changed == []          # 无法判定内容变没变 -> 不得谎报"没变"


@pytest.mark.asyncio
async def test_waf_on_level3_does_not_lose_level12(monkeypatch):
    """第 3 级被拦：第 1/2 级的结果仍在，只是无法判定正文指纹。"""
    local = {"三月七": {"page_id": 1, "content_hash": "oldhash",
                        "revision_ts": "2026-10-01T00:00:00Z"}}
    _local(monkeypatch, local)
    wiki = FakeWiki(
        changes=[{"title": "三月七"}],
        metas={"三月七": _meta("三月七", "2026-10-12T00:00:00Z")},
        pages={}, fail_at="L3",
    )
    rep = await ds.detect_changes("sr", SINCE, wiki=wiki)

    assert rep.intersect == 1
    assert rep.changed == [], "取不到正文时不得凭空报出变更页"
    assert rep.waf_hits >= 1
