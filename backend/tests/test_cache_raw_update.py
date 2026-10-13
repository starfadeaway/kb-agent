"""守卫：原始缓存必须「存在则更新」，不能是「不存在才写」。

缺陷
----
`cache_raw()` 原先写的是

    if not p.is_file():
        p.write_text(page.wikitext, encoding="utf-8")

看上去是「没有就写一份」，实际后果是**缓存一旦落盘就永不更新**。
重抓到页面出了新版本时，缓存仍停在第一次抓到的内容上。

两个后果都很难当场发现：
  1. 离线重建索引用旧正文 → MySQL（每次抓取都 upsert）与向量库**分叉**，
     而索引本身看起来完全正常；
  2. `last_revision_ts` 停在旧值 —— 它正是版本感知检索 `min_revision_ts`
     的输入，等于拿过期时间戳做版本过滤。

修复后还要能回答「这轮抓取到底更新了几页」，
所以函数改为返回是否发生变化。

已对修复前的快照实跑确认能抓到缺陷。
"""
from __future__ import annotations

import json

import pytest
from app.ingest.pipeline import cache_raw
from app.ingest.wiki_client import PageContent


def _page(**kw) -> PageContent:
    base = dict(
        title="太空封印站", page_id=12345,
        wikitext="第一版正文", last_revision="2024-01-01T00:00:00Z",
        revision_id=100,
    )
    base.update(kw)
    return PageContent(**base)  # type: ignore[arg-type]


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """把缓存路径重定向到 tmp —— 测试绝不碰真实的 data/raw。

    两个细节都是踩出来的：
    1. `cache_raw` 内部是模块级查找 `raw_cache_path(...)`，
       所以替换模块属性即可生效，不需要改生产代码；
    2. 但**测试自己不能在顶层 `from ... import raw_cache_path`** ——
       那样拿到的是本模块的名字，monkeypatch 改不到它，
       断言就会去读真实的 518 个 meta.json。
       所以这里把两个路径函数一并返回给测试用。
    """
    import app.ingest.pipeline as pl
    game = "sr"
    (tmp_path / game).mkdir(parents=True, exist_ok=True)

    def cache_path(page, _g=game):
        return tmp_path / _g / f"{page.page_id}.txt"

    def meta_path(pid, _g=game):
        return tmp_path / _g / f"{pid}.meta.json"

    monkeypatch.setattr(pl, "raw_cache_path", cache_path)
    monkeypatch.setattr(pl, "raw_meta_path", meta_path)
    return game, cache_path, meta_path


# ---------- 首次写入 ----------

def test_first_write_creates_files(paths):
    game, cache_path, meta_path = paths
    page = _page()
    assert cache_raw(page, game) is True
    assert cache_path(page).read_text(encoding="utf-8") == "第一版正文"
    meta = json.loads(meta_path(12345).read_text(encoding="utf-8"))
    assert meta["revision_id"] == 100
    assert meta["last_revision"] == "2024-01-01T00:00:00Z"


def test_second_identical_write_reports_no_change(paths):
    """内容没变就不要再写盘（抓取全库 1038 页，无谓 IO 也要避免）。"""
    game, _, _ = paths
    cache_raw(_page(), game)
    assert cache_raw(_page(), game) is False, "内容未变却报告已更新"


# ---------- 核心：必须能更新 ----------

def test_changed_body_updates_cache(paths):
    """**这条是缺陷本体**：早先版本下缓存永不更新。"""
    game, cache_path, _ = paths
    cache_raw(_page(), game)
    changed = cache_raw(_page(wikitext="第二版正文"), game)
    assert changed is True, "正文变了却没有更新缓存"
    assert cache_path(_page()).read_text(encoding="utf-8") == "第二版正文"


def test_changed_revision_updates_meta(paths):
    """`last_revision_ts` 是版本感知检索的输入，过期会让版本过滤失效。"""
    game, _, meta_path = paths
    cache_raw(_page(), game)
    changed = cache_raw(
        _page(last_revision="2026-08-02T04:52:16Z", revision_id=100353), game)
    assert changed is True, "revision 变了却没有更新侧车"
    meta = json.loads(meta_path(12345).read_text(encoding="utf-8"))
    assert meta["last_revision_ts"] == "2026-08-02T04:52:16Z"
    assert meta["revision_id"] == 100353


def test_body_changed_but_revision_same_still_updates(paths):
    """正文改了而 revision 没变（理论上不该发生，但守卫不该依赖它成立）。"""
    game, _, _ = paths
    cache_raw(_page(), game)
    assert cache_raw(_page(wikitext="修订过"), game) is True


# ---------- 回归保护 ----------

def test_meta_path_is_stable_across_title_change(paths):
    """正文文件名不含标题，标题变了不该产生第二个缓存文件。"""
    game, _, meta_path = paths
    cache_raw(_page(), game)
    cache_raw(_page(title="太空封印站（重定向）"), game)
    files = sorted(p.name for p in meta_path(1).parent.glob("*.meta.json"))
    assert len(files) == 1, f"标题变化产生了额外缓存文件：{files}"