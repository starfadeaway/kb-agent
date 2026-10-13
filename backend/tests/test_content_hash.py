"""守卫：内容指纹（版本更新链路的第 1 步）。

`docs/delta_sync.md` 定的顺序是：先落指纹 → 再做增量检测 → 再按页重建。
本文件守的是第 1 步，且必须满足三条纪律：

1. **双向验证**：对"没有指纹"的旧代码实跑必须变红。
2. **顶层不 import 修复后才存在的符号** ——
   本文件只 import `pipeline` / `models` 里**本来就有的**模块与类，
   新增的字段一律用属性访问（`hasattr` / `__table__.columns`）。
   否则旧代码会在 collection 阶段直接 ImportError，
   一条行为断言都跑不到，等于没验证。
3. **断言行为，不断言实现**：能通过改调用方式绕过，才算真的被守住。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.ingest import pipeline  # noqa: E402
from app.ingest.wiki_client import PageContent  # noqa: E402
from app.store.models import ChunkMeta, Page  # noqa: E402

WIKITEXT = "{{Infobox\r\n| name = 三月七\r\n}}\r\n\r\n\r\n命途：存护、巡猎。\r\n"


@pytest.fixture()
def raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把 raw 缓存重定向到临时目录。

    不重定向的话测试会往真实的 data/raw 写文件 ——
    守卫测试污染生产数据，代价比它验证的 bug 还大。
    """
    root = tmp_path / "raw"

    def _cache_path(page: PageContent, game: str) -> Path:
        p = root / game
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{page.page_id}_x.txt"

    def _meta_path(page_id: int, game: str) -> Path:
        p = root / game
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{page_id}.meta.json"

    monkeypatch.setattr(pipeline, "raw_cache_path", _cache_path)
    monkeypatch.setattr(pipeline, "raw_meta_path", _meta_path)
    return root


def _page() -> PageContent:
    return PageContent(
        title="三月七", page_id=1494, wikitext=WIKITEXT,
        last_revision="2026-10-01T00:00:00Z", revision_id=12345,
    )


# --------------------------------------------------------------------------
# 表结构：字段必须真的存在（改 model 不等于 ALTER 生效，但缺字段更早暴露）
# --------------------------------------------------------------------------

def test_page_model_has_content_hash():
    assert "content_hash" in Page.__table__.columns, \
        "Page 模型缺 content_hash —— 增量检测将无从查询"


def test_chunk_meta_model_has_content_hash():
    assert "content_hash" in ChunkMeta.__table__.columns, \
        "ChunkMeta 模型缺 content_hash —— 无法判断哪些块可跳过 embedding"


def test_content_hash_is_indexed():
    """页级指纹会被高频查询（"哪些页变了"），没有索引等于全表扫。"""
    idx = {c.name for c in Page.__table__.indexes}
    assert idx, "Page 上应保留索引定义"
    # 具体索引名由迁移脚本创建，这里只断言字段确实带 index=True
    col = Page.__table__.columns["content_hash"]
    assert col.index is True, "content_hash 应建索引"


# --------------------------------------------------------------------------
# sidecar：指纹必须落盘，否则"不读正文判断变没变"做不到
# --------------------------------------------------------------------------

def test_sidecar_records_content_hash(raw_dir):
    pipeline.cache_raw(_page(), "sr")
    meta = json.loads((raw_dir / "sr" / "1494.meta.json").read_text(encoding="utf-8"))
    assert meta.get("content_hash"), "sidecar 未记录内容指纹"
    assert len(meta["content_hash"]) == 16


def test_sidecar_hash_readable_without_reading_body(raw_dir):
    """`cached_page_hash` 只读 sidecar。增量检测要按批遍历上千页，
    每次读全文等于把检测变成重扫语料，那就没有省下任何东西。"""
    pipeline.cache_raw(_page(), "sr")
    fn = getattr(pipeline, "cached_page_hash", None)
    assert callable(fn), "缺少 cached_page_hash：只能读全文才能判断变没变"
    assert fn(1494, "sr"), "sidecar 有指纹但读不出来"
    assert fn(999999, "sr") == "", "不存在的页面应返回空串而不是抛错"


def test_cache_raw_detects_real_change(raw_dir):
    """正文实质变了 -> sidecar 指纹必须变。"""
    pipeline.cache_raw(_page(), "sr")
    before = pipeline.cached_page_hash(1494, "sr")

    changed = _page()
    changed.wikitext = WIKITEXT + "\n终结技倍率提升。\n"
    pipeline.cache_raw(changed, "sr")
    after = pipeline.cached_page_hash(1494, "sr")

    assert before != after, "正文实质变更后指纹未变 —— 增量检测会漏掉这次更新"


def test_cache_raw_ignores_format_only_change(raw_dir):
    """纯格式变动（CRLF -> LF、行尾空格、多余空行）不算变更。

    这一条直接决定增量还省不省钱：wiki 上改错别字/调排版很常见，
    若这些都算"变了"，实测那 20.9% 的交集会被放大到接近全量。
    """
    pipeline.cache_raw(_page(), "sr")
    before = pipeline.cached_page_hash(1494, "sr")

    reformatted = _page()
    reformatted.wikitext = "{{Infobox\n| name = 三月七\n}}\n\n\n命途：存护、巡猎。\n\n"
    pipeline.cache_raw(reformatted, "sr")
    after = pipeline.cached_page_hash(1494, "sr")

    assert before == after, "纯格式变动被误判为内容变更"
