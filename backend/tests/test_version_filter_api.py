"""版本感知检索的**入口**测试。

这个能力早就做完了 —— `RetrievalConfig.min_revision_ts` 一路接到两个 store 的
`build_where()`，实测也确实生效（`scripts/probe_version_filter.py`）。
但它长期**没有任何入口能设置**：HTTP 请求模型没这个字段、前端没控件、
`run_eval.py` 反而显式把它置 None。

症状是"说得到、演示不出来"：README/ARCHITECTURE 都把版本感知当卖点写着，
面试被问"怎么演示"却只能在 Python 里跑。这组测试盯住入口真的存在。
"""

from __future__ import annotations

import dataclasses

import pytest
from app.main import RetrieveRequest, _normalize_ts
from app.retrieval.hybrid import RetrievalConfig


def test_retrieve_request_exposes_version_filter():
    """HTTP 入口必须能传 min_revision_ts —— 这是"可演示"的前提。"""
    req = RetrieveRequest(query="三月七", game="sr", min_revision_ts="2025-01-01")
    assert req.min_revision_ts == "2025-01-01"


def test_retrieve_request_exposes_page_title_filter():
    req = RetrieveRequest(query="三月七", game="sr", page_titles=["三月七"])
    assert req.page_titles == ["三月七"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-02-01", "2026-02-01T00:00:00Z"),
        ("2026-02-01T16:53:50Z", "2026-02-01T16:53:50Z"),
        ("  2026-02-01  ", "2026-02-01T00:00:00Z"),
        ("", None),
        (None, None),
        ("   ", None),
    ],
)
def test_date_normalization(raw, expected):
    """纯日期必须补成当天 00:00:00Z。

    不补就是直接拿 `"2026-02-01"` 去和库里的 `"2026-02-01T16:53:50Z"` 比大小，
    结果会把当天所有内容都排除掉 —— **结果是错的但不报错**。
    """
    assert _normalize_ts(raw) == expected


def test_retrieval_config_roundtrips_version_filter():
    cfg = RetrievalConfig(game="sr", min_revision_ts="2026-01-01T00:00:00Z")
    assert cfg.as_dict().get("min_revision_ts") == "2026-01-01T00:00:00Z"


def test_as_dict_covers_every_dataclass_field():
    """**新增字段不允许从 as_dict() 里静默掉出去。**

    这正是上一条测试暴露的真缺陷：`as_dict()` 曾是手写字典，
    加 `min_revision_ts` / `page_titles` 时忘了同步 ——
    `run_eval.py` 落库的 config 快照因此看不出这轮跑带没带版本过滤，
    而类注释写的「所有字段都会随评测结果落库，保证可复现」成了假承诺。

    这条断言是跨层的：只要有人把 as_dict 改回手写清单而漏字段，立刻变红。
    """
    declared = {f.name for f in dataclasses.fields(RetrievalConfig)}
    emitted = set(RetrievalConfig(game="sr").as_dict())
    assert declared == emitted, (
        f"as_dict() 与 dataclass 字段不一致，差集={declared ^ emitted}。"
        f"新增 RetrievalConfig 字段后必须同步出现在 as_dict() 里"
    )


def test_store_build_where_supports_version_filter(stores_probe):
    """两个 store 都必须理解 min_revision_ts，否则换后端行为会变。"""
    np_store, lance = stores_probe
    kw = {"game": "sr", "min_revision_ts": "2026-01-01T00:00:00Z"}
    a = np_store.build_where(**kw)
    b = lance.build_where(**kw)
    assert a == b, f"两端 build_where 不一致：{a} vs {b}"
    assert "last_revision_ts" in str(a), f"过滤条件里没有版本键：{a}"