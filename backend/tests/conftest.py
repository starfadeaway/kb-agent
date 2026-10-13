"""pytest 共享夹具。

关键约束：**测试不得联网**。
`smoke_refactor.py` 里有大量真实检索/真实 LLM 调用的断言，那是手动自检；
进 CI 的这批必须离线可跑，否则每次推送都要等 API、配额，还不稳定。
需要网络的用 `smoke_refactor.py`，需要语料的用 `needs_corpus` marker 跳过。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session")
def settings():
    from app.core.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def covered(settings) -> dict[str, str]:
    """已收录游戏 slug → 显示名。"""
    return settings.games


@pytest.fixture()
def stores_probe():
    """一套**互相隔离**的 (numpy_store, lance_store)，指向临时目录。

    两个坑都在这里挡掉，新写测试的人不用再踩：
    1. `Settings` 是 frozen dataclass，只能用 dataclasses.replace 改字段；
    2. **数据根是 `chroma_path.parent`，不是 `chroma_path` 本身**
       （两个 store 都是 `chroma_abs_path.parent / "vectors"|"lance"`）。
       把 chroma_path 直接指向 mkdtemp 会让所有测试共用系统 Temp 目录，
       数据互相污染 —— 表现为条数莫名变大、schema 被上一个测试写死。
    """
    import dataclasses
    import shutil
    import tempfile

    from app.core.config import get_settings
    from app.store.lance_store import LanceVectorStore
    from app.store.vector_store import VectorStore

    tmp = Path(tempfile.mkdtemp(prefix="kb_store_"))
    s = dataclasses.replace(get_settings(), chroma_path=str(tmp / "data" / "chroma"))
    try:
        yield VectorStore(s), LanceVectorStore(s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture()
def chunk_meta():
    """生产实际产出的 metadata 键集（`ingest_pages` 里那一组）。

    不能随手写 `{"page_title": ...}`：Lance 表的列名由**首批 upsert 的键**决定，
    少写一个 game 就会让后续 `query(game=...)` 直接报 schema error。
    """
    def _meta(page_title: str, **extra) -> dict:
        return {
            "game": "sr",
            "page_id": 1,
            "page_title": page_title,
            "section": "",
            "chunk_index": 0,
            "last_revision": 1,
            "last_revision_ts": "2025-01-01T00:00:00Z",
            "source_url": "https://example.invalid/p",
            "truncated": 0,
            **extra,
        }

    return _meta