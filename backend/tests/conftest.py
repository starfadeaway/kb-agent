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


@pytest.fixture(scope="session")
def raw_corpus() -> Path:
    """本地已抓语料目录；没有就 skip，而不是让 CI 变红。"""
    d = PROJECT_ROOT / "data" / "raw"
    if not d.is_dir() or not any(d.iterdir()):
        pytest.skip("本地没有语料（data/raw 为空），先跑 scripts/crawl_all.py")
    return d