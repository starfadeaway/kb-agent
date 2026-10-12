"""删除指定 collection（重新入库前清场，避免旧的坏 chunk 混进评测）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.store.vector_store import get_vector_store  # noqa: E402


def main() -> None:
    s = get_settings()
    game = sys.argv[1] if len(sys.argv) > 1 else s.game_primary
    model = sys.argv[2] if len(sys.argv) > 2 else None
    variant = sys.argv[3] if len(sys.argv) > 3 else "base"
    store = get_vector_store(s)
    name = store.collection_name(game, model, variant)
    before = store.count(game, model, variant)
    try:
        store.drop(game, model, variant)
        print(f"已删除 collection {name}（原 {before} 条）")
    except Exception as exc:  # noqa: BLE001
        print(f"删除 {name} 失败（可能本就不存在）：{type(exc).__name__}: {exc}")
    print("现存 collection:", [c["name"] for c in store.list_collections()])


if __name__ == "__main__":
    main()