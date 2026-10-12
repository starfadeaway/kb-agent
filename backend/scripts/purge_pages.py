"""从已建好的索引里精确删掉某个 page_id 的全部 chunk。

**为什么不直接重建索引**：重建要把全部 5000+ chunk 重新过一遍 embedding
（≈20 分钟 + API 费用），而要删的只有 4 个模板页的十几个 chunk。
chunk_id 由 page_id + chunk_index 决定、内容没变，所以按 page_id 定点删除
和"重建后自然不存在"是等价的，且零 embedding 开销。

删完**必须重启后端**：索引缓存是进程内的，不重启还在服务旧数据。

用法：
    python scripts/purge_pages.py --dry-run            # 只看会删什么
    python scripts/purge_pages.py --apply              # 真删
    python scripts/purge_pages.py --apply --game sr --page-id 1273
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.ingest.page_filter import should_index  # noqa: E402
from app.store.vector_store import get_vector_store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="", help="限定某个游戏（默认全部）")
    ap.add_argument("--page-id", type=int, default=0, help="限定单个 page_id")
    ap.add_argument("--apply", action="store_true", help="真删；不给就是 dry-run")
    args = ap.parse_args()

    s = get_settings()
    store = get_vector_store(s)

    # (collection, game, page_id) -> [rows]，按页面分组便于报告
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for col in store.list_collections():
        game = col.get("game")
        if not game or (args.game and game != args.game):
            continue
        # 只处理生产索引：实验变体留着做历史对照，不动
        if col.get("variant") != "base" or col.get("embed_model") != s.embed_model_main:
            continue
        rows = store.get_collection(game, col.get("embed_model"), "base").all_rows()
        for r in rows:
            pid = int(r.get("page_id") or 0)
            if args.page_id and pid != args.page_id:
                continue
            keep, _why = should_index(str(r.get("page_title") or ""))
            if not keep:
                grouped.setdefault((col["name"], game, pid), []).append(r)

    if not grouped:
        print("索引里没有需要清理的模板页（可能已经清过了）")
        return 0

    total = sum(len(v) for v in grouped.values())
    print(f"待清理 {total} chunk，分布在 {len(grouped)} 个页面")
    for (name, game, pid), rows in sorted(grouped.items()):
        titles = sorted({str(r.get("page_title")) for r in rows})
        print(f"  [{name}] {game} page_id={pid}: {len(rows)} chunk  {titles}")

    if not args.apply:
        print("\n这是 dry-run。加 --apply 才会真删。")
        return 0

    # 按 collection 聚合 id：同一个 collection 下多个模板页要一次删完，
    # 逐页循环会拿同一批 id 反复删（幂等，但日志会骗人）
    by_collection: dict[str, tuple[str, set[str]]] = {}
    for (name, game, _pid), rows in grouped.items():
        _prev_game, ids = by_collection.setdefault(name, (game, set()))
        ids.update(str(r.get("id")) for r in rows if r.get("id"))

    for name, (game, ids) in sorted(by_collection.items()):
        col = store.get_collection(game, s.embed_model_main, "base")
        before = col.count()
        quoted = ", ".join("'" + i.replace("'", "''") + "'" for i in sorted(ids))
        col._table().delete(f"id IN ({quoted})")
        print(f"  {name}: 目标 {len(ids)} chunk，{before} → {col.count()}")

    print("\n完成。请重启后端让缓存失效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())