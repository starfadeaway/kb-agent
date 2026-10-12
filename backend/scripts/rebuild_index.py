"""离线重建向量索引：只读本地原始缓存，不碰 wiki。

存在的理由有两个：
1. 清洗/分块逻辑改动后要重跑全量索引，重抓既慢又会再吃一轮 WAF 配额；
2. 向量库出问题时能在不丢数据的前提下重建（原始文本始终以文件形式落盘，
   索引是可再生的派生物）。

用法：
    python scripts/rebuild_index.py --game sr --fresh
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.ingest.pipeline import (  # noqa: E402
    IngestConfig,
    Ingestor,
    load_cached_page,
    raw_meta_path,
)
from app.store.vector_store import get_vector_store  # noqa: E402


def cached_ids(game: str) -> list[int]:
    d = Path(__file__).resolve().parents[2] / "data" / "raw" / game
    if not d.is_dir():
        return []
    return sorted(
        int(m.group(1)) for m in (re.match(r"^(\d+)_.+\.txt$", f.name) for f in d.glob("*.txt")) if m
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sr")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--embed-mode", default="title")
    # 实验 D 用：不传就用 .env 的 EMBED_MODEL_MAIN。
    # 换模型必须配独立 variant，否则两套维度的向量会写进同一个 collection。
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--fresh", action="store_true", help="先删除同名 collection")
    ap.add_argument("--limit", type=int, default=0, help="只重建前 N 页，调试用")
    args = ap.parse_args()

    s = get_settings()
    cfg = IngestConfig.from_settings(
        args.game,
        variant=args.variant,
        embed_mode=args.embed_mode,
        embed_model=args.embed_model or s.embed_model_main,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    store = get_vector_store(s)
    if args.fresh:
        name = store.collection_name(args.game, cfg.embed_model, cfg.variant)
        print(f"已清空 collection {name}" if store.drop(args.game, cfg.embed_model, cfg.variant)
              else f"collection {name} 本就不存在")

    ids = cached_ids(args.game)
    if args.limit:
        ids = ids[: args.limit]
    missing_meta = [i for i in ids if not raw_meta_path(i, args.game).is_file()]
    print(f"缓存页 {len(ids)}，缺元数据 {len(missing_meta)}")

    ing = Ingestor(settings=s, llm=None, store=store)
    pages = []
    for pid in ids:
        p = load_cached_page(pid, args.game)
        if p is not None:
            pages.append(p)
    print(f"可复原页面 {len(pages)}")

    t0 = time.perf_counter()
    stats = await ing.ingest_pages(
        pages, cfg, source_url_for=lambda p: f"{s.wiki_base}/{args.game}/{p.title}"
    )
    await ing.llm.aclose()
    out = {
        "collection": store.collection_name(args.game, cfg.embed_model, cfg.variant),
        "count": store.count(args.game, cfg.embed_model, cfg.variant),
        "stats": stats.as_dict(),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())