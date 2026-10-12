"""批量入库多个分类（共用一个客户端与限流节流器）。

顺序执行而不是并发：并发只会更快触发 EdgeOne 限流，不会更快完成。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.ingest.pipeline import IngestConfig, Ingestor  # noqa: E402
from app.ingest.wiki_client import WikiClient  # noqa: E402
from scripts.crawl import checkpoint_path, load_ckpt, save_ckpt  # noqa: E402

logger = logging.getLogger("bulk")

DEFAULT_PLAN = [
    ("光锥", 90),
    ("位面饰品", 29),
    ("凝滞虚影", 29),
    ("NPC", 80),
    ("敌人", 60),
    ("奇物", 60),
    ("事件", 60),
    ("关卡", 50),
    ("交互事件", 40),
    ("开拓任务", 40),
]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sr")
    ap.add_argument("--interval", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--embed-mode", default="title")
    ap.add_argument("--categories", default="", help="形如 光锥:90,位面饰品:29，覆盖默认计划")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = get_settings()
    cfg = IngestConfig.from_settings(args.game, variant=args.variant, embed_mode=args.embed_mode)
    ing = Ingestor(settings=s)

    plan = DEFAULT_PLAN
    if args.categories:
        plan = []
        for item in args.categories.split(","):
            name, _, lim = item.partition(":")
            plan.append((name, int(lim or 30)))

    summary = []
    t0 = time.perf_counter()
    async with WikiClient(
        args.game, base_url=s.wiki_base, min_interval=args.interval, max_retry=s.wiki_max_retry
    ) as wc:
        for category, limit in plan:
            ck = checkpoint_path(args.game, category)
            done = load_ckpt(ck)
            try:
                titles = await wc.list_category_pages(category, limit=limit)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] 取分类失败: %s", category, exc)
                summary.append({"category": category, "error": str(exc)[:120]})
                continue

            pending = [t for t in titles if t not in done]
            logger.info("=== %s: 共 %d 页，待抓 %d 页 ===", category, len(titles), len(pending))
            cat_pages = cat_chunks = 0
            for i in range(0, len(pending), args.batch):
                batch = pending[i : i + args.batch]
                try:
                    pages = await wc.fetch_pages(batch, batch=args.batch)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] 批次失败，跳过后续: %s", category, exc)
                    break
                stats = await ing.ingest_pages(
                    pages, cfg, source_url_for=lambda p: f"{s.wiki_base}/{args.game}/{p.title}"
                )
                cat_pages += stats.pages
                cat_chunks += stats.chunks
                done.update(p.title for p in pages)
                save_ckpt(ck, done, {"pages": cat_pages, "chunks": cat_chunks, "waf_hits": wc.waf_hits})
                logger.info(
                    "[%s] %d/%d | +%d 页 +%d chunk | WAF %d",
                    category, len(done), len(titles), stats.pages, stats.chunks, wc.waf_hits,
                )
                if wc.waf_hits >= 3:
                    logger.warning("WAF 频繁拦截，本分类先到这，剩下的下次续传")
                    break
            summary.append(
                {"category": category, "pages": cat_pages, "chunks": cat_chunks, "done": len(done)}
            )

    await ing.llm.aclose()
    result = {
        "collection": ing.store.collection_name(args.game, cfg.embed_model, cfg.variant),
        "collection_count": ing.store.count(args.game, cfg.embed_model, cfg.variant),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "per_category": summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = Path(__file__).resolve().parents[2] / "data" / "ingest_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())