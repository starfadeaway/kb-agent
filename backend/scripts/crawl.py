"""可断点续传的抓取 CLI。

用法（PowerShell）:
    .venv\\Scripts\\python.exe backend\\scripts\\crawl.py --game sr --category 光锥 --limit 40

为什么这么设计：
1. **批量取页**：MediaWiki 的 titles 支持 `A|B|C`，一次请求拿 10 页，
   请求数降一个数量级 —— 这是对付 EdgeOne 限流最有效的一招，
   比单纯把间隔从 2s 拉到 5s 更管用。
2. **断点续传**：WAF 随时可能拦一道，限流窗口又不可预测，
   所以每批落一次 checkpoint，被打断后重跑不重复抓。
3. **embedding 分批**：向量化走另一条链路（不打 wiki），不受 WAF 影响。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.ingest.pipeline import IngestConfig, Ingestor  # noqa: E402
from app.ingest.wiki_client import WikiClient  # noqa: E402

logger = logging.getLogger("crawl")


def checkpoint_path(game: str, category: str) -> Path:
    root = Path(__file__).resolve().parents[2] / "data" / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{game}_{category}.json"


def load_ckpt(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        # utf-8-sig：PowerShell 的 Set-Content -Encoding UTF8 会写 BOM，
        # 用普通 utf-8 读会整个 JSON 解析失败，进而把 checkpoint 静默清空。
        return set(json.loads(path.read_text(encoding="utf-8-sig")).get("done_titles", []))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("checkpoint %s 解析失败(%s)，按空处理", path, exc)
        return set()


def save_ckpt(path: Path, done: set[str], stats: dict) -> None:
    path.write_text(
        json.dumps(
            {"done_titles": sorted(done), "stats": stats, "updated_at": datetime.now().isoformat()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


async def crawl(
    game: str,
    category: str,
    limit: int,
    interval: float,
    batch: int,
    variant: str,
    embed_mode: str,
    dry_run: bool,
) -> dict:
    s = get_settings()
    ck = checkpoint_path(game, category)
    done = load_ckpt(ck)
    cfg = IngestConfig.from_settings(game, variant=variant, embed_mode=embed_mode)

    ing = Ingestor(settings=s)
    total_pages = 0
    total_chunks = 0
    failed: list[str] = []
    waf_hits = 0
    t0 = time.perf_counter()

    async with WikiClient(
        game, base_url=s.wiki_base, min_interval=interval, max_retry=s.wiki_max_retry
    ) as wc:
        titles = await wc.list_category_pages(category, limit=limit)
        pending = [t for t in titles if t not in done]
        logger.info("分类 %s 共 %d 页，待抓 %d 页（已跳过 %d）", category, len(titles), len(pending), len(done))

        for i in range(0, len(pending), batch):
            chunk_titles = pending[i : i + batch]
            try:
                pages = await wc.fetch_pages(chunk_titles, batch=batch)
            except Exception as exc:  # noqa: BLE001
                logger.warning("批次失败 %s: %s", chunk_titles[:3], exc)
                failed.extend(chunk_titles)
                waf_hits = wc.waf_hits
                save_ckpt(ck, done, {"pages": total_pages, "chunks": total_chunks, "failed": failed[:50]})
                if wc.waf_hits and wc.waf_hits > 0 and len(failed) >= batch:
                    logger.warning("连续被拦，先跑一半的量，剩下的下次续传")
                    break
                continue

            waf_hits = wc.waf_hits
            if dry_run:
                total_pages += len(pages)
                done.update(chunk_titles)
                logger.info("[dry] 抓到 %d 页", len(pages))
                continue

            stats = await ing.ingest_pages(
                pages, cfg, source_url_for=lambda p: f"{s.wiki_base}/{game}/{p.title}"
            )
            total_pages += stats.pages
            total_chunks += stats.chunks
            failed.extend(stats.failed)
            done.update(p.title for p in pages)

            save_ckpt(
                ck,
                done,
                {
                    "pages": total_pages,
                    "chunks": total_chunks,
                    "failed": failed[:50],
                    "waf_hits": waf_hits,
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                },
            )
            logger.info(
                "进度 %d/%d 页 | 本批 %d 页 %d chunk | 累计 %d 页 %d chunk | WAF %d",
                len(done), len(titles), stats.pages, stats.chunks, total_pages, total_chunks, waf_hits,
            )

    await ing.llm.aclose()
    return {
        "game": game,
        "category": category,
        "variant": variant,
        "embed_mode": embed_mode,
        "collection": ing.store.collection_name(game, cfg.embed_model, variant),
        "collection_count": ing.store.count(game, cfg.embed_model, variant),
        "pages_this_run": total_pages,
        "chunks_this_run": total_chunks,
        "done_titles": len(done),
        "failed": failed[:20],
        "waf_hits": waf_hits,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="biligame wiki 抓取 + 入库")
    ap.add_argument("--game", default="sr", choices=["sr", "zzz", "ys"])
    ap.add_argument("--category", required=True, help="分类名，如 光锥 / 位面饰品 / NPC")
    ap.add_argument("--limit", type=int, default=40, help="该分类最多取多少页")
    ap.add_argument("--interval", type=float, default=5.0, help="请求最小间隔(秒)")
    ap.add_argument("--batch", type=int, default=10, help="一次请求取几页")
    ap.add_argument("--variant", default="base", help="collection 变体名，用于隔离分块/embedding 实验")
    ap.add_argument("--embed-mode", default="title", choices=["text", "title"])
    ap.add_argument("--dry-run", action="store_true", help="只抓不入库，验证限流参数")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(
        crawl(
            args.game, args.category, args.limit, args.interval,
            args.batch, args.variant, args.embed_mode, args.dry_run,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()