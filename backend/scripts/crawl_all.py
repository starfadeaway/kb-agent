"""按需入库：把一个尚未收录的游戏灌进知识库。

这是回答"用户问别的游戏怎么办"的**第 3 层**答案，也是这套架构可扩展性的证据：
**加一个游戏不需要改任何检索/Agent 代码，只需要一条抓取命令。**

用法：
    python scripts/crawl_all.py --game ys --categories 圣遗物套装,命之座激活 --limit 25

为什么限流参数是 batch=10 / interval=6：
实测「一次请求 10 页 + 间隔 6s + 保持 Cookie」连续抓 518 页 0 次拦截。
间隔拉到 20s 只是更慢，并没有把拦截率降到 0，所以不取那个值。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crawl import crawl

logger = logging.getLogger("crawl_all")

# 每个游戏的推荐分类。挑的是"和现有两个库能形成对照"的体系，
# 而不是随便抓一堆 —— 演示时能问出「同一个问题、两个游戏各有各的答案」才有说服力。
#
# 页数是实测的分类总量（cmlimit=500 探过），不是拍脑袋：
#   圣遗物套装 65 / 冒险道具 43 / 传说任务 500 / 委托任务 182 / 书籍 105 / 世界任务 91
# 传说任务和委托任务特意只取一部分 —— 它们是长文本任务条目，
# 全量抓对"数值问答"型演示没有增量，却会让入库时间翻几倍。
PRESETS: dict[str, list[tuple[str, int]]] = {
    "ys": [
        # 圣遗物套装 ↔ 星铁的「位面饰品」，是最直观的一对
        ("圣遗物套装", 65),
        ("冒险道具", 43),
        ("合成图纸", 9),
        ("七天神像", 7),
        ("七国徽印", 10),
        ("命之座激活", 20),
        ("传说任务", 40),
        ("委托任务", 30),
        ("天赋培养素材", 20),
        ("奇偶互动", 15),
        ("可钓游鱼", 12),
    ],
    "sr": [("位面饰品", 40), ("NPC", 25)],
    "zzz": [("音擎", 30), ("邦布", 20)],
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ys", choices=sorted(PRESETS))
    ap.add_argument("--categories", default="", help="逗号分隔；不给则用预设")
    ap.add_argument("--limit", type=int, default=0, help="覆盖预设里的每类页数")
    ap.add_argument("--interval", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.categories:
        cats = [(c.strip(), args.limit or 20) for c in args.categories.split(",") if c.strip()]
    else:
        cats = [
            (c, args.limit or n) for c, n in PRESETS[args.game]
        ]

    print(f"游戏：{args.game}")
    print(f"分类：{', '.join(f'{c}×{n}' for c, n in cats)}")
    print(f"限流：batch={args.batch} interval={args.interval}s")
    print(f"预计页数：{sum(n for _, n in cats)}\n")

    # crawl() 的真实返回字段是 pages_this_run / chunks_this_run / failed(list) / waf_hits，
    # 不是 pages/chunks/failed(int)。按错的 schema 聚合会直接 TypeError，
    # 而且是跑完全部请求才炸 —— 白等一趟。
    totals = {"pages": 0, "chunks": 0, "failed": 0, "waf_hits": 0}
    per_cat = []
    for i, (cat, limit) in enumerate(cats, 1):
        print(f"--- [{i}/{len(cats)}] {cat} (limit={limit}) ---", flush=True)
        try:
            r = await crawl(args.game, cat, limit, args.interval, args.batch,
                            "base", "title", args.dry_run)
            totals["pages"] += r.get("pages_this_run", 0)
            totals["chunks"] += r.get("chunks_this_run", 0)
            totals["failed"] += len(r.get("failed") or [])
            totals["waf_hits"] += r.get("waf_hits", 0) or 0
            per_cat.append({"category": cat, **r})
            print(f"    -> pages={r.get('pages_this_run')} chunks={r.get('chunks_this_run')} "
                  f"failed={len(r.get('failed') or [])} WAF={r.get('waf_hits')} {r.get('elapsed_s')}s",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            # 单个分类失败不能中断整轮：限流通常只影响某一批
            logger.warning("分类 %s 失败：%s: %s", cat, type(exc).__name__, exc)
            per_cat.append({"category": cat, "error": f"{type(exc).__name__}: {exc}"})
            await asyncio.sleep(args.interval * 2)

    print("\n" + "=" * 70)
    print(f"  {args.game} 入库完成：{totals['pages']} 页 / {totals['chunks']} chunk / "
          f"{totals['failed']} 失败 / WAF 拦截 {totals['waf_hits']} 次")
    print("=" * 70)
    out = Path(__file__).resolve().parents[2] / "data" / "eval" / f"crawl_{args.game}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"game": args.game, "totals": totals, "categories": per_cat},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())
