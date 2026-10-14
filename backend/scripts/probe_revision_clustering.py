"""修订时间的**聚类度**分析：量化 wiki 到底是「随手维护」还是「事件驱动」。

## 为什么要有这个脚本

2026-10-14 我在实现「披露页面修订日期」时，写下了这么一段 UI 逻辑：

> 按 12 个月为界分三档，超过阈值标红，提示「来源均较早」

用户当场反问：**「游戏 Wiki 中的信息一般不会更改，这样的说法还是不能解决
时间戳的问题吗」** —— 他是对的，而我当时没有数据，只是凭直觉。

于是有了这个脚本。它不回答「内容会不会过时」，它回答一个更基础的问题：

> **wiki 页面的编辑，是零散的日常维护，还是跟着版本更新批量触发的？**

答案直接决定修订时间能不能当「内容时效」的代理指标：

- 若编辑**均匀分散** → 维护者天天在改 → 「久未编辑」≈「被遗忘」→ 时间戳有预警价值；
- 若编辑**高度聚类** → 批量触发（版本更新） → 「久未编辑」≈「没人觉得需要改」
  → **时间戳推不出内容过时**，用它标红是制造无法支撑的焦虑。

## 结论（sr 库实测，见 data/eval/probe_revision_clustering.json）

706 页分布在 210 个日期上，但 **Top12 个日期占了全部页面的 40.5%**
（2026-02-13 一天改了 63 页，2026-03-16 改了 41 页）。

这是**事件驱动的批量维护**。所以：

> 页面两年没被编辑，最可能的解释是「wiki 认为这部分内容不需要改」，
> 而不是「内容悄悄过期了」。

因此 UI 里的陈旧告警被撤掉了，只保留溯源陈述。
这个脚本存在的意义就是**守住那个否定结论** ——
将来谁想再加陈旧告警，先跑一遍这个脚本看看数据变了没有。

## 口径

- 只读 `pages` 表，**不调 LLM、不调 embedding**，几秒钟跑完；
- 按 `game` 分库统计，避免混算；
- Gini 系数取「按日期分组的页面数」分布，0 = 完全均匀，1 = 全部集中在一天。
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

from sqlalchemy import text

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import get_settings  # noqa: E402
from app.store.db import session_scope  # noqa: E402
from exp_common import build_meta  # noqa: E402

EVAL_DIR = BACKEND.parent / "data" / "eval"


def gini(counts: list[int]) -> float:
    """基尼系数：0 = 完全均匀，接近 1 = 全部挤在一天。

    用它把「Top-N 占多少」这种口算变成一个能跨语料比较的标量。
    """
    vals = sorted(counts)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * cum) / (n * total) - (n + 1) / n


async def collect(game: str) -> list[tuple[str, str]]:
    async with session_scope(get_settings()) as s:
        rows = (
            await s.execute(
                text(
                    "SELECT title, last_revision_ts FROM pages "
                    "WHERE game = :g AND last_revision_ts <> ''"
                ),
                {"g": game},
            )
        ).all()
    return [(r[0], r[1]) for r in rows]


def analyze(rows: list[tuple[str, str]], top_n: int = 12) -> dict:
    by_day = collections.Counter(ts[:10] for _, ts in rows)
    by_month = collections.Counter(ts[:7] for _, ts in rows)
    n = len(rows)
    top = by_day.most_common(top_n)
    newest = max(by_day) if by_day else ""
    oldest = min(by_day) if by_day else ""
    share = sum(c for _, c in top) / n if n else 0.0
    g = gini(list(by_day.values()))
    return {
        "pages": n,
        "distinct_days": len(by_day),
        "pages_per_day": round(n / len(by_day), 2) if by_day else 0,
        "oldest_day": oldest,
        "newest_day": newest,
        f"top{top_n}_days": [{"day": d, "pages": c} for d, c in top],
        f"top{top_n}_share": round(share, 4),
        "gini_by_day": round(g, 4),
        "by_month": dict(sorted(by_month.items())),
        # 判定阈值写在这里而不是文档里：将来数据变了，结论会自己先变红，
        # 而不是等某人想起来去翻文档。
        "clustered": bool(share >= 0.30),
        "verdict": (
            "编辑呈事件驱动聚集（Top12 日期占 >= 30%），说明 wiki 是跟随版本更新"
            "批量维护的。**「页面久未编辑」推不出「内容已过时」**，它更可能意味着"
            "「wiki 认为这部分内容不需要改」。因此不得用修订时间做内容陈旧告警。"
            if share >= 0.30 else
            "编辑较分散（Top12 日期占比 < 30%），维护是零散的。"
            "此时「久未编辑」可能意味着「被遗忘」，用修订时间提示陈旧是有依据的 —— "
            "但仍需注意它只是页面级信号，不是内容级信号。"
        ),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", default="sr")
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--out", default="probe_revision_clustering.json")
    args = ap.parse_args()

    report = {}
    for game in [g.strip() for g in args.game.split(",") if g.strip()]:
        rows = await collect(game)
        if not rows:
            print(f"[{game}] 没有带 last_revision_ts 的页面，跳过")
            continue
        report[game] = analyze(rows, args.top_n)
        r = report[game]
        print(f"[{game}] {r['pages']} 页 / {r['distinct_days']} 个日期 "
              f"（{r['pages_per_day']} 页/天）")
        print(f"    Top{args.top_n} 日期占 {r[f'top{args.top_n}_share']:.1%}  "
              f"Gini={r['gini_by_day']}  clustered={r['clustered']}")
        for item in r[f"top{args.top_n}_days"][:5]:
            print(f"      {item['day']}  {item['pages']:3} 页")

    if not report:
        print("没有任何库可分析")
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"_meta": build_meta(args), "by_game": report}
    out = EVAL_DIR / args.out
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())
