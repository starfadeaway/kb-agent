"""版本感知检索：把「通用 RAG 在会过期的语料上会失效」这件事量化。

场景：wiki 每个页面记录的是**当前版本**的内容，但页面之间的修订时间
跨度极大 —— 本项目实测 sr 库 518 页横跨 2023-07 到 2026-10，
其中 2023/2024 年的 155 页（30%）是两年前的内容。

用户问的是**当前版本**的问题，检索却可能把两年前写的条目排进来。
本脚本量化两件事：

  1. 不加版本过滤时，top-5 里混入多少"陈旧页面"（修订时间早于阈值）；
  2. 加上 `min_revision_ts` 之后 top-5 变成什么，gold 命中率如何变化。

第 2 点必须**两个方向都报**：过滤掉陈旧页面会同时过滤掉 gold 页
（如果 gold 页本身就是旧的），只看"陈旧变少了"就宣布胜利是自欺欺人。

用法：
    .venv\\Scripts\\python.exe scripts\\probe_version_story.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import get_settings  # noqa: E402
from app.eval.cases import cases_for  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig  # noqa: E402
from app.store.vector_store import get_vector_store  # noqa: E402
from probe_io import save_probe  # noqa: E402

#: 阈值取年初：把 2025 及更早的内容全部排除在外。
CUTOFF = "2026-01-01T00:00:00Z"


def page_titles(game: str) -> dict[str, str]:
    """page_title → last_revision_ts，来自本地原始缓存的侧车元数据。"""
    out: dict[str, str] = {}
    for f in (ROOT / "data" / "raw" / game).glob("*.meta.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        ts = m.get("last_revision_ts") or ""
        if m.get("title"):
            out[m["title"]] = ts
    return out


def year_of(ts: str) -> str:
    return ts[:4] if ts else "?"


async def main() -> None:
    s = get_settings()
    store = get_vector_store(s)
    ts_by_title = page_titles("sr")
    retriever = HybridRetriever(settings=s, llm=get_llm(), store=store)

    cases = [c for c in cases_for("sr") if c.gold_pages]
    print(f"评测题 {len(cases)} 条（有 gold 的），版本阈值 {CUTOFF}\n")

    rows = []
    stale_without = stale_with = 0
    hit_without = hit_with = n = 0
    for case in cases:
        cfg_base = RetrievalConfig(game="sr", final_top_k=5)
        cfg_new = RetrievalConfig(game="sr", final_top_k=5, min_revision_ts=CUTOFF)

        h_base = await retriever.retrieve(case.question, cfg_base)
        h_new = await retriever.retrieve(case.question, cfg_new)

        # RetrievalHit 没有 page_title，标题挂在它带的 chunk 上。
        tb = [h.chunk.page_title for h in h_base]
        tn = [h.chunk.page_title for h in h_new]
        sb = [t for t in tb if year_of(ts_by_title.get(t, "")) != "?"
              and ts_by_title.get(t, "") < CUTOFF]
        sn = [t for t in tn if year_of(ts_by_title.get(t, "")) != "?"
              and ts_by_title.get(t, "") < CUTOFF]

        gold_hit_b = any(g in tb for g in case.gold_pages)
        gold_hit_n = any(g in tn for g in case.gold_pages)
        hit_without += gold_hit_b
        hit_with += gold_hit_n
        stale_without += len(sb)
        stale_with += len(sn)
        n += 1

        rows.append({
            "case_id": case.case_id,
            "gold_pages": case.gold_pages,
            "gold_year": [year_of(ts_by_title.get(g, "")) for g in case.gold_pages],
            "top5_without": [{"title": t, "rev": ts_by_title.get(t, "")} for t in tb],
            "top5_with": [{"title": t, "rev": ts_by_title.get(t, "")} for t in tn],
            "stale_count_without": len(sb),
            "stale_count_with": len(sn),
            "gold_hit_without": gold_hit_b,
            "gold_hit_with": gold_hit_n,
        })

        if sb or gold_hit_b != gold_hit_n:
            print(f"  {case.case_id}  陈旧页 {len(sb)}→{len(sn)}  "
                  f"gold {'✓' if gold_hit_b else '✗'}→{'✓' if gold_hit_n else '✗'}"
                  f"  gold 年份={year_of(ts_by_title.get(case.gold_pages[0],''))}")

    total_slots = n * 5
    print(f"\n=== 汇总（{n} 条题，top-5 共 {total_slots} 个槽位）===")
    print(f"  陈旧页面（修订早于 {CUTOFF[:4]}）占比："
          f"{stale_without / total_slots:.1%} → {stale_with / total_slots:.1%}")
    print(f"  gold 命中率：{hit_without}/{n} → {hit_with}/{n}")

    gain = hit_with - hit_without
    print(f"\n  → gold 变化 {gain:+d} 条。"
          + ("过滤确实换来了召回代价，必须如实报。" if gain < 0 else
             "过滤没有伤到召回。" if gain == 0 else "过滤反而提升了召回。"))

    save_probe(
        "probe_version_story",
        {
            "cutoff": CUTOFF,
            "n_cases": n,
            "stale_slot_share_without": round(stale_without / total_slots, 4),
            "stale_slot_share_with": round(stale_with / total_slots, 4),
            "gold_hit_without": hit_without,
            "gold_hit_with": hit_with,
            "rows": rows,
        },
        context={"game": "sr", "top_k": 5, "variable": "min_revision_ts",
                 "corpus_year_histogram": dict(Counter(year_of(v) for v in ts_by_title.values()))},
    )


if __name__ == "__main__":
    asyncio.run(main())