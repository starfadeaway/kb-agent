"""验证「版本感知检索」是真的能用，还是只存在于代码里。

背景：`RetrievalConfig` 有 `min_revision_ts` / `page_titles` 两个过滤维度，
一路接到 `store.build_where()`，两个 store 实现都写了。
但 `RetrieveRequest`（HTTP 入口）没有这两个字段 —— 也就是说
README 里"版本感知检索"这个卖点，**用接口根本调不到**，只有写 Python 才能用。

这里先用 Python 直接验证它确实能工作（否则不该往上暴露）。
如果通过，下一步才是把它加进 RetrieveRequest。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.retrieval.hybrid import RetrievalConfig, get_retriever


async def main() -> int:
    s = get_settings()
    r = get_retriever(s)
    cfg = RetrievalConfig(game="sr", final_top_k=5)

    base = await r.retrieve("三月七", cfg)
    print(f"基线（无过滤）           : {len(base)} 命中")
    if not base:
        print("✗ 基线就召不回，无法验证过滤")
        return 1

    revs = [h.chunk.metadata.get("last_revision_ts", "") for h in base]
    revs = sorted(x for x in revs if x)
    print(f"  命中页修订时间范围      : {revs[0]} .. {revs[-1]}")
    titles = [h.chunk.page_title for h in base]
    print(f"  命中页面: {titles}")

    # 1) min_revision_ts：设成比所有命中都新的时间戳，应当全部被过滤掉
    cutoff = "2099-01-01T00:00:00Z"
    cfg2 = RetrievalConfig(game="sr", final_top_k=5, min_revision_ts=cutoff)
    filtered = await r.retrieve("三月七", cfg2)
    print(f"\nmin_revision_ts={cutoff}: {len(filtered)} 命中")
    if filtered:
        print("✗ 设了未来时间戳仍有命中 —— 过滤没生效")
        for h in filtered:
            print(f"    {h.chunk.page_title} @ {h.chunk.metadata.get('last_revision_ts')}")
        return 1
    print("  ✓ 未来时间戳能过滤掉全部命中（过滤生效）")

    # 2) 放宽到 2000 年，应当能召回（证明不是"永远返回空"）
    cfg3 = RetrievalConfig(game="sr", final_top_k=5, min_revision_ts="2000-01-01T00:00:00Z")
    loose = await r.retrieve("三月七", cfg3)
    print(f"\nmin_revision_ts=2000     : {len(loose)} 命中")
    if not loose:
        print("✗ 放宽阈值后反而召不回 —— build_where 条件写错了")
        return 1
    print(f"  ✓ 放宽后能召回 {len(loose)} 条，命中: {[h.chunk.page_title for h in loose]}")

    # 3) page_titles：限定到具体页面，应当只剩那一页
    cfg4 = RetrievalConfig(game="sr", final_top_k=5, page_titles=[titles[0]])
    only = await r.retrieve("三月七", cfg4)
    print(f"\npage_titles=[{titles[0]}]: {len(only)} 命中 -> {[h.chunk.page_title for h in only]}")
    if any(h.chunk.page_title != titles[0] for h in only):
        print("✗ page_titles 过滤没有生效")
        return 1
    print("  ✓ page_titles 过滤生效")

    print("\n结论：版本感知检索在检索层是真实可用的，"
          "只是没有 HTTP 入口 —— 属于「实现了但没人能用」。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))