"""增量变更检测：把「哪些页变了」变成一个可以回答的问题。

这是 `docs/delta_sync.md` 六步方案的第 2 步，依赖第 1 步的内容指纹。

## 为什么是三级筛，不是直接抓

`recentchanges` 能一次列出 500 条"被编辑过"的页面，成本极低。
但**被编辑过 ≠ 内容变了** —— wiki 上改错别字、调排版都会进这个列表。
所以不能拿着它就开抓，那等于花 WAF 配额去重抓没变的内容。

三级筛，每级都比上一级贵一个数量级，能不进下一级就不进：

| 级 | 手段 | 单次成本 | 覆盖量 |
|---|---|---|---|
| 1 | `recentchanges` 时间窗 | 1 请求 / 500 条 | 全站被编辑过的页 |
| 2 | 交集到本地收录 + `fetch_page_meta` 比对 `revision_ts` | 1 请求 / 20 页 | 只有本地有的页 |
| 3 | 取正文算指纹比对 | 1 请求 / 10 页（带正文） | 只有指纹真的变的页 |

第 2 级比对 `revision_ts` 的意义：它能挡掉**"编辑过但正文没变"**之外的
另一类噪音 —— 页面被重新保存、模板被触碰这类 revision 变了但内容没动的情况。
只有 revision 也变了的，才值得付取正文的代价。

## 三个必须显式说出来的边界

1. **窗口没翻完时，所有计数都是下界。**
   实测 7 天窗口在 WAF 下会被截断（`exhausted=false`），
   此时"变更页 148"是至少 148，不是恰好 148。
2. **本模块只检测，不重建。** 重建是第 3 步（按页先删后建），
   在那里才处理 chunk 级脏数据 —— 在这里重建就等于提前踩坑。
3. **指纹为空的本地页一律判为"变了"**，因为无法证明它没变。
   这会在第 1 步刚跑完、但还有漏网页面时造成一次性全量重建，属预期行为。
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.ingest.fingerprint import is_stale, page_hash
from app.ingest.wiki_client import PageContent, WAFBlocked, WikiClient
from app.store.db import init_db, session_scope
from app.store.models import Page

logger = logging.getLogger(__name__)

# 翻页中断的哨兵：**不是** None。
# None 在这个循环里表示「窗口已翻完」，用它表示中断会把一次失败的检测
# 报成一次完整扫描 —— 那比报错更危险。
_INTERRUPTED = "__waf_interrupted__"


@dataclass
class DeltaReport:
    """一次检测的结果。计数都是**下界**（窗口没翻完时）。"""

    game: str
    since: str
    changed: list[dict] = field(default_factory=list)
    revision_only: list[str] = field(default_factory=list)
    new_pages: list[str] = field(default_factory=list)
    local_pages: int = 0
    scanned_changes: int = 0
    unique_changed_site: int = 0
    intersect: int = 0
    requests_level1: int = 0
    requests_level2: int = 0
    requests_level3: int = 0
    window_exhausted: bool = True
    waf_hits: int = 0

    def as_dict(self) -> dict:
        return {
            "game": self.game,
            "since": self.since,
            "local_pages": self.local_pages,
            "scanned_changes": self.scanned_changes,
            "unique_changed_site": self.unique_changed_site,
            "intersect": self.intersect,
            "changed": self.changed,
            "changed_count": len(self.changed),
            "revision_only_count": len(self.revision_only),
            "new_pages_count": len(self.new_pages),
            "requests": {
                "level1_recentchanges": self.requests_level1,
                "level2_revision_meta": self.requests_level2,
                "level3_content": self.requests_level3,
            },
            "window_exhausted": self.window_exhausted,
            "lower_bound": not self.window_exhausted,
            "waf_hits": self.waf_hits,
        }


async def _local_pages(game: str, settings: Settings) -> dict[str, dict]:
    """本地收录的 title -> {page_id, content_hash, last_revision_ts}。

    用 MySQL 的 `pages` 表而不是 checkpoint：checkpoint 只是抓取时的
    "别重复抓"清单，它不记录指纹，回答不了"变了没有"。
    """
    out: dict[str, dict] = {}
    async with session_scope(settings) as session:
        rows = await session.execute(
            select(Page.title, Page.page_id, Page.content_hash, Page.last_revision_ts)
            .where(Page.game == game)
        )
        for title, pid, ch, ts in rows.all():
            out[title] = {"page_id": int(pid), "content_hash": ch or "", "revision_ts": ts or ""}
    return out


def _iso(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def detect_changes(
    game: str,
    since: datetime,
    *,
    settings: Settings | None = None,
    wiki: WikiClient | None = None,
    max_level1_pages: int = 8,
    max_level3_pages: int = 40,
    dry_run: bool = True,
) -> DeltaReport:
    """检测 `[since, now]` 区间内**内容实质变化**的页面。

    `dry_run=True` 时只检测不落任何状态 —— 步 3 才允许真正写入。
    """
    s = settings or get_settings()
    rep = DeltaReport(game=game, since=_iso(since))

    local = await _local_pages(game, s)
    rep.local_pages = len(local)
    if not local:
        logger.warning("%s 本地无收录页面，跳过检测", game)
        return rep

    client = wiki or WikiClient(game, base_url=s.wiki_base)
    owns = wiki is None

    try:
        # ---- 第 1 级：时间窗内被编辑过的页面 ----
        # `cursor is None` 的含义必须区分清楚：**「还没翻第一页」** vs
        # 「已经翻完，没有下一页」。混为一谈会让循环反复请求第一页 ——
        # 配额白花、scanned_changes 虚高，而且不报错。
        first_page = True
        cursor: str | None = None
        titles: set[str] = set()
        for _ in range(max_level1_pages):
            if first_page:
                first_page = False
                try:
                    rows, cursor = await client.recent_changes(since)
                except WAFBlocked as exc:
                    # 已翻到的页要留下，不能因为后面被拦就整堆丢掉 ——
                    # 实测撞过：翻页第 6 页吃满 4 次退避后抛异常，
                    # 前面 2500 条已读到的变更记录全部作废。
                    logger.warning("第 1 级中断（WAF）：%s", exc)
                    rep.waf_hits += 1
                    cursor = _INTERRUPTED
                    break
            elif cursor and cursor != _INTERRUPTED:
                try:
                    rows, cursor = await client.recent_changes_page(since, cursor)
                except WAFBlocked as exc:
                    logger.warning("第 1 级翻页中断（WAF）：%s", exc)
                    rep.waf_hits += 1
                    cursor = _INTERRUPTED
                    break
            else:
                cursor = None      # 没有续页游标 = 窗口已翻完
                break
            rep.requests_level1 += 1
            if not rows:
                cursor = None
                break
            rep.scanned_changes += len(rows)
            titles.update(r["title"] for r in rows)
        rep.window_exhausted = cursor is None
        rep.unique_changed_site = len(titles)

        # ---- 交集到本地收录 ----
        hit = [t for t in titles if t in local]
        rep.intersect = len(hit)
        if not rep.window_exhausted:
            logger.warning(
                "窗口未翻完（WAF 截断）：变更页 %d、需重建集合均为**下界**", rep.intersect,
            )
        if not hit:
            return rep

        # ---- 第 2 级：批量查 revision，不取正文 ----
        try:
            metas = await client.fetch_page_meta(hit, batch=20)
        except WAFBlocked as exc:
            logger.warning("第 2 级中断（WAF），本轮只能报第 1 级的结果：%s", exc)
            rep.waf_hits += 1
            metas = []
        rep.requests_level2 += max(1, -(-len(hit) // 20))

        suspects: list[tuple[str, PageContent]] = []
        for m in metas:
            info = local.get(m.title)
            if not info:
                continue
            if (m.last_revision or "") == info["revision_ts"] and info["content_hash"]:
                # revision 没变 -> 内容不可能变，连指纹都不用算
                continue
            suspects.append((m.title, m))

        # ---- 第 3 级：只有 revision 变了的才取正文算指纹 ----
        suspects = suspects[:max_level3_pages]
        fetched: list[PageContent] = []
        if suspects:
            try:
                fetched = await client.fetch_pages([t for t, _ in suspects], batch=10)
            except WAFBlocked as exc:
                # 同样的道理：第 3 级被拦不该让第 1/2 级的结果一起消失。
                # 取到一半也保留 —— fetch_pages 是按批次的，已成功的批次有效。
                logger.warning("第 3 级中断（WAF），只报告取到正文的那些页：%s", exc)
                rep.waf_hits += 1
            rep.requests_level3 += max(1, -(-len(suspects) // 10))

        for m in fetched:
            info = local.get(m.title)
            if not info:
                continue
            new_hash = page_hash(m.wikitext)
            if not is_stale(info["content_hash"], new_hash):
                # revision 变了但指纹没变：编辑过，内容没动（错别字/排版）
                rep.revision_only.append(m.title)
                continue
            rep.changed.append({
                "page_id": info["page_id"],
                "title": m.title,
                "old_hash": info["content_hash"],
                "new_hash": new_hash,
                "old_revision_ts": info["revision_ts"],
                "new_revision_ts": m.last_revision or "",
            })
    finally:
        if owns:
            await client.aclose()

    return rep


async def main(game: str, days: int) -> int:
    s = get_settings()
    if not await init_db(s):
        print("!! MySQL 不可用")
        return 1
    since = datetime.now(UTC) - __import__("datetime").timedelta(days=days)
    rep = await detect_changes(game, since, settings=s)
    d = rep.as_dict()
    print(f"== {game} 近 {days} 天 ==")
    print(f"  本地收录 {d['local_pages']} 页")
    print(f"  全站变更 {d['unique_changed_site']} 页（扫到 {d['scanned_changes']} 条）")
    print(f"  交集 {d['intersect']} 页")
    print(f"  revision 变了但内容没变 {d['revision_only_count']} 页")
    print(f"  **内容实质变化 {d['changed_count']} 页**")
    print(f"  请求量 {d['requests']}")
    if d["lower_bound"]:
        print("  ⚠️ 窗口未翻完，以上均为**下界**")
    for c in d["changed"][:10]:
        print(f"    {c['title'][:28]}  {c['old_hash']} -> {c['new_hash']}")

    # 产物必须落盘：这些数字要写进文档与简历，只留在终端输出里等于无法追溯。
    # 路径是 parents[3] 不是 parents[2]：本文件在 backend/app/ingest/ 下，
    # parents[2] 指向的是 backend/，产物会落到 backend/data/eval/ ——
    # 那个路径不匹配任何 .gitignore 规则，会被误提交进仓库。
    out = (Path(__file__).resolve().parents[3] / "data" / "eval"
           / f"delta_sync_{game}_{days}d.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物已写 {out.name}")
    return 0


if __name__ == "__main__":
    import sys

    g = sys.argv[1] if len(sys.argv) > 1 else "sr"
    dd = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    raise SystemExit(asyncio.run(main(g, dd)))
