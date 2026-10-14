"""biligame MediaWiki 抓取客户端。

关键约束(实测得出):
- biligame 由腾讯云 EdgeOne 防护,密集请求会返回 567 挑战页而非数据。
- 挑战页是 HTML(Content-Type: text/html),正常 API 响应是 application/json。
  因此不能用「拿到响应」判定成功,必须按 Content-Type + 内容特征区分。
- 限制在所有请求之间维持最小间隔,配合 Cookie 会话保持。

设计要点:
- 全站共享一个节流器(按 host),避免多协程各自绕过间隔。
- WAF 拦截不计入普通重试,走更长的退避,因为它通常需要"放慢"而非"重试"。
- 页面正文与其修订时间戳一并返回 —— 时间戳是版本时效性方案的输入。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# EdgeOne 拦截页的特征串,任一命中即判定为「未拿到真实数据」
_WAF_MARKERS = (
    "567",
    "EdgeOne",
    "__tst_status",
    "请求已被站点的安全策略拦截",
    "Restricted Access",
)


class WAFBlocked(RuntimeError):
    """被 EdgeOne 拦截。与普通 HTTP 错误区分,因为处置策略不同。"""


class WikiError(RuntimeError):
    """wiki 侧的确定性错误(页面不存在、参数错误等)。"""


def data_has_error(payload: dict[str, Any]) -> bool:
    """MediaWiki 把错误放在 query.errors 里，返回体依然是 200，不能只看状态码。

    只有 `list_category_pages` 在用这个判定：分类不存在时 API 照样返 200 +
    `query.errors`，不查就会把「分类写错了」当成「这个分类是空的」。
    其他方法靠 MediaWiki 的 `pages[].missing` 字段识别不存在的页面，
    没有再查一遍 errors —— 依赖的是 formatversion=2 的 missing 约定。
    """
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("error") or (payload.get("query") or {}).get("errors"))


@dataclass(slots=True)
class PageContent:
    """一个 wiki 页面的原始内容。"""

    title: str
    page_id: int
    wikitext: str
    last_revision: str | None
    """ISO8601 UTC 时间戳,例如 2026-07-15T04:48:04Z。版本时效性过滤依赖此字段。"""
    revision_id: int = 0
    """MediaWiki 修订号(单调递增),用于判断"这条内容比某版本新还是旧"。"""

    @property
    def last_revision_ts(self) -> float:
        """修订时间的 Unix 时间戳;解析失败返回 0。"""
        if not self.last_revision:
            return 0.0
        try:
            from datetime import datetime

            return datetime.strptime(
                self.last_revision, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=None).timestamp()
        except (ValueError, TypeError):
            return 0.0


class _Throttle:
    """按 host 维护最小请求间隔的节流器。"""

    def __init__(self, min_interval: float) -> None:
        self._min = min_interval
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, host: str) -> asyncio.Lock:
        if host not in self._locks:
            self._locks[host] = asyncio.Lock()
        return self._locks[host]

    async def wait(self, host: str) -> None:
        """排队 + 睡到距上次请求满 min_interval。

        锁是**按 host** 的，且只在 sleep 期间持有：多个协程会依次算出
        各自的等待时间，而不是一起冲出去。用 `time.monotonic()` 而不是
        `time.time()`，是为了不受系统时间被调整（NTP 校时）的影响 ——
        用墙钟的话一次校时就可能让间隔瞬间变成 0。
        """
        async with self._lock(host):
            delta = time.monotonic() - self._last.get(host, 0.0)
            if delta < self._min:
                await asyncio.sleep(self._min - delta)
            self._last[host] = time.monotonic()


class WikiClient:
    """biligame 单站点的异步客户端。"""

    def __init__(
        self,
        slug: str,
        base_url: str = "https://wiki.biligame.com",
        min_interval: float = 2.0,
        max_retry: int = 4,
        timeout: float = 20.0,
        waf_backoff_base: float = 20.0,
    ) -> None:
        self.slug = slug
        self.api = f"{base_url}/{slug}/api.php"
        self.index = f"{base_url}/{slug}/index.php"
        self._throttle = _Throttle(min_interval)
        self._max_retry = max_retry
        self.waf_backoff_base = waf_backoff_base
        """WAF 退避基数。实测密集请求会被连续拦截,20s 起跳比 8s 稳定。"""
        self._client = httpx.AsyncClient(
            # 「保持 Cookie」这一条没有显式代码，靠的就是复用同一个 client：
            # httpx 的 cookie jar 会把挑战页下发的 cookie 带到下一次请求。
            # 每次现 new 一个 client 就等于每次都从零开始，518 页 0 拦截那组
            # 参数也就白测了 —— 所以抓取全程必须共用一个 WikiClient 实例。
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
        )
        self.waf_hits = 0
        """累计被拦截次数,用于观察限流强度。"""

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> WikiClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # 底层请求
    # ------------------------------------------------------------------ #

    async def _get(self, url: str, params: dict[str, str]) -> Any:
        """发起一次受节流保护的请求,返回解析后的 JSON。

        遇到 WAF 拦截时以更长的退避重试 —— 拦截通常意味着「请求太快」,
        而不是「这一次运气不好」,所以重试间隔要显著长于普通错误。
        """
        host = httpx.URL(url).host or ""
        last_exc: Exception | None = None

        # 进入本循环之前，调用方要清楚两件事：
        # 1. 间隔不由本函数决定 —— 节流器的 min_interval 来自构造参数，
        #    库内实测值是 6s（CLI `--interval 6` / WIKI_MIN_INTERVAL），
        #    WikiClient 的默认值 2.0 只是兜底，不是实测值。
        # 2. 最坏耗时由 max_retry × 退避基数决定：WAF 连续命中时
        #    单次请求最坏要等 20+40+80 ≈ 140s 才抛异常，所以
        #    上层（crawl / delta_sync）必须能容忍这种阻塞。
        for attempt in range(self._max_retry):
            await self._throttle.wait(host)
            try:
                resp = await self._client.get(url, params=params)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                await self._sleep_backoff(attempt, base=1.0)
                continue

            # WAF 拦截:返回的是挑战页(常见状态码 567),不是 API 数据
            if self._is_waf(resp):
                self.waf_hits += 1
                logger.warning(
                    "WAF 拦截(第 %d 次,status=%s),slug=%s,放慢后重试",
                    self.waf_hits,
                    resp.status_code,
                    self.slug,
                )
                await self._sleep_backoff(attempt, base=self.waf_backoff_base)
                continue

            # 注意这里**不** return 也不 raise：拦截是「没拿到数据」，
            # 不是「这个页面不存在」。真正的失败信号在循环跑完之后
            # 统一以 WAFBlocked 抛出，上层据此把本轮记成中断而不是 0 结果。

            if resp.status_code == 404:
                raise WikiError(f"404 Not Found: {url} params={params}")

            if resp.status_code >= 500:
                last_exc = WikiError(f"HTTP {resp.status_code}")
                await self._sleep_backoff(attempt, base=2.0)
                continue

            try:
                return resp.json()
            except ValueError:
                last_exc = WikiError(
                    f"响应不是 JSON(可能仍是挑战页): {resp.text[:120]!r}"
                )
                await self._sleep_backoff(attempt, base=4.0)

        raise WAFBlocked(
            f"重试 {self._max_retry} 次仍未取得数据,slug={self.slug},url={url}"
        ) from last_exc

    @staticmethod
    def _is_waf(resp: httpx.Response) -> bool:
        """判定「拿到响应但不是数据」。

        两种踩过的坑：
        1. EdgeOne 有时把挑战页标成 application/json，只看 Content-Type 会漏判，
           所以状态码 567 直接判死；
        2. 更早的一次误判是「567 = 站点不存在」，实际它只是限流，
           所以这里绝不能把拦截当成 404 处理。
        """
        if resp.status_code == 567:
            return True
        head = resp.text[:2000]
        ctype = resp.headers.get("content-type", "")
        if "json" in ctype:
            try:
                payload = resp.json()
            except ValueError:
                return True
            if isinstance(payload, dict) and ("query" in payload or "batchcomplete" in payload):
                return False  # 真 API 响应
        return any(m in head for m in _WAF_MARKERS)

    @staticmethod
    async def _sleep_backoff(attempt: int, base: float) -> None:
        """指数退避 + 抖动,避免多协程同步重试。"""
        delay = base * (2**attempt) + random.uniform(0, 1.0)
        await asyncio.sleep(delay)

    # ------------------------------------------------------------------ #
    # 业务方法
    # ------------------------------------------------------------------ #

    async def site_info(self) -> dict[str, Any]:
        """站点信息,用于确认 slug 指向的是正确的游戏站。"""
        data = await self._get(
            self.api, {"action": "query", "meta": "siteinfo", "siprop": "general", "format": "json"}
        )
        return data.get("query", {}).get("general", {})

    async def list_categories(self, limit: int = 50) -> list[dict[str, Any]]:
        """顶层分类列表,含页面数,用于评估语料规模与结构。"""
        data = await self._get(
            self.api,
            {"action": "query", "list": "allcategories", "aclimit": str(limit), "acprop": "size", "format": "json"},
        )
        return data.get("query", {}).get("allcategories", [])

    async def list_category_pages(
        self, category: str, limit: int = 500, offset: int = 0
    ) -> list[str]:
        """列出某分类下的页面标题。

        这个 wiki 的分类命名空间前缀是中文「分类:」而不是「Category:」，
        所以两种前缀都要试 —— 只写一种会静默返回空列表，
        看起来像"这个分类没内容"，实际是路径错了。
        """
        raw = category.split(":", 1)[1] if ":" in category else category
        prefixes = (category,) if ":" in category else ("分类:", "Category:")
        last: dict[str, Any] = {}

        # **本方法只取一页，不翻页。** 匿名用户的 cmlimit 上限是 500，
        # 传再大的 limit 也会被服务端截断，所以返回的标题集合
        # 在大分类下是**下界而不是全集** —— 上层用它估规模没问题，
        # 拿它当"这个分类就这些页"就会漏抓且不报错。
        for prefix in prefixes:
            data = await self._get(
                self.api,
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": f"{prefix}{raw}",
                    "cmlimit": str(limit),
                    "cmoffset": str(offset),
                    "format": "json",
                },
            )
            members = data.get("query", {}).get("categorymembers", [])
            last = data
            if members:
                return [m["title"] for m in members]

        if data_has_error(last):
            raise WikiError(f"分类不存在或前缀不对: {category}")
        return []


    async def page_categories(self, title: str) -> list[str]:
        """查页面所属分类，用来定位「角色」这类数据到底挂在哪个分类下。"""
        data = await self._get(
            self.api,
            {
                "action": "query",
                "prop": "categories",
                "cllimit": "50",
                "titles": title,
                "format": "json",
                "formatversion": "2",
            },
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return []
        return [c["title"].replace("Category:", "") for c in pages[0].get("categories", [])]

    async def fetch_page_meta(self, titles: list[str], batch: int = 20) -> list[PageContent]:
        """只取页面 id / 标题 / 修订信息，不要正文。

        用途：补齐缓存的侧车元数据。正文已经在本地，
        为此重新下载全文既慢又占 WAF 配额。
        """
        out: list[PageContent] = []
        for i in range(0, len(titles), batch):
            chunk = titles[i : i + batch]
            data = await self._get(
                self.api,
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "ids|timestamp",
                    "titles": "|".join(chunk),
                    "format": "json",
                    "formatversion": "2",
                },
            )
            for page in data.get("query", {}).get("pages", []):
                if page.get("missing"):
                    continue
                revs = page.get("revisions") or []
                if not revs:
                    continue
                out.append(
                    PageContent(
                        title=page.get("title", ""),
                        page_id=page.get("pageid", 0),
                        wikitext="",
                        last_revision=revs[0].get("timestamp"),
                        revision_id=int(revs[0].get("revid") or 0),
                    )
                )
        # 出口的集合**可能短于入参**：页面被删/被保护/无修订都会被 continue 掉，
        # 且不报错。delta_sync 靠 `local.get(m.title)` 兜底跳过这些页，
        # 所以最终 changed 少掉的部分会被当成"没变"而不是"查不到"。
        return out

    async def fetch_meta_by_ids(self, page_ids: list[int], batch: int = 20) -> list[PageContent]:
        """按 **page_id** 取元数据，不要标题。

        为什么必须走 id 而不是标题：本地缓存的文件名把标题里的非法字符替换成了 `_`
        （`raw_cache_path` 里的 `re.sub(r'[\\/:*?"<>|]', "_", ...)`），所以
        `模板:遗器套装` 落盘成 `1273_模板_遗器套装.txt`。拿文件名反推标题再按标题查，
        对任何标题里带 `:` 的页面都会查不到 —— 补元数据时静默写 0 份，
        看着像"没有缺失"，实际是查错了键。page_id 是落盘时保留下来的唯一无歧义标识。
        """
        out: list[PageContent] = []
        for i in range(0, len(page_ids), batch):
            chunk = [str(p) for p in page_ids[i : i + batch]]
            data = await self._get(
                self.api,
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "ids|timestamp",
                    "pageids": "|".join(chunk),
                    "format": "json",
                    "formatversion": "2",
                },
            )
            for page in data.get("query", {}).get("pages", []):
                if page.get("missing") or page.get("invalid"):
                    continue
                revs = page.get("revisions") or []
                if not revs:
                    continue
                out.append(
                    PageContent(
                        title=page.get("title", ""),
                        page_id=page.get("pageid", 0),
                        wikitext="",
                        last_revision=revs[0].get("timestamp"),
                        revision_id=int(revs[0].get("revid") or 0),
                    )
                )
        return out

    async def recent_changes(
        self, since: str, *, limit: int = 500, namespace: int = 0,
    ) -> tuple[list[dict], str | None]:
        """列出 `since` 之后被改动过的页面，返回 (记录列表, 续页游标)。

        增量同步的第一级筛子：一次请求能列出整个时间窗内被编辑过的页面，
        而不必逐页去问。实测它能穿过 EdgeOne（`probe_delta_sync.py` 阶段一）。

        **它给的是"被编辑过"，不是"内容变了"** —— wiki 上改个错别字、
        调一下排版都会进这个列表。真正的内容判定交给第二级（比对指纹），
        两级分工是省 WAF 配额的关键：
        第一级便宜（每请求 500 条），第二级贵（要取正文）。

        `since` 用 MediaWiki 的时间戳格式（`2026-10-06T00:00:00Z`），
        配合 `rcdir=older` 表示"从该时刻往回翻"。

        翻页用返回的游标；**游标非空说明窗口没翻完**，调用方必须把它当
        下界而不是精确值 —— 实测 7 天窗口在 WAF 下会被截断。
        """
        data = await self._get(
            self.api,
            {
                "action": "query",
                "list": "recentchanges",
                "rcnamespace": str(namespace),
                "rcstart": since,
                "rcdir": "older",
                "rclimit": str(limit),
                "rcprop": "title|timestamp|ids",
                "format": "json",
                "formatversion": "2",
            },
        )
        rows = data.get("query", {}).get("recentchanges", [])
        cursor = data.get("continue", {}).get("rccontinue")
        return rows, cursor

    async def recent_changes_page(
        self, since: str, cursor: str, *, limit: int = 500, namespace: int = 0,
    ) -> tuple[list[dict], str | None]:
        """`recent_changes` 的翻页版。"""
        data = await self._get(
            self.api,
            {
                "action": "query",
                "list": "recentchanges",
                "rcnamespace": str(namespace),
                "rcstart": since,
                "rcdir": "older",
                "rclimit": str(limit),
                "rcprop": "title|timestamp|ids",
                "rccontinue": cursor,
                "format": "json",
                "formatversion": "2",
            },
        )
        rows = data.get("query", {}).get("recentchanges", [])
        # 翻页版的出口与首页版同义：cursor 非空 = 还没翻完。
        # 调用方（delta_sync）必须自己决定翻到哪儿为止，不能把
        # 「拿到一页」当成「窗口已遍历完」—— 后者会把下界当结论。
        return rows, data.get("continue", {}).get("rccontinue")

    async def search(self, keyword: str, limit: int = 10) -> list[str]:
        data = await self._get(
            self.api,
            {
                "action": "query",
                "list": "search",
                "srsearch": keyword,
                "srlimit": str(limit),
                "format": "json",
            },
        )
        return [h["title"] for h in data.get("query", {}).get("search", [])]

    async def fetch_page(self, title: str) -> PageContent:
        """取页面 wikitext 与最后修订时间。"""
        data = await self._get(
            self.api,
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content|timestamp|ids",
                "rvslots": "main",
                "titles": title,
                "format": "json",
                "formatversion": "2",
            },
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            raise WikiError(f"页面不存在: {title}")

        page = pages[0]
        revs = page.get("revisions") or []
        if not revs:
            raise WikiError(f"页面无修订记录: {title}")

        rev = revs[0]
        wikitext = (rev.get("slots", {}).get("main", {}) or {}).get("content", "")
        return PageContent(
            title=page.get("title", title),
            page_id=page.get("pageid", 0),
            wikitext=wikitext,
            last_revision=rev.get("timestamp"),
            revision_id=int(rev.get("revid") or 0),
        )

    async def fetch_pages(self, titles: list[str], batch: int = 10) -> list[PageContent]:
        """批量取页。MediaWiki 单次 query 支持多 title,分批以控制 URL 长度。

        `batch=10` 不是随手取的整数：它和请求间隔 6s 是一组实测参数
        （518 页 / 0 次拦截，见 `crawl_all.py` 顶部）。一次塞更多页
        会让单请求体积变大，EdgeOne 的判定阈值比间隔更早触发 ——
        实测对策优先级是「批量取页 > 拉长间隔 > 退避重试」。

        与 `fetch_page_meta` 的 batch=20 的差别：这里带正文（rvprop 含
        content），响应体大得多，所以批量更小。
        """
        out: list[PageContent] = []
        for i in range(0, len(titles), batch):
            chunk = titles[i : i + batch]
            data = await self._get(
                self.api,
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "content|timestamp|ids",
                    "rvslots": "main",
                    "titles": "|".join(chunk),
                    "format": "json",
                    "formatversion": "2",
                },
            )
            for page in data.get("query", {}).get("pages", []):
                if page.get("missing"):
                    continue
                revs = page.get("revisions") or []
                if not revs:
                    continue
                wikitext = (revs[0].get("slots", {}).get("main", {}) or {}).get("content", "")
                out.append(
                    PageContent(
                        title=page.get("title", ""),
                        page_id=page.get("pageid", 0),
                        wikitext=wikitext,
                        last_revision=revs[0].get("timestamp"),
                        revision_id=int(revs[0].get("revid") or 0),
                    )
                )
        # 出口可能短于入参：missing / 无修订的页被 continue 掉且不报错。
        # delta_sync 第 3 级直接拿它算指纹，缺的页会静默不进 changed；
        # 那级带 WAF 计数，看 requests_level3 与 waf_hits 才分得清
        # 「内容没变」和「压根没取到」。
        return out
