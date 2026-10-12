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
    """MediaWiki 把错误放在 query.errors 里，返回体依然是 200，不能只看状态码。"""
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
        """批量取页。MediaWiki 单次 query 支持多 title,分批以控制 URL 长度。"""
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
        return out
