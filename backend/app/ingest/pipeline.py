"""入库管线：抓取 → 清洗 → 分块 → 向量化 → 落库。

一次运行的配置快照会写进 MySQL 的 eval_runs.config，
保证"某个 chunk 是哪个版本的 chunker / 哪个 embedding 模型产出的"可追溯——
这是后面做单变量消融实验的前提，否则两组数据不可比。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings, get_settings
from app.ingest.chunker import Chunk, chunk_page, chunk_to_index_text
from app.ingest.page_filter import should_index
from app.ingest.template_parser import find_infobox
from app.ingest.wiki_client import PageContent, WikiClient
from app.llm.client import LLMClient, get_llm
from app.store.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class IngestStats:
    pages: int = 0
    chunks: int = 0
    facts: int = 0
    failed: list[str] = field(default_factory=list)
    waf_hits: int = 0
    #: 被页面过滤器跳过的标题（模板定义页等），便于事后核对"为什么少了这些页"
    skipped: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "pages": self.pages,
            "chunks": self.chunks,
            "facts": self.facts,
            "failed": self.failed,
            "waf_hits": self.waf_hits,
            "skipped": self.skipped,
            "elapsed_s": round(self.elapsed_s, 1),
        }


@dataclass
class IngestConfig:
    """入库配置。variant 决定写进哪个 collection（分块/embedding 实验要分库）。"""

    game: str = "sr"
    embed_model: str | None = None
    variant: str = "base"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embed_mode: str = "title"      # text | title（实验 B 的第二个变量，单独一维）
    embed_batch: int = 16

    @classmethod
    def from_settings(cls, game: str, **over) -> IngestConfig:
        s = get_settings()
        base = {
            "game": game,
            "embed_model": s.embed_model_main,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
        }
        # 显式传入的实验参数必须覆盖 .env 默认值，否则调参根本调不动
        base.update(over)
        return cls(**base)

    def as_dict(self) -> dict:
        return {
            "game": self.game,
            "embed_model": self.embed_model,
            "variant": self.variant,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embed_mode": self.embed_mode,
        }


def raw_cache_path(page: PageContent, game: str) -> Path:
    root = Path(__file__).resolve().parents[3] / "data" / "raw" / game
    root.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>|]', "_", page.title).strip()[:80]
    return root / f"{page.page_id}_{safe}.txt"


def raw_meta_path(page_id: int, game: str) -> Path:
    root = Path(__file__).resolve().parents[3] / "data" / "raw" / game
    return root / f"{page_id}.meta.json"


def cache_raw(page: PageContent, game: str) -> bool:
    """落原始 wikitext + 侧车元数据。返回**本次是否真的更新了缓存**。

    有了它，清洗/分块的任何改动都能离线重跑验证 ——
    既省 WAF 配额，也让"同一份数据在不同 chunk 参数下对比"成为可能。

    侧车文件单独存元数据，是因为正文文件名里只编码了 page_id 和标题，
    而 last_revision 是版本时效性设计的输入，不能丢。

    **必须是「存在则更新」，不是「不存在才写」。**
    早先写的是 `if not p.is_file(): p.write_text(...)`，
    看上去像「没有就写一份」，实际后果是**缓存一旦落盘就永不更新**：
    重抓到一个页面出了新版本（正文改了、last_revision 变了），
    缓存里仍停在第一次抓到的内容上。

    两个后果都很难在当场发现：
      1. 离线重建索引用的是旧正文 → MySQL（每次抓取都会 upsert）
         与向量库内容**分叉**，而索引看起来完全正常；
      2. `last_revision_ts` 停在旧值 —— 而它正是版本感知检索
         `min_revision_ts` 的输入，等于拿过期的时间戳做版本过滤。

    > 2026-10-13 实测核对：当前 1034 个可比页面，两处的 `revision_id` 与
    > `last_revision_ts` **完全一致，尚未分叉** —— 因为这批语料抓取后没有重抓过。
    > 所以这是**潜在缺陷，不是已发生的损失**。

    只保留最新版、无法对比历史版本，是本项目的既有取舍
    （见 `docs/version_staleness.md`：语料每页只有最新版，做不到版本对比）。

    返回是否变化，是为了让抓取器能统计「这轮真的更新了几页」——
    如果全程都是 False，说明这轮抓取没有任何实质进展。
    """
    changed = False

    p = raw_cache_path(page, game)
    if not p.is_file() or p.read_text(encoding="utf-8") != page.wikitext:
        p.write_text(page.wikitext, encoding="utf-8")
        changed = True

    meta_p = raw_meta_path(page.page_id, game)
    meta = json.dumps(
        {
            "page_id": page.page_id,
            "title": page.title,
            "last_revision": page.last_revision or "",
            "last_revision_ts": page.last_revision or "",
            "revision_id": page.revision_id,
        },
        ensure_ascii=False,
        indent=2,
    )
    if not meta_p.is_file() or meta_p.read_text(encoding="utf-8") != meta:
        meta_p.write_text(meta, encoding="utf-8")
        changed = True

    return changed


def load_cached_page(page_id: int, game: str) -> PageContent | None:
    """从本地缓存复原一个页面（离线重建索引用）。"""
    meta_p = raw_meta_path(page_id, game)
    if not meta_p.is_file():
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    safe = re.sub(r'[\\/:*?"<>|]', "_", meta["title"]).strip()[:80]
    txt_p = raw_meta_path(page_id, game).parent / f"{page_id}_{safe}.txt"
    if not txt_p.is_file():
        return None
    return PageContent(
        title=meta["title"],
        page_id=page_id,
        wikitext=txt_p.read_text(encoding="utf-8"),
        last_revision=meta.get("last_revision") or None,
        revision_id=int(meta.get("revision_id") or 0),
    )


def page_to_chunks(page: PageContent, cfg: IngestConfig, source_url: str) -> list[Chunk]:
    from app.ingest.template_parser import clean_wikitext

    text = clean_wikitext(page.wikitext)
    return chunk_page(
        page_id=page.page_id,
        page_title=page.title,
        text=text,
        game=cfg.game,
        last_revision=int(page.revision_id or 0),
        last_revision_ts=page.last_revision or "",
        source_url=source_url,
        chunk_size=cfg.chunk_size,
        overlap=cfg.chunk_overlap,
    )


#: facts 表各列的容量，直接从 ORM 模型读，不在这里另抄一份 ——
#: 抄一份的话，改了模型宽度这里不会跟着变，超长又是一次
#: "Data too long" 崩溃 + 半个库回填失败。
_FACT_COL_LEN: dict[str, int] = {}


def _fact_col_len(field: str, fallback: int) -> int:
    if not _FACT_COL_LEN:
        from app.store.models import Fact
        for c in Fact.__table__.c:
            length = getattr(c.type, "length", None)
            if isinstance(length, int):
                _FACT_COL_LEN[c.name] = length
    return _FACT_COL_LEN.get(field, fallback)


#: MediaWiki 模板名里夹带的注释。原神语料实测出现过：
#:   `多重系列任务<!-- 森林书等嵌套系列任务可将模板名改为多重系列任务 -->`
#: 这是编辑者写给人看的说明，不是类型名。原样入库有两个问题：
#: 1. 撑爆 `entity_type VARCHAR(32)`，整条 INSERT 失败 —— 已经因为它
#:    让原神整库回填崩过一次（MySQL 1406 Data too long）；
#: 2. 就算列够宽，存进去的也是「模板名+注释」这种没法当类型查的值。
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def _clean_entity_type(name: str) -> str:
    """模板名 → 可入库、可当类型查的实体类型。"""
    s = _COMMENT_RE.sub("", name or "")
    s = _flatten_template_name(s)
    s = s.strip().strip("|").strip()
    limit = _fact_col_len("entity_type", 32)
    return s[:limit] if s else ""


def _flatten_template_name(s: str) -> str:
    """`{{a|b}}` / `[[链接|文字]]` 取可读部分。

    注释剥掉之后偶尔还剩模板或链接语法 —— 实体类型应该是纯名字，
    留着 `{{...}}` 只会让唯一键 `uq_fact_key` 带上噪声。
    """
    for _ in range(3):          # 最多剥 3 层，够应付真实语料
        new = re.sub(r"\{\{[^{}]*\|([^{}|]*)}}\}", r"\1", s)
        new = re.sub(r"\[\[[^[\]|]*\|([^\[\]]*)]]", r"\1", new)
        new = re.sub(r"\{\{([^{}|]*)[|}\n]", r"\1", new)
        if new == s:
            break
        s = new
    # 正则只能成对吃掉 `}}`；`{{某某模板}}` 这种没管道参数的会剩一个孤儿 `}`。
    # 实体类型应该是纯名字，把残留的成对符号一并去掉。
    return s.replace("{", "").replace("}", "").replace("[", "").replace("]", "")


def page_to_facts(page: PageContent, cfg: IngestConfig, source_url: str) -> list[dict]:
    """从 infobox 抽结构化数值。抽不出来就返回空，不硬凑。

    值必须走和 clean_wikitext 同一套展平：否则 MySQL 里存的是
    「使装备者的攻击力提高{{颜色|描述2|12%}}」这种带模板语法的原始串，
    看着像结构化数据、实际没法当数值用 —— 向量库干净而 SQL 库是脏的，
    比两边都脏更糟，因为会让人误以为 SQL 侧已经可用。
    """
    from app.ingest.template_parser import _flatten_value

    tpl = find_infobox(page.wikitext)
    if tpl is None or not tpl.named:
        return []
    entity_name = (tpl.get("名称") or page.title).strip()
    entity_name = entity_name[:_fact_col_len("entity_name", 128)]
    entity_type = _clean_entity_type(tpl.name) or "未知"
    facts: list[dict] = []
    for attr, raw_val in tpl.named.items():
        val = _flatten_value(str(raw_val), 0).strip()
        if not val or len(val) > 300:
            continue
        facts.append(
            {
                "game": cfg.game,
                "entity_type": entity_type,
                "entity_name": entity_name,
                "attr": attr.strip()[:_fact_col_len("attr", 64)],
                "value": val,
                "value_num": _extract_number(val),
                "page_id": page.page_id,
                "source_url": source_url,
                "last_revision_ts": page.last_revision or "",
            }
        )
    return facts


def _extract_number(text: str) -> float | None:
    """能从文本里取到唯一数字就返回它，方便 SQL 侧直接做范围/排序查询。

    取到多个数字（比如 12% / 120 / 12%）就返回 None ——
    硬取第一个会造出错误数据，那比没有更糟。
    """
    nums = _NUM_RE.findall(text.replace(",", ""))
    if len(nums) != 1:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


class Ingestor:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.s = settings or get_settings()
        # 同 ReActAgent：兜底用单例，避免和 /chat 各持一份 embedding 缓存。
        self.llm = llm or get_llm(self.s)
        self.store = store or get_vector_store(self.s)

    async def ingest_pages(
        self, pages: Sequence[PageContent], cfg: IngestConfig, *, source_url_for=None
    ) -> IngestStats:
        stats = IngestStats()
        t0 = time.perf_counter()

        all_chunks: list[Chunk] = []
        for page in pages:
            # 模板定义页（`模板:邦布图鉴` 这种）不含游戏事实，只是 MediaWiki 语法。
            # 放进来会占 embedding 成本并挤召回位；跳过的标题记进 stats，
            # 不静默丢弃 —— 静默丢弃比丢弃本身更难查。
            keep, why = should_index(page.title)
            if not keep:
                stats.skipped.append(f"{page.title}（{why}）")
                continue
            url = source_url_for(page) if source_url_for else f"{self.s.wiki_base}/{cfg.game}/{page.title}"
            try:
                cache_raw(page, cfg.game)
                chunks = page_to_chunks(page, cfg, url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("分块失败 %s: %s", page.title, exc)
                stats.failed.append(page.title)
                continue
            stats.pages += 1
            stats.chunks += len(chunks)
            stats.facts += len(page_to_facts(page, cfg, url))
            all_chunks.extend(chunks)

        if not all_chunks:
            stats.elapsed_s = time.perf_counter() - t0
            return stats

        index_texts = [chunk_to_index_text(c, cfg.embed_mode) for c in all_chunks]
        vectors = await self.llm.embed(index_texts, model=cfg.embed_model, batch_size=cfg.embed_batch)

        metadatas = [
            {
                "game": c.game,
                "page_id": int(c.page_id),
                "page_title": c.page_title,
                "section": c.section,
                "chunk_index": int(c.chunk_index),
                "last_revision": int(c.last_revision),
                "last_revision_ts": c.last_revision_ts or "",
                "source_url": c.source_url,
                "truncated": int(bool(c.truncated)),
            }
            for c in all_chunks
        ]

        # 分批 upsert：一次几千条会把底层存储的写入缓冲顶到内存峰值
        for i in range(0, len(all_chunks), 256):
            sl = slice(i, i + 256)
            self.store.upsert(
                game=cfg.game,
                model=cfg.embed_model,
                variant=cfg.variant,
                ids=[c.chunk_id for c in all_chunks[sl]],
                embeddings=vectors[sl],
                documents=index_texts[sl],
                metadatas=metadatas[sl],
            )

        stats.elapsed_s = time.perf_counter() - t0
        return stats

    async def crawl_category(
        self,
        category: str,
        cfg: IngestConfig,
        *,
        limit: int = 60,
        on_progress=None,
    ) -> IngestStats:
        """抓一个分类下的页面并入库。

        抓取请求严格串行（节流器保证间隔），只对 embedding 走批量并发 —— 
        抓取并发只会招来 WAF，不会更快。
        """
        stats = IngestStats()
        t0 = time.perf_counter()
        async with WikiClient(
            cfg.game,
            base_url=self.s.wiki_base,
            min_interval=self.s.wiki_min_interval,
            max_retry=self.s.wiki_max_retry,
        ) as wc:
            titles = await wc.list_category_pages(category, limit=limit)
            logger.info("分类 %s 取到 %d 个页面", category, len(titles))
            if on_progress:
                on_progress({"stage": "listed", "category": category, "titles": len(titles)})

            pages: list[PageContent] = []
            for title in titles:
                try:
                    pages.append(await wc.fetch_page(title))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("抓取失败 %s: %s", title, exc)
                    stats.failed.append(title)
                    continue
                if on_progress:
                    on_progress({"stage": "fetched", "title": title, "done": len(pages)})
            stats.waf_hits = wc.waf_hits

        sub = await self.ingest_pages(
            pages, cfg, source_url_for=lambda p: f"{self.s.wiki_base}/{cfg.game}/{p.title}"
        )
        sub.failed = stats.failed + sub.failed
        sub.waf_hits = stats.waf_hits
        sub.elapsed_s = time.perf_counter() - t0
        return sub


async def smoke_ingest(game: str = "sr", limit: int = 3) -> dict:
    """端到端冒烟：抓几页 → 入库 → 用一句查询验证能召回。"""
    from app.core.config import get_settings as gs

    s = gs()
    cfg = IngestConfig.from_settings(game)
    ing = Ingestor(settings=s)
    async with WikiClient(
        game, base_url=s.wiki_base, min_interval=s.wiki_min_interval, max_retry=s.wiki_max_retry
    ) as wc:
        titles = await wc.search("刻晴", limit=limit) or ["刻晴"]
        pages = [await wc.fetch_page(t) for t in titles]
    stats = await ing.ingest_pages(pages, cfg)
    q = await ing.llm.embed(["刻晴的元素属性是什么"], model=cfg.embed_model)
    hits = ing.store.query(
        query_embedding=q[0], game=game, model=cfg.embed_model, variant=cfg.variant, top_k=5
    )
    await ing.llm.aclose()
    return {
        "ingest": stats.as_dict(),
        "collection": ing.store.collection_name(game, cfg.embed_model, cfg.variant),
        "count": ing.store.count(game, cfg.embed_model, cfg.variant),
        "hits": [
            {"score": h.score, "page": h.page_title, "section": h.metadata.get("section"), "text": h.text[:120]}
            for h in hits
        ],
    }


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(asyncio.run(smoke_ingest()), ensure_ascii=False, indent=2))