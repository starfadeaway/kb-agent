"""MySQL 持久化：结构化数据 + Agent 运行轨迹。

分工（这是整个存储设计的核心）：
- **向量库**管 chunk 正文与 embedding（语义召回）
- **MySQL** 管页面元数据、infobox 结构化数值、Agent 轨迹、评测结果（精确查询与可观测性）

两类查询走不同存储是刻意的：
"太空封印站几件套"要的是精确字段 → SQL；
"攻击力不够又想更快该带哪个"要的是语义 → 向量。
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert

from app.core.config import Settings, get_settings

# 这里原本还要 `from app.eval.cases import EvalCase as EvalCaseDef` ——
# 评测集定义（dataclass）与数据库表（SQLAlchemy model）都叫 EvalCase，
# 同名裸导入时后者会把前者顶掉，于是 `EvalCase(...)` 实际在构造 ORM 对象。
# 当时靠别名区分，后来 load_eval_cases()（唯一的用户）被删成死代码，
# 别名与它一起走了，**陷阱本身也一起消失了** —— 这比留一个未使用的别名更干净。
# 真要再从库里读评测集时，记得把别名加回来。
from app.ingest.fingerprint import chunk_hash, page_hash
from app.ingest.pipeline import IngestConfig, load_cached_page
from app.store.db import MySQLUnavailable, init_db, session_scope
from app.store.models import ChunkMeta, EvalResult, Fact, Message, Page, Run, Session, ToolCallLog
from app.store.models import EvalCase as EvalCaseRow

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 会话（多轮对话）
# --------------------------------------------------------------------------- #
async def ensure_session(session_key: str, game: str = "sr", settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    try:
        async with session_scope(s) as session:
            ins = mysql_insert(Session).values(session_key=session_key, game=game)
            await session.execute(ins.on_duplicate_key_update(game=Session.game))
        return True
    except MySQLUnavailable:
        return False


async def load_history(session_key: str, limit: int = 8, settings: Settings | None = None) -> list[dict]:
    """取最近若干轮对话，供 Agent 的 history 用。

    只保留 user/assistant 两种角色：tool 消息不带 tool_call_id 重放给模型会直接 400。
    """
    s = settings or get_settings()
    try:
        async with session_scope(s) as session:
            stmt = (
                select(Message)
                .where(Message.session_key == session_key, Message.role.in_(("user", "assistant")))
                .order_by(Message.id.desc())
                .limit(limit)
            )
            rows = list((await session.execute(stmt)).scalars())
    except MySQLUnavailable:
        return []
    # SQL 是倒序取出来的，回填给模型要按时间正序
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


async def save_turn(
    session_key: str,
    *,
    question: str,
    answer: str,
    refs: list | None = None,
    settings: Settings | None = None,
) -> bool:
    s = settings or get_settings()
    try:
        await ensure_session(session_key, settings=s)
        async with session_scope(s) as session:
            await session.execute(
                mysql_insert(Message).values([
                    {"session_key": session_key, "role": "user", "content": question[:8000]},
                    {"session_key": session_key, "role": "assistant", "content": answer[:8000],
                     "refs": refs or None},
                ])
            )
        return True
    except MySQLUnavailable:
        return False


# --------------------------------------------------------------------------- #
# 评测结果落库
# --------------------------------------------------------------------------- #
async def save_eval_run(
    *,
    exp_label: str,
    config: dict,
    rows: list[dict],
    settings: Settings | None = None,
) -> int:
    """把一次评测跑分写进 eval_results。

    落库而不只是写 JSON 文件，是为了能做回归：
    改了分块参数后跑一遍，直接对比两次的 recall@3 差了多少。
    """
    s = settings or get_settings()
    if not await init_db(s):
        return 0
    try:
        async with session_scope(s) as session:
            payload = []
            for r in rows:
                k = r.get("k", 5)
                payload.append({
                    "exp_label": exp_label,
                    "case_id": r.get("case_id", ""),
                    "config": config,
                    "retrieved_ids": r.get("top_titles", []),
                    "recall_at_k": float(r.get(f"recall@{k}", 0.0)),
                    "mrr": float(r.get("mrr", 0.0)),
                    "ndcg": float(r.get("ndcg", 0.0)),
                    "hit_rate": 1.0 if r.get("page_hit") else 0.0,
                    "latency_ms": int(r.get("latency_ms", 0)),
                })
            if payload:
                ins = mysql_insert(EvalResult).values(payload)
                await session.execute(
                    ins.on_duplicate_key_update(
                        recall_at_k=ins.inserted.recall_at_k,
                        mrr=ins.inserted.mrr,
                        ndcg=ins.inserted.ndcg,
                        hit_rate=ins.inserted.hit_rate,
                        latency_ms=ins.inserted.latency_ms,
                        config=ins.inserted.config,
                    )
                )
        return len(payload)
    except MySQLUnavailable as exc:
        logger.warning("评测结果未落库：%s", exc)
        return 0


async def save_eval_cases(game: str, settings: Settings | None = None) -> int:
    """把代码里定义的评测集 upsert 进 `eval_cases` 表，返回落库条数。

    单独抽出来是因为原先这段逻辑内嵌在 `run_eval.py` 的落库分支里，
    而它只处理 `args.game` —— 于是只有"正好被实验跑到过的那个库"才有评测集。
    实验 A–E 全跑 sr，于是 zzz / ys 的 24 条评测集永远进不了库，
    表就少了一大半，看起来像"评测集就这么大"。

    upsert 而非 insert：重复执行是幂等的，改了 gold 也会刷新。
    """
    from sqlalchemy.dialects.mysql import insert as mysql_insert

    from app.eval.cases import cases_for

    cases = cases_for(game)
    if not cases:
        return 0
    s = settings or get_settings()
    async with session_scope(s) as session:
        ins = mysql_insert(EvalCaseRow).values([c.as_dict() for c in cases])
        await session.execute(
            ins.on_duplicate_key_update(
                question=ins.inserted.question,
                category=ins.inserted.category,
                gold_answer=ins.inserted.gold_answer,
                gold_pages=ins.inserted.gold_pages,
                difficulty=ins.inserted.difficulty,
                notes=ins.inserted.notes,
            )
        )
    return len(cases)


# --------------------------------------------------------------------------- #
# 语料回填
# --------------------------------------------------------------------------- #
def cached_page_ids(game: str) -> list[int]:
    """扫描本地原始缓存。

    注意层级：persist.py 位于 backend/app/store/，所以项目根是 parents[3]，
    写成 parents[2] 会指到 backend/，data/ 在那里不存在，
    表现为"回填了 0 页且不报错"—— 这种静默失败最难查。
    """
    import re
    from pathlib import Path

    d = Path(__file__).resolve().parents[3] / "data" / "raw" / game
    if not d.is_dir():
        return []
    return sorted(
        int(m.group(1)) for m in (re.match(r"^(\d+)_.+\.txt$", f.name) for f in d.glob("*.txt")) if m
    )


def empty_corpus_refusal(page_ids: list[int], existing_pages: int, game: str) -> str | None:
    """语料为空但库里已有数据时，给出拒绝理由；不该拒绝时返回 None。

    抽成纯函数是为了能脱离数据库单测 —— 这个判断一旦失效，
    后果是整个库的语料被静默删光，所以它必须有测试守着，
    而不能只在跑真回填时才"顺便验证一下"。
    """
    if page_ids:
        return None
    if existing_pages:
        return (
            f"缓存目录里没有 {game} 的任何页面，但库里已有 {existing_pages} 页 —— "
            f"拒绝执行，以免剪枝把整库删光。先确认 data/raw/{game} 是否存在"
            f"（当前工作目录：{Path.cwd()}）"
        )
    return None


async def backfill_corpus(game: str = "sr", settings: Settings | None = None) -> dict:
    """把本地缓存的页面 / chunk / infobox 事实全量写入 MySQL。

    走 upsert（ON DUPLICATE KEY UPDATE），可以反复执行不产生重复。
    末尾还会**剪掉本轮语料不再产生的残留**，所以它是「以当前语料为准的全量替换」
    而不是「往里塞」。

    **必须复用入库时的同一套页面过滤器**，否则两个库对"收录哪些页"各说各话：
    `pipeline.Ingestor` 用 `should_index` 跳过模板定义页（`模板:邦布图鉴` 这种），
    而这里原先对每个缓存页都直接切 chunk —— 于是 MySQL 里会出现
    `sr-1273-*`（模板:遗器套装的分片）和一堆 `entity_type='#set:'` 的垃圾 fact，
    而这些页面在向量库里**根本不存在**。
    一边查 SQL 一边查向量，对不上时根本不知道该信哪个。

    注意：on_duplicate_key_update 里必须用 ORM 属性（如 Page.title），
    不能用 sqlalchemy.column("title") —— 后者没有表上下文，
    MySQL 会报 1052 Column 'title' in field list is ambiguous。
    """
    s = settings or get_settings()
    if not await init_db(s):
        return {"ok": False, "reason": "MySQL 不可用"}

    # **语料为空时必须拒绝执行，而不是"回填 0 页"**。
    #
    # 下面的剪枝逻辑是"以本轮语料为准的全量替换"：keep_* 三个集合为空时，
    # 库里现有的每一行都会被判成"残留"而删除。
    # 于是只要 cached_page_ids 返回空（cwd 不对、data/raw 被清、game 拼错），
    # 一次回填就会把这个 game 的 Page / ChunkMeta / Fact **全部删光**，
    # 而且照旧返回 ok=True、pages=0 —— 不报错、不告警、数据不可逆。
    #
    # 触发它不需要什么罕见条件，跑错一次工作目录就够了。
    page_ids = cached_page_ids(game)
    if not page_ids:
        async with session_scope(s) as probe:
            existing = (
                await probe.execute(select(func.count()).select_from(Page).where(Page.game == game))
            ).scalar() or 0
        refusal = empty_corpus_refusal(page_ids, existing, game)
        if refusal:
            return {"ok": False, "reason": refusal}
        return {"ok": True, "pages": 0, "note": f"缓存与库里都没有 {game}，无需回填"}

    cfg = IngestConfig.from_settings(game)
    from app.ingest.page_filter import should_index
    from app.ingest.pipeline import page_to_chunks, page_to_facts

    pages_w = chunks_w = facts_w = 0
    errors: list[str] = []
    skipped: list[str] = []
    # 本轮**实际产生**的 key 集合，剪枝时用它们当"应该存在的全集"。
    keep_chunk_ids: set[str] = set()
    keep_fact_keys: set[tuple] = set()
    keep_page_ids: set[int] = set()

    async with session_scope(s) as session:
        for pid in page_ids:
            page = load_cached_page(pid, game)
            if page is None:
                errors.append(f"page {pid} 无法复原")
                continue
            # 与入库管线同一套判据：被过滤的模板页在 MySQL 里也不该有分片。
            keep, why = should_index(page.title)
            if not keep:
                skipped.append(f"{page.title}（{why}）")
                continue
            url = f"{s.wiki_base}/{game}/{page.title}"
            chunks = page_to_chunks(page, cfg, url)
            facts = page_to_facts(page, cfg, url)

            # upsert 写法必须是两步：先拿到 ins，再用 ins.inserted.xxx 引用"新值"。
            # 直接在链式调用里写 ins.value 会 UnboundLocalError；
            # 写 Model.value 则生成 `table`.`col`，MySQL 读的是**旧值** ——
            # 两种写法都不报错，但数据永远不更新，非常难查。
            ins = mysql_insert(Page).values(
                game=game,
                page_id=page.page_id,
                title=page.title,
                url=url,
                last_revision=page.revision_id,
                last_revision_ts=page.last_revision or "",
                content_hash=page_hash(page.wikitext),
                n_chars=sum(c.char_len for c in chunks),
                n_chunks=len(chunks),
            )
            await session.execute(
                ins.on_duplicate_key_update(
                    title=ins.inserted.title,
                    url=ins.inserted.url,
                    last_revision=ins.inserted.last_revision,
                    last_revision_ts=ins.inserted.last_revision_ts,
                    content_hash=ins.inserted.content_hash,
                    n_chars=ins.inserted.n_chars,
                    n_chunks=ins.inserted.n_chunks,
                )
            )
            pages_w += 1
            keep_page_ids.add(page.page_id)

            if chunks:
                ins = mysql_insert(ChunkMeta).values(
                    [
                        {
                            "chunk_id": c.chunk_id, "game": game, "page_id": c.page_id,
                            "page_title": c.page_title, "section": c.section[:250],
                            "chunk_index": c.chunk_index, "n_chars": c.char_len,
                            "last_revision": c.last_revision,
                            "last_revision_ts": c.last_revision_ts or "",
                            "content_hash": chunk_hash(c.text),
                            "source_url": c.source_url, "truncated": int(bool(c.truncated)),
                        }
                        for c in chunks
                    ]
                )
                await session.execute(
                    ins.on_duplicate_key_update(
                        section=ins.inserted.section,
                        n_chars=ins.inserted.n_chars,
                        last_revision=ins.inserted.last_revision,
                        last_revision_ts=ins.inserted.last_revision_ts,
                        content_hash=ins.inserted.content_hash,
                        source_url=ins.inserted.source_url,
                    )
                )
                chunks_w += len(chunks)
                keep_chunk_ids.update(c.chunk_id for c in chunks)

            if facts:
                ins = mysql_insert(Fact).values(facts)
                await session.execute(
                    ins.on_duplicate_key_update(
                        value=ins.inserted.value,
                        value_num=ins.inserted.value_num,
                        source_url=ins.inserted.source_url,
                        last_revision_ts=ins.inserted.last_revision_ts,
                    )
                )
                facts_w += len(facts)
                keep_fact_keys.update(
                    (f["entity_type"], f["entity_name"], f["attr"]) for f in facts
                )

        # ---------------- 剪掉本轮语料不再产生的残留 ----------------
        # upsert 天然**不会删除**。改了 page_filter 或 chunk_size 再回填，
        # 旧行会永久留在库里：实测 sr 多出 26 个 chunk、3688 条 fact，
        # 样本是 `sr-1273-*` —— page 1273 是「模板:遗器套装」，
        # 向量库里早就没有它了，MySQL 里却还留着它的分片。
        #
        # 后果不只是"数字虚高"：MySQL 里会出现向量库里**根本不存在**的 chunk_id。
        # 一边查 SQL 一边查向量，对不上时根本不知道该信哪个。
        # 所以回填必须是"以当前语料为准的全量替换"，而不是"往里塞"。
        #
        # **必须留在这个 async with 里面**：`session_scope` 在块退出时就 commit
        # 并关闭 session，写在外面的话这些 DELETE 永远不会被提交 ——
        # 表现为"剪枝逻辑明明跑了、库里一行没少"，比不写还难查。
        stale_chunks = await session.execute(
            select(ChunkMeta.chunk_id).where(ChunkMeta.game == game)
        )
        drop_chunks = [cid for cid in stale_chunks.scalars() if cid not in keep_chunk_ids]
        if drop_chunks:
            await session.execute(
                delete(ChunkMeta).where(
                    ChunkMeta.game == game, ChunkMeta.chunk_id.in_(drop_chunks)
                )
            )

        stale_facts = (await session.execute(
            select(Fact.entity_type, Fact.entity_name, Fact.attr).where(Fact.game == game)
        )).all()
        drop_facts = [tuple(r) for r in stale_facts if tuple(r) not in keep_fact_keys]
        for et, en, at in drop_facts:
            await session.execute(
                delete(Fact).where(Fact.game == game, Fact.entity_type == et,
                                  Fact.entity_name == en, Fact.attr == at)
            )

        # 页级剪枝要放在 chunk/fact 之后：它们都带 page_id，
        # 先删页会让上面的比对失据（虽然这里没有外键约束，
        # 但顺序表达了"先删细节、再删主体"的意图）。
        stale_pages = (await session.execute(
            select(Page.page_id).where(Page.game == game)
        )).scalars().all()
        drop_pages = [pid for pid in stale_pages if pid not in keep_page_ids]
        if drop_pages:
            await session.execute(
                delete(Page).where(Page.game == game, Page.page_id.in_(drop_pages))
            )

    return {
        "ok": True,
        "pages": pages_w,
        "chunks": chunks_w,
        "facts": facts_w,
        "pruned_pages": len(drop_pages),
        "pruned_chunks": len(drop_chunks),
        "pruned_facts": len(drop_facts),
        # 不静默丢弃：跳过/错误都列出来，被过滤的模板页是预期行为，
        # 静默丢掉反而让人以为"回填漏了页面"。
        "skipped_pages": skipped[:10],
        "errors": errors[:10],
    }


# --------------------------------------------------------------------------- #
# Agent 运行轨迹
# --------------------------------------------------------------------------- #
async def save_run(state, settings: Settings | None = None) -> bool:
    """把一次 Agent 运行落库。

    MySQL 不可用时返回 False 而不是抛异常 —— 轨迹持久化是加分项，
    不该因为它让整个问答接口挂掉。
    """
    s = settings or get_settings()
    try:
        async with session_scope(s) as session:
            ins = mysql_insert(Run).values(
                run_id=state.run_id,
                session_key=state.session_key,
                question=state.question,
                answer=state.answer[:8000],
                game=state.game,
                model=s.deepseek_chat_model,
                steps=len(state.steps),
                prompt_tokens=state.prompt_tokens,
                completion_tokens=state.completion_tokens,
                tool_calls=len(state.steps),
                tool_failures=state.tool_failures,
                stop_reason=state.stop_reason,
                latency_ms=state.elapsed_ms,
                refs=state.refs or None,
                config={"guard": state.guard_stats()} if state.guard else None,
            )
            await session.execute(
                ins.on_duplicate_key_update(
                    answer=ins.inserted.answer,
                    steps=ins.inserted.steps,
                    latency_ms=ins.inserted.latency_ms,
                )
            )
            if state.steps:
                await session.execute(
                    mysql_insert(ToolCallLog).values(
                        [
                            {
                                "run_id": state.run_id, "step": st.step, "tool_name": st.tool_name,
                                "args": st.args, "ok": int(bool(st.ok)),
                                "error": st.error, "latency_ms": st.latency_ms,
                                "result_chars": st.result_chars,
                            }
                            for st in state.steps
                        ]
                    )
                )
        return True
    except MySQLUnavailable as exc:
        logger.warning("运行轨迹未落库（MySQL 不可用）：%s", exc)
        return False


async def recent_runs(limit: int = 20, settings: Settings | None = None) -> list[dict]:
    s = settings or get_settings()
    try:
        async with session_scope(s) as session:
            stmt = (
                select(Run)
                .order_by(Run.id.desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()
    except MySQLUnavailable:
        return []
    return [
        {
            "run_id": r.run_id, "game": r.game, "steps": r.steps,
            "tool_failures": r.tool_failures, "stop_reason": r.stop_reason,
            "latency_ms": r.latency_ms, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "question": r.question[:120], "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


async def run_detail(run_id: str, settings: Settings | None = None) -> dict | None:
    s = settings or get_settings()
    try:
        async with session_scope(s) as session:
            run = (await session.execute(select(Run).where(Run.run_id == run_id))).scalar_one_or_none()
            if run is None:
                return None
            calls = (
                await session.execute(
                    select(ToolCallLog).where(ToolCallLog.run_id == run_id).order_by(ToolCallLog.step)
                )
            ).scalars().all()
            return {
                "run_id": run.run_id, "question": run.question, "answer": run.answer,
                "game": run.game, "model": run.model, "stop_reason": run.stop_reason,
                "latency_ms": run.latency_ms, "usage": {
                    "prompt": run.prompt_tokens, "completion": run.completion_tokens,
                },
                "refs": run.refs or [], "config": run.config or {},
                "tool_calls": [
                    {"step": c.step, "tool": c.tool_name, "args": c.args, "ok": bool(c.ok),
                     "ms": c.latency_ms, "chars": c.result_chars, "error": c.error}
                    for c in calls
                ],
            }
    except MySQLUnavailable:
        return None
