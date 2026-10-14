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
    """确保会话行存在。返回 True=写成功，**False 只表示 MySQL 不可用**。

    不是"会话是否新建"：走的是 upsert 语义，每次都会执行一次写。
    这个区分对调用方很重要 —— `save_turn` 就故意不看返回值直接写 Message，
    靠的是外键/`session_key` 冗余列兜住，而不是靠这里失败时报错。
    """
    s = settings or get_settings()
    try:
        async with session_scope(s) as session:
            # on_duplicate_key_update 里给 game 赋它自己，等价于"存在就不动"：
            # 目的是让同一 session_key 被别的 game 复用时也不会被悄悄改写。
            ins = mysql_insert(Session).values(session_key=session_key, game=game)
            await session.execute(ins.on_duplicate_key_update(game=Session.game))
        return True
    except MySQLUnavailable:
        return False


async def load_history(session_key: str, limit: int = 8, settings: Settings | None = None) -> list[dict]:
    """取最近若干轮对话，供 Agent 的 history 用。

    只保留 user/assistant 两种角色：tool 消息不带 tool_call_id 重放给模型会直接 400。

    返回值**顺序敏感**：调用方直接拼进 messages 数组喂模型，
    顺序错了模型就会把"先问后答"读成倒叙 —— 所以最后那步 reversed 不能省。
    MySQL 不可用时返回空列表（而不是抛错）：多轮上下文是加分项，
    丢了它问答照样能用，只是退化成单轮。
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
    """把一轮 user/assistant 成对落库。True=已提交，False=MySQL 不可用。

    两个要点：
    - **两条消息在同一次 execute 里写入**，所以一定同生共死，
      不会出现"存了回答没存提问"导致历史错位的半截状态。
    - 正文截断到 8000 字符。截断是**静默**的，长对话回放时拿到的历史
      可能已在句中被切断；要复现原对话得另找途径，别以为库里存了全文。
    返回值只用于健康检查，不影响问答主流程。
    """
    s = settings or get_settings()
    try:
        # 不检查 ensure_session 的返回值：会话缺失时 Message 自己的 session_key
        # 足以让查询工作，这里少一次写、少一个失败点。
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

    **入参 `rows` 的键名是硬约定**，下面那段 dict 取值全靠字符串匹配：
    `f"recall@{k}"`、`mrr`、`ndcg`、`page_hit`、`latency_ms`、`top_titles`。
    任何一个键名对不上都会 `float(r.get(..., 0.0))` → 记成 **0.0**，
    不报错、不告警，落库后就是一个看起来很正常的"零分实验"。
    所以改评测脚本里的输出字段名时必须同步改这里；
    记 0 还是在记真实值，取决于键名匹配，这一点没有任何运行时保护。

    返回值 = 实际写入条数（`len(payload)`），调用方用它判断"这次到底落没落库"：
    0 既可能是没有结果，也可能是 MySQL 不可用。
    """
    s = settings or get_settings()
    if not await init_db(s):
        return 0
    try:
        async with session_scope(s) as session:
            payload = []
            # 逐行转成表字段。`k` 决定 recall_at_k 取哪一档的指标（默认 5），
            # 存进同一列的是"不同 k 混在一起"的结果 —— 所以对比实验时
            # 必须同时看 config 快照里的 k，别把 recall@3 和 recall@5 放一行比。
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
                # upsert 的意义是**重跑同一实验覆盖旧值**，而不是追加：
                # 同 case_id 重跑时靠唯一键定位，用新值整体替换（含 config）。
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

    ⚠️ 这条路径**不捕 MySQLUnavailable**，与同文件其他函数不同：
    它是一次性的运维/初始化动作，报错比静默返回更可取。
    代价是调用方必须自己准备好"数据库不存在"的情况。
    返回值直接等于用例条数（0 = 该游戏没定义评测集），
    调用方用它来发现"这个库压根没评测集"而不是"评测跑完了"。
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

    两个入参各自的含义（`backfill_corpus` 是唯一调用方）：
    - `page_ids`：本轮缓存目录里解析出的页面 id 列表，空 = 语料没找到；
    - `existing_pages`：库里该游戏已有的页数，用来区分
      "本来就没有数据"（无事可做）和"有数据却找不到语料"（危险）。
    返回 None 有两种含义：语料非空（正常回填），或两边都空（无害空跑）——
    调用方靠 `backfill_corpus` 里先判 `if not page_ids` 的顺序把两者区分开。
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

    # 三个计数只统计"**真正写进去**"的行，continue 掉的页不计入。
    # 所以 pages_w < len(page_ids) 是正常的（模板页、被跳过的页）。
    pages_w = chunks_w = facts_w = 0
    errors: list[str] = []
    skipped: list[str] = []
    # 本轮**实际产生**的 key 集合，剪枝时用它们当"应该存在的全集"。
    # 关键：这三个集合是后面 DELETE 的**唯一依据** ——
    # 循环一旦有页没跑成（errors 非空），该页的 chunk/fact 就会被当成残留删掉。
    # 这是"以当前语料为准"必须付出的代价，容忍不了部分失败，所以失败项会被显式返回。
    keep_chunk_ids: set[str] = set()
    keep_fact_keys: set[tuple] = set()
    keep_page_ids: set[int] = set()

    # 整个循环 + 剪枝在**同一个 session_scope** 里：一个事务。
    # 中途抛异常时前面所有 upsert 和后面的 delete 一起回滚，
    # 不会留下"写了一半、剪了一半"的中间态。
    async with session_scope(s) as session:
        # 终止条件由 `page_ids` 保证：它来自一次性目录快照，长度有限，
        # 循环内没有任何地方往里 append。三种 continue（复原失败/被过滤/无 chunk）都不会提前结束整批。
        for pid in page_ids:
            page = load_cached_page(pid, game)
            if page is None:
                errors.append(f"page {pid} 无法复原")
                continue
            # 与入库管线同一套判据：被过滤的模板页在 MySQL 里也不该有分片。
            # 注意 continue 发生在 keep_page_ids.add 之前，
            # 所以它既不会被写入，也不会被剪枝碰到（库里本来就没有它）。
            keep, why = should_index(page.title)
            if not keep:
                skipped.append(f"{page.title}（{why}）")
                continue
            url = f"{s.wiki_base}/{game}/{page.title}"
            # 下面这两个调用的参数必须与 `pipeline.Ingestor` 完全一致（同一个 cfg），
            # 否则 chunk_id / 切分边界会对不上，剪枝就会把向量库里真存在的分片当成残留删掉。
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
        # 剪枝的判据只有一条：**库里现有 id 不在本轮 keep 集合里**。
        # 所以它对 keep 集合的正确性零容错 —— 集合漏一个，就是删一条有效数据。
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
        # fact 的"身份"是三元组（类型+名称+属性），不是自增 id ——
        # 换版本后同一实体同一属性会被视为同一条而刷新，而不是新增。
        drop_facts = [tuple(r) for r in stale_facts if tuple(r) not in keep_fact_keys]
        # 逐条 DELETE 是因为 (entity_type, entity_name, attr) 上没有唯一索引，
        # 无法用一条 IN 批量。3688 条量级下这是最坏几百毫秒，可接受。
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

    # 返回值是 API `/api/backfill` 的响应体，同时也是人工核对回填是否正常的凭据。
    # 三个 pruned_* 是本次删除量，正常重跑应为 0；突然非 0 说明 page_filter 或
    # chunk_size 变了（或本轮有页复原失败，见 errors），要去看 skipped/errors 而不是直接接受。
    # 退出时 drop_chunks/drop_facts/drop_pages 仍在作用域内 —— 它们只在**有 page_ids 时**定义，
    # 所以函数开头的空语料早返回不能落到这里，否则会 NameError。
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

    两处结构上的要点：
    - `Run` 与 `ToolCallLog` 写在**同一个 session** 里，所以详情页不会读到
      "有运行记录但一条工具调用都没有"的半截数据。
    - `answer[:8000]` 是静默截断，跟消息表同一个口径；轨迹详情页拿到的
      长回答因此可能不完整，看轨迹时别把它当权威原文。
    """
    s = settings or get_settings()
    try:
        async with session_scope(s) as session:
            # config 只在有 guard 时落：没有护栏的那次运行存 `None` 而不是 `{}`，
            # 这样统计"开启护栏的比例"时分母才不含无关样本。
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
            # 工具调用是**纯 insert**：一次 run_id 的轨迹是完整的，
            # 重跑覆盖用的是上面 Run 的 upsert，这条依赖"要么整批写、要么都没写"，
            # 因此不能也换成 on_duplicate_key_update（那会把上次的步骤混进来）。
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
    """轨迹列表页数据：按时间倒序，question 截断到 120 字供列表展示。

    **刻意不返回 answer** —— 列表页不需要全文，少读一个大字段差别很明显。
    要正文走 `run_detail`。MySQL 不可用时返回空列表（列表页显示空态，不报错）。
    """
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
    """单次运行的完整轨迹。⚠️ 返回 None 有**两种**含义，调用方区分不了：

    1. 这个 run_id 不存在（用户点了个旧链接 / 数据被清）；
    2. MySQL 整体不可用。

    这里选择不抛异常，是因为前端只想渲染一个"没有该记录"的空态；
    代价是数据库挂了也被渲染成"记录不存在"，排查时要看日志里的 warning。
    `tool_calls` 按 step 正序，消费方（轨迹时间线）依赖这个顺序。
    """
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
