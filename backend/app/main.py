"""FastAPI 入口。

设计取向：接口先于前端交付。每个端点都能用 curl 直接验证，
面试演示时不用依赖前端起服务。
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.react import UNCOVERED_GAMES, ReActAgent
from app.agent.tools import ToolContext, get_schemas
from app.core.config import get_settings
from app.llm.client import get_llm
from app.retrieval.hybrid import RetrievalConfig, get_retriever, reset_retriever
from app.store.db import close_db, db_status, init_db
from app.store.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("api")

_settings = get_settings()
_semaphore = asyncio.Semaphore(_settings.concurrency_limit)
_warmup_state: dict = {"state": "pending"}
api_router = APIRouter()


async def _do_warmup(retriever) -> None:
    """后台预热 BM25 索引（阻塞 CPU，放线程里跑）。

    实测冷启动 2.8s = 拉全量语料 717ms + jieba 分词建索引 2132ms。
    这个索引整个进程只建一次，如果留给第一个用户请求触发，
    他看到的就是"系统卡了两秒然后才开始答"。改成启动时后台建，
    端口一监听就已经有索引了。
    """
    _warmup_state["state"] = "running"
    try:
        # 遍历 settings.games 而不是写死两个 —— 加第三个库时预热自动跟上，
        # 漏掉的话第一个查该库的用户会独自吃掉 1.8s 建索引。
        for game in _settings.game_slugs:
            cfg = RetrievalConfig(game=game, use_bm25=True, use_vector=True, use_rerank=True)
            # to_thread：jieba 分词是 CPU 密集的纯 Python，不让出事件循环会卡住健康检查
            info = await asyncio.to_thread(retriever.warmup, cfg)
            _warmup_state[game] = info
            logger.info("预热 %s: %s", game, info)
        _warmup_state["state"] = "ready"
    except Exception as exc:  # noqa: BLE001
        # 预热失败不能让服务起不来 —— 检索器会在首次请求时懒加载兜底
        _warmup_state["state"] = f"failed: {type(exc).__name__}"
        logger.warning("预热失败，降级为懒加载：%s", exc)


async def _warmup_after_ingest(game: str) -> None:
    """入库后重建该库的 BM25 索引（后台）。

    失败不抛 —— 检索器会在下次请求时懒加载兜底，
    这里只是把"下一个用户卡 1.8s"变成"用户无感"。
    """
    try:
        cfg = RetrievalConfig(game=game, use_bm25=True, use_vector=True, use_rerank=True)
        info = await asyncio.to_thread(get_retriever(_settings).warmup, cfg)
        logger.info("入库后重建索引 %s: %s", game, info)
    except Exception as exc:  # noqa: BLE001
        logger.warning("入库后重建索引失败，将降级为懒加载：%s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭两侧资源。端口能监听**不代表**这两样已经就绪。

    启动时建立：
    - `init_db` 建库建表。它按设计**不抛**（除非 MYSQL_REQUIRED=true），
      返回 False 时服务照常起来、MySQL 相关能力整体降级 —— 所以端点里不能假设
      DB 可用，一律以 `db_status()` / 捕获 MySQLUnavailable 为准。
    - 预热任务 `create_task` 后台跑，**这里不等它**：冷启动要 2.8s，
      等于端口就绪时间被 BM25 建索引拖住。副作用是刚起来的瞬间
      第一个请求可能撞上尚未建好的索引，检索器内部有懒加载兜底。

    关闭时释放（finally，顺序即依赖顺序）：
    先取消预热任务，避免关库期间它还在往 store 上打；
    再 `close_db` dispose 连接池并把引擎/会话工厂置回 None；
    最后 `aclose` 关 LLM 客户端的 httpx 连接。
    少了 close_db 会在退出时挂住等连接池回收，少了 aclose 会留 warning。

    结果消费：`lifespan` 返回后端点才可被调用，因此所有 `get_*` 单例
    （retriever / llm / vector_store）在模块级初始化是安全的，不存在竞态。
    """
    # 用 get_retriever() 单例而不是 HybridRetriever(...)：
    # BM25 索引和语料缓存在实例级，新建实例等于每个请求重付 ~1800ms 建索引。
    # 预热和 /chat 的 ToolContext.build() 共用同一份缓存，才有意义。
    # 所以不再保留模块级 _retriever —— 直接调 get_retriever() 拿即可，
    # 少一个可能忘记赋值的全局状态。
    await init_db(_settings)
    task = asyncio.create_task(_do_warmup(get_retriever(_settings)))
    try:
        yield
    finally:
        task.cancel()
        await close_db()
        await get_llm().aclose()


app = FastAPI(title="kb-agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# 请求 / 响应模型
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    """`/chat` 与 `/chat/stream` 共用的入参。

    进入端点前已经成立的前提：MySQL 可能缺席（`init_db` 不抛），
    所以 `session_key` 给了也不保证落库成功，只保证「尽力记录」。
    `game` 决定检索哪个语料库，缺省用 `.env` 的主库；传了不存在的库会在
    检索阶段才失败，这属于当前已知的行为差异而非校验。

    落库语义：`session_key` 是多轮上下文与历史记录的唯一钥匙，
    不传则这一轮**不留痕**（save_turn 被跳过），前端历史也就断了。
    """

    question: str = Field(min_length=1, max_length=1000)
    session_key: str | None = None
    game: str | None = None
    history: list[dict] | None = None
    # Chat 基座（deepseek / minimax）。传错或未配置由 LLMClient 抛错，
    # 这里统一转成 400，让前端能给出可读提示而不是 500。
    backend: str | None = None
    # 本轮允许使用的工具。留空 = 用 Settings.agent_tools（生产默认只给 search_kb）。
    # 显式传 ["search_kb","get_page","calculator","list_games"] 可复现实验 C 的 C2 档。
    tools: list[str] | None = None


class RetrieveRequest(BaseModel):
    """`/retrieve` 入参：只做检索、不生成，用来单独观察检索层。

    契约前提：这个模型同时是**检索实验的对照组入口** ——
    实验 A/B 就是靠这里逐项传不同参数跑的，所以 `None` 与写死值的区别很重要（见下）。

    响应形状：`{query, label, config, hits[]}`，其中 `label`/`config` 是这次
    实际生效的配置快照（已被 from_settings 用 `.env` 兜底后的值），
    落进实验产物后就是「这个数字是哪档配置跑出来的」的唯一凭据。

    `page_titles` 与 `min_revision_ts` 是两个**约束**（过滤掉候选）而非加权，
    用来验证「某页是不是唯一来源」和「只看某时间之后的内容」，
    它们会同时压低 recall，属预期行为，读指标时要和基线档分开列。
    """
    query: str = Field(min_length=1, max_length=500)

    # 以下几项默认 **None 而非写死数值**：写死的话 Pydantic 每次都会把默认值
    # 传给下游，于是 `.env` 里的 RETRIEVE_TOP_K / RETRIEVE_CANDIDATE / RRF_K
    # 永远被覆盖 —— 改配置不生效，而且不报任何错。
    # None = "没指定"，由 RetrievalConfig.from_settings() 用 .env 兜底。
    game: str | None = None
    top_k: int | None = None
    use_bm25: bool = True
    use_vector: bool = True
    use_rerank: bool = True
    embed_model: str | None = None
    variant: str = "base"
    # 0 = 不裁剪（全量送重排），用于和优化前做 A/B 对照
    rerank_candidates: int | None = None
    # 版本感知检索：只保留 last_revision_ts >= 该值的 chunk。
    # 之前这个能力在检索层是通的（RetrievalConfig → build_where → 两个 store 都实现了），
    # 但**没有任何入口能设置它** —— 只能写 Python 才能演示，文档里却当卖点写着。
    # 接受 `YYYY-MM-DD` 或完整 ISO8601，统一补成 ISO 字符串再比大小。
    min_revision_ts: str | None = None
    # 只在指定页面里检索（用于验证"某页是不是唯一来源"）
    page_titles: list[str] | None = None


class IngestRequest(BaseModel):
    """`/ingest` 入参：抓一个分类下的页面并重建索引。

    前提与副作用（这是全 API 里唯一会**写坏缓存**的端点）：
    - `game` + `category` 决定抓取范围，`limit` 限条数。
      限流间隔来自 `settings.wiki_min_interval`，不是本模型里的字段。
    - 成功后调用 `reset_retriever()` 丢弃检索器缓存，所以**抓完之后**
      之前的 BM25 索引不再代表语料；不重置就表现为「入库成功却查不到」。
    - `variant` / `embed_mode` 决定切块与向量化方式，改它们等于要重建全库向量，
      只对新增页面生效的假设在这里不成立。

    同步返回抓取统计；耗时较长的抓取请走 `scripts/crawl.py`，
    否则会占住 `_semaphore`（`concurrency_limit`）把问答请求一起堵住。
    """
    game: str = "sr"
    category: str
    limit: int = 30
    # 早先这里有 interval: float = 5.0 —— 收下后**从未使用**：
    # WikiClient 的 min_interval 来自 settings.wiki_min_interval，
    # 所以改这个字段不生效也不报错。CLI 的 crawl.py 有自己的 --interval，
    # 那个是真在用的，别混为一谈。
    variant: str = "base"
    embed_mode: str = "title"


# --------------------------------------------------------------------------- #
# 端点
# --------------------------------------------------------------------------- #
@api_router.get("/health")
async def health() -> dict:
    """进程级健康快照：前端的所有下拉框数据都从这里来。

    进入这里的前提：MySQL 可能根本不在（`init_db` 不抛）。

    **`ok` 与 `ready` 是两件事，别混用**：
    - `ok` 只表示「HTTP 进程活着」——进程还在跑就是 true，因此它**不能**当探针；
    - `ready` 才反映依赖健康（MySQL 就绪 + 至少一个 chat 基座配好密钥），
      负载均衡 / k8s 探针应当断言 `ready`，MySQL 挂掉时会诚实地变成 false。

    此前这里只有一个恒为 true 的 `ok`，MySQL 全挂、LLM 密钥全缺也照样返回 ok:true，
    等于探针永远认为后端可用。现在把两种语义分开。

    结果消费：`fallback` 与 `cache` 是降级留痕，`warmup` 是预热状态，
    三者都是「不报错但行为变了」的唯一出口 —— 它们必须出现在响应里，
    否则降级这件事没有任何人会知道。
    """
    store = get_vector_store(_settings)
    llm = get_llm()
    mysql = db_status()
    llm_keys = {
        "deepseek": bool(_settings.deepseek_api_key),
        "siliconflow": bool(_settings.siliconflow_api_key),
        "judge": bool(_settings.judge_model),
    }
    return {
        "ok": True,
        # 依赖就绪度：探针断言这个，不是上面的 ok。
        # 纯向量检索链路不依赖 MySQL，所以 mysql 未就绪只降级为 false、不抛异常；
        # 判定口径写在这里而不是散在探针配置里，避免各处对「什么算健康」理解不一致。
        "ready": bool(mysql.get("state") == "ready" and any(llm_keys.values())),
        "mysql": mysql,
        "llm_configured": llm_keys,
        # 前端模型下拉框的数据源。只列出配好密钥的，不会泄露密钥内容。
        "chat_backends": llm.available_backends(),
        # 降级留痕。降级不报错，所以它必须自己出现在健康检查里 ——
        # 否则「今天有 40% 的回答其实来自备用模型」这件事根本不会有人知道。
        "fallback": llm.fallback_stats(),
        "embed_model": _settings.embed_model_main,
        "rerank_model": _settings.rerank_model,
        # slug → 显示名。前端游戏切换器直接渲染这个，不再在前端写死两个游戏。
        "games": _settings.games,
        # 前端需要知道有哪些游戏"存在但没收录"，才能给出「暂未收录」的提示而不是假装检索过
        "known_uncovered": {
            slug: name for slug, (name, _) in UNCOVERED_GAMES.items()
            if slug not in _settings.games
        },
        "collections": store.list_collections(),
        # 报**实际生效**的工具集，不是注册表全量 ——
        # 生产默认只挂 search_kb，报全量会让侧栏看起来像"4 个工具都开着"。
        "tools": [s["function"]["name"] for s in get_schemas(_settings.agent_tool_names)],
        # 延迟优化的可观测性：预热状态 + 缓存命中率
        "warmup": _warmup_state,
        "cache": llm.cache_stats(),
        # 并发上限：前端可据此提示"当前最多同时处理 N 个请求"
        "concurrency_limit": _settings.concurrency_limit,
    }


def _normalize_ts(raw: str | None) -> str | None:
    """把用户输入的时间戳归一成能直接做字符串比较的 ISO 形式。

    为什么必须归一：`last_revision_ts` 在库里长这样 `2026-02-01T16:53:50Z`，
    而前端传的自然是一个很顺手的 `2026-02-01`。直接拿去比会得到
    `"2026-02-01" >= "2026-02-01T16:53:50Z"` 为 True —— 恰好把当天所有内容
    都排除掉，**结果是错的但不报错**。纯日期补成当天 00:00:00Z 才是"从这天起"。
    """
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return f"{s}T00:00:00Z"
    return s


@api_router.post("/retrieve")
async def retrieve(req: RetrieveRequest) -> dict:
    """纯检索端点：不调 LLM，延迟只反映检索层，适合做 A/B 与排障。

    进入前的两个前提：
    - `get_retriever(_settings)` 是进程级单例，BM25 索引只建一次；
      刚 `reset_retriever()` 之后第一次调用要重建（约 1.8s），所以这里的
      延迟数字只在缓存热着的时候才有横向可比性。
    - 整段检索占用 `_semaphore`（全局并发上限），慢查询会把 `/chat` 一起挤住，
      所以面板上反复点不会更快。

    结果消费：命中以 `hits[]` 返回，`rank` 从 1 开始、已按最终分数排好序；
    `source` 标明这条是 bm25 / vector / rerank 哪一路贡献的 —— 重排前后的
    来源分布比总分更能说明「提升来自哪里」。
    """
    retriever = get_retriever(_settings)
    # 只把**请求里真的给了**的字段传下去（None 的一律丢掉），
    # 其余交给 from_settings() 用 .env 兜底。
    #
    # 注意 `game` 不在这份 dict 里：它下面已经单独算好了
    # （req.game or .env 的主库），放两遍会直接
    # "got multiple values for keyword argument 'game'" —— 500。
    # 这个 bug 已经犯过一次：验收脚本只测了不带 game 的请求，
    # 而前端**每次都带 game**，于是检索面板整个是坏的。
    given = {k: v for k, v in {
        "embed_model": req.embed_model,
        "final_top_k": req.top_k,
        "rerank_candidates": req.rerank_candidates,
    }.items() if v is not None}
    cfg = RetrievalConfig.from_settings(
        settings=_settings,
        game=req.game or _settings.game_primary,
        variant=req.variant,
        use_bm25=req.use_bm25,
        use_vector=req.use_vector,
        use_rerank=req.use_rerank,
        min_revision_ts=_normalize_ts(req.min_revision_ts),
        page_titles=req.page_titles or None,
        **given,
    )
    async with _semaphore:
        try:
            hits = await retriever.retrieve(req.query, cfg)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"检索失败：{type(exc).__name__}: {exc}") from exc
    return {
        "query": req.query,
        "label": cfg.label(),
        "config": cfg.as_dict(),
        "hits": [
            {
                "rank": i,
                "chunk_id": h.chunk_id,
                "score": h.score,
                "source": h.rank_source,
                "page_title": h.chunk.page_title,
                "section": h.chunk.metadata.get("section", ""),
                "text": h.chunk.text,
                "url": h.chunk.source_url,
                "last_revision_ts": h.chunk.metadata.get("last_revision_ts", ""),
            }
            for i, h in enumerate(hits, 1)
        ],
    }


def _build_agent(req: ChatRequest) -> tuple[ReActAgent, ToolContext]:
    """构造本次请求的 Agent 与工具上下文（非流式 / 流式共用）。

    基座在这里就校验完：传了没配密钥的 provider，等 Agent 跑起来才报错的话，
    用户已经等了好几秒才看到一句 500 —— 明显该在入口处 400 掉。
    """
    llm = get_llm()
    if req.backend:
        try:
            llm.chat_backend_name(req.backend)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
    agent = ReActAgent(
        settings=_settings,
        llm=llm,
        backend=req.backend,
        tools=get_schemas(req.tools) if req.tools else None,
    )
    ctx = ToolContext.build(req.game or _settings.game_primary)
    if req.game:
        ctx.game = req.game
    return agent, ctx


@api_router.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """非流式问答：一次跑完 ReAct 全过程后整体返回。

    进入前的前提（都由 `_build_agent` 保证）：`backend` 若显式指定必须是**已配密钥**
    的基座，否则这里就 400 —— 不等 Agent 跑起来才报错。
    多轮历史优先用请求里的 `history`，没给且带 `session_key` 时才查库。

    结果消费：`fallback` 信息不在响应体里（那是 chat 基座层的字段），
    想知道这一轮是不是被降级过，只能对 `/api/health` 的 `fallback_stats`
    按时间对账 —— 所以跨模型对比实验跑完后必须核这一项，
    否则"降级样本混进对比"这个隐藏变量就查不出来了。
    `trace[]` 是逐步工具调用，前端用它渲染过程面板，`run_id` 用来回查 `/api/runs/{id}`。
    """
    from app.store.persist import load_history, save_run, save_turn

    agent, ctx = _build_agent(req)

    # 多轮上下文：优先用请求里带的 history，没有就从库里取最近几轮
    history = req.history
    if history is None and req.session_key:
        history = await load_history(req.session_key, settings=_settings)

    async with _semaphore:
        state = await agent.run(req.question, session_key=req.session_key, history=history, ctx=ctx)

    # 轨迹与会话落库是"加分项"，失败不能影响问答本身
    await save_run(state, settings=_settings)
    if req.session_key:
        await save_turn(req.session_key, question=req.question, answer=state.answer,
                        refs=state.refs, settings=_settings)

    return {
        "run_id": state.run_id,
        "game": state.game,
        "backend": agent.backend or "deepseek",
        "answer": state.answer,
        "steps": len(state.steps),
        "tool_failures": state.tool_failures,
        "stop_reason": state.stop_reason,
        "elapsed_ms": state.elapsed_ms,
        "usage": {"prompt": state.prompt_tokens, "completion": state.completion_tokens},
        "guard": state.guard_stats(),
        "injection": state.injection_summary(),
        "refs": state.refs,
        "trace": [
            {"step": s.step, "tool": s.tool_name, "args": s.args, "ok": s.ok, "ms": s.latency_ms}
            for s in state.steps
        ],
    }


@api_router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式问答：与 `/chat` 走同一个 Agent，差别只在**什么时候**拿结果。

    进入前的前提同上（`_build_agent` 已完成基座校验），另外要注意
    **流一旦发出就无法改状态码**：`_build_agent` 在返回响应之前跑，
    所以参数错误仍然是干净的 400；但生成过程中出错只能以 `event: error`
    的形式补发，前端必须单独处理这个事件，不能只等 `done`。

    结果消费：`done` 事件里额外带 `persisted`（落库结果）和 `backend`，
    前端据此提示「已存档」并标出实际用的模型 ——
    这是唯一能在流式响应里看出发生了降级的位置。
    """
    agent, ctx = _build_agent(req)
    backend = agent.backend or "deepseek"

    async def gen():
        from app.store.persist import load_history, save_run, save_turn

        history = req.history
        if history is None and req.session_key:
            history = await load_history(req.session_key, settings=_settings)
        try:
            async with _semaphore:
                async for event in await agent.run(
                    req.question, session_key=req.session_key, history=history, ctx=ctx, stream=True
                ):
                    # done 事件里带上落库结果，前端能提示"本次运行已存档"
                    if event["event"] == "done":
                        persisted = await save_run(agent.last_state, settings=_settings)
                        if req.session_key:
                            await save_turn(req.session_key, question=req.question,
                                            answer=event["data"].get("answer", ""),
                                            refs=event["data"].get("refs", []), settings=_settings)
                        event = {
                            **event,
                            "data": {**event["data"], "persisted": persisted, "backend": backend},
                        }
                    yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@api_router.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    """触发一次分类抓取。同步返回，长任务请用 scripts/crawl.py。"""
    from app.ingest.pipeline import IngestConfig, Ingestor

    cfg = IngestConfig.from_settings(req.game, variant=req.variant, embed_mode=req.embed_mode)
    ing = Ingestor(settings=_settings)
    async with _semaphore:
        stats = await ing.crawl_category(req.category, cfg, limit=req.limit)

    # 入库后必须丢弃检索器缓存：_text_cache / _bm25_cache 里存的是
    # **入库那一刻的语料快照**，不清的话新抓的页面检索不到，
    # 要等进程重启才生效 —— 表现为"明明入库成功了却查不到"。
    # 这里选择整体失效而不是按 game 精确失效：入库是低频操作，
    # 多付一次建索引（约 1.8s）换实现简单可靠，是划算的。
    reset_retriever()
    stats_dict = stats.as_dict()
    stats_dict["retriever_cache_reset"] = True

    # 重建索引挪到后台，而不是留给下一个请求。
    # 原因不是"慢"，是"卡"：_get_bm25 是同步且内部无 await 的，
    # 一旦在请求路径上触发，整个事件循环会被占住 1.8s ——
    # 那一刻所有并发中的请求全部停摆，表现为服务整体无响应。
    # 放到 to_thread 里后台重建，既不阻塞，也不让用户等。
    asyncio.create_task(_warmup_after_ingest(req.game))
    return stats_dict


@api_router.get("/stats")
async def stats() -> dict:
    """最轻量的一条可观测端点：向量库里的集合列表 + MySQL 状态。

    与 `/health` 的区别：不调 LLM、不读配置，所以可以放心地高频轮询；
    判断「语料在不在」用这里，判断「回答为什么变了」用 /health 的 fallback。
    """
    store = get_vector_store(_settings)
    return {"collections": store.list_collections(), "mysql": db_status()}


# --------------------------------------------------------------------------- #
# 运行轨迹（可观测性）：MySQL 可用时才有内容，不可用时返回空列表而不是报错
# --------------------------------------------------------------------------- #
@api_router.get("/runs")
async def runs(limit: int = 20) -> list[dict]:
    """最近的运行列表。

    进入前的前提：MySQL 可能缺席 —— 此时 `recent_runs` 返回**空列表而不是报错**，
    所以「列表为空」有两种含义：真的没跑过，或库没连上。
    要区分必须同时看 `/api/health` 的 `mysql.state`，别把降级读成「没人用过」。
    """
    from app.store.persist import recent_runs

    return await recent_runs(limit=limit, settings=_settings)


@api_router.get("/runs/{run_id}")
async def run_detail(run_id: str) -> dict:
    """单次运行的完整轨迹（含逐步工具调用），是复现问题的那一屏。

    注意这里的 404 混了两种原因：run 真不存在，或者 MySQL 没连上导致压根没落库。
    两者返回同一个 404，排查时得先看 `/api/health` 的 `mysql.state`。
    """
    from app.store.persist import run_detail as _detail

    detail = await _detail(run_id, settings=_settings)
    if detail is None:
        raise HTTPException(404, f"run {run_id} 不存在（MySQL 未启用或未落库）")
    return detail


# 前端开发期通过 Vite 代理访问，统一挂到 /api 下，
# 这样生产部署到别的域名时前端代码里全是相对路径。
# 同时保留无前缀版本，方便 curl 直接调试。
app.include_router(api_router, prefix="/api")
app.include_router(api_router)