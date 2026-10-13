"""Agent 可用工具集。

每个工具都必须满足三条，否则不注册：
1. **可失败**：参数非法、检索为空、超时，都要返回可读的错误字符串而不是抛异常 ——
   因为工具错误是要喂回给模型让它自我修正的，抛异常会直接中断整条链；
2. **有预算**：返回体必须截断，否则一个大页面的全文能把上下文撑爆；
3. **可观测**：返回体里带够定位问题的信息（命中数、耗时、被过滤的条件）。
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig
from app.store.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    settings: Settings
    retriever: HybridRetriever
    store: VectorStore
    game: str = "sr"

    @classmethod
    def build(cls, game: str = "sr") -> ToolContext:
        s = get_settings()
        # 必须用 get_retriever() 单例，不能 HybridRetriever(...) 新建 ——
        # 后者的 BM25 索引是实例级的，新建一次就要重付 ~1800ms 的建索引成本，
        # 而 /chat 每个请求都会调这里。实测对照见 retrieval/hybrid.py:345。
        from app.retrieval.hybrid import get_retriever

        return cls(
            settings=s,
            retriever=get_retriever(s),
            store=get_vector_store(s),
            game=game,
        )


ToolFn = Callable[[dict, ToolContext], Awaitable[str]]

REGISTRY: dict[str, ToolFn] = {}
SCHEMAS: list[dict] = []


def tool(
    name: str,
    description: str,
    props: dict[str, str],
    required: list[str],
    types: dict[str, str] | None = None,
) -> Callable[[ToolFn], ToolFn]:
    """注册一个工具。

    props 是 {参数名: 描述}；types 可选，指定 {参数名: json_type}，默认全部按 string 声明。
    注意必须产出合规的 JSON Schema —— 少一层 {"type": ...} 包装，
    DeepSeek 会直接 400 "Invalid schema for function"。
    """
    types = types or {}

    def deco(fn: ToolFn) -> ToolFn:
        REGISTRY[name] = fn
        SCHEMAS.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            pname: {"type": types.get(pname, "string"), "description": pdesc}
                            for pname, pdesc in props.items()
                        },
                        "required": required,
                    },
                },
            }
        )
        return fn

    return deco


def _truncate(text: str, limit: int) -> str:
    """截断并**显式告知原长度**。

    静默截断会让模型以为自己已经看全了正文，于是基于半截内容作答
    或者反复检索同一页 —— 尾部那句标记就是给模型看的信号。
    limit 由 .env 提供（TOOL_CHUNK_CHARS / TOOL_PAGE_CHARS），
    早先这里是字面量 500 / 4000，是全项目唯一改不了的预算。
    """
    return text if len(text) <= limit else text[:limit] + f"\n...[已截断，原长度 {len(text)}]"


# --------------------------------------------------------------------------- #
@tool(
    "search_kb",
    "在游戏知识库中做语义+关键词混合检索，返回带出处的文本片段。问任何游戏内数值、机制、角色信息前都应先调用它。",
    {
        "query": "检索问题，用完整句子描述需要的信息",
        "game": "可选，星铁填 sr，绝区零填 zzz。默认用当前会话的游戏",
        "top_k": "可选，返回条数，默认 5，最大 10",
    },
    ["query"],
    types={"top_k": "integer"},
)
async def search_kb(args: dict, ctx: ToolContext) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "[错误] query 不能为空。"
    s = ctx.settings
    # top_k 的默认值取自 `.env`，不是写死的 5。
    # 写死的话改 .env 完全不生效，而且**不报任何错** ——
    # 恰好 .env 的默认值就是 5，测的时候根本看不出来（假绿）。
    top_k = min(int(args.get("top_k") or s.retrieve_top_k), 10)
    game = str(args.get("game") or ctx.game)
    available = ctx.settings.games
    if game not in available:
        return (
            f"[错误] 知识库里没有游戏 {game}，已收录："
            f"{' / '.join(available.values())}。"
        )

    # 从 .env 派生候选池与 rrf_k；显式的 final_top_k 压过 .env。
    # 早先这里是 `RetrievalConfig(game=game, final_top_k=top_k)`，
    # 其余字段全走 dataclass 默认值 —— 于是 RETRIEVE_CANDIDATE / RRF_K
    # 在 Agent 这条路径上是**死的**，改 .env 检索照常跑、不报任何错。
    # （`/api/retrieve` 早就改用 from_settings 了，所以 check_config_wiring.py
    #  验不出来 —— 它只验 API 路径，正好漏掉这条。）
    cfg = RetrievalConfig.from_settings(settings=s, game=game, final_top_k=top_k)
    t0 = time.perf_counter()
    try:
        hits = await ctx.retriever.retrieve(query, cfg)
    except Exception as exc:  # noqa: BLE001
        return f"[检索失败] {type(exc).__name__}: {exc}"
    dt = int((time.perf_counter() - t0) * 1000)

    # 第一行是给 Agent 循环看的机器可读状态（status / top_score / top_page），
    # 它要靠这个判断"这次检索到底有没有找到东西"，从而决定要不要继续搜。
    # 没有这一层，Agent 只能看到一段文本，就会把"召回了但都不相关"当成"再搜一次说不定有"。
    if not hits:
        header = f"[search_kb] status=empty hits=0 top_score=0.000 {dt}ms"
        return f"{header}\n知识库中没有与「{query}」相关的内容。"

    top = hits[0]
    header = (
        f"[search_kb] status=ok hits={len(hits)} top_score={top.score:.3f} "
        f"top_page={top.chunk.page_title} {dt}ms 来源={top.rank_source}"
    )
    lines = [header]
    for i, h in enumerate(hits, 1):
        c = h.chunk
        lines.append(
            f"\n[{i}] 分数 {h.score:.4f} | {c.page_title} > {c.metadata.get('section', '')}\n"
            f"{_truncate(c.text, s.tool_chunk_chars)}\n"
            f"出处：{c.source_url}"
        )
    return "\n".join(lines)


@tool(
    "get_page",
    "按页面标题取该页面的完整正文，用于 search_kb 片段不够用时补全上下文。",
    {"title": "wiki 页面标题，如 三月七 / 光锥", "game": "可选游戏 slug"},
    ["title"],
)
async def get_page(args: dict, ctx: ToolContext) -> str:
    """取页面全文。

    这里**直接现抓 wiki 而不是读向量库**，是一个刻意的取舍：
    - 向量库里只有分块后的片段，拿不到"这个页面的完整结构"；
    - 现抓能拿到最新修订，而向量库可能是几小时前的快照。

    代价是每次调用都要过一遍 WAF，且可能限流。
    所以 Agent 的正常路径应该只调一次 search_kb（一次就够），
    只有片段确实不够时才调这个 —— 限流是最容易在演示时翻车的地方。
    """
    title = str(args.get("title", "")).strip()
    if not title:
        return "[错误] title 不能为空。"
    game = str(args.get("game") or ctx.game)
    t0 = time.perf_counter()
    try:
        from app.ingest.wiki_client import WikiClient

        async with WikiClient(
            game,
            base_url=ctx.settings.wiki_base,
            min_interval=ctx.settings.wiki_min_interval,
            max_retry=ctx.settings.wiki_max_retry,
        ) as wc:
            page = await wc.fetch_page(title)
    except Exception as exc:  # noqa: BLE001
        # 错误文案里要明确告诉模型"这是限流"和"还有别的路可走"，
        # 否则它会把网络错误当成"页面不存在"，转而去反复 search_kb
        return f"[抓取失败] {type(exc).__name__}: {exc}。若提示被限流，请稍后重试或改用 search_kb。"

    from app.ingest.template_parser import clean_wikitext

    text = clean_wikitext(page.wikitext)
    dt = int((time.perf_counter() - t0) * 1000)
    return (
        f"《{page.title}》最后修订 {page.last_revision}（{dt}ms，共 {len(text)} 字）\n"
        f"出处：{ctx.settings.wiki_base}/{game}/{page.title}\n\n{_truncate(text, ctx.settings.tool_page_chars)}"
    )


@tool(
    "calculator",
    "计算游戏内数值（伤害、倍率、遗器分数、养成资源消耗）。支持 + - * / ^ 与括号。",
    {"expression": "算式，例如 (0.8 + 0.2) * 1000"},
    ["expression"],
)
async def calculator(args: dict, ctx: ToolContext) -> str:
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return "[错误] expression 不能为空。"
    if not re.fullmatch(r"[0-9+\-*/().%^ ]+", expr):
        return f"[错误] 算式含非法字符，只允许数字和 + - * / ^ % ( ) 。收到的：{expr[:80]}"
    try:
        # 只用 AST 白名单求值，避免 eval 的代码执行风险
        import ast
        import operator as op

        binops = {
            ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
            ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
        }
        unops = {ast.UAdd: op.pos, ast.USub: op.neg}

        def ev(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return ev(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
                return float(node.value)
            if isinstance(node, ast.BinOp) and type(node.op) in binops:
                return binops[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in unops:
                return unops[type(node.op)](ev(node.operand))
            raise ValueError("算式含不支持的语法")

        val = ev(ast.parse(expr, mode="eval"))
        return f"{expr} = {val:g}"
    except ZeroDivisionError:
        return f"[错误] {expr} 出现除零。"
    except (SyntaxError, ValueError) as exc:
        return f"[错误] 无法解析算式 {expr}：{exc}"


@tool(
    "list_games",
    "列出当前知识库覆盖的游戏，以及各自的语料规模。用于路由问题到正确的知识库。",
    {},
    [],
)
async def list_games(args: dict, ctx: ToolContext) -> str:
    games = ctx.settings.games
    lines = ["知识库覆盖的游戏："]
    for slug, label in games.items():
        try:
            n = ctx.store.count(slug, None, "base")
        except Exception:  # noqa: BLE001
            n = -1
        lines.append(f"- {slug}（{label}）：{'索引未建立' if n < 0 else f'{n} 个片段'}")
    lines.append(
        "判断规则：术语体系不同，别混用 —— "
        "原神/星铁是「圣遗物 / 位面饰品」，绝区零是「音擎 / 驱动盘」；"
        "命座 / 命途、专精 / 行迹 也各指一套。"
    )
    return "\n".join(lines)


def get_schemas(names: Sequence[str] | None = None) -> list[dict]:
    """工具 schema。

    `names` 非空时**只返回这些工具** —— 这是实验 C 的消融开关：
    决定"模型能看到哪些工具声明"，而不是"模型调用了哪些工具"
    （后者会被模型的随机选择污染，测不出能力差异）。

    生产默认只给 search_kb：实验 C 实测 C1(4.60) ≥ C2(4.47)，
    p=0.47 未达显著，也就是**多挂 3 个工具没有可测收益，却要多付 22.6% token**。
    要恢复全工具设 `AGENT_TOOLS=search_kb,get_page,calculator,list_games`。
    """
    if not names:
        return list(SCHEMAS)
    wanted = list(names)
    unknown = [n for n in wanted if n not in REGISTRY]
    if unknown:
        raise ValueError(f"未知工具 {unknown}，可选：{', '.join(REGISTRY)}")
    return [s for s in SCHEMAS if s.get("function", {}).get("name") in wanted]


# 工具自己返回错误文案时统一用的前缀。
# [无结果] 不算失败 —— 检索为空是合法结论，不是错误。
#
# 这里是一份**约定**而不是类型标记：工具既不抛异常也不返回结构化结果，
# 而是返回一段给模型看的文本。那 ok 就只能靠前缀判断。
# 代价是新增工具必须遵守前缀规范，否则失败率指标会静默失真。
# 另一个约束：熔断文案（「[重复查询已拦截]」「[熔断]」）刻意不在这个列表里 ——
# 它们确实没执行成功，但 Agent 层单独按 `ok=False` 记，不重复计入工具失败。
_ERROR_PREFIXES = ("[错误]", "[超时]", "[异常]", "[检索失败]", "[抓取失败]")

#: 上面的**子集**：属于基础设施失败（网络/超时/后端异常）而不是参数用错。
#:
#: 区分它们的理由：Agent 的熔断要掐的是「连续拿不到可信信息」。
#: 基础设施失败期间继续检索只会重复失败，所以这类必须计入熔断计数；
#: 而 `[错误]`（如 query 为空）是模型自己传错参数，改一下就能重试，
#: 计入熔断等于惩罚一次可自愈的错误。SearchGuard 依赖这个区分。
_INFRA_FAILURE_PREFIXES = ("[超时]", "[异常]", "[检索失败]", "[抓取失败]")


async def call_tool(name: str, args: dict, ctx: ToolContext, timeout: float) -> tuple[str, bool]:
    """执行工具，返回 (结果文本, 是否成功)。

    ok 的含义是"这次工具调用得到了有效结果"，而不是"没抛异常"：
    工具按约定把失败转成可读文案返回，若 ok 一律为 True，
    Agent 轨迹里的 tool_failures 就永远是 0，失败率这个指标也就废了。
    超时和异常同样转成可读文本交给模型自我修正 —— 这是 ReAct 能自我纠错的关键。
    """
    fn = REGISTRY.get(name)
    if fn is None:
        return f"[错误] 未知工具 {name}。可用工具：{', '.join(REGISTRY)}", False
    try:
        text = await asyncio.wait_for(fn(args, ctx), timeout=timeout)
    except TimeoutError:
        return f"[超时] 工具 {name} 超过 {timeout:.0f}s 未返回，请换用更简单的查询。", False
    except Exception as exc:  # noqa: BLE001
        logger.exception("工具 %s 执行失败", name)
        return f"[异常] 工具 {name} 失败：{type(exc).__name__}: {exc}", False
    return text, not text.lstrip().startswith(_ERROR_PREFIXES)