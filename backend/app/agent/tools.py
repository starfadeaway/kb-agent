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
    #: 实验 F 的消融开关：`search_kb` / `get_page` 是否输出页面修订日期。
    #: 默认 True —— 生产行为就是披露。
    #:
    #: 与已删除的 `TOOL_TRANSPORT` 有本质区别，必须说清：
    #: 那个字段全仓**没有一处分支读它**，改 .env 完全不生效；
    #: 这个字段在 `search_kb` / `get_page` 里都有**真实分支**，
    #: 且 `test_revision_disclosure.py` 断言两条路径的输出确实不同。
    #: 守卫断言的是行为，不是「某个值等于某状态」。
    disclose_revision: bool = True

    @classmethod
    def build(cls, game: str = "sr") -> ToolContext:
        s = get_settings()
        # 必须用 get_retriever() 单例，不能 HybridRetriever(...) 新建 ——
        # 后者的 BM25 索引是实例级的，新建一次就要重付 ~1800ms 的建索引成本，
        # 而 /chat 每个请求都会调这里。实测对照见 retrieval/hybrid.py:345。
        from app.retrieval.hybrid import get_retriever

        # 三个分量的**消费面很窄**，写清楚是为了避免"这里也能注入"这类误解：
        #   retriever → 只有 search_kb 用（get_page 走 wiki 现抓，不碰向量库）
        #   store     → 只有 list_games 用（count）
        #   game      → 所有工具在模型没显式传 game 时的默认库
        # 刻意不在这里塞检索参数（top_k / 阈值 / rrf_k）：
        # 那些是 search_kb 校验完参数后的职责，抽到这一层会让"这次检索用了什么配置"
        # 变成一个只能靠猜的属性 —— 之前把 RetrievalConfig 留在这里就是那个问题。
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
        # 下面是 **import 期副作用**，不是运行期注册：
        # 任何工具只要模块被 import 就已经在表里了，运行时再注册是来不及的
        # （ReActAgent 在 __init__ 里就把 schemas 快照走了）。
        # REGISTRY 与 SCHEMAS 必须成对写入，缺一半的表现完全不同：
        #   只写 REGISTRY → 模型看不见它，会幻觉出这个名字并被 call_tool 拒绝
        #   只写 SCHEMAS  → 模型看得见、调用时却找不到实现，同样返回"未知工具"
        # 两种都不报错，只有跑起来才发现，所以放在一处写。
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
    # 前提：调用方已经把这个 limit 的**来源**说清楚了
    # （search_kb 用 chunk 级、get_page 用 page 级，都来自 .env）。
    # 这里只截断不编号、不分页：分页要模型再调一次工具，
    # 而多一次调用的思考 + 工具声明 token 通常比多给 2000 字符正文还贵。
    # 换来的代价是模型可能刚好丢掉答案所在的那一句 ——
    # 所以尾部那句"已截断"必须留着，它是模型知道"还有内容没看到"的唯一信号。
    return text if len(text) <= limit else text[:limit] + f"\n...[已截断，原长度 {len(text)}]"


def _fmt_revision(ts: object) -> str:
    """ISO8601 UTC 时间戳 → `YYYY-MM-DD`。取不到就返回空串。

    为什么只保留日期：模型要判断的是「这条内容大概多新」，
    `2026-07-15T04:48:04Z` 里的时分秒对它没有决策价值，却要占 token。

    **不要在这里做任何"智能"转换**：解析不了就返回空，
    空着比给一个猜出来的日期安全 —— 编一个日期比不给日期坏得多。
    """
    s = str(ts or "").strip()
    return s[:10] if len(s) >= 10 else ""


def _revision_span(hits) -> str:
    """这批命中页面的修订时间跨度，给模型一个整体的新旧感知。

    为什么放在 header 而不是每条都写全：逐条写会让每条多 20+ 字符，
    而模型对"这批数据整体有多新"和"具体某条多新"是两个不同需求 ——
    前者决定要不要声明时效，后者放在每条行内。

    ⚠ 用词必须是「页面修订」而不是「数据截至」。
    `last_revision_ts` 是**页面最后一次编辑时间**，一页 2020 年建的 wiki
    昨天改了个错别字它就是昨天 —— 写成「数据截至」会造出比不给日期
    更隐蔽的误导。
    """
    # 返回的片段由 search_kb 直接拼在 header 末尾（` 来源={rank_source}{span}`），
    # 所以它**必须自带前导空格**，不能返回裸日期 ——
    # 否则 header 会粘成 `来源=hybrid2026-07-15`，日期被吞进字段值里。
    #
    # 另一条隐性契约：span 出现在 `top_score=` **之后**，
    # 而 SearchGuard.observe 是在前 200 字符里按空格切 token、找 `top_score=` 开头的那个。
    # 两者的相对顺序不能对调，否则熔断读到的分数会变成 `0.812` 后面粘着的别的东西，
    # parse 失败按 0.0 处理 → 一次好检索被当成 miss 累加。
    stamps = sorted(
        {d for d in (_fmt_revision(h.chunk.metadata.get("last_revision_ts"))
                     for h in hits) if d}
    )
    if not stamps:
        return ""
    if len(stamps) == 1:
        return f" 页面修订 {stamps[0]}"
    return f" 页面修订 {stamps[0]}~{stamps[-1]}"


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
    # 前置：ctx.game 必须已经过 `_prepare` 的路由校验，模型显式传的 game 在这里再校验一次。
    # 两次校验不是冗余：模型可以主动传一个越界 slug，而 `_prepare` 只看用户问题。
    # 这里返回错误而不是兜底去搜默认库 —— 搜错库然后报"未收录"，
    # 会让用户以为知识库里真没有这条内容，而事实是我们查错了地方。
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
    span = _revision_span(hits) if ctx.disclose_revision else ""
    header = (
        f"[search_kb] status=ok hits={len(hits)} top_score={top.score:.3f} "
        f"top_page={top.chunk.page_title} {dt}ms 来源={top.rank_source}{span}"
    )
    lines = [header]
    # 循环不做二次限量：条数上限由上面的 top_k（≤10）兜住，
    # 而每条正文已经被 _truncate 压到 tool_chunk_chars。
    # 真正撑爆上下文的是「条数 × 步数」，那属于 agent_token_budget 的职责；
    # 工具层再截一次条数，模型只会以为"库里的结果就这些了"——
    # 而它其实连 10 条里的第 7 条都没看到。
    for i, h in enumerate(hits, 1):
        c = h.chunk
        rev = (
            _fmt_revision(c.metadata.get("last_revision_ts"))
            if ctx.disclose_revision else ""
        )
        # 修订日期同时出现在**行内**和**出处行**：
        #   行内 → 模型读到这条内容时就知道它多新，就近判断，不用来回对照 header；
        #   出处行 → `_collect_refs` 靠「出处：」前缀扫引用，顺带把它带进
        #           前端的引用列表（否则前端拿不到时间戳，右下角就无从展示）。
        # 两处同源，不要在别处再算一次。
        rev_tail = f" | 修订 {rev}" if rev else ""
        section = c.metadata.get("section", "")
        lines.append(
            f"\n[{i}] 分数 {h.score:.4f} | {c.page_title} > {section}"
            f"{f' | 页面修订 {rev}' if rev else ''}\n"
            f"{_truncate(c.text, s.tool_chunk_chars)}\n"
            f"出处：{c.source_url}{rev_tail}"
        )
    return "\n".join(lines)
    # ↑ 出口契约 —— 这一段文本有**三个互相独立的消费方**，改格式要三处一起改：
    #   1) 模型：只看文本。第一行的 status/top_score 是它判断"要不要继续搜"的信号，
    #      尾部每条的「出处：」是它编造引用时的素材来源。
    #   2) SearchGuard.observe：解析同一行里的 `status=empty` 与 `top_score=`
    #      来决定 miss 计数 —— 格式一变，熔断会静默失效而不是报错。
    #   3) ReActAgent._collect_refs：按行首「出处：」前缀扫引用，并从竖线后面
    #      取修订日期。少一处输出格式，引用列表就会和模型的正文出处对不上，
    #      表现为"看起来像幻觉"，但责任其实在工具层。
    # 反过来说：ok=False 的判断**不**看这段文本（那由 call_tool 按前缀做），
    # 所以上面空结果那一支也必须自己写出 header，不能提前 return 一句人话。


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
    # 前置：这是整条链上**唯一真正出网**的步骤，也是唯一可能触发 WAF/限流的点。
    # SearchGuard 也会拦它（去重 + 熔断），但熔断拦不住限流本身 ——
    # 限流的表现是每次都返回抓取失败文案，miss 照涨，最终被熔断收尾。
    # 所以真正的防线是上面 docstring 里那句"正常路径只调一次 search_kb"，
    # 而不是这里的任何检查。
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
    # 早先这里直接把 `page.last_revision` 拼进句子 —— 它是 ISO 串
    # （2026-07-15T04:48:04Z），模型既读不出时效，时分秒还在吃 token。
    # 换成只给日期，并在措辞上说清是「页面修订」而不是「数据截至」。
    rev = _fmt_revision(page.last_revision) if ctx.disclose_revision else ""
    rev_txt = (
        f"页面最后修订 {rev}" if rev
        else ("页面修订时间未知" if ctx.disclose_revision else "")
    )
    return (
        f"《{page.title}》{rev_txt}（{dt}ms，共 {len(text)} 字）\n"
        # 「出处：」这一行的格式必须与 search_kb **逐字符一致**
        #（<url>[ | 修订 YYYY-MM-DD]），因为 `_collect_refs` 是按前缀统一扫的，
        # 并不区分产出者是哪个工具。
        # 这里漏掉前缀的后果特别隐蔽：页面内容照样喂给了模型、模型照样会在正文里
        # 标出处，但 refs 里就是没有这一条 —— 用户点开发现"引用的地方查无此文"，
        # 而模型那边一切正常。看起来和幻觉一模一样。
        # 同理，「修订」两个字不能改写：那是 `_collect_refs` 的解析锚点。
        f"出处：{ctx.settings.wiki_base}/{game}/{page.title}"
        f"{f' | 修订 {rev}' if rev else ''}\n\n"
        f"{_truncate(text, ctx.settings.tool_page_chars)}"
    )
    # 出口与 search_kb 的差别：这里**没有 header 行**，所以 SearchGuard.observe
    # 读不到 status/top_score，它对 get_page 只做一件事 ——
    # 判断结果是不是熔断/失败文案。是的话不计数，是正文的话也不计数。
    # 也就是说 get_page 既不会累加 miss 也不会清零：清零只在 search_kb 出好消息时发生。


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
    # 正则与 AST 白名单是**两层**的，故意都留着，但职责不同：
    #   正则层 → 给模型一条可读的定位反馈（哪个字符不对），成本 O(n)
    #   AST 层 → 真正决定求值语义，并把"能过正则但算不了"的输入
    #             （超长表达式、深度嵌套）变成明确的错误而不是慢
    # 少了 AST 层，光靠正则推断 `1/0` 该返回什么、括号该左结合还是右结合，
    # 这些都要在正则里补规则，而正则表达不了优先级 —— 补得越复杂越容易漏。
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
    # 出口契约：只回一句 `<算式> = <结果>`，没有出处、也没有 header。
    # 三个后果都是刻意的：
    #   - 不进 refs        → 算术结论没有来源可引，模型不应把它包装成"知识库说"
    #   - 不影响熔断计数   → SearchGuard 只统计 search_kb，算对了不该抵消一次检索 miss
    #   - ok 由前缀判定    → 成功文案不以 [错误]/[异常] 开头，所以 call_tool 判 ok=True


@tool(
    "list_games",
    "列出当前知识库覆盖的游戏，以及各自的语料规模。用于路由问题到正确的知识库。",
    {},
    [],
)
async def list_games(args: dict, ctx: ToolContext) -> str:
    games = ctx.settings.games
    lines = ["知识库覆盖的游戏："]
    # 每个游戏各查一次 count，并**逐个吞掉异常**：
    # 这个工具的用途是回答"知识库里有什么"，如果因为某一个库索引没建好就整体失败，
    # 模型连"有哪几个库"都不知道，反而会去反复 search_kb 一个个试探 ——
    # 那是把一个便宜的查询换成 N 次昂贵的检索。
    # 失败用 -1 兜住并渲染成"索引未建立"，它是**可用信息**而不是错误：
    # 至少模型知道该去问别的库，而不是以为这个库是空的。
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
    # 出口是纯文本列表，**不产生任何「出处：」行**，所以这次调用之后
    # state.refs 不会增长。
    # 评测里看到 refs 为空但 steps > 0 是正常的（calculator / list_games 都如此），
    # 不要当成引用抽取漏了 —— 判断依据是「steps 里有没有 search_kb / get_page」，
    # 而不是 refs 本身。
    # 同样没有 header，所以不参与 SearchGuard 的 miss 计数。


def get_schemas(names: Sequence[str] | None = None) -> list[dict]:
    """工具 schema。

    `names` 非空时**只返回这些工具** —— 这是实验 C 的消融开关：
    决定"模型能看到哪些工具声明"，而不是"模型调用了哪些工具"
    （后者会被模型的随机选择污染，测不出能力差异）。

    生产默认只给 search_kb：实验 C 实测 C1(4.60) ≥ C2(4.47)，
    p=0.47 未达显著，也就是**多挂 3 个工具没有可测收益，却要多付 22.6% token**。
    要恢复全工具设 `AGENT_TOOLS=search_kb,get_page,calculator,list_games`。
    """
    # 出口是 SCHEMAS 的**过滤结果**，不是重新构造 —— 顺序严格等于装饰器定义顺序。
    # 这点对实验有意义：实验 C 记录的"给了模型哪些工具"包含顺序，
    # 哪天改成按 names 顺序重排，历史实验的"唯一变量"就不唯一了。
    # 消费方只有两处：ReActAgent.__init__（存成 self.tools，每轮决策都带上）
    # 和 mcp_server 的参数对照。返回空列表不是错误，
    # ReActAgent._tool_schemas 会把它转成 None 表示"不给任何工具"。
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
    # 前置：name 来自**模型输出的 tool_calls**，不是我们自己校验过的值。
    # 所以"未知工具"是预期内的一类输入（模型幻觉出一个函数名），
    # 必须返回可读文案让它自我纠正，而不是 KeyError 把整条链打断。
    if fn is None:
        return f"[错误] 未知工具 {name}。可用工具：{', '.join(REGISTRY)}", False
    try:
        text = await asyncio.wait_for(fn(args, ctx), timeout=timeout)
    except TimeoutError:
        return f"[超时] 工具 {name} 超过 {timeout:.0f}s 未返回，请换用更简单的查询。", False
    except Exception as exc:  # noqa: BLE001
        logger.exception("工具 %s 执行失败", name)
        return f"[异常] 工具 {name} 失败：{type(exc).__name__}: {exc}", False
    # 唯一一次"成功"的出口，也是全仓唯一判断工具成败的地方。
    # 两个分量的消费方是分开的，别把它们混用：
    #   ok   → StepRecord.ok 与 state.tool_failures —— 指标，只看真假
    #   text → 包成 tool 消息喂回模型 + `_collect_refs` 扫里面的「出处：」行
    #
    # 前提：所有工具的失败文案都必须以 `_ERROR_PREFIXES` 里的前缀开头。
    # 这层判定是**文本约定**而不是类型标记，工具抛异常/漏前缀都不会报错，
    # 只会让失败被算成 ok=True —— 指标失真且没有任何告警，是最难发现的一类。
    #
    # 熔断文案刻意不在前缀表里：它压根没走到这个函数
    # （`_exec_call` 在调之前就判掉了），所以不会和这里的判定打架。
    #
    # lstrip 是为了容忍工具自己在文案前面加了换行或空格。
    return text, not text.lstrip().startswith(_ERROR_PREFIXES)