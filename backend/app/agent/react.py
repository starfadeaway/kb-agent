"""手搓 ReAct 内核（约 380 行，不依赖 LangGraph）。

为什么不用框架：JD 里明确写了"关注上下文、工具调度、运行状态、异常处理和效果评测，
而不只是完成模型 API 调用"。手写一版能把这些点全部暴露出来，
面试时能讲清楚"框架替我做了什么、我为什么还是自己写了一遍"。

主循环在 `_run_once`，结构就是一个显式的 while：
    查安全阀 → 问模型 → 有 tool_calls 就执行并回填 → 循环
                └─ 没有 tool_calls 就是最终答案，退出
没有隐藏的调度层，状态全在 `AgentState` 里，出问题能直接断点看。

安全阀（任一触发即停机，并让模型基于已有信息收尾）：
- 最大步数     防止工具调用死循环
- token 预算   防止上下文/费用失控
- 墙钟超时     防止慢工具拖死整条链
- 单工具超时   防止某个检索卡住

外加一道 SearchGuard 熔断（第 53 行），针对的是另一类问题：
不是"停不下来"，而是"停得太晚" —— 知识库里没有的问题它会一直检索到顶满步数。
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from app.agent.tools import ToolContext, call_tool, get_schemas
from app.core.config import Settings, get_settings
from app.llm.client import LLMClient, get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个游戏知识库助手，知识库覆盖这些游戏：{covered}。

工作方式：
1. 涉及游戏内事实（数值、机制、角色、装备、任务）时，**必须先调用 search_kb**，不要凭记忆作答。
2. search_kb 结果不足时，用 get_page 补全页面正文；不确定该查哪个库时先 list_games 确认。
3. 需要算数时用 calculator，不要心算。
4. 引用信息时在句末标注出处，格式如「[三月七]」。

约束：
- 知识库之外的最新版本改动、B站攻略作者观点，一律说明"知识库未收录"。
- 各游戏术语不要混用：{terms}
- 最后用中文回答，简洁分点。"""


def build_system_prompt(covered: dict[str, str], terms: dict[str, str]) -> str:
    """按当前已收录的游戏生成系统提示词。

    之前这里是写死两个游戏的常量字符串，结果加了第三个库之后，
    模型拿着"你是星铁与绝区零的助手"这份提示词，把原神的问题当成越界问题拒答了
    ——检索层能召回，Agent 层自己把门关上了。游戏集合必须从 settings.games 注入。
    """
    return SYSTEM_PROMPT.format(
        covered="、".join(covered.values()) or "（未配置任何游戏）",
        terms="；".join(terms.values()) or "按各游戏官方叫法",
    )

FINALIZE_PROMPT = """已达安全阀上限。请停止调用工具，基于已经拿到的信息给出最终回答；
如果信息不足以回答，就明确说明缺少哪部分信息。"""

OUT_OF_SCOPE_PROMPT = """你是一个游戏知识库助手，但当前知识库**没有收录**用户问的这个游戏。

已收录：{covered}
用户问的：{asked}

要求：
1. 用中文，**两到三句话**，直接说明「{asked}」不在当前知识库范围内。
2. 明确列出已收录的游戏，让用户知道能问什么。
3. **不要**凭记忆回答该游戏的内容 —— 你的记忆可能过时，而用户分不清
   哪句是查到的、哪句是你猜的。宁可说"未收录"，也不要给一个看起来合理的错答案。
4. 语气自然，不要道歉、不要长篇解释。可以顺带提一句知识库可以扩充。"""

# 连续多少次"检索了但没找到可信答案"后，强制让模型收尾。
# 实测不设这道闸时，问一个知识库里没有的问题，Agent 会连开 7 次检索
# （30317 prompt tokens / 12.7s）才承认查不到 —— 行为没错，成本完全失控。
MISS_LIMIT = 2
# 低于这个 top_score 认为"检索命中了但都不相关"，同样计入 miss
LOW_SCORE = 0.55


class SearchGuard:
    """给检索类工具上熔断：查询去重 + 连续低置信度强制收尾。

    为什么放在 Agent 层而不是工具里：去重和计数都是**跨调用**的状态，
    工具自身每次调用都是无状态的。
    """

    def __init__(self, miss_limit: int = MISS_LIMIT, low_score: float = LOW_SCORE) -> None:
        self.miss_limit = miss_limit
        self.low_score = low_score
        self.seen: dict[str, str] = {}      # 指纹 -> 原始 query
        self.consecutive_misses = 0
        self.dedup_blocked = 0
        self.forced_finishes = 0

    @staticmethod
    def fingerprint(query: str) -> str:
        """字符二元组指纹，与分词无关。

        一开始按空格/标点切词做指纹，结果「遗器副词条强化规则」和
        「遗器 副词条 强化 规则」被判成两个查询 —— 中文不分词就没法这么切。
        去掉所有空白标点后取字符 bigram，两种写法会落到同一个指纹上。
        """
        clean = re.sub(r"[\s,，。；;、?!？!.:：\-\"'“”‘’()（）\[\]【】]+", "", query.lower())
        if len(clean) < 2:
            return clean[:200]
        grams = {clean[i : i + 2] for i in range(len(clean) - 1)}
        return "|".join(sorted(grams))

    def check(self, name: str, args: dict) -> str | None:
        """命中熔断条件时返回替代文本，否则 None。"""
        if name not in {"search_kb", "get_page"}:
            return None
        raw = str(args.get("query") or args.get("title") or "").strip()
        if not raw:
            return None
        key = f"{name}:{self.fingerprint(raw)}"
        if key in self.seen:
            self.dedup_blocked += 1
            return (
                f"[重复查询已拦截] 「{raw}」与本轮之前的检索等价（指纹 {key}），"
                f"不会产生新信息。请换一个**实质不同**的检索角度，"
                f"或者直接基于已有信息作答。"
            )
        if self.consecutive_misses >= self.miss_limit:
            self.forced_finishes += 1
            return (
                f"[熔断] 此前已连续 {self.consecutive_misses} 次检索未找到可信内容，"
                f"本次「{raw}」不再执行。请停止检索，直接说明知识库未收录该部分信息。"
            )
        self.seen[key] = raw
        return None

    def observe(self, name: str, result: str) -> None:
        """根据工具结果更新连续 miss 计数。

        被熔断拦截的结果不能参与计数 —— 它既不是成功也不是"没找到"，
        如果走兜底分支把 misses 清零，熔断会在下一次调用时自我解除。
        """
        if name != "search_kb":
            return
        if result.startswith("[重复查询已拦截]") or result.startswith("[熔断]"):
            return
        head = result[:200]
        if "status=empty" in head:
            self.consecutive_misses += 1
            return
        for token in head.split():
            if token.startswith("top_score="):
                try:
                    score = float(token.split("=", 1)[1])
                except ValueError:
                    score = 0.0
                self.consecutive_misses = self.consecutive_misses + 1 if score < self.low_score else 0
                return
        self.consecutive_misses = 0

    def stats(self) -> dict:
        return {
            "unique_queries": len(self.seen),
            "dedup_blocked": self.dedup_blocked,
            "forced_finishes": self.forced_finishes,
            "consecutive_misses": self.consecutive_misses,
        }


@dataclass
class StepRecord:
    step: int
    tool_name: str
    args: dict
    ok: bool
    latency_ms: int
    result_chars: int
    error: str | None = None


@dataclass
class AgentState:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    session_key: str | None = None
    game: str = "sr"
    question: str = ""
    """用户原始提问。必须单独存：运行结束后 messages[-1] 是模型的回复，
    直接用它当 question 会把答案存成问题，落库后完全没法复现。"""
    messages: list[dict] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    answer: str = ""
    refs: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_failures: int = 0
    stop_reason: str = "answer"
    elapsed_ms: int = 0
    guard: SearchGuard | None = None
    #: 路由判定为「提到了未收录的游戏」时的 slug；None 表示在覆盖范围内。
    #: 单独存一份而不是塞进 stop_reason，是为了让评测能把"越界拒答"
    #: 和"检索不到"统计成两件不同的事 —— 前者是正确行为，后者是缺陷。
    uncovered_game: str | None = None

    def tool_call_count(self) -> int:
        return len(self.steps)

    def guard_stats(self) -> dict:
        return self.guard.stats() if self.guard else {}


_GAME_HINTS = {
    "zzz": ("绝区零", "音擎", "驱动盘", "邦布", "邦布等级", "代理人", "艾莲", "比利"),
    "sr": ("星穹铁道", "星铁", "光锥", "命途", "行迹", "位面饰品", "忘却之庭", "开拓者", "三月七"),
    "ys": ("原神", "提瓦特", "璃月", "须弥", "稻妻", "圣遗物", "大伟丘", "须弥蕈"),
}

# 明确知道存在、但知识库尚未收录的游戏。
#
# 只放**游戏名**和绝不歧义的专有词，刻意不放角色名 —— 代价是不对称的：
# 漏判只是多搜一次库；误判会把一个本来能答的问题错误地拒答掉。
UNCOVERED_GAMES: dict[str, tuple[str, tuple[str, ...]]] = {
    "hhzr": ("崩坏3", ("崩坏3", "崩坏三", "第三次崩坏")),
    "lysk": ("英雄联盟", ("英雄联盟", "召唤师峡谷")),
    "bzhy": ("明日方舟", ("明日方舟",)),
    "wmy": ("王者荣耀", ("王者荣耀",)),
    "dmzz": ("动物森友会", ("动森", "动物森友会")),
    "jsr": ("塞尔达传说", ("塞尔达", "王国之泪", "旷野之息")),
    "sw": ("三角洲行动", ("三角洲行动",)),
    "mc": ("鸣潮", ("鸣潮",)),
}


def game_display_name(slug: str | None, covered: dict[str, str]) -> str:
    """游戏 slug → 中文显示名。未收录的走 UNCOVERED_GAMES，最后兜底用 slug。"""
    if not slug:
        return "未知游戏"
    if slug in covered:
        return covered[slug]
    return UNCOVERED_GAMES.get(slug, (slug, ()))[0]


def route_game(question: str, covered: dict[str, str]) -> str | None:
    """判断问题属于哪个游戏，返回 slug。

    返回 `None` = **没有提到任何已知游戏**，此时沿用调用方的默认值。

    刻意不返回默认值兜底。原来这里"两个都不匹配就返回 sr"，
    后果是「原神甘雨的突破材料」被当成星铁问题去搜星铁库，
    搜不到后报"知识库未收录" —— **一个建立在错误前提上的结论**：
    库里其实压根没有原神这个库，问题不是"没收录这条"，而是"没有这个游戏"。

    覆盖范围从 `covered` 参数来而不是写死常量，所以 `GAME_TERTIARY=ys` 配上
    灌好的语料，`ys` 的关键词会自动从"未收录"变成"可路由"。
    """
    for game in covered:
        hints = _GAME_HINTS.get(game)
        if hints and any(h in question for h in hints):
            return game
    for slug, (_, hints) in UNCOVERED_GAMES.items():
        if slug not in covered and any(h in question for h in hints):
            return slug
    return None


def _split_for_stream(text: str, min_len: int = 6) -> list[str]:
    """把完整答案切成适合打字机效果的片段（优先按中文标点断句）。"""
    if not text:
        return []
    pieces: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in "。！？；\n" or len(buf) >= min_len * 4:
            pieces.append(buf)
            buf = ""
    if buf:
        pieces.append(buf)
    return pieces


class ReActAgent:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        tools: list[dict] | None = None,
        chat_fn: Callable[..., Awaitable[dict]] | None = None,
        backend: str | None = None,
    ) -> None:
        self.s = settings or get_settings()
        # 兜底用 get_llm() 而不是 LLMClient()：各 new 一个的话，
        # embedding/rerank 缓存就分裂成两个池子 —— /api/health 报的那个是空的，
        # 而同一条 query 在 /chat 和 /retrieve 之间也就不再互相命中了。
        # 需要独立实例的实验脚本仍可显式传 llm= 覆盖。
        self.llm = llm or get_llm(self.s)
        self.backend = backend
        self.last_state: AgentState | None = None
        """最近一次运行的状态。流式接口拿不到返回值，靠它把轨迹落库。"""
        # 可选：限制本轮允许使用的工具。实验 C（C0/C1/C2）靠它做能力消融 ——
        # 消融的变量必须是「模型能看到哪些工具声明」，
        # 而不是「模型调用了哪些工具」（后者会被模型的随机选择污染）。
        # 默认取 Settings.agent_tools（生产默认只开 search_kb，依据见 config 里的注释）。
        self.tools = tools if tools is not None else get_schemas(self.s.agent_tool_names)
        # 可选：替换 chat 调用。实验 E 靠它切到 MiniMax 等别的基座，
        # 好处是实验跑的是**生产代码本身**这条循环，而不是脚本里抄一份副本 ——
        # 抄一份的话，测出来的就不是线上真正在跑的东西了。
        self.chat_fn = chat_fn or self._default_chat

    async def _default_chat(self, messages: list[dict], **kw) -> dict:
        return await self.llm.chat(messages, backend=self.backend, **kw)

    @property
    def _tool_schemas(self) -> list[dict] | None:
        """空列表 = 不给任何工具（退化成单轮问答）。"""
        return self.tools or None

    async def run(
        self,
        question: str,
        *,
        session_key: str | None = None,
        history: list[dict] | None = None,
        ctx: ToolContext | None = None,
        stream: bool = False,
    ) -> AgentState | AsyncIterator[dict]:
        if stream:
            return self._run_stream(question, session_key, history, ctx)
        return await self._run_once(question, session_key, history, ctx)

    # ------------------------------------------------------------------ #
    def _prepare(self, question: str, session_key: str | None, history: list[dict] | None, ctx: ToolContext | None):
        context = ctx or ToolContext.build()
        covered = self.s.games
        routed = route_game(question, covered)
        # 未提到游戏 → 用上下文里的默认；提到了未收录的游戏 → 标记出来走拒答
        uncovered = routed is not None and routed not in covered
        game = routed if routed is not None and not uncovered else context.game
        context.game = game

        if uncovered:
            system = OUT_OF_SCOPE_PROMPT.format(
                asked=game_display_name(routed, covered),
                covered="、".join(covered.values()),
            )
        else:
            system = build_system_prompt(covered, self.s.game_terms)
        messages: list[dict] = [{"role": "system", "content": system}]
        # 历史只保留最近 4 轮，避免长会话把预算吃光
        if history:
            messages.extend(history[-8:])
        messages.append({"role": "user", "content": question})
        state = AgentState(
            session_key=session_key, game=game, messages=messages, question=question
        )
        state.uncovered_game = routed if uncovered else None
        return context, messages, state

    async def _refuse_uncovered(self, state: AgentState) -> None:
        """未收录游戏：无工具、单轮，直接给出边界说明。

        为什么不让模型去搜再自己承认搜不到：那条路径会产生
        「知识库未收录 X」这种**看起来合理但前提错误**的回答 ——
        真实情况是压根没有这个游戏的库。两者的区别用户分辨不出来，
        而我们必须让他分辨得出来。
        """
        t0 = time.perf_counter()
        try:
            out = await self.chat_fn(
                state.messages, temperature=0.0, max_tokens=512  # 不传 tools
            )
            state.answer = out["content"]
            state.prompt_tokens += out["usage"]["prompt_tokens"]
            state.completion_tokens += out["usage"]["completion_tokens"]
        except Exception as exc:  # noqa: BLE001
            logger.exception("拒答生成失败")
            state.answer = (
                f"[系统] 当前知识库未收录该游戏，且生成说明失败：{type(exc).__name__}: {exc}"
            )
        state.stop_reason = "out_of_scope"
        state.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        state.messages = state.messages + [{"role": "assistant", "content": state.answer}]
        self.last_state = state

    def _budget_exceeded(self, state: AgentState, deadline: float) -> str | None:
        """返回触发的安全阀名称，没触发返回 None。

        三道预算里 token 预算用**累计值**而不是单步值：
        单步 token 少但步数多，照样能把上下文撑爆。
        墙钟用 monotonic 而非 time.time()，后者会被系统时钟调整影响。
        """
        if len(state.steps) >= self.s.agent_max_steps:
            return "max_steps"
        if state.prompt_tokens + state.completion_tokens >= self.s.agent_token_budget:
            return "token_budget"
        if time.monotonic() > deadline:
            return "wall_timeout"
        return None

    async def _finalize(self, messages: list[dict], state: AgentState, reason: str) -> None:
        """安全阀触发后的强制收尾：不带工具再问一次，让它把话说完。"""
        state.stop_reason = reason
        try:
            # 走 chat_fn 而不是 self.llm.chat：切到别的基座时，
            # 收尾这一步也必须用同一个模型，否则实验测的是"半程换模型"。
            out = await self.chat_fn(
                messages + [{"role": "user", "content": FINALIZE_PROMPT}],
                temperature=0.0,
                max_tokens=1024,
            )
            state.answer = out["content"]
            state.prompt_tokens += out["usage"]["prompt_tokens"]
            state.completion_tokens += out["usage"]["completion_tokens"]
        except Exception as exc:  # noqa: BLE001
            logger.exception("收尾失败")
            state.answer = f"[系统] 达到安全阀上限({reason})，且生成最终回答失败：{exc}"
        finally:
            self._collect_refs(state)

    def _collect_refs(self, state: AgentState) -> None:
        """从工具结果里抽引用，供前端展示与忠实度评测。"""
        seen: set[str] = set()
        for msg in state.messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            for line in content.splitlines():
                if line.startswith("出处："):
                    url = line.replace("出处：", "").strip()
                    if url and url not in seen:
                        seen.add(url)
                        state.refs.append({"url": url})

    async def _exec_call(
        self, call: dict, ctx: ToolContext, guard: SearchGuard, state: AgentState
    ) -> tuple[str, StepRecord]:
        """执行一次工具调用，返回 (工具输出文本, 已登记的 StepRecord)。

        非流式与流式两条路径共用这一份。顺序很关键，不能调换：
          check（熔断）→ 执行 → observe（计数）→ 记 StepRecord
        observe 必须在 check 之后，它靠返回值前缀区分
        「真去查了没查到」和「压根没查」，两者对 miss 计数的意义相反。

        任何异常都在这里兜住并转成可读文本喂回模型 —— 抛出去会直接中断整条链，
        模型也就失去了自己调整策略的机会。
        """
        ts = time.perf_counter()
        blocked = guard.check(call["name"], call["args"])
        if blocked is not None:
            # 熔断命中：不执行工具，但仍然把"为什么没执行"喂回去，
            # 让模型读到之后能自己调整策略，而不是看到空结果反复重试。
            text, ok = blocked, False
        else:
            try:
                text, ok = await call_tool(call["name"], call["args"], ctx, timeout=self.s.tool_timeout)
            except Exception as exc:  # noqa: BLE001
                text, ok = f"[异常] {exc}", False
        guard.observe(call["name"], text)
        dt = int((time.perf_counter() - ts) * 1000)
        if not ok:
            state.tool_failures += 1
        rec = StepRecord(
            step=len(state.steps),
            tool_name=call["name"],
            args=call["args"],
            ok=ok,
            latency_ms=dt,
            result_chars=len(text),
            error=None if ok else text[:200],
        )
        state.steps.append(rec)
        return text, rec

    def _tool_result_message(self, call: dict, text: str) -> dict:
        """工具结果消息。tool_call_id 必须与 assistant 里的 id 一致，否则消息链断掉。"""
        return {"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": text}

    # ------------------------------------------------------------------ #
    async def _run_once(
        self,
        question: str,
        session_key: str | None,
        history: list[dict] | None,
        ctx: ToolContext | None,
    ) -> AgentState:
        context, messages, state = self._prepare(question, session_key, history, ctx)
        # 越界问题：不进 ReAct 循环，直接拒答（见 _refuse_uncovered 的理由）
        if state.uncovered_game:
            await self._refuse_uncovered(state)
            return state
        guard = SearchGuard()
        state.guard = guard
        t0 = time.perf_counter()
        # deadline 相对 monotonic 起算，不受系统时间调整影响
        deadline = t0 + self.s.agent_wall_timeout

        # ---- ReAct 主循环：安全阀 → 决策 → 执行 → 回填 ----
        while True:
            # 1) 安全阀检查放在每轮开头：工具执行完、下一轮决策前，
            #    这样"最后一轮已超出预算"的情况也能被拦住。
            reason = self._budget_exceeded(state, deadline)
            if reason:
                await self._finalize(messages, state, reason)
                break

            # 2) 决策：把完整 messages 交给模型，带工具 schema
            try:
                out = await self.chat_fn(messages, tools=self._tool_schemas, temperature=0.0)
            except Exception as exc:  # noqa: BLE001
                # 模型调用失败不重试：重试只会放大成本，且同样的输入大概率同样失败。
                # 直接把错误如实返回给用户，比假装还在思考要好。
                logger.exception("模型调用失败")
                state.stop_reason = "error"
                state.answer = f"[系统] 模型调用失败：{type(exc).__name__}: {exc}"
                break

            state.prompt_tokens += out["usage"]["prompt_tokens"]
            state.completion_tokens += out["usage"]["completion_tokens"]

            # 3) 没有 tool_calls = 模型认为信息够了，这是终止信号
            if not out["tool_calls"]:
                messages.append({"role": "assistant", "content": out["content"] or ""})
                state.answer = out["content"]
                state.stop_reason = "answer"
                state.messages = messages
                state.elapsed_ms = int((time.perf_counter() - t0) * 1000)
                self._collect_refs(state)
                self.last_state = state
                return state

            # 工具调用：assistant 消息必须带 tool_calls，后面的 tool 消息才有锚点。
            # 只发一句纯文本 assistant，下一轮 API 会报
            # "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"。
            messages.append(self.llm.assistant_tool_message(out["content"], out["tool_calls"]))

            # 4) 执行工具。失败也把错误文本喂回去让模型自己纠正 ——
            #    检索不到和报错的处理路径不一样，直接抛异常中断会丢掉模型自我调整的机会。
            for call in out["tool_calls"]:
                text, _ = await self._exec_call(call, context, guard, state)
                messages.append(self._tool_result_message(call, text))

        # 只有走到安全阀才会到这里
        state.messages = messages
        state.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return state

    # ------------------------------------------------------------------ #
    async def _run_stream(
        self,
        question: str,
        session_key: str | None,
        history: list[dict] | None,
        ctx: ToolContext | None,
    ) -> AsyncIterator[dict]:
        """SSE 事件流：token / tool_start / tool_end / done。

        工具调用过程也推给前端 —— 用户能看到"它在检索什么"，
        这是 Agent 产品化和"调 API"最直观的差别。
        """
        context, messages, state = self._prepare(question, session_key, history, ctx)
        if state.uncovered_game:
            t0 = time.perf_counter()
            await self._refuse_uncovered(state)
            state.elapsed_ms = int((time.perf_counter() - t0) * 1000)
            yield {"event": "out_of_scope", "data": {"game": state.uncovered_game}}
            yield {"event": "done", "data": {
                "run_id": state.run_id, "game": state.game, "answer": state.answer,
                "steps": 0, "tool_failures": 0, "stop_reason": state.stop_reason,
                "elapsed_ms": state.elapsed_ms, "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens, "refs": [], "trace": [],
                "out_of_scope": state.uncovered_game,
            }}
            return
        guard = SearchGuard()
        state.guard = guard
        t0 = time.perf_counter()
        deadline = t0 + self.s.agent_wall_timeout

        try:
            while True:
                reason = self._budget_exceeded(state, deadline)
                if reason:
                    yield {"event": "limit", "data": {"reason": reason}}
                    break

                out = await self.chat_fn(messages, tools=self._tool_schemas, temperature=0.0)
                state.prompt_tokens += out["usage"]["prompt_tokens"]
                state.completion_tokens += out["usage"]["completion_tokens"]

                if not out["tool_calls"]:
                    # 答案已经随本次带 tools 的调用完整返回了。
                    # 再发一次流式请求等于把同一段话生成两遍：多花一份 token，
                    # 还会因为要重放 tool 消息而触发
                    # "role='tool' must follow a message with 'tool_calls'"。
                    # 所以直接按标点切片推送，复用已有文本。
                    messages.append({"role": "assistant", "content": out["content"]})
                    for piece in _split_for_stream(out["content"]):
                        state.answer += piece
                        yield {"event": "token", "data": {"text": piece}}
                    state.stop_reason = "answer"
                    break

                messages.append(self.llm.assistant_tool_message(out["content"], out["tool_calls"]))
                for call in out["tool_calls"]:
                    yield {"event": "tool_start", "data": {"name": call["name"], "args": call["args"]}}
                    text, rec = await self._exec_call(call, context, guard, state)
                    yield {
                        "event": "tool_end",
                        "data": {"name": call["name"], "ok": rec.ok, "ms": rec.latency_ms, "chars": rec.result_chars},
                    }
                    messages.append(self._tool_result_message(call, text))
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式运行失败")
            yield {"event": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}}
        finally:
            self._collect_refs(state)
            state.elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self.last_state = state
            yield {
                "event": "done",
                "data": {
                    "run_id": state.run_id,
                    "game": state.game,
                    "answer": state.answer,
                    "steps": len(state.steps),
                    "tool_failures": state.tool_failures,
                    "stop_reason": state.stop_reason,
                    "elapsed_ms": state.elapsed_ms,
                    "guard": state.guard_stats(),
                    "prompt_tokens": state.prompt_tokens,
                    "completion_tokens": state.completion_tokens,
                    "refs": state.refs,
                    "trace": [
                        {
                            "step": s.step,
                            "tool": s.tool_name,
                            "args": s.args,
                            "ok": s.ok,
                            "ms": s.latency_ms,
                            "chars": s.result_chars,
                        }
                        for s in state.steps
                    ],
                },
            }