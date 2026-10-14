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

from app.agent.tools import (
    _INFRA_FAILURE_PREFIXES,  # noqa: F401 单一来源，避免两处前缀不同步
    ToolContext,
    call_tool,
    get_schemas,
)
from app.core.config import Settings, get_settings
from app.llm.client import LLMClient, get_llm
from app.security.guard import INSTRUCTION_NOTICE, scan_and_wrap

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个游戏知识库助手，知识库覆盖这些游戏：{covered}。

工作方式：
1. 涉及游戏内事实（数值、机制、角色、装备、任务）时，**必须先调用 search_kb**，不要凭记忆作答。
2. search_kb 结果不足时，用 get_page 补全页面正文；不确定该查哪个库时先 list_games 确认。
3. 需要算数时用 calculator，不要心算。
4. 引用信息时在句末标注出处，格式如「[三月七]」。
5. 检索结果带有**页面最后修订日期**。游戏数值会随版本改动，
   涉及具体数值时优先采信修订较新的条目；如果引用的条目修订日期明显早于其他条目，
   在回答末尾用一句话说明该数据的页面修订日期，提醒用户可能已过时。

约束：
- 知识库之外的最新版本改动、B站攻略作者观点，一律说明"知识库未收录"。
- 各游戏术语不要混用：{terms}
- 最后用中文回答，简洁分点。

{security}"""


def build_system_prompt(covered: dict[str, str], terms: dict[str, str],
                        *, disclose_revision: bool = True) -> str:
    """按当前已收录的游戏生成系统提示词。

    之前这里是写死两个游戏的常量字符串，结果加了第三个库之后，
    模型拿着"你是星铁与绝区零的助手"这份提示词，把原神的问题当成越界问题拒答了
    ——检索层能召回，Agent 层自己把门关上了。游戏集合必须从 settings.games 注入。

    `disclose_revision=False` 时整条第 5 点消失（连编号都不占位）。
    它必须与 `ToolContext.disclose_revision` 同源取值：
    提示词说有日期、工具不给日期，比两者都没有更糟——
    模型会去找那个字段，找不到就当成「没有时效信息」，
    于是披露层被消融掉的那一半反而制造了新的噪声。
    """
    if not disclose_revision:
        stripped = _strip_revision_clause(SYSTEM_PROMPT)
        return stripped.format(
            covered="、".join(covered.values()) or "（未配置任何游戏）",
            terms="；".join(terms.values()) or "按各游戏官方叫法",
            security=INSTRUCTION_NOTICE,
        )
    return SYSTEM_PROMPT.format(
        covered="、".join(covered.values()) or "（未配置任何游戏）",
        terms="；".join(terms.values()) or "按各游戏官方叫法",
        security=INSTRUCTION_NOTICE,
    )


#: 删披露层那一条时用**行首带编号的整段**做锚点，而不是只匹配半句话。
#: 只匹配「5. 检索结果带有**页面最后修订日期**。」的话，后面那两行续行
#: 会留在提示词里变成一段没有编号的悬空文字 —— 看起来像正常提示词，
#: 但语义已经废了，而且没有任何东西会报错。
_REVISION_CLAUSE_ANCHOR = "5. 检索结果带有**页面最后修订日期**。"


def _strip_revision_clause(template: str) -> str:
    """从系统提示词模板里去掉披露层那一整条，返回**未格式化**的模板。

    为什么单独一个函数：删条款这件事本身需要一个**失败可见**的行为。
    锚点一旦对不上（比如有人重排了条款编号），静默返回原模板的话，
    消融档就会带着完整披露层去跑 —— 实验照常出结果，
    F1 与 F2 两档其实一样，最后得出「披露层无影响」的错误结论。
    所以这里直接抛，而不是兜底。
    """
    lines = template.split("\n")
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith(_REVISION_CLAUSE_ANCHOR))
    except StopIteration:
        raise ValueError(
            f"SYSTEM_PROMPT 里找不到以 {_REVISION_CLAUSE_ANCHOR!r} 开头的那一行 —— "
            "改 SYSTEM_PROMPT 时必须同步改这个锚点，"
            "否则消融档会带着一条指向不存在字段的指令去跑实验，"
            "而结果看起来完全正常。"
        ) from None
    # 往后只吃**缩进续行**，不吃后面的空行。
    # 把空行一起吃掉会让「工作方式：」那一段和「约束：」贴在一起，
    # 提示词还能用，但版式和生产的差了一处 —— 而实验 F 的两档提示词
    # 应当只差那一条条款，版式差异属于没控制住的变量。
    end = start + 1
    while end < len(lines) and lines[end].startswith(" "):
        end += 1
    rest = lines[:start] + lines[end:]
    return "\n".join(rest)


def production_system_prompt() -> str:
    """**已格式化**的生产系统提示词（披露层开启）。

    单独提供这个函数，是因为仓库里长期存在一处口径混用：
    `SYSTEM_PROMPT` 是**未格式化的模板**，而 `scripts/exp_c.py` 的单轮档
    直接把它当系统消息发了出去 —— 于是 C0 档模型收到的是
    「知识库覆盖这些游戏：{covered}」，连 `{security}` 注入防护那段都没被替换。

    凡是断言「提示词里有什么/没有什么」的地方，都应该查这个函数的返回值，
    而不是查模板 —— 模板里那句「检索结果带有页面最后修订日期」是**永远**在的，
    拿它做断言只能证明模板没被改，证明不了生产提示词里有那句话。
    """
    # 必须原样传 `settings.games`（slug → **显示名**）。
    # 曾经写成 `{g: g for g in settings.games}`，提示词里就出现
    # 「知识库覆盖这些游戏：sr、zzz、ys」—— 全是 slug，
    # 而旧版硬编码提示词写的是「崩坏：星穹铁道、绝区零、原神」。
    # 这类错误不会让任何东西报错，只是把模型认游戏的关键线索换成了缩写。
    s = get_settings()
    return build_system_prompt(s.games, s.game_terms)


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
        # 调用方 `_exec_call` 在**执行工具之前**问这一句：拿到非 None 就完全不调工具，
        # 把返回的这段文本直接当工具输出喂回模型。所以这里的措辞是写给模型看的，
        # 不是给人看的错误提示。
        #
        # 前提：`self.seen` 与两个计数器是**每轮运行**新建的（见 `_run_once` 里的
        # `guard = SearchGuard()`），不是跨请求共享。
        # 这个前提一旦破（把 guard 挂到 session 上），A 用户查过的指纹会拦掉
        # B 用户完全正常的第一次检索，而且两边都看不到任何异常。
        #
        # 只管检索类工具：calculator / list_games 没有"换个说法再搜一遍"这个失败模式，
        # 用指纹去拦它们会把模型的正常重算判成重复调用。
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

        # 返回 None = 放行，并且这次查询**已经登记进 seen**（上面最后一行）。
        # 登记放在放行之后是个有意的顺序：被去重/熔断拦下的查询不该占 seen，
        # 否则模型改一次措辞就可能撞上自己上一轮被拦的那条，白白少一次检索机会。
        #
        # 出口只有两种语义，调用方必须靠 `blocked is None` 区分：
        # 非 None → 不执行工具、ok=False、但**照常喂回模型**。
        # 丢弃这个返回值等于把熔断变成静默丢消息，模型只会看到"工具没回话"。

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
        # 下面三条分支的**顺序就是优先级**，且全都建立在一个前提上：
        # 「判据一定在结果的前 200 个字符里」。
        # 这个前提由 search_kb 的输出格式保证（header 是第一行），不是由这里保证 ——
        # 任何改动 header 位置/长度的改动都必须同步放宽 head，否则熔断会静默失效。
        if result.startswith(_INFRA_FAILURE_PREFIXES):
            # 基础设施失败期间继续检索只会重复失败，必须计入 miss。
            # 原先这里没有这一分支，失败会落到末尾把 misses 清零 ——
            # 熔断因此自我解除，Agent 一路空转到 agent_max_steps，
            # 而且从 guard_stats 完全看不出来（consecutive_misses 一直是 0）。
            self.consecutive_misses += 1
            return
        if "status=empty" in head:
            self.consecutive_misses += 1
            return
        # 命中这一段的前提：正常路径下 `top_score=` 是**按空格分隔**的一个独立 token
        # （search_kb 的 header 就是这么拼的）。分隔符一变（改成逗号），
        # 这里会静默走到最后一行把 misses 清零 —— 熔断看着还在，实则永不触发。
        for token in head.split():
            if token.startswith("top_score="):
                try:
                    score = float(token.split("=", 1)[1])
                except ValueError:
                    score = 0.0
                # 注意这里是"有则清零"，不是"累加"：一次好结果就足以证明
                # 前一次的低分可能只是这次 query 写得不好，重新给了机会。
                self.consecutive_misses = self.consecutive_misses + 1 if score < self.low_score else 0
                return
        # 兜底：完全没有可判定的分数（格式被改过 / 工具换了输出形态）。
        # 选择清零而不是累加 —— 宁可熔断晚触发，也不要在判据失效时
        # 把 Agent 提前掐死在还有希望的一轮上。
        self.consecutive_misses = 0

    def stats(self) -> dict:
        # 唯一消费方是 `AgentState.guard_stats()`，再往上是 /api/chat 响应、
        # SSE 的 done 事件和评测产物里的 guard 字段。
        #
        # 刻意只导出这四个计数：它们对应三种**可区分**的失败模式 ——
        # 重复查询（模型在原地打转）/ 熔断触发（查了但没查到）/
        # 连续低置信（快到了但对不上）。少导一个就没法判断
        # "Agent 是省了 token 还是压根查不到"。
        return {
            "unique_queries": len(self.seen),
            "dedup_blocked": self.dedup_blocked,
            "forced_finishes": self.forced_finishes,
            "consecutive_misses": self.consecutive_misses,
        }

        # 读这四个数时的一条推论：`forced_finishes > 0` 一定意味着"查了但没查到"，
        # 不可能是"压根没查"。因为 `check()` 只在 seen 已有一次真实查询之后才可能
        # 走到熔断分支，而 `observe()` 只在真跑过 search_kb 时才动计数器。
        # 评测脚本靠这个区分"正确收尾"和"被熔断掐掉的缺陷"，不要合并成一个数。


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
    #: 本次运行里从工具结果中检出的疑似注入，[(tool_name, ScanResult), ...]。
    #: 只留摘要不进答案、不进引用 —— 检测结果既不是知识库内容也不是用户诉求，
    #: 把它当答案的一部分复述出去，等于二次执行攻击。
    injections: list[tuple[str, object]] = field(default_factory=list)

    def note_injection(self, tool_name: str, result) -> None:
        """记账只记真命中。

        防御性判断放在方法内部而不是只靠调用方：
        一个"没有命中也记账"的接口，会让 suspect_tools 这个数
        变成"扫过多少次"而不是"命中多少次"，指标当场失去意义。
        """
        if not result.suspect:
            return
        self.injections.append((tool_name, result))
        logger.warning(
            "工具结果命中注入规则 tool=%s severity=%s rules=%s",
            tool_name, result.severity, result.summary()["rules"],
        )

    def injection_summary(self) -> dict:
        """给 `/api/chat` 响应与评测产物用的计数视图。"""
        if not self.injections:
            return {"suspect_tools": 0, "severity": "none", "rules": []}
        rules: set[str] = set()
        worst = "warn"
        for _name, res in self.injections:
            rules.update(res.summary()["rules"])
            if res.severity == "block":
                worst = "block"
        return {
            "suspect_tools": len(self.injections),
            "severity": worst,
            "rules": sorted(rules),
        }

    def guard_stats(self) -> dict:
        # guard 为 None 只发生在"越界拒答"那条路径 —— 它压根没进 ReAct 循环，
        # SearchGuard 也没被创建。此时返回 **空 dict** 而不是四个 0，
        # 与 `_run_stream` 越界分支刻意不发 guard 字段是同一个理由：
        # 「没测到」和「测到是 0」在指标上必须能区分开。
        # 消费方要按 `.get()` 取，别假设四个 key 一定在。
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
    # 两趟扫描的分工：第一趟只看**已收录**游戏（词表 = _GAME_HINTS ∩ covered），
    # 第二趟只看**明确知道存在但未收录**的游戏。两张词表目前不相交，
    # 所以顺序交换不会改变结果 —— 但保持"先覆盖内、后覆盖外"是为了将来
    # 某个游戏既进了 _GAME_HINTS 又可能未收录时，语义仍然成立。
    #
    # 覆盖不到的第三种情况：slug 既不在 covered、也不在 UNCOVERED_GAMES
    # （新库在语料灌好前的过渡期）。两趟都扫不到 → 返回 None →
    # 调用方按"没提到任何游戏"处理、退回默认库。这是可接受的降级：
    # 再"猜"一个就会重演拿星铁库去答原神问题的那个 bug。
    for game in covered:
        hints = _GAME_HINTS.get(game)
        if hints and any(h in question for h in hints):
            return game
    for slug, (_, hints) in UNCOVERED_GAMES.items():
        if slug not in covered and any(h in question for h in hints):
            return slug
    # 返回 None 在下游**不是拒答**，而是"沿用调用方的默认库"
    # （见 `_prepare`）。用户问「光锥有哪些？」这种不点名游戏的问题就走这条路：
    # 先拿默认库搜一次，搜不到会在 ReAct 循环里被 SearchGuard 收尾，
    # 用户看到的是"知识库未收录该部分信息"，而不是一句莫名其妙的路由失败。
    return None


def trim_history(history: list[dict], max_chars: int) -> list[dict]:
    """把多轮历史裁到 `max_chars` 字符以内，**整轮丢弃**，返回时间正序。

    从最新一条往回累加：最新那轮才是当前问题的直接上文，砍它比砍三轮前的更伤。

    不做摘要：历史是「问 / 答」成对的，摘要会把引用出处、已确认/未确认这类
    结构信息抹平，而出处正是忠实度评测要对齐的东西。宁可少记几轮，
    也不要把问答对压成一篇看不出对错的散文。

    至少留最后一条（下面的 `if kept`）：极端情况下单条就超预算，
    全丢等于把多轮会话静默降级成单轮；留一条的代价有上界，
    真超了 `agent_token_budget` 也会在循环里被拦。
    """
    kept: list[dict] = []
    used = 0
    # 循环的终止由三件事共同保证，两个出口都必要：
    #   a) `used + n > max_chars` → 预算耗尽，break（`kept` 非空时才判，所以单条超限仍会留下）
    #   b) history 走完 → 自然结束
    #   c) 最坏情况 a 与 b 都不成立 → 整段历史都塞得下，原样返回（不算裁剪失败）
    # 退出时 kept 是**逆序**的，下面 reverse 之后才允许被拼接进 messages。
    for msg in reversed(history):
        n = len(str(msg.get("content") or ""))
        if kept and used + n > max_chars:
            break
        kept.append(msg)
        used += n
    # 反转的必要性：这里是从最新往回取的，直接 extend 会得到
    # "最新问答 → 上一轮问答 → 更早问答"的时间倒序，
    # 模型会把后一轮当成对前一轮的追问，忠实度评测里的引用归属也会整体错位。
    kept.reverse()
    return kept


def _split_for_stream(text: str, min_len: int = 6) -> list[str]:
    """把完整答案切成适合打字机效果的片段（优先按中文标点断句）。"""
    # 出口有一条硬约束：**"".join(pieces) 必须严格等于 text**。
    # 消费方 `_run_stream` 是逐片 `state.answer += piece` 累加的，
    # 任何"顺手 strip 一下"或去重相邻片段的优化，都会让最终答案与模型输出对不上，
    # 而这种偏差在评测里表现为莫名其妙的忠实度扣分，很难归因。
    # 空串是唯一允许的特例（`not text` 那条），调用方拿到 [] 就不会发任何 token 事件。
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
        # 出口有**两种形状**，且是这个返回类型（而不是拆成 run/run_stream）的原因：
        # 非流式拿 AgentState 直接落库；流式拿一个异步迭代器边跑边推。
        # 调用方必须显式传 stream=True 才知道拿的是哪种 ——
        # 误用（比如对迭代器做 len()）不会报错，只会拿到一个看不懂的 TypeError。
        if stream:
            return self._run_stream(question, session_key, history, ctx)
        return await self._run_once(question, session_key, history, ctx)

    # ------------------------------------------------------------------ #
    def _prepare(self, question: str, session_key: str | None, history: list[dict] | None, ctx: ToolContext | None):
        context = ctx or ToolContext.build()
        covered = self.s.games
        routed = route_game(question, covered)
        # routed 有三个落点，决定完全不同的后续路径，进入下面的分支前先钉清楚：
        #   1) 命中 covered              → 正常 system prompt + 工具默认检索该库
        #   2) 命中 UNCOVERED_GAMES      → uncovered=True，换拒答 prompt，不进 ReAct 循环
        #   3) routed is None            → 沿用 ctx.game（用户没点名游戏）
        # 第 2 种情况下 game **刻意退回** context.game 而不是那个越界 slug：
        # 真实目标由 state.uncovered_game 单独承载，state.game 保持成一个合法的库，
        # 这样 done 事件里的 game 字段不会变成一个库里根本不存在的值。
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
            # 披露层两半必须同源：工具侧 `ToolContext.disclose_revision` 决定
            # search_kb / get_page 输不输出日期，这里决定提示词要不要教模型用日期。
            # 两处读同一个字段，实验 F 才能用一次赋值完成整层消融。
            system = build_system_prompt(covered, self.s.game_terms,
                                         disclose_revision=context.disclose_revision)
        messages: list[dict] = [{"role": "system", "content": system}]
        # 历史有两道限制，职责不同，别混成一道：
        #   load_history(limit=8)  控 DB IO —— 最多取多少行；
        #   trim_history(...)      控上下文预算 —— 取回来的能塞多少字符。
        # 早先只有前者：8 条短问答和 8 条长回答占的预算差好几倍，
        # 而 agent_token_budget 只在 ReAct 循环里检查、不管历史，
        # 表现为"没怎么检索就被判 token_budget 收尾"。
        if history:
            messages.extend(trim_history(history, self.s.agent_history_chars))
        # 从这一行起，messages 只增不改 —— ReAct 循环里一路 append，没有任何地方回删。
        # 这是一个刻意的约束：超预算只能在 `_budget_exceeded` 里靠"不再调模型"处理，
        # 靠裁剪已经发出去的消息是做不到的（上下文里已经包含了被裁的部分）。
        #
        # user 消息追加在**最后**：它是模型的直接任务，写在历史前面会让模型
        # 把历史里的上一轮问题当成本轮问题回答。
        messages.append({"role": "user", "content": question})
        state = AgentState(
            session_key=session_key, game=game, messages=messages, question=question
        )
        state.uncovered_game = routed if uncovered else None
        # 三个返回值的分工（`_run_once` 与 `_run_stream` 完全一致）：
        #   context → 只被透传给 `_exec_call` 里的 call_tool，本函数之外没人读它的字段
        #   messages → 循环的工作副本，跑完才整体赋给 state.messages
        #   state    → 循环内被逐步写入，最后 `return state` 返回的就是它
        # 刻意不让循环内直接读 state.messages 写答案：跑到一半时它还没有 final answer，
        # 落库/推流只能看到半条消息链。要看轨迹就读 state.steps 与 state.refs。
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
        # 把这次回复也拼回消息链：落库要的是完整的一问一答，
        # 只存 question + answer 两列的话，多轮历史就没有上下文可拼。
        # 用 `+` 而不是 append：state.messages 是 AgentState 构造时接进来的那个
        # list（与 _prepare 返回的 messages 同一对象），就地改会让调用方的
        # 局部变量多出这条消息 —— 这里改副本，图的是"这一步只有 state 该变"。
        state.messages = state.messages + [{"role": "assistant", "content": state.answer}]
        # 刻意**不**调 _collect_refs：这条路径没有工具消息可扫，
        # refs 保持空列表正是"没检索过"这个事实的准确表达。
        # 走一遍空扫描没坏处，但会让读者误以为这里也可能有引用。
        self.last_state = state

    def _budget_exceeded(self, state: AgentState, deadline: float) -> str | None:
        """返回触发的安全阀名称，没触发返回 None。

        三道预算里 token 预算用**累计值**而不是单步值：
        单步 token 少但步数多，照样能把上下文撑爆。
        墙钟用 monotonic 而非 time.time()，后者会被系统时钟调整影响。
        """
        # 三个 if 的顺序 = 优先级，返回值只回答"最先撞上的是哪一道"。
        # 调用方（两条 run 路径）拿它只做分支、不做数值比较，所以契约很松；
        # 但把步数放第一是有意的：它是唯一一条**与墙钟和 token 都无关**的确定性上界，
        # 放最后的话一次慢工具就能让 max_steps 整轮失效。
        #
        # 刻意的反面：这里**不**检查 len(messages)。上下文膨胀由 token_budget 兜，
        # 而 token_budget 读的是累计 usage —— 用消息条数当预算等于用一个
        # 和实际费用不相关的量来停机。
        if len(state.steps) >= self.s.agent_max_steps:
            return "max_steps"
        if state.prompt_tokens + state.completion_tokens >= self.s.agent_token_budget:
            return "token_budget"
        if time.monotonic() > deadline:
            return "wall_timeout"
        return None

    async def _finalize(self, messages: list[dict], state: AgentState, reason: str) -> None:
        """安全阀触发后的强制收尾：不带工具再问一次，让它把话说完。"""
        # 前置：传进来的 messages 末尾一定是 **tool 消息**而不是 assistant ——
        # 安全阀在每轮开头检查，此刻模型还没给出这一轮的答案。
        # 这个前置同时解释了为什么收尾提示要作为 user 消息追加：
        # tool 消息后面必须跟一条非 tool 消息，多数基座还要求是 user，
        # 直接把 FINALIZE_PROMPT 塞进 system 会被部分 API 判成消息链非法。
        #
        # 出口：只写 state 上的标量字段（answer / token 计数 / stop_reason）。
        # **不**把 messages 赋给 state.messages，也不调 _collect_refs 之外的落库动作 ——
        # 前者由调用方在循环外统一赋值（避免同一份引用被赋两次），
        # 后者在 finally 里做，所以即使 LLM 调用抛异常，引用照样收得到。
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
            # 这里能直接扫 state.messages，是因为 AgentState 构造时接的就是同一个
            # list 对象（`AgentState(messages=messages)`）—— 引用共享，不是拷贝。
            # 这个"偷懒"是有价值的：换成真正拷贝的实现，循环里 append 的
            # tool 消息就不会出现在 state.messages 里，引用会静默变成空列表，
            # 而且不报任何错。
            self._collect_refs(state)

    def _collect_refs(self, state: AgentState) -> None:
        """从工具结果里抽引用，供前端展示与忠实度评测。

        顺带把**页面修订日期**一起抽出来：`出处：` 行现在的格式是
        `出处：<url> | 修订 YYYY-MM-DD`，那个修订时间是前端气泡右下角
        要展示的东西。没有它，引用就只有链接没有时效，
        而一份没有时效标注的引用，正是「用户把旧数据当新数据」的开端。

        注意：URL 后面挂的是**竖线分隔的修订**，不是 query 参数。
        这里用 `partition("|")` 而不是 split，避免 URL 里出现 `|` 时截错。
        """
        # 调用时机是唯一的前置条件：state.messages 里必须已经有 tool 消息。
        # 三个调用点都在"消息链定型之后"（`_finalize` 的 finally、`_run_once` 返回前、
        # 流式的 finally）。若改成循环里每步调一次，seen 这个局部 set 每次重建，
        # 同一条出处行会被反复收进 refs，引用列表里出现重复项 ——
        # 而前端和评测都不会报错，只会安静地展示成"引用了 5 个来源"。
        seen: set[str] = set()
        for msg in state.messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            for line in content.splitlines():
                if not line.startswith("出处："):
                    continue
                url, _, tail = line[len("出处：") :].partition("|")
                url = url.strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                ref: dict = {"url": url}
                rev = tail.strip()
                if rev.startswith("修订"):
                    ref["last_revision_ts"] = rev[len("修订") :].strip()
                state.refs.append(ref)

        # 每条 ref 只有两种形状：拿到修订 → {"url", "last_revision_ts"}，
        # 没拿到 → 只有 {"url"}。**刻意不给空值占位**：
        # 前端按 key 是否存在决定要不要渲染时效标签，塞空串会渲染出一个空标签；
        # 忠实度评测按这个 key 判定"这条引用有没有时效标注"，空串会被算成有。
        #
        # 消费方：/api/chat 与 SSE done.refs、以及忠实度评测的引用对齐。
        # 注意去重按 URL 走：同一个页面被 search_kb 和 get_page 各引用一次
        # 只会留下一条，但两条的修订日期同源（都来自 metadata.last_revision_ts）。
        # 丢弃某个分量是安全的 —— tools.py 里行内与出处行的日期是同一份计算。
        # 前提是所有工具都遵守「出处：<url>[ | 修订 date]」这一条格式约定；
        # 少一个工具遵守，引用就会对不上模型的正文出处，看起来和幻觉一样。

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
        # 返回值的两个分量，消费方不同：
        #   text → 调用方立刻包成 tool 消息进 messages（下一步）；
        #   rec  → 流式路径读 ok/ms/chars 推 tool_end 事件，
        #           非流式路径直接 `_` 丢弃。
        # 丢弃 rec 是安全的：它已经被 append 进 state.steps，
        # 而非流式的出口只有 AgentState，steps 就在里面。
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
        # dt 从这里才开始算，**包含熔断拦截的那 0ms 调用**：
        # 被拦下的步骤 latency≈0，这是刻意让它看起来"没花时间"，
        # 评测里才能区分"查了但被熔断"和"查了很慢"。
        if not ok:
            state.tool_failures += 1
        # rec.step 用当前 steps 长度当序号，前提是**每个 call 恰好 append 一条** ——
        # 包括熔断拦下的那步。不记它就看不出模型被拦过，而模型并不知道自己被拦了
        # （它只是读到一段文本）。
        # error 截 200 字符：完整文案已经进 messages，trace 再存一份全量会让落库翻倍。
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

    def _tool_result_message(self, call: dict, text: str, state: AgentState) -> dict:
        """工具结果消息。tool_call_id 必须与 assistant 里的 id 一致，否则消息链断掉。

        这里同时是**间接 Prompt 注入的唯一入口**：search_kb / get_page 的返回
        是外部正文（wiki 页面），一旦某页被投毒，它的祈使句会带着指令语法进上下文。
        所以工具结果一律经 `scan_and_wrap`：扫描留痕 + 包进 `<retrieved_data>` 数据区，
        系统提示里已声明该区内是数据、不得执行。

        注意是**包裹**不是删除：命中就丢掉内容，等于亲手造出
        「检索到了却不给看」那种假阴性（见 docs/injection_guard.md）。

        `state` 必须显式传进来 —— `self.last_state` 在这一步还指着上一次运行，
        用它记账会把本次的命中记到上一个 run_id 上。
        """
        # 前置：`text` 必须是**未经处理的工具原始输出**。扫描只在这里做一次，
        # 任何调用方提前改写/截断/拼接文本都会让注入检测漏扫那一段，
        # 而漏扫是静默的 —— 返回值看起来完全正常。
        # 两条 run 路径共用这一个入口正是为了堵住这个口子：
        # 只要有一处绕过（比如流式路径直接 push 原文），注入防护就变成
        # "只在某个接口上生效"的假防护。
        content, result = scan_and_wrap(call["name"], text)
        # note_injection 内部已过滤未命中，所以这里不需要再判 result.suspect；
        # ScanResult 不进消息链，只通过 state.injections 旁路留痕。
        state.note_injection(call["name"], result)
        # 出口是唯一进 messages 的形态。name 字段不是所有基座都认，
        # 但带上它才能在出错信息里区分是哪个工具出的问题。
        return {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": call["name"],
            "content": content,
        }

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
        # guard **每轮运行新建**，不跨请求共享：seen 与两个计数器的语义都是"本次运行"。
        # 挂到 session 上会让同一个指纹在第二次提问时被当成重复查询拦掉，
        # 而用户看到的只是一次莫名其妙的"重复查询已拦截"。
        # 代价是同一 session 内熔断不复位 —— 但换个问法本来就该重新给机会。
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

            # 3) 没有 tool_calls = 模型认为信息够了，这是终止信号。
            #    注意这是**唯一**的正常终止路径：安全阀触发走 break，
            #    模型异常走另一个 break —— 只有这里 return。
            # 出口时 state 已经完整：answer / refs / elapsed / last_state 全部就位，
            # 所以调用方拿到 state 就能直接落库，不需要再补任何字段。
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
            # 这个 for 遍历的是**模型一次请求里返回的全部 tool_calls**（可能有多个）。
            # 前置假设：一个 assistant 消息的每个 tool_call 后面都紧跟一条 tool 消息 ——
            # 所以循环体里两行必须成对出现，中途 continue/break 会让消息链断掉，
            # 下一轮直接被 API 判成非法。
            for call in out["tool_calls"]:
                text, _ = await self._exec_call(call, context, guard, state)
                messages.append(self._tool_result_message(call, text, state))

        # 只有走到安全阀才会到这里 —— 三个出口里只有安全阀那个 break 会落到循环外，
        # 「模型调用失败」和「没有 tool_calls」都在循环内 return。
        # 所以落到这里时：state.answer 一定已被 _finalize 填过（成功或失败兜底文案），
        # state.refs 也已被 _finalize 的 finally 收过 —— 循环外**不能**再收一次，
        # 否则引用会重复（虽然 url 去重能挡住，但 wasted 一遍全量扫描）。
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
            # 越界分支：不进 ReAct 循环，直接给边界说明。
            # 出口只有 done 这一个事件（外加一条 out_of_scope 让前端能换文案），
            # 而且刻意**不发** guard / injection 两个字段、refs 与 trace 硬写成空列表：
            # 这条路径压根没调工具，对应计数恒为 0，
            # 发一串 0 值会让前端把"没有这个数据"和"数据全是零"混为一谈，
            # 评测侧也会把不存在的 guard 当成"熔断了 0 次"。
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

        # try 包住整个循环：流式路径里任何一步抛异常都不能让 SSE 连接裸断，
        # 前端会一直转圈直到超时。异常统一转成 error 事件再往下走。
        #
        # 前置：三条出口（正常收尾、limit 的 break、except）**全部**汇聚到 finally，
        # 所以落库四件事（_collect_refs / elapsed / last_state / done 事件）
        # 只能写在 finally 里 —— 写在循环里就等于有路径漏掉，
        # 而漏掉的路径正好是最需要留痕的失败路径。
        try:
            while True:
                reason = self._budget_exceeded(state, deadline)
                if reason:
                    # 熔断/超时出口：与 `_run_once` 的对应分支有一处**故意的差异** ——
                    # 这里只推一条 limit 事件就 break，不再像非流式那样发一次
                    # FINALIZE_PROMPT 收尾。原因是 SSE 已经开始吐 token，
                    # 中途再插一轮"不带工具的完整生成"会让前端先收到一段停顿、
                    # 再收到整段答案，体验上像是卡住了。
                    # 代价：limit 结束时 state.answer 可能还是空的，
                    # 前端必须能处理「只有 limit、没有 token」的组合。
                    yield {"event": "limit", "data": {"reason": reason}}
                    # 必须落到 state 上：AgentState.stop_reason 的默认值是 "answer"，
                    # 不写的话最终 done 事件会把「被安全阀掐掉」报成「正常回答」。
                    # 非流式路径没有这个问题 —— 它走 _finalize()，那里第一件事就是
                    # state.stop_reason = reason。两条路径此前在这里是分叉的。
                    state.stop_reason = reason
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
                # tool_start / tool_end 必须成对且**夹住** _exec_call：
                # start 推的是模型的意图（name/args），end 推的是真实结果
                # （ok/ms/chars 都取自 rec，而不是 text 的长度 ——
                #   熔断拦下时 text 是我们自己拼的文案，它的长度不代表任何耗时）。
                # 只推 start 不推 end，前端那个"正在检索…"的动画就永远不会停。
                for call in out["tool_calls"]:
                    yield {"event": "tool_start", "data": {"name": call["name"], "args": call["args"]}}
                    text, rec = await self._exec_call(call, context, guard, state)
                    yield {
                        "event": "tool_end",
                        "data": {"name": call["name"], "ok": rec.ok, "ms": rec.latency_ms, "chars": rec.result_chars},
                    }
                    # 与非流式路径同一个入口：注入检测不能只在一个接口上生效。
                    messages.append(self._tool_result_message(call, text, state))
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式运行失败")
            yield {"event": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}}
        finally:
            self._collect_refs(state)
            state.elapsed_ms = int((time.perf_counter() - t0) * 1000)
            # 流式接口拿不到返回值，落库全靠这一行把状态交出去 ——
            # 调用方（main.py 的 SSE 分支）读 self.last_state 写 messages + steps。
            # 放在 finally 而不是循环成功出口，是为了 error/limit 两条路径也留痕。
            self.last_state = state
            # done 是**唯一**保证会发出的事件，前端按它收尾。
            # 字段来源一一对应到 state：`answer` 由 token 事件逐片累加而来，
            # `trace` 是 state.steps 的投影（刻意丢掉 StepRecord.error，
            #   完整错误文案已经在 messages 里，再发一遍只是占带宽），
            # `guard` / `injection` 是两份**聚合视图**而不是原始列表 ——
            #   评测读的是聚合后的计数，per-step 明细在 messages/trace 里。
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
                    "injection": state.injection_summary(),
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