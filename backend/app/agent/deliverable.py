"""交付物任务：一次提问，产出多份结构化交付物（Markdown + JSON）。

## 它是什么

普通问答一次调用只回一段话。交付物任务把「整理三月七的养成指南」这类请求
拆成若干**子任务**，每个子任务独立检索、独立抽取结构化事实，最后汇总成
一份带出处、带时效、带冲突标注的交付物。

    规划(LLM) → 逐子任务(检索 → 抽取事实) → 合并 → 冲突检测 → 渲染 Markdown/JSON

## 为什么不拆多 Agent

五个子任务用的是**同一套工具、同一个知识库、同一个基座**，
没有分解的必要性；拆开还会引入跨 agent 的失败归因问题，
而本项目没有任何多 agent 评测集 —— 拆了更好还是更差，**测不出来**，
那就是没有依据的架构决定。

所以这里用**单 Agent + Plan-and-Execute**，但每个子任务都做成独立的
`async def _subtask_xxx(...)`，签名统一为「吃一份上下文、回一份结构化结果」。
将来真有数据支撑要拆多 agent，把这些方法挂到不同 worker 上即可，
规划/合并/冲突检测/渲染这四层**完全不用动**。

## 冲突检测为什么不用 LLM

让模型判断「这两条数值是不是冲突」，等于让它自己给自己打分，
不可复现、不可测、也没有基准。所以冲突检测是一个**纯函数**：

    同一个 key 下出现 ≥2 个不同的规范化值 → 冲突

没有 LLM 参与，没有时间戳参与，因此：
- 结果确定：同样的输入永远得到同样的冲突集；
- 可以写守卫：给定构造好的 facts，冲突集是什么是算得出来的。

## 冲突为什么不按时间戳判谁对

**修订时间只告诉你「这条记录何时录进 wiki」，推不出「这个数值现在还对不对」。**
实测三库的修订日期都按版本节点聚集（Top12 日期占 40%–60%），
说明 wiki 是版本更新驱动的批量维护 —— 页面久未编辑更可能意味着
「wiki 认为它不需要改」，而不是「内容已过期」。

所以 `detect_conflicts` 只做三件事：标出冲突、摆出各来源及其修订时间、
**明确说系统无法判断哪个正确**。详细依据见 `docs/version_staleness.md`。

## 已知限制（写在代码里，不藏着）

1. **主语靠标题前缀推**（见 `subject_of`）：子页面用 `/` 分隔所以够用，
   但同一对象若有两页标题毫无公共前缀，会被当成不同主语 → **漏报**。
2. **冲突检测依赖 key 对齐**：key 由模型抽取，同一个概念若被抽成
   `生命值_80级` 和 `80级生命值`，就会被当成两个 key → **漏报**。
   规范化只处理空白/全角/标点/大小写，不做语义归并。
3. **同页不同修订不算冲突**：同一页面在不同日期给出的不同值是**正常**的
   （wiki 改过），不是矛盾。
4. **抽取阶段可能漏事实**：模型只报它在证据里看到的，没检索到就不会有。
   空交付物和「真的没有」目前无法区分，靠 `status=empty` 标注。
5. **检索会召回无关对象**：真实运行里「长夜月」被召回到三月七的检索结果中。
   现在靠主语限定让这类噪声不产生假冲突，但它仍会占证据预算。
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Protocol

from app.agent.tools import ToolContext
from app.core.config import get_settings
from app.llm.client import LLMClient, get_llm
from app.retrieval.hybrid import RetrievalConfig, RetrievalHit

logger = logging.getLogger(__name__)

# 事实抽取的输出契约。
#
# **只让模型输出证据编号 `[i]`，出处由我们自己从 chunk 解析。**
# 早先的版本要求模型逐字复制 source_url / page_title / revision，
# 实测直接把输出打爆：`max_tokens=1536` 时 completion_tokens 顶到 1536、
# JSON 被截断，解析失败，5 个子任务全部返回 0 条事实 ——
# 而且**不报任何错**，只表现为「召回到 N 个 chunk 但没有事实」。
#
# 改成编号后每条事实少掉三个长字符串，输出体积降一大截；
# 更重要的是「模型抄错 URL」这一整类幻觉被**从根上消掉**了：
# 它没机会编 URL，只能写一个范围校验能挡住的整数。
FACT_SCHEMA_HINT = (
    '{"facts": [{"i": 证据编号整数, "key": "字段标识(英文/数字/下划线, 如 hp_80)", '
    '"label": "字段中文名", "value": "原始值字符串", "unit": "单位或空"}]}'
)

PLAN_PROMPT = """把下面这个请求拆成 3-6 个可以**分别检索**的子任务。

请求：{question}

规则：
1. 每个子任务只问一件事，不要把「数值」和「推荐」混在一起；
2. 每个子任务给出 2-4 个检索关键词（中文，用于向量检索）；
3. 不要写「总结」「综述」这类无法检索的子任务；
4. 知识库只覆盖 {covered}，不要规划超出这个范围的内容。
5. 如果请求指向一个**具体对象**（角色/装备/道具），把它填进 `subject`，
   用知识库里那个对象的正式名称，不要带「•命途」「/子页」等后缀
   （例如「三月七」而不是「三月七•存护」）。
   如果请求不是围绕某个具体对象（例如「整理遗器套装有哪些」），
   `subject` 留空字符串 —— 留空会让主题闸门放行全部证据，那比乱猜一个名字好。

只输出 JSON：{schema}
{{"topic": "交付物标题", "subject": "本次交付物讨论的核心对象，无则空字符串", "subtasks": [
  {{"name": "子任务名", "goal": "这一子任务要拿到什么", "queries": ["关键词1", "关键词2"]}}
]}}"""

EXTRACT_PROMPT = """从下面的检索证据里抽取**结构化事实**。

目标：{goal}

规则（违反任何一条都算失败）：
1. 只输出证据里**明确写着**的值，不要推算、不要补全、不要用你的先验知识；
2. `i` 填证据的编号（`[1]` 就填 1），出处由系统按编号回填，你**不要**自己写 URL；
3. 同一个字段若在不同证据里出现不同数值，**两条都输出**（key 用相同值），
   不要自己挑一个「更对」的；
4. 每条证据最多贡献 3 条事实，**总数不超过 20 条** —— 输出有长度上限，
   写太长会被截断，截断后整批事实全部作废，宁可少写几条；
5. 没找到就返回空数组，不要为了凑数编。

【证据】
{contexts}

只输出 JSON：{schema}"""


# --------------------------------------------------------------------------- #
# 结构化结果（纯数据，不含行为，便于 JSON 序列化与断言）
# --------------------------------------------------------------------------- #
@dataclass
class Fact:
    """一条事实及其出处。

    `revision` 是页面级修订时间，**不是数据生效时间** —— 只作溯源展示，
    任何判断「哪个值更新更可信」的逻辑都不允许出现在本模块。
    """
    key: str
    label: str
    value: str
    unit: str = ""
    source_url: str = ""
    page_title: str = ""
    revision: str | None = None
    subtask: str = ""

    def norm(self) -> str:
        return normalize_value(self.value)


@dataclass
class Conflict:
    """同一**主语**下、同一 key 上互不相容的取值。

    `subject` 是从页面标题推出来的主语（见 `subject_of`）——
    没有它，「三月七存护是冰元素」与「三月七巡猎是虚数元素」会被当成冲突，
    而那是两个对象各自的属性。

    `verdict` 恒为「无法判断」—— 系统**不**按修订时间给结论，理由见模块 docstring。
    """
    key: str
    subject: str
    label: str
    variants: list[Fact]
    # 注意：括号不能省。dataclass 字段的默认值不会隐式跨行续接字符串，
    # 写成两行裸字面量是 SyntaxError（而且报错位置在下一行，很容易看走眼）。
    verdict: str = (
        "系统无法判断哪个取值正确：各来源的页面修订时间只反映录入时刻，"
        "不能推出内容新旧；请以官方最新公告为准。"
    )
    kind: str = "value_mismatch"


@dataclass
class SubtaskResult:
    name: str
    goal: str
    status: str = "ok"          # ok / empty / error
    queries: list[str] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    notes: str = ""
    pages: list[str] = field(default_factory=list)
    latency_ms: int = 0
    #: 被主题闸门拦下的事实条数，以及它们的来源页名。
    #: **必须有**而不是只丢不记：闸门会误杀标题不含主题词的合法页面
    #: （见 subject_matches 的已知代价），不记下来就等于
    #: 「交付物少了几条数据但没人知道为什么」。
    offtopic_dropped: int = 0
    offtopic_pages: list[str] = field(default_factory=list)
    #: 同页同 key 的截断回声被合并掉的数量（见 dedupe_same_page_facts）。
    #: 同样必须留痕：静默去重会让 facts_total 凭空变小。
    duplicate_dropped: int = 0


@dataclass
class DeliverableResult:
    run_id: str
    topic: str
    question: str
    game: str
    plan: list[dict] = field(default_factory=list)
    subtasks: list[SubtaskResult] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    generated_at: str = ""
    #: 本次交付物讨论的核心对象（规划器给的），为空表示主题闸门未启用。
    subject: str = ""
    elapsed_ms: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def refs(self) -> list[dict]:
        """去重后的引用列表，供前端复用现有的气泡出处展示。"""
        seen: dict[str, dict] = {}
        for st in self.subtasks:
            for f in st.facts:
                if f.source_url and f.source_url not in seen:
                    seen[f.source_url] = {
                        "url": f.source_url,
                        "title": f.page_title,
                        **({"last_revision_ts": f.revision} if f.revision else {}),
                    }
        return list(seen.values())


# --------------------------------------------------------------------------- #
# 纯函数区：冲突检测与渲染。不碰 LLM、不碰 DB，全部可单测。
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[\s,，。、;；:：!！?？'\"“”‘’()（）\[\]【】~～\-_.]+")


def normalize_value(v: Any) -> str:
    """把取值规范化到可比较的形式。

    **只做无语义的清洗**：去空白、去标点、全角转半角、英文小写。
    刻意**不做**单位换算、不做数字大小比较 —— 那会引入「1万 = 10000」
    这类语义假设，而这类假设出错时不会报错，只会让冲突被悄悄漏掉。
    """
    s = str(v or "").strip().lower()
    s = s.translate(str.maketrans("０１２３４５６７８９（）％－＋", "0123456789()%-+"))
    return _PUNCT.sub("", s)


def subject_of(page_title: str) -> str:
    """从页面标题推出**主语**，用于给 key 加限定。

    wiki 的子页面约定用 `/` 分隔：`三月七•存护/攻略` 与 `三月七•存护` 是
    同一个主语的两页。所以把末尾的 `/xxx` 段去掉，得到主语。

    ## 为什么必须有这一步（实测踩过的坑）

    最早的版本直接用裸 key 判冲突，一次真实运行报出 **17 处冲突，
    逐条看下来全是误报**：

        [eidolon_1_name] 记忆中的你@三月七•存护  vs  初花学剑动星芒@三月七•巡猎
        [element]        冰@三月七•存护         vs  虚数@三月七•巡猎

    三月七有**存护 / 巡猎两条命途**，是两个页面、两个对象，
    `eidolon_1_name` 在它们上面本来就该不同 —— 那不是矛盾，是两个事实。
    还有更离谱的：检索把**另一个角色「长夜月」**召回来了，
    于是「风雪之角 vs 海妖残鳍」也被当成同一字段的冲突。

    根因是裸 key 没有主语：`element=冰` 与 `element=虚数` 并不是
    对同一件事的两个答案。**没有主语限定，就没有可比较的对象。**

    ## 这条启发式的已知代价

    靠标题前缀推主语，同一个对象若有两页标题毫无公共前缀，
    它们会被当成不同主语 → **漏报**。当前语料里子页面普遍带 `/前缀`，
    所以够用；真要更准得上语义聚类，那又是另一件事（且需要评测集）。
    """
    t = str(page_title or "").strip()
    # 用第一个 "/" 切而不是 rsplit：`A/B/C` 的主语是 `A`，不是 `A/B`。
    return t.split("/", 1)[0].strip() if "/" in t else t


#: wiki 角色页用 `•` 分隔命途：`三月七•存护` 的主语是 `三月七`。
#: 比较主语时必须先把它去掉，否则交付物主题写「三月七」、
#: 来源页写「三月七•存护」时判不出关系，会被闸门误杀。
_SUBJECT_SEP = "•"


def _stamp_now() -> str:
    """交付物生成时间：`2026-10-14 17:01:24`，**本地时间、无时区后缀**。

    为什么不用 ISO8601（早先用的是 `...T09:01:24+00:00`）：
    这个字段是给**人**看的交付物抬头，读者要拿它和自己的钟对照。
    `+00:00` 会让人以为显示的是 UTC，于是把「下午 5 点生成的交付物」
    读成「早上 9 点生成的」—— 一次纯粹由格式造成的误读。
    去掉后缀又保留 UTC 数字更糟：那等于把一个不带说明的时区当成本地时间。

    机器可读的那一份仍然是 ISO（`build_meta` 里的 `generated_at`），
    两者用途不同，不要互相套用格式。

    用本地时间还意味着产物**不可跨时区比较**。交付物是单次运行的快照，
    不参与任何跨库/跨档对比，所以这个代价可以接受；
    真要参与对比的时间戳一律走 `exp_common.build_meta`。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def subject_matches(subject: str, page_title: str) -> bool:
    """这条事实的**来源页**和交付物主题是不是同一个对象。

    ## 要挡的是什么（实测）

    交付物「整理三月七的养成指南」跑出 77 条事实，其中 5 条来自
    **另一个角色「长夜月」**：检索因为「冰属性 / 存护命途」这些共享词
    把它召回来了，抽取层忠实于证据，于是长夜月的命途、稀有度、
    属性全被当成三月七的事实写进了交付物。

    根因不是抽取跑偏（它确实忠实于召回），而是**没有任何一层问过
    「这条证据是不是在说本次要讲的那个对象」**。
    `detect_conflicts` 里的 `subject_of` 只用来给 key 加限定，
    挡得住「把两个对象当同一字段比」，挡不住「把别人的字段搬进来」。

    ## 为什么用标题包含而不是语义相似度

    纯函数、可单测、无额外模型调用，而且这个语料的约定足够规整
    （角色页 = 角色名 + 可选 `•命途` + 可选 `/子页`）。
    上语义相似度会引入一个新的不可复现环节，而这个位置要的是
    「宁可漏也不要把别人的数据混进来」——**精度优先于召回**。

    ## 已知代价（必须说清）

    1. **会误杀标题不含主题词的合法页面**。比如主题是「三月七」，
       但某条「存护命途机制」讲的是通用机制、标题里没有「三月七」，
       这条事实会被丢掉。当前语料里这种情况不多，但确实存在 ——
       所以 `_run_subtask` 会把被丢掉的页名记进 notes 并计入 meta，
       让误杀**可见**，而不是悄悄少一条数据。
    2. `subject` 为空时**一律放行**（`return True`）。闸门拿不到主题
       就等于没有闸门，此时若选择「全部拦掉」，交付物会直接变成空的；
       放行 + 在 meta 里标注「本轮未启用闸门」才是诚实的结果。
    """
    s = str(subject or "").strip()
    if not s:
        return True
    title = subject_of(page_title).split(_SUBJECT_SEP, 1)[0].strip()
    if not title:
        return True
    s_norm = _PUNCT.sub("", s.lower())
    t_norm = _PUNCT.sub("", title.lower())
    if not s_norm or not t_norm:
        return True
    # 双向包含：主题可能比标题更完整（「三月七•巡猎」vs「三月七」），
    # 也可能只是标题的一部分（「钟离」vs「岩王钟离攻略」）。
    return s_norm in t_norm or t_norm in s_norm


def dedupe_same_page_facts(facts: list[Fact]) -> tuple[list[Fact], int]:
    """同一页 + 同一 key + 同一 label 上，**一个取值是另一个的前缀**时，只留更完整的那个。

    ## 实测到的现象

    「长夜月的星魂」子任务抽出了两条 `星魂6描述`，同一个页面、同一个 key：

        长夜月在场时，我方全体目标的全属性抗性穿透提高20%。
        长夜月在场时，我方全体目标的全属性抗性穿透提高20%。忆灵「长夜」施放
        【迷梦，流失，如露】后，长夜月获得本次攻击消耗【忆质】数量的30%。

    短的那条是 wiki 列表页里的**摘要**，长的是详情页的**全文** ——
    两条 chunk 都被召回了，模型各抽了一次。表里于是出现两行，
    一行少一半内容，看起来像「数据自相矛盾」，其实是一条被截断了。

    ## 为什么规则这么窄

    只在「归一化后互为前缀」时才合并。两条同 key 的取值如果
    **谁也不包含谁**，那是真正的不确定（同一页不同段落给了不同说法），
    合并掉就是把矛盾藏起来 —— 而这个模块的整个立场是
    「只陈述、不替用户判断」。所以那种情况**两条都保留**。

    宽一点的规则（比如「同 key 就留最长的」）会把上面那种真歧义也吃掉，
    而且吃得更隐蔽：表里干干净净，看不出曾经有过分歧。

    ⚠️ 比较维度必须包含 `label`，这一点也是实测踩出来的：
    同一页同 key 但 label 不同（「隧洞遗器1 / 2 / 3」）是**模型在区分
    不同槽位**，不是重复抽取。只比 key+value 会把合法的多值字段误吃，
    表现是「表里只剩一个遗器位」，看起来像数据缺失而不是去重。

    ## 为什么在子任务内去重而不是全局

    同一个页面出现在两个子任务是正常的（角色页同时供「基础属性」和
    「星魂」两个子任务使用），那两条记录的 `subtask` 不同，
    放进各自的段落里是对的。只有**同一个子任务内**同页同 key 的重复
    才是抽取噪声。

    返回合并掉的数量，是为了让这件事在产物里可见 ——
    静默去重会让「事实数」这个数字凭空变小而没人知道为什么。
    """
    kept: list[Fact] = []
    dropped = 0
    for f in facts:
        hit_i = -1
        for i, k in enumerate(kept):
            # 页面标题与 key 都归一化后比较：模型可能对同一页
            # 写出「星魂6描述」和「星魂6 描述」两种 key。
            if normalize_value(k.page_title) != normalize_value(f.page_title):
                continue
            if normalize_value(k.key) != normalize_value(f.key):
                continue
            # label 也要一起比，否则会吃掉**合法的多值字段**。
            # 实测踩到：三月七的「遗器推荐」子任务里，同一页同 key `relic_set`
            # 抽出三条，label 分别是「隧洞遗器1 / 2 / 3」、取值都是「4件套」。
            # 这三条是**三个不同的遗器位**，各自独立，不是重复抽取 ——
            # 只比 key+value 会把它们当成「同 key 同值」判成重复，
            # 一次吃掉两条，表里只剩一个遗器位，看着像数据缺失。
            #
            # label 不同 = 模型在区分不同的槽位/条目，此时即使取值相同
            # 也必须都保留；只有 label 也相同才谈得上前缀回声。
            if normalize_value(k.label) != normalize_value(f.label):
                continue
            a, b = k.norm(), f.norm()
            # 前缀关系：b 是 a 的真前缀或反之。相等时也当重复（只留一条）。
            if a == b or a.startswith(b) or b.startswith(a):
                hit_i = i
                break
        if hit_i < 0:
            kept.append(f)
            continue
        dropped += 1
        # 留长的那一条。label/unit 一并跟过来，短的那条往往是列表摘要，
        # 它的 label 更笼统（「星魂6」而不是「星魂6描述」）。
        #
        # 用下标而不是 `kept.index(hit)`：Fact 是普通 dataclass，
        # `index()` 走字段级 __eq__，语义上依赖「表里不存在两条字段全同的事实」
        # 这个不变量。那条不变量确实成立（第二条会被上面的前缀判据立刻合并），
        # 所以这里用下标**不是**在修一个可达的 bug，只是让「覆盖哪一条」
        # 在代码里就一眼可见，不必读者去推那个不变量。
        if len(f.norm()) > len(kept[hit_i].norm()):
            kept[hit_i] = f
    return kept, dropped


def detect_conflicts(facts: list[Fact]) -> list[Conflict]:
    """**同一主语 + 同一 key** 下出现不同规范化取值才是冲突。

    纯函数：同样的输入永远得到同样的输出，因此可以写守卫锁住。

    两条排除规则，缺一不可：

    1. **主语不同不算冲突**（见 `subject_of`）。三月七存护页和巡猎页上的
       同名字段是两个对象各自的属性，不互斥。
    2. **同一页面内部的不同取值也不算冲突** —— wiki 改过内容就会这样，
       那是正常的历史漂移，混进 conflicts 只会制造指向不存在问题的噪音。

    只有「同一主语、不同页面、同名字段给出互不相容的取值」才是真冲突，
    而它**不判对错**，只标出来让人核对。
    """
    groups: dict[tuple[str, str], list[Fact]] = {}
    for f in facts:
        if not f.key or not str(f.value).strip():
            continue
        groups.setdefault((subject_of(f.page_title), f.key), []).append(f)

    out: list[Conflict] = []
    for (subject, key), group in sorted(groups.items()):
        variants: dict[str, Fact] = {}
        for f in group:
            variants.setdefault(f.norm(), f)
        if len(variants) < 2:
            continue
        # 全都来自同一个页面 -> 不是冲突，是同页历史漂移
        if len({f.page_title or f.source_url for f in variants.values()}) < 2:
            continue
        out.append(Conflict(
            key=key,
            subject=subject,
            label=next((f.label for f in group if f.label), key),
            variants=sorted(variants.values(), key=lambda x: (x.page_title, x.norm())),
        ))
    return out


def render_markdown(res: DeliverableResult) -> str:
    """渲染人读版。

    措辞上守一条线：全篇只说「页面修订」，不说「数据截至」「可能过时」——
    后两者是数据支撑不了的推断（见 docs/version_staleness.md）。
    """
    L: list[str] = [f"# {res.topic}", ""]
    L.append(f"> 生成于 {res.generated_at}　·　知识库：{res.game}　·　"
             f"子任务 {len(res.subtasks)} 个　·　耗时 {res.elapsed_ms}ms")
    L.append("")
    L.append(f"原始请求：{res.question}")
    L.append("")

    if res.conflicts:
        L.append("## ⚠️ 检测到数据冲突")
        L.append("")
        L.append("以下字段在不同来源里给出了**互不相容的取值**。")
        L.append("本系统**不判断哪个正确** —— 各来源的修订时间只反映记录何时录入 wiki，"
                 "推不出内容新旧（实测三库均为版本更新驱动的批量维护）。")
        L.append("")
        for c in res.conflicts:
            L.append(f"### {c.subject} · {c.label}（`{c.key}`）")
            L.append("")
            L.append("| 取值 | 来源页面 | 页面修订 | 所属子任务 |")
            L.append("|---|---|---|---|")
            for v in c.variants:
                unit = f" {v.unit}" if v.unit else ""
                L.append(f"| {v.value}{unit} | {v.page_title or '—'} | "
                         f"{v.revision or '—'} | {v.subtask or '—'} |")
            L.append("")
            L.append(f"**结论**：{c.verdict}")
            L.append("")

    L.append("## 结构化事实")
    L.append("")
    any_fact = False
    for st in res.subtasks:
        if not st.facts:
            continue
        any_fact = True
        L.append(f"### {st.name}")
        L.append("")
        L.append(f"> 目标：{st.goal}")
        if st.notes:
            L.append(f"> 说明：{st.notes}")
        L.append("")
        L.append("| 字段 | 取值 | 单位 | 来源 | 页面修订 |")
        L.append("|---|---|---|---|---|")
        for f in st.facts:
            L.append(f"| {f.label} | {f.value} | {f.unit or '—'} | "
                     f"{f.page_title or '—'} | {f.revision or '—'} |")
        L.append("")
    if not any_fact:
        L.append("**没有抽到任何事实。**")
        L.append("")

    empty = [st.name for st in res.subtasks if st.status != "ok"]
    if empty:
        L.append("## 未能完成的子任务")
        L.append("")
        for st in res.subtasks:
            if st.status != "ok":
                L.append(f"- **{st.name}**（{st.status}）：{st.notes or '无可用证据'}")
        L.append("")

    if res.refs:
        L.append("## 出处")
        L.append("")
        for r in res.refs:
            rev = f"　修订 {r['last_revision_ts']}" if r.get("last_revision_ts") else ""
            L.append(f"- [{r.get('title') or r['url']}]({r['url']}){rev}")
        L.append("")

    L.append("---")
    L.append("")
    L.append("**时效说明**：上表的「页面修订」是 wiki 页面最后一次被编辑的时间，"
             "**不是数据生效日期**。本页两年没被动过，通常意味着 wiki 认为它不需要改，"
             "而不是内容已过时。")
    L.append("")
    return "\n".join(L)


def build_json(res: DeliverableResult) -> dict:
    """机器可读版。键名即字段名，形状与 Markdown 一一对应。"""
    return {
        "run_id": res.run_id,
        "topic": res.topic,
        "question": res.question,
        "game": res.game,
        "subject": res.subject,
        "generated_at": res.generated_at,
        "elapsed_ms": res.elapsed_ms,
        "plan": res.plan,
        "sections": [
            {
                "name": st.name,
                "goal": st.goal,
                "status": st.status,
                "queries": st.queries,
                "pages": st.pages,
                "notes": st.notes,
                "latency_ms": st.latency_ms,
                # 闸门留痕要进 JSON 产物，不能只活在运行时对象上：
                # 下载下来的 .json 是别人复核的依据，
                # 只在服务端日志里记等于没记。
                "offtopic_dropped": st.offtopic_dropped,
                "offtopic_pages": st.offtopic_pages,
                "duplicate_dropped": st.duplicate_dropped,
                "facts": [asdict(f) for f in st.facts],
            }
            for st in res.subtasks
        ],
        "conflicts": [
            {
                "key": c.key,
                "subject": c.subject,
                "label": c.label,
                "kind": c.kind,
                "verdict": c.verdict,
                "variants": [
                    {
                        "value": v.value, "unit": v.unit, "page_title": v.page_title,
                        "source_url": v.source_url, "revision": v.revision,
                        "subtask": v.subtask,
                    }
                    for v in c.variants
                ],
            }
            for c in res.conflicts
        ],
        "refs": res.refs,
        "meta": res.meta,
    }


# --------------------------------------------------------------------------- #
# 编排层
# --------------------------------------------------------------------------- #
class Retriever(Protocol):
    async def retrieve(self, query: str, config: RetrievalConfig) -> list[RetrievalHit]: ...


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def contexts_from_hits(hits: list[RetrievalHit], *, max_chars: int = 6000) -> str:
    """把检索结果拼成抽取 prompt 里的证据段。

    每段带上 source_url / page_title / 修订日期 —— 抽取模型要**逐字复制**它们，
    所以证据里必须有这些字段，且格式固定（否则模型会开始改写 URL）。
    """
    out, total = [], 0
    for i, h in enumerate(hits, 1):
        c = h.chunk
        rev = c.metadata.get("last_revision_ts") or ""
        rev = f"修订 {str(rev)[:10]}" if rev else "修订 无"
        block = (
            f"[{i}] 页面：{c.page_title}\n"
            f"source_url：{c.source_url}\n"
            f"页面标题：{c.page_title}\n"
            f"{rev}\n"
            f"正文：{c.text}"
        )
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n\n".join(out)


class DeliverableAgent:
    """单 Agent 的规划-执行编排。

    每个 `_run_subtask` 都是独立 coroutine，输入一份 goal、输出一份 SubtaskResult。
    这就是为将来拆多 agent 预留的接缝：**方法签名不用改，只换谁来调**。
    """

    def __init__(
        self, *, settings=None, llm: LLMClient | None = None,
        ctx: ToolContext | None = None, top_k: int = 5,
    ) -> None:
        self.s = settings or get_settings()
        self.llm = llm or get_llm(self.s)
        self.ctx = ctx
        self.top_k = top_k

    # ---------------- 规划 ---------------- #
    async def plan(self, question: str) -> tuple[str, str, list[dict]]:
        """产出 (topic, subject, subgoals)。失败时返回空计划，**不**退回硬编码模板。

        退回模板看起来更稳，但那样就分不清「模型规划失败」和「真的只有这几项」，
        而交付物的完整性恰恰取决于计划 —— 所以宁可显式空着。

        `subject` 是主题闸门（`subject_matches`）的唯一输入。
        它必须来自规划器而不是正则抠题面：中文里「整理三月七的养成指南」
        的主语既不总是请求开头，也不总是第一个名词短语，
        正则抠出来的名字错一个字，闸门就会把**全部**事实拦掉，
        表现为交付物空了 —— 而闸门本身没有任何报错。
        """
        covered = "、".join(self.s.games.values()) or "（未配置）"
        prompt = PLAN_PROMPT.format(
            question=question, covered=covered, schema=FACT_SCHEMA_HINT
        )
        try:
            raw = await self.llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1024
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("交付物规划失败: %s: %s", type(exc).__name__, exc)
            return question, "", []
        data = _parse_json(raw["content"]) or {}
        subs = [s for s in (data.get("subtasks") or []) if isinstance(s, dict) and s.get("name")]
        subject = str(data.get("subject") or "").strip()
        if subject:
            logger.info("交付物主题闸门已启用：subject=%r", subject)
        else:
            # 不是 warning 级的噪音，是需要让人知道的事实：
            # subject 为空时闸门放行全部证据（见 subject_matches 的第 2 条代价）。
            logger.info("交付物未取到 subject，本轮主题闸门放行全部证据")
        return str(data.get("topic") or question), subject, subs[:6]

    # ---------------- 单个子任务 ---------------- #
    async def _run_subtask(self, spec: dict, ctx: ToolContext,
                            subject: str = "") -> SubtaskResult:
        """检索 → 抽取事实 → 回填出处。**一个子任务的完整生命周期。**

        这里的每个 Fact 都能追溯到具体 chunk，而且**模型全程没碰过 URL**：
        它只输出证据编号，出处由本方法按编号回填。
        """
        name = str(spec.get("name") or "未命名子任务")
        goal = str(spec.get("goal") or "")
        queries = [str(q) for q in (spec.get("queries") or [])][:4] or [goal]
        t0 = time.perf_counter()
        st = SubtaskResult(name=name, goal=goal, queries=queries)

        # --- 检索：每个关键词各跑一次，证据合并去重 ---
        seen_ids: set[str] = set()
        hits: list[RetrievalHit] = []
        for q in queries:
            if not q.strip():
                continue
            try:
                got = await ctx.retriever.retrieve(
                    q, RetrievalConfig(game=ctx.game, final_top_k=self.top_k)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("子任务「%s」检索失败 q=%s: %s: %s", name, q, type(exc).__name__, exc)
                continue
            for h in got or []:
                if h.chunk_id not in seen_ids:
                    seen_ids.add(h.chunk_id)
                    hits.append(h)
        st.pages = sorted({h.chunk.page_title for h in hits if h.chunk.page_title})

        contexts = contexts_from_hits(hits)
        if not contexts.strip():
            st.status = "empty"
            st.notes = "本子任务的关键词没有召回到任何 chunk。"
            st.latency_ms = int((time.perf_counter() - t0) * 1000)
            return st

        # --- 抽取 ---
        prompt = EXTRACT_PROMPT.format(goal=goal, contexts=contexts, schema=FACT_SCHEMA_HINT)
        max_tokens = 2048
        data, truncated = None, False
        # 截断了就加预算重试一次。只重试一次：再长说明这个子任务本身摊子太大
        # （证据太多），那应该拆子任务，而不是无限往上加 token。
        for attempt in range(2):
            try:
                raw = await self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                st.status = "error"
                st.notes = f"事实抽取失败：{type(exc).__name__}: {exc}"
                st.latency_ms = int((time.perf_counter() - t0) * 1000)
                return st
            data = _parse_json(raw["content"])
            # completion_tokens 顶到上限 == 输出被截断 == JSON 必然不完整。
            # 这是判断截断唯一可靠的信号：`chat()` 不返回 finish_reason，
            # 而截断后的半截 JSON 恰好是最容易被 _parse_json 吞掉的形态 ——
            # 它长得像「模型返回了空结果」，于是空结果被当成事实采信。
            # 实测就是这么变成 5/5 子任务全空的，且不报任何错。
            truncated = (raw.get("usage", {}).get("completion_tokens") or 0) >= max_tokens
            if data is not None and not truncated:
                break
            if attempt == 0:
                logger.warning(
                    "子任务「%s」抽取输出被截断（completion_tokens>=%d），加预算重试",
                    name, max_tokens,
                )
                max_tokens *= 2

        if truncated:
            st.status = "truncated"
            st.notes = (f"模型输出达到 {max_tokens} token 上限仍被截断，JSON 不完整，"
                        "本子任务的事实未能采纳。")
        elif data is None:
            st.status = "error"
            st.notes = "事实抽取输出无法解析为 JSON。"

        # --- 校验：证据编号必须落在本轮真实召回范围内 ---
        # 出处**不**由模型提供，而是按编号从 chunk 反查 ——
        # 所以模型没机会编造 URL，「抄错出处」这一整类问题从根上消失了。
        # 编号越界就整条丢弃：越界通常说明模型已经数错、跑偏了。
        dropped = 0
        dropped_pages: set[str] = set()
        for item in (data or {}).get("facts") or []:
            if not isinstance(item, dict):
                continue
            value = str(item.get("value") or "").strip()
            key = str(item.get("key") or "").strip()
            if not value or not key:
                continue
            try:
                idx = int(item.get("i"))
            except (TypeError, ValueError):
                logger.warning("丢弃缺证据编号的事实 key=%s", key)
                continue
            if not 1 <= idx <= len(hits):
                logger.warning("丢弃越界证据编号的事实 key=%s i=%s (共%d条)", key, idx, len(hits))
                continue
            chunk = hits[idx - 1].chunk
            # --- 主题闸门 ---
            # 放在**回填出处之后、写入 facts 之前**：
            # 出处已经从 chunk 解析好了，丢弃时不产生任何未回填的中间态。
            if not subject_matches(subject, chunk.page_title):
                dropped_pages.add(chunk.page_title)
                dropped += 1
                continue
            st.facts.append(Fact(
                key=key,
                label=str(item.get("label") or key),
                value=value,
                unit=str(item.get("unit") or ""),
                source_url=chunk.source_url,
                page_title=chunk.page_title,
                revision=(str(chunk.metadata.get("last_revision_ts") or "")[:10] or None),
                subtask=name,
            ))
        st.offtopic_dropped = dropped
        st.offtopic_pages = sorted(dropped_pages)
        # 去重在主题闸门之后：闸门挡掉的是「别人的字段」，
        # 这里处理的是「同一个字段被抽了两次」，两件事互不干扰，
        # 顺序换了也不会互相掩盖。
        st.facts, st.duplicate_dropped = dedupe_same_page_facts(st.facts)
        if st.offtopic_dropped:
            st.notes = (
                f"主题闸门拦下 {st.offtopic_dropped} 条来自无关页面的事实（"
                f"{'、'.join(st.offtopic_pages)}）。"
                "这些页被共享属性词召回，内容讲的是别的对象。")
        if st.duplicate_dropped:
            st.notes = (
                (st.notes + " ") if st.notes else "") + (
                f"另有 {st.duplicate_dropped} 条同一页同字段的截断回声"
                "（短版本是列表摘要、长版本是正文全文），已保留更完整的那条。")
        if not st.facts and st.status == "ok":
            st.status = "empty"
            st.notes = (st.notes + " " if st.notes else "") + (
                f"召回到 {len(hits)} 个 chunk，但没有抽到可用于「{subject}」的"
                "结构化事实。")
        st.latency_ms = int((time.perf_counter() - t0) * 1000)
        return st

    # ---------------- 主流程 ---------------- #
    async def run(self, question: str, *, game: str | None = None) -> DeliverableResult:
        t0 = time.perf_counter()
        ctx = self.ctx or ToolContext.build(game or self.s.game_primary)
        if game:
            ctx.game = game
        res = DeliverableResult(
            run_id=uuid.uuid4().hex[:16], topic=question, question=question,
            game=ctx.game, generated_at=_stamp_now(),
        )
        # 时间戳用**本地时间**且不带时区后缀，理由见 `_stamp_now` 的 docstring。
        res.topic, subject, plan = await self.plan(question)
        res.subject = subject
        for spec in plan:
            res.subtasks.append(await self._run_subtask(spec, ctx, subject))

        res.conflicts = detect_conflicts([f for st in res.subtasks for f in st.facts])
        res.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        res.meta = {
            "subtasks_total": len(res.subtasks),
            "subtasks_ok": sum(1 for s in res.subtasks if s.status == "ok"),
            "subtasks_empty": sum(1 for s in res.subtasks if s.status == "empty"),
            "subtasks_error": sum(1 for s in res.subtasks if s.status == "error"),
            # 截断单列：它既不是「没查到」也不是「抽取失败」，
            # 而是「本来能抽到但输出超预算被砍了」，混进 empty 会低估覆盖率。
            "subtasks_truncated": sum(1 for s in res.subtasks if s.status == "truncated"),
            "facts_total": sum(len(s.facts) for s in res.subtasks),
            "subject": subject,
            # 闸门是否真的在工作，必须能从产物里一眼看出来。
            # 「开了闸门但一条没拦到」和「根本没开闸门」在事实数上长得一样，
            # 只有记下开关状态才分得清。
            "topic_gate_enabled": bool(subject),
            "facts_offtopic_dropped": sum(s.offtopic_dropped for s in res.subtasks),
            "offtopic_pages": sorted({p for s in res.subtasks for p in s.offtopic_pages}),
            "facts_duplicate_dropped": sum(s.duplicate_dropped for s in res.subtasks),
            "conflicts_total": len(res.conflicts),
            "conflict_policy": "只标出冲突与各来源修订时间，不按时间判对错",
            "known_limits": (
                "主语靠页面标题去掉 / 之后的部分推出（wiki 子页面约定）；"
                "同一对象若有两页标题无公共前缀会被当成不同主语 -> 漏报。"
                "key 由模型抽取，同义 key 不归并 -> 漏报。"
                "同页不同修订不算冲突（wiki 改过内容属正常）。"
                "「推荐类」子任务（配什么遗器/光锥）通常抽不到可核对的事实，"
                "会落在 empty —— 这是如实反馈，不是失败。"
                + (
                    f"主题闸门已启用（subject={subject}）：来源页标题不含主题词的"
                    "事实会被拦下，代价是标题不含主题词的**合法**页面也会被误杀。"
                    if subject else
                    "本轮未取到 subject，主题闸门未启用："
                    "别的对象的字段可能被当作本主题的事实写进来。"
                )
            ),
        }
        return res
