"""生成后引用校验 —— C 类失败（答案对，但引用了不存在的页面）的结构性缺口。

## 要挡的到底是什么

`docs/failure_analysis.md` 第四节记的 C 类失败，形态是：

| case | correctness | faithfulness |
|---|---|---|
| sr-007 | 5.0 | 2.0 |
| sr-011 | 5.0 | 2.0 |
| sr-013 | 5.0 | 2.0 |

答案本身完全正确（correctness 满分），但正文里出现了**本轮检索结果中根本没有**
的页面名（不老者的仙舟、劫火莲灯…）。危险在于：这种失败在均值里完全看不出来，
用户点开引用却发现查无此文 —— 可核查性直接崩塌。

系统提示词要求模型「引用信息时在句末标注出处，格式如「[三月七]」」，
而**那个方括号里的页面名是模型自己写的**。`state.refs` 不可伪造
（出处行由工具层产出、`_collect_refs` 按 `出处：` 前缀扫出来），
但**正文里的方括号可以编**。本模块补的就是这一层。

## 刻意分成两档，因为它们的误报率完全不同

### L1 页面级 —— 硬拦截，剥掉引用

正文引用的页面名，是否出现在**本轮实际检索到的页面集合**里。

零误报：本轮检索到的页面就是全部合法引用对象，不在里面的页面名
用户点开必然是死链。所以不在里面就是错的，直接剥掉并计数。

### L2 数值级 —— 只记录，不改文本

带引用的句子里出现的数字，是否在该页正文里出现过。

**这一档有已知误报，所以只当信号、不当判据**：模型可能写出**推导值**
而不是引用值 —— 「攻击力 1000 的角色加多少攻击力」这类题，答案里的 120
是 1000 × 12% 算出来的，页面正文只会有「12%」而没有「120」。
把它当成幻觉去剥，会把**正确的计算过程**削掉，那是比 C 类更糟的失败。

所以 L2 只写进 `citation_audit.unsupported_numbers`，供评测与人工复核看，
**绝不改动 `state.answer`**。这个取舍是本模块最重要的一行说明：
校验器的价值取决于它不冤枉多少好答案，不取决于它抓了多少坏答案。

## 为什么不用「模型自查」

让模型自己检查自己有没有编引用是不可靠的：它会同样流畅地为编造的页面名背书。
这里的判据全部来自**工具层已经落地的证据**（本轮检索到的页面与其正文），
与模型的自我陈述无关。
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# 正文里的引用标记。
#
# 两个否定前瞻/后顾都是必要的：
#   `(?!\()`  排除 markdown 链接 `[文字](url)` —— 那种自带可点击 URL，
#             不属于「凭记忆编造出处」这一类，误剥会把正常排版毁掉。
#   `(?!\[)`  排除嵌套/相邻的方括号，避免 `[[a]]` 被切成 `[a]`。
_CITE_RE = re.compile(r"(?<!\!)\[([^\[\]\n]{1,60})\](?!\()")

# 归一化时要抹掉的装饰性字符。
# wiki 标题的花样比人想的��：「凝滞虚影 • 炎华之形」「那刻夏/攻略」
# 「位面饰品·太空封印站」。不归一化的话同一个页面会有三四种写法，
# 匹配不上就会被当成编造 —— 那是纯误伤。
_NOISE_RE = re.compile(r"[\s·•・\-—_「」『』《》（）()\[\]【】:：,，。、/|｜]+")

# 短于这个长度的方括号内容一律不当引用（`[1]`、`[x]`）。
MIN_CITATION_LEN = 2

# search_kb 的结果块首行：`[3] 分数 0.9984 | 太空封印站 > 正文`
_HIT_HEADER_RE = re.compile(r"^\[(\d+)\]\s+分数\s+([\d.]+)\s\|\s(.+?)\s>\s(.*)$")
# get_page 的首行：`《卡芙卡/攻略》页面修订 2026-02-18（312ms，共 2044 字）`
_PAGE_HEADER_RE = re.compile(r"^《(.+?)》")
_SOURCE_PREFIX = "出处："

# 句子里的数值。含小数与百分号（百分号本身不在匹配里，`12%` 取到 `12`）。
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# 句子边界。中英文标点都算。
_SENT_SPLIT_RE = re.compile(r"[。！？!?\n；;]")


def normalize_title(s: str) -> str:
    """页面名归一化：抹掉装饰性字符，用于比对。"""
    return _NOISE_RE.sub("", s or "").strip()


def parse_citations(answer: str) -> list[tuple[str, int, int]]:
    """抽出正文里的引用标记，返回 [(原文, 起, 止)]，按出现顺序、不去重。

    保留起止位置是因为 L1 要**就地剥掉**编造的引用 —— 只拿到名字的话
    就没法从原文里精确删除，整篇重写又容易顺手改坏别的内容。
    """
    out: list[tuple[str, int, int]] = []
    for m in _CITE_RE.finditer(answer or ""):
        raw = m.group(1).strip()
        # 全是数字或标点的方括号不是页面引用（编号、脚注、装饰）
        if len(raw) < MIN_CITATION_LEN or not any(ch.isalnum() for ch in raw):
            continue
        out.append((raw, m.start(), m.end()))
    return out


def is_supported(citation: str, titles: Iterable[str]) -> bool:
    """这个引用是否指向本轮检索到的某个页面。

    判定用**双向包含**而不是全等：
    模型写「太空封印站」而检索到的标题是「太空封印站（位面饰品）」，
    或者反过来 —— 全等判定会把它们当成编造，那是纯粹的误伤。
    两侧都要求归一化后长度 >= 2，否则一个字的标题会被任意长引用命中。
    """
    cit = normalize_title(citation)
    if len(cit) < MIN_CITATION_LEN:
        return False
    for t in titles:
        norm = normalize_title(t)
        if len(norm) < MIN_CITATION_LEN:
            continue
        if cit == norm or cit in norm or norm in cit:
            return True
    return False


def harvest_retrieved_pages(content: str) -> dict[str, str]:
    """从工具输出里收「本轮真正取到正文的页面 → 该页正文」。

    同时认两种格式（都由 tools.py 写死，改那边必须同步改这里）：

    search_kb 的结果块：
        [1] 分数 0.9984 | 太空封印站 > 正文
        <截断后的正文>
        出处：https://... | 修订 2025-04-13

    get_page 的单页：
        《卡芙卡/攻略》页面修订 2026-02-18（312ms，共 2044 字）
        出处：https://...
        <页面正文>

    **两个工具都要认**：模型 get_page 打开过的页面再引用它是合法出处，
    漏掉这一路会把正确引用判成编造 —— 那是纯粹的误伤，而误伤比漏报更贵。

    同页多个 chunk 的正文用换行拼起来 —— L2 要在**整页**里找数值，
    只看某一个 chunk 会把「数值在另一个 chunk 里」误判成没有出处。

    解析不出来就返回空 dict（不抛异常）：这是校验用的辅助信息，
    拿不到就等于不做 L2，不该因此让整轮运行失败。
    """
    out: dict[str, str] = {}

    # --- search_kb：正文在「命中头」与「出处行」之间 ---
    lines = (content or "").splitlines()
    i = 0
    while i < len(lines):
        hit = _HIT_HEADER_RE.match(lines[i])
        if not hit:
            i += 1
            continue
        title = hit.group(3).strip()
        i += 1
        buf: list[str] = []
        while i < len(lines) and not lines[i].startswith(_SOURCE_PREFIX):
            buf.append(lines[i])
            i += 1
        if title:
            _merge(out, title, "\n".join(buf).strip())

    # --- get_page：正文在「出处行」**之后**，直到下一个块头 ---
    i = 0
    while i < len(lines):
        page = _PAGE_HEADER_RE.match(lines[i])
        if not page:
            i += 1
            continue
        title = page.group(1).strip()
        i += 1
        # 跳过这一页的出处行
        while i < len(lines) and not lines[i].startswith(_SOURCE_PREFIX):
            i += 1
        i += 1
        buf = []
        while i < len(lines) and not (
            _HIT_HEADER_RE.match(lines[i]) or _PAGE_HEADER_RE.match(lines[i])
        ):
            buf.append(lines[i])
            i += 1
        if title:
            _merge(out, title, "\n".join(buf).strip())

    return out


def _merge(out: dict[str, str], title: str, body: str) -> None:
    """同页多个 chunk / 多次 get_page 的正文拼起来。

    L2 要在**整页**里找数值，只看某一个 chunk 会把
    「数值恰好在同页另一个 chunk 里」误判成没有出处。
    """
    if not title:
        return
    body = body.strip()
    out[title] = f"{out[title]}\n{body}".strip() if title in out else body


@dataclass
class CitationAudit:
    """一轮运行的引用校验结果。

    三态而不是两态 —— 这是本模块最容易出错、也最容易被下游误读的地方：

    | checked | fabricated | 含义 |
    |---|---|---|
    | True  | []        | 校验过，没发现问题 |
    | True  | 非空      | 校验过，**发现了问题**（已剥掉） |
    | False | []        | **压根没法校验** |

    第三种绝不能混进第一种。本轮一个页面都没取到时（检索全挂、
    或者工具输出格式改了导致 harvest 解析不出来），`retrieved` 是空的，
    此时**每一个**引用都会被判成编造 —— 但那是判据缺失，不是模型撒谎。
    照着剥会把一次工具故障的回答削成满身弹孔的文本。

    所以 `retrieved` 为空时一律不剥，`checked=False` 如实记下。
    这条规则与本项目 `SearchGuard.observe` 的兜底是同一个原则：
    **判据失效时宁可不动，也不要在没有判据的情况下动手** ——
    SearchGuard 那里是「宁可熔断晚触发，也不要把 Agent 提前掐死」，
    这里是「宁可留着可疑引用，也不要削掉正确答案」。
    """

    #: 正文里抽到的全部引用（原文形态，含重复）
    cited: list[str] = field(default_factory=list)
    #: 本轮没检索到、已从正文剥掉的页面名（去重、保持出现顺序）
    fabricated: list[str] = field(default_factory=list)
    #: L2 信号：数值所在句子引用的页面正文里查不到这个数
    unsupported_numbers: list[dict] = field(default_factory=list)
    #: 本轮检索到的页面名，去重后升序
    retrieved_titles: list[str] = field(default_factory=list)
    #: L1 是否拿到了可用的判据（本轮真的取到过页面正文）
    checked: bool = False

    @property
    def ok(self) -> bool:
        """**校验过且没发现问题**才算 ok。

        没校验过（`checked=False`）返回 False：把「没法判断」报成「没问题」，
        正是评测里最容易把盲区当成达标的地方 —— 本项目的
        `test_revision_disclosure` 就栽过一次同类的坑（判据写错导致恒红/假绿）。
        """
        return self.checked and not self.fabricated

    def to_dict(self) -> dict:
        return {
            "cited": self.cited,
            "fabricated": self.fabricated,
            "unsupported_numbers": self.unsupported_numbers,
            "retrieved_titles": self.retrieved_titles,
            "checked": self.checked,
            "ok": self.ok,
        }


def _sentence_around(text: str, start: int, end: int) -> str:
    """取引用所在的整句话（不含方括号本身）。

    向前找最近的句读、向后找最近的句读；找不到就用整段。
    只做**加粗不做截断**：宁可多带几个字，也不要因为切句规则变化
    把一个明明有出处的数字判成没出处 —— L2 宁可漏报也不误报。
    """
    left = max((text.rfind(m, 0, start) for m in "。！？!?\n；;"), default=-1)
    right_candidates = [text.find(m, end) for m in "。！？!?\n；;"]
    right_candidates = [c for c in right_candidates if c != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right]


def audit_answer(answer: str, retrieved: Mapping[str, str]) -> tuple[str, CitationAudit]:
    """校验正文引用。返回 (处理后的正文, 审计结果)。

    `retrieved` 是「页面名 → 该页正文」，来自 `harvest_retrieved_pages`。

    对 L1（页面不存在于本轮检索）：**剥掉**方括号里的编造引用。
    为什么剥而不是标红：标红要改渲染层，而引用列表 `refs` 本来就只有
    真实检索结果 —— 正文里留着一个点不开的 `[xxx]`，用户点它没有任何反馈，
    那不是提示，是悬案。剥掉之后正文与可点击引用列表一致，
    剥掉了几处在 `CitationAudit.fabricated` 里留痕、可查、可评测。

    对 L2（数值无出处）：**只记录**，不动正文（见模块 docstring 的取舍说明）。
    """
    audit = CitationAudit(retrieved_titles=sorted(retrieved.keys()))
    answer = answer or ""
    cites = parse_citations(answer)
    if not cites:
        # 没有引用就没有可校验的东西，但 checked 仍要如实反映
        # 「本轮有没有取到过页面」—— 空引用与「无从校验」是两件事。
        audit.checked = bool(retrieved)
        return answer, audit

    audit.cited = [c[0] for c in cites]

    # 判据缺失时不剥。理由见 CitationAudit 的 docstring：
    # retrieved 为空既可能是「真的没检索到」，也可能是「工具输出格式变了导致
    # harvest 解析不出来」，两种情况的正确反应都不是「把正文里所有引用削掉」。
    if not retrieved:
        return answer, audit

    audit.checked = True

    # 从后往前替换，避免前面的替换把后面的下标顶偏
    drop_spans: list[tuple[int, int]] = []
    for raw, s, e in cites:
        if is_supported(raw, retrieved.keys()):
            # L2：只看被认定为真实引用的那些句子
            page = _match_page(raw, retrieved)
            if page:
                body = retrieved[page] or ""
                for num in _NUM_RE.findall(_sentence_around(answer, s, e)):
                    if num not in body:
                        audit.unsupported_numbers.append(
                            {"page": page, "number": num, "citation": raw}
                        )
            continue
        audit.fabricated.append(raw)
        drop_spans.append((s, e))

    for s, e in sorted(drop_spans, reverse=True):
        answer = answer[:s] + answer[e:]
    # 去重但保持出现顺序：同一个编造页面被引用三次只报一次
    seen: set[str] = set()
    audit.fabricated = [x for x in audit.fabricated if not (x in seen or seen.add(x))]
    return answer, audit


def _match_page(citation: str, retrieved: Mapping[str, str]) -> str | None:
    """在 retrieved 里找到这个引用实际指向的页面（取最长匹配，避免短标题抢先）。"""
    best: str | None = None
    for t in retrieved:
        if is_supported(citation, [t]) and (best is None or len(t) > len(best)):
            best = t
    return best
