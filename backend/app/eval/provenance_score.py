"""溯源题的客观判据（纯函数，不经 LLM）。

## 为什么判据必须与实验 F 的结论范围一致

实验 F 能诚实回答的问题只有一个：**披露层给出的日期，模型有没有用上。**
所以这里的每个判据都只回答这一个问题的某个切面：

- ``mentions_date``      —— 具体日期有没有被复述出来（信息有没有传到答案里）
- ``has_revision_wording`` —— 有没有说出「修订」这个动作（有没有意识到有这回事）
- ``has_staleness_wording`` —— 有没有主动声明「可能已过时」（有没有意识到风险）
- ``declares_uncertainty`` —— 有没有表达「我不能确定哪条更新」（自我认知）

## 刻意**没有**的判据

没有「回答是否正确」。本语料只有最新版快照，没有任何历史版本可比对，
`docs/version_staleness.md` 第四节已经把这个能力边界钉死了。
在这里造一个「正确性」判据，只能靠猜 —— 而猜出来的正确率
会被当成实验结论引用，比没有这个数字坏得多。

## 日期匹配为什么要容忍多种写法

模型写同一个日期可能是 `2026-07-15`、`2026年7月15日`、`2026/07/15`、
`2026.07.15`、甚至 `20260715`。只认一种格式的话，
「模型复述了日期但格式不同」会被判成没复述 ——
那是在测正则表达式，不是在测披露层。
"""
from __future__ import annotations

import re

from app.eval.provenance_cases import ProvenanceCase

#: 「修订」类措辞：说的是机制本身（这个页面被编辑过）。
REVISION_WORDS = ("修订", "最后编辑", "编辑时间", "版本时间", "更新时间")

#: 「陈旧」类措辞：说的是风险判断（这个数据可能不行了）。
#: 与 REVISION_WORDS 分开统计，因为前者是**如实转述**，
#: 后者是**主动推断** —— 后者恰好是 docs/version_staleness.md 第五节
#: 实测认为「原理上就不该由页面修订时间推出」的那件事。
STALENESS_WORDS = (
    "过时", "过期", "不是最新", "旧版本", "已失效", "可能已变",
    "可能过时", "内容可能", "仅供参考", "谨慎参考",
)

#: 强不确定性标记：出现即判定为「模型承认自己没把握」，
#: 因为这些词在游戏知识问答里几乎只用于自我评估。
STRONG_UNCERTAINTY_MARKERS = (
    "不确定", "无法判断", "无法确定", "把握", "不敢保证", "存疑",
    "不太确定", "建议以", "请以", "应以", "仅供参考", "不一定",
)

#: 弱标记：「可能」在中文里绝大多数时候修饰的是**机制**而不是**版本**
#: （「可能叠加两次」「可能触顶」），拿来当不确定性信号会系统性虚高。
#: 所以它只在附近出现时效语境词时才算数。
WEAK_UNCERTAINTY_MARKER = "可能"

#: 弱标记的语境词。命中任意一个，才认为那个「可能」说的是版本时效。
PROVENANCE_CONTEXT = (
    "版本", "最新", "更新", "旧", "新", "哪一条", "哪条", "哪个", "哪一",
    "来源", "修订", "过时", "有效",
)

#: 弱标记的语境窗口（字符数）。取 24 而不是更大：
#: 句子是「以上信息来自 X 页面，可能不是最新」，24 字足够覆盖，
#: 再大就会把整段里跟时效无关的「可能」也捞进来。
CONTEXT_WINDOW = 24


#: 明确**不算**不确定表达的高频词。
#: 这层是对 STRONG 列表的加固：万一哪天有人往强标记里加了「可能」，
#: 这里的剔除还能兜住一部分。
UNCERTAINTY_FALSE_FRIENDS = ("可能值", "可能触", "可能对", "可能高", "可能低")


def _date_pattern(iso_date: str) -> re.Pattern[str] | None:
    """`2026-07-15` → 能匹配各种常见写法的正则。日期不合法就返回 None。"""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(iso_date or "").strip())
    if not m:
        return None
    y, mo, d = m.groups()
    # 分隔符类必须同时含「年」和「月」：中文写法是「2026年7月15日」，
    # 只放「年」的话 `2026年7` 之后遇到「月」就断了 ——
    # 而这种写法恰恰是模型输出中文日期时最常见的一种。
    # 写漏「月」时正则不报错，只是静默匹配不到，于是所有中文日期
    # 一律判成「没复述」，指标会显示成一个很低的、看起来很合理的数。
    sep = r"\s*[-/年月.\s]\s*"
    return re.compile(
        rf"{y}{sep}0?{int(mo)}{sep}0?{int(d)}"          # 2026-07-15 / 2026年7月15日 / 2026 07 15
        rf"|{y}0?{int(mo)}0?{int(d)}"                   # 20260715
    )


def mentions_date(answer: str, iso_date: str) -> bool:
    """答案里有没有出现这个具体日期（容忍多种书写格式）。"""
    pat = _date_pattern(iso_date)
    if pat is None:
        return False
    return bool(pat.search(answer or ""))


def has_revision_wording(answer: str) -> bool:
    return any(w in (answer or "") for w in REVISION_WORDS)


def has_staleness_wording(answer: str) -> bool:
    return any(w in (answer or "") for w in STALENESS_WORDS)


def declares_uncertainty(answer: str) -> bool:
    """答案里有没有表达「我不能确定哪条更新」。

    分强弱两档，避免被高频的「可能」撑起来：

    - 强标记直接算数（「不确定」「把握」「建议以…为准」）；
    - 弱标记「可能」必须**落在时效语境窗口内**才算 ——
      「该效果可能叠加两次」说的是机制，不是版本时效。

    为什么不用「事后解释阈值」的方式调参：那样阈值就是照着结果挑的，
    判据会不知不觉变成「怎么写都能得分」。这里把规则写死，
    并让 `test_provenance_scoring.py` 同时锁住正反两面的例子。
    """
    text = answer or ""
    for bad in UNCERTAINTY_FALSE_FRIENDS:
        text = text.replace(bad, "")
    if any(w in text for w in STRONG_UNCERTAINTY_MARKERS):
        return True
    for m in re.finditer(WEAK_UNCERTAINTY_MARKER, text):
        lo = max(0, m.start() - CONTEXT_WINDOW)
        hi = min(len(text), m.end() + CONTEXT_WINDOW)
        if any(w in text[lo:hi] for w in PROVENANCE_CONTEXT):
            return True
    return False


def ref_revision_coverage(refs) -> float:
    """引用里带修订日期的比例。没有引用时返回 0.0，不返回 NaN。

    为什么必须是 0 而不是 NaN：0 意味着「引用了但没带日期」，
    能和「压根没引用」区分开。NaN 往下游一传就会变成 `if x != x` 这类判断，
    而 None 在 JSON 里会变成 null，和 0 混在一起看不出来。
    """
    refs = list(refs or [])
    if not refs:
        return 0.0
    return round(sum(1 for r in refs if r.get("last_revision_ts")) / len(refs), 4)


def score_case(case: ProvenanceCase, answer: str, refs, gold_dates: dict[str, str]) -> dict:
    """给单条 case 打分。返回的每一项都能独立解释，不做加权总分。

    为什么不给总分：这四类指标是**不同性质**的
    （日期复述 / 机制意识 / 风险声明 / 自我认知），
    加权求和等于替读者做了「哪个更重要」的判断，
    而那个权重在本实验里没有任何依据支撑。
    """
    answer = answer or ""
    hit_pages = [
        p for p in case.target_pages
        if gold_dates.get(p) and mentions_date(answer, gold_dates[p])
    ]
    miss_pages = [p for p in case.target_pages if p not in hit_pages]
    row = {
        "case_id": case.case_id,
        "game": case.game,
        "kind": case.kind,
        "target_pages": list(case.target_pages),
        "gold_dates": {p: gold_dates.get(p, "") for p in case.target_pages},
        # 逐页命中数 / 应命中数。分母为 0（selfrate 题）时给 None 而不是 0，
        # 否则它会被算成「一个都没答对」，方向完全相反。
        "date_hits": len(hit_pages),
        "date_targets": len(case.target_pages),
        "date_hit_rate": (round(len(hit_pages) / len(case.target_pages), 4)
                          if case.target_pages else None),
        # 全对才算 1：多来源题里答对 3/4 和答对 4/4 不是一回事。
        "all_dates_hit": (bool(case.target_pages) and not miss_pages),
        "missed_pages": miss_pages,
        "has_revision_wording": has_revision_wording(answer),
        "has_staleness_wording": has_staleness_wording(answer),
        "declares_uncertainty": declares_uncertainty(answer),
        "ref_count": len(list(refs or [])),
        "ref_revision_coverage": ref_revision_coverage(refs),
        "answer_chars": len(answer),
    }
    return row


def aggregate_rows(rows) -> dict:
    """聚合一档。

    分母逐指标各算各的：`date_hit_rate` 的分母是「有 target 的 case 数」，
    `declares_uncertainty` 的分母是「selfrate case 数」，
    两者不同 —— 合成一个 n 会让某个指标的分母悄悄变大。
    """
    rows = list(rows)
    dated = [r for r in rows if r["date_targets"]]
    selfrate = [r for r in rows if r["kind"] == "selfrate"]

    def avg(key, subset):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n": len(rows),
        "n_dated": len(dated),
        "n_selfrate": len(selfrate),
        # 逐页命中率：把「命中页数 / 应命中页数」在所有多页题上摊平。
        # 不能用 case 级 0/1 的平均代替 —— 一条 4 页题答对 3 页，
        # 在 case 级记 0，在页级记 0.75，两者不是同一个量。
        "page_level_date_hit_rate": avg("date_hit_rate", dated),
        "case_level_all_hit_rate": avg("all_dates_hit", dated),
        "has_revision_wording_rate": avg("has_revision_wording", rows),
        "has_staleness_wording_rate": avg("has_staleness_wording", rows),
        # 分母只算 selfrate 题：其它题问的是「日期是多少」，
        # 答完之后说不说「我不确定」不是判据。
        "declares_uncertainty_rate": avg("declares_uncertainty", selfrate),
        "ref_revision_coverage": avg("ref_revision_coverage", rows),
        "avg_answer_chars": int(avg("answer_chars", rows) or 0),
    }