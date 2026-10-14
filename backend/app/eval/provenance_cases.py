"""溯源题集（实验 F 专用）。

## 为什么要单独一个题集，不能往 `cases.py` 里加

`cases.py` 是 A/B/C/D/E 全部历史实验的题源，`exp_common.SUBSET_IDS`
按 case_id 引用它。往里加题会改动历史实验的可用题集合 ——
分母一变，「C0 vs C2」和「E1 vs E2」的历史结论就不再能与新产物对照。
所以这里是独立模块，由 exp_f.py 单独加载。

## 这组题测的是什么，不测什么

**测**：披露层给出的页面修订日期，模型有没有真的用上
（复述出来 / 主动声明时效 / 主动表达不确定）。

**不测**：披露层有没有让回答**变准**。

第二件事在本语料上**测不了**，原因写在 `docs/version_staleness.md`
第三、四节：语料只有最新版快照，没有历史版本，所以「这个数值现在还对吗」
这个问题根本无法验证。把「披露层提升了准确率」写成实验结论属于无源推断。

## 判据为什么用正则而不用 judge

judge 的三个维度是 correctness / faithfulness / completeness，
**没有一维评时效敏感度**。拿它评这类题只会得到一个与时效无关的分数，
再拿那个分数论证「披露层有效」是循环论证。
所以这里全部用客观判据：答案里有没有出现那个具体日期、有没有时效措辞。

## gold 日期不写死在题里

`target_pages` 只记页面标题，具体日期在跑的时候从 MySQL 现读。
理由：手打日期常量会和语料漂移 —— 页面被重抓一次，
题里的 gold 就成了一个早已不存在的日期，而题目看起来完全正常。
现读则保证「gold = 模型看到工具输出里的那个日期」，
这正是「有没有忠实复述」这件事的准绳。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProvenanceCase:
    """一条溯源题。

    `kind` 决定判据用哪一组：

    - ``restate``     单页日期复述 —— 答案里必须出现那个页面的修订日期；
    - ``multisource`` 多页日期对应 —— 逐页判定，外加一个「全部命中」；
    - ``selfrate``    让模型自己判断哪条更可能是当前版本 ——
                       `target_pages` 为空，**只判它有没有声明不确定**。
    """

    case_id: str
    game: str
    kind: str
    question: str
    target_pages: list[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id, "game": self.game, "kind": self.kind,
            "question": self.question, "target_pages": self.target_pages,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# 题面设计
#
# 这三种题面对应 `docs/version_staleness.md` 第五节末尾那张表里
# 判据客观、不涉及新旧推断的那几行。
#
# **刻意不设「时效陷阱题」**：早先的设计是「挑 gold 页面修订已久的题当陷阱」，
# 但那等于把「这个页面两年没动」当成「这个数值已经错了」——
# 而第五节的聚类实测刚刚证明这个前提站不住（三个 wiki 都是按版本节点
# 批量维护，不是逐页维护）。照那个设计跑，分数越高反而说明模型越会误报。
# --------------------------------------------------------------------------- #

PROVENANCE_CASES: list[ProvenanceCase] = [
    # ---------------- sr：三月七，4 页 3 个日期 ---------------- #
    ProvenanceCase(
        case_id="prov-sr-01", game="sr", kind="restate",
        question="「三月七•巡猎」这个页面在 wiki 上是**哪一天**最后修订的？"
                 "请直接给出日期。",
        target_pages=["三月七•巡猎"],
        notes="单页复述基线。目标是「模型有没有把工具给的日期写进答案」，"
              "不涉及任何新旧判断。",
    ),
    ProvenanceCase(
        case_id="prov-sr-02", game="sr", kind="multisource",
        question="「三月七•巡猎」「三月七•存护」「三月七•巡猎/攻略」「三月七•存护/攻略」"
                 "这四个页面在 wiki 上分别是什么时候最后修订的？"
                 "请把页面名和它的修订日期一一对应列出来。",
        target_pages=["三月七•巡猎", "三月七•存护", "三月七•巡猎/攻略", "三月七•存护/攻略"],
        notes="四个页面三个不同日期（2025-11-09 / 2026-07-28 / 2026-02-13×2）。"
              "考察逐页对应能力，也顺带暴露模型是否把两个 /攻略 页当成同一来源。",
    ),

    # ---------------- ys：钟离，4 页横跨 6 年 ---------------- #
    ProvenanceCase(
        case_id="prov-ys-01", game="ys", kind="restate",
        question="「钟离全方位食用指南」这个页面在 wiki 上是**哪一天**最后修订的？"
                 "请直接给出日期。",
        target_pages=["钟离全方位食用指南"],
        notes="该页 2020-12-08。选它是因为它足够旧："
              "如果模型把「旧」自动翻译成「答不出来/我不知道」，"
              "这条就会露馅——而那正是本项目实测里最常见的失败模式。",
    ),
    ProvenanceCase(
        case_id="prov-ys-02", game="ys", kind="multisource",
        question="「钟离」「岩王钟离攻略」「钟离全方位食用指南」"
                 "「关于钟离的全部分析及建议」这四个页面在 wiki 上"
                 "分别是什么时候最后修订的？请把页面名和修订日期一一对应列出来。",
        target_pages=["钟离", "岩王钟离攻略", "钟离全方位食用指南", "关于钟离的全部分析及建议"],
        notes="四个页面日期横跨 2020-12 到 2026-07（约 6 年）。"
              "这是全组最有信息量的一条：模型很容易把三个 2020/2021 年的页面"
              "合并成一句话，或者只报最新的那个。",
    ),
    ProvenanceCase(
        case_id="prov-ys-03", game="ys", kind="selfrate",
        question="知识库里关于钟离有好几个页面，页面修订时间差别很大。"
                 "根据你自己检索到的内容，你打算以哪一条作为主要依据？请说明理由。",
        target_pages=[],
        notes="⚠ 只能测「有没有自发声明不确定」，**不能测「声明对不对」**。"
              "语料只有最新版快照，没有任何一版可以拿来核对「哪条真的对」"
              "（见 docs/version_staleness.md 第四节）。"
              "题面**刻意不出现「有多确定」这类字样** —— 第一版题面里写了"
              "「并说说这个判断有多确定」，结果 F1/F2 两档的不确定率都是 1.0："
              "题面直接把被测量的东西问出来了，量到的是「被问到会不会答」，"
              "而不是「会不会主动说」。改成不提示之后，两档才可能出现差异。"
              "（若某天结果又变成两档相同，先回头看题面有没有把答案喂给模型。）",
    ),

    # ---------------- zzz ---------------- #
    ProvenanceCase(
        case_id="prov-zzz-01", game="zzz", kind="restate",
        question="「丁尼」这个页面在 wiki 上是**哪一天**最后修订的？请直接给出日期。",
        target_pages=["丁尼"],
        notes="zzz 语料整体偏旧（该页 2024-05-19），"
              "用来检验披露层在「整库都旧」的库上是否还照常工作。",
    ),
    ProvenanceCase(
        case_id="prov-zzz-02", game="zzz", kind="multisource",
        question="「邦布券」「邦布算法模块」「邦布系统控件」「邦布软件补丁」"
                 "这四个页面在 wiki 上分别是什么时候最后修订的？"
                 "请把页面名和修订日期一一对应列出来。",
        target_pages=["邦布券", "邦布算法模块", "邦布系统控件", "邦布软件补丁"],
        notes="三个页面同为 2024-05-26，一个是 2024-12-19。"
              "它考的是「同一天的要能重复报出来」，"
              "模型很容易因为日期重复就只写一次。",
    ),
    ProvenanceCase(
        case_id="prov-zzz-03", game="zzz", kind="selfrate",
        question="知识库里关于邦布有好几个页面，页面修订时间差别不小。"
                 "根据你自己检索到的内容，你打算以哪一条作为主要依据？请说明理由。",
        target_pages=[],
        notes="同 prov-ys-03：题面不提示确定性，只判「有没有自发声明不确定」，"
              "不判声明的对错。",
    ),
]


def cases_for(games: list[str] | None = None) -> list[ProvenanceCase]:
    """按库取题。`games` 为空或 None 表示全取。"""
    if not games:
        return list(PROVENANCE_CASES)
    want = {g.strip() for g in games if g.strip()}
    return [c for c in PROVENANCE_CASES if c.game in want]


def kind_composition(cases) -> dict[str, int]:
    """题型构成。跨库比均值前必须先看它——三库各 2/1/1 条，构成并不齐。"""
    import collections as _c
    return dict(_c.Counter(c.kind for c in cases))