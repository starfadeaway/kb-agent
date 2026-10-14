"""守卫：溯源判据本身。

## 为什么判据模块自己要被测

判据是实验 F 的**测量仪器**。仪器坏了，实验照样跑完、照样出一张漂亮的表，
而那些数字描述的是正则表达式的脾气，不是模型的行为。

这类故障有一个特别难受的性质：**它不会让测试变红，只会让结论变错**。
所以这里的断言大多写成「构造一个模型会怎么答的字符串，看判据给不给 1」，
而不是「断言某个常量等于某个常量」。

## 几个真实的坑，都是在写判据时预判出来的

1. 日期格式：模型写 `2026年7月15日` 和 `2026-07-15` 是同一件事。
   只认一种格式的话，「复述了但格式不同」会被判成没复述。
2. `可能`：数值描述里到处都是（「可能值 +5%」），
   拿它当不确定性信号会让 selfrate 达成率虚高。
3. `date_hit_rate` 分母为 0（selfrate 题）时必须给 None 而不是 0 ——
   给 0 会被算成「一个都没答对」，方向完全相反。
4. 多页题「答对 3/4」：case 级记 0 还是 0.75，两个口径不能混。
"""
from __future__ import annotations

import pytest
from app.eval.provenance_cases import (
    PROVENANCE_CASES,
    ProvenanceCase,
    cases_for,
    kind_composition,
)
from app.eval.provenance_score import (
    aggregate_rows,
    declares_uncertainty,
    has_revision_wording,
    has_staleness_wording,
    mentions_date,
    ref_revision_coverage,
    score_case,
)

DATE = "2026-07-15"


# --------------------------------------------------------------------------- #
# 1. 日期匹配
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("answer", [
    "该页面最后修订于 2026-07-15。",
    "最后修订 2026/07/15。",
    "修订日期是 2026年7月15日。",
    "2026.07.15 更新。",
    "日期 2026-7-15。",
    "紧凑写法 20260715。",
    "带空格 2026 年 7 月 15 日。",
])
def test_mentions_date_accepts_common_writings(answer):
    assert mentions_date(answer, DATE), f"没认出这是同一个日期：{answer}"


@pytest.mark.parametrize("answer", [
    "该页面最后修订于 2025-07-15。",   # 年份不同
    "该页面最后修订于 2026-07-16。",   # 日不同
    "该页面最后修订于 2026-08-15。",   # 月不同
    "该页面很久没动过了。",              # 压根没日期
    "",                                 # 空回答
])
def test_mentions_date_rejects_wrong_or_absent(answer):
    assert not mentions_date(answer, DATE)


def test_mentions_date_rejects_malformed_date_instead_of_guessing():
    """日期本身不合法时必须判 False，不能「尽力匹配」。

    这里最容易写成 `int(m[1:3])` 然后 try/except —— 而 gold 日期来自
    数据库现读，万一某页的 last_revision_ts 是空串或格式异常，
    判据就会把整条 case 静默算成「没命中」，看起来像模型没复述。
    """
    assert mentions_date("2026-07-15", "") is False
    assert mentions_date("2026-07-15", "unknown") is False
    assert mentions_date("2026-07-15", "2026-7") is False
    assert mentions_date("2026-07-15", None) is False


# --------------------------------------------------------------------------- #
# 2. 措辞三类必须分开
# --------------------------------------------------------------------------- #
def test_revision_wording_is_not_staleness_wording():
    """「最后修订 2026-07-15」是**如实转述**，不是风险判断。

    两者混成一个「时效措辞率」就会掩盖本项目最关键的那个区分：
    `last_revision_ts` 是页面编辑时间，陈述它没问题，
    由它推出「内容可能过时」则是聚类实测否掉的那个推断。
    """
    answer = "三月七•巡猎 页面最后修订 2026-07-15。"
    assert has_revision_wording(answer)
    assert not has_staleness_wording(answer)


def test_staleness_wording_detected():
    for text in ("这个数值可能已经过时。", "该数据不是最新版本。", "仅供参考，谨慎参考。"):
        assert has_staleness_wording(text), text


def test_plain_answer_scores_zero_on_all_three_wording_checks():
    plain = "三月七是星穹铁道的四星冰属性存护命途角色，命途为存护。"
    assert not has_revision_wording(plain)
    assert not has_staleness_wording(plain)
    assert not declares_uncertainty(plain)


# --------------------------------------------------------------------------- #
# 3. 不确定性判据：阈值不能被「可能」这类高频词撑起来
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("answer", [
    "我不太确定，可能有出入。",
    "无法判断哪个更新。",
    "建议以官方最新公告为准。",
    "这个判断我把握不大。",
    # 弱标记 + 时效语境 —— 这条锁的是「弱标记不是一律不算」：
    # 只测反面的话，把弱标记整个删掉也能让那几条测试全绿。
    "以上信息来自三月七•巡猎页面，可能不是最新版本。",
])
def test_declares_uncertainty_accepts(answer):
    assert declares_uncertainty(answer)


@pytest.mark.parametrize("answer", [
    # 「可能值」是数值描述，不是自我认知 —— 不排除掉的话
    # selfrate 的达成率会被这类句子凭空抬上去。
    "该天赋可能值 5% 攻击力。",
    "该效果可能叠加两次。",
    "充能上限可能触顶。",
])
def test_declares_uncertainty_rejects_numeric_friends(answer):
    assert not declares_uncertainty(answer), (
        f"把数值描述误判成自我认知了：{answer}"
    )


# --------------------------------------------------------------------------- #
# 4. 引用修订覆盖率
# --------------------------------------------------------------------------- #
def test_ref_revision_coverage_distinguishes_empty_from_zero():
    assert ref_revision_coverage([]) == 0.0
    assert ref_revision_coverage(None) == 0.0
    # 引用了但没带日期 —— 与「压根没引用」是两回事
    assert ref_revision_coverage([{"url": "u"}]) == 0.0
    assert ref_revision_coverage([{"url": "u", "last_revision_ts": "2026-07-15"}]) == 1.0
    mixed = [{"url": "a"}, {"url": "b", "last_revision_ts": "2026-07-15"}]
    assert ref_revision_coverage(mixed) == 0.5


# --------------------------------------------------------------------------- #
# 5. score_case 的分母
# --------------------------------------------------------------------------- #
def _restate_case() -> ProvenanceCase:
    return ProvenanceCase(case_id="t-1", game="sr", kind="restate",
                          question="q", target_pages=["A", "B"])


def test_partial_hit_is_not_full_hit():
    c = _restate_case()
    row = score_case(c, "A 是 2026-07-15", [], {"A": "2026-07-15", "B": "2026-01-01"})
    assert row["date_hits"] == 1 and row["date_targets"] == 2
    assert row["date_hit_rate"] == 0.5
    assert row["all_dates_hit"] is False
    assert row["missed_pages"] == ["B"]


def test_full_hit():
    c = _restate_case()
    row = score_case(c, "A 2026-07-15，B 2026-01-01",
                     [{"url": "u", "last_revision_ts": "2026-07-15"}],
                     {"A": "2026-07-15", "B": "2026-01-01"})
    assert row["all_dates_hit"] is True
    assert row["missed_pages"] == []
    assert row["ref_revision_coverage"] == 1.0


def test_selfrate_case_has_none_denominator_not_zero():
    """selfrate 题没有目标页 —— date_hit_rate 必须是 None，不能是 0。

    0 会被 `avg()` 当成「这条 0% 命中」拉低整个平均值，
    而实际上这条题根本没被考复述能力，它被考的是 `declares_uncertainty`。
    """
    c = ProvenanceCase(case_id="t-2", game="sr", kind="selfrate",
                       question="q", target_pages=[])
    row = score_case(c, "我不太确定哪个更新。", [], {})
    assert row["date_hit_rate"] is None
    assert row["date_targets"] == 0
    assert row["all_dates_hit"] is False
    assert row["declares_uncertainty"] is True


def test_missing_gold_date_does_not_count_as_hit():
    """gold 日期取不到时不能算命中，也不能算「答错」——它压根不可判。"""
    c = _restate_case()
    row = score_case(c, "答案是 2026-07-15", [], {"A": "2026-07-15", "B": ""})
    assert row["missed_pages"] == ["B"]
    assert row["gold_dates"]["B"] == ""


# --------------------------------------------------------------------------- #
# 6. 聚合的分母逐指标不同
# --------------------------------------------------------------------------- #
def test_aggregate_uses_per_metric_denominators():
    rows = [
        # dated 题，答对一半
        {"kind": "restate", "date_targets": 1, "date_hit_rate": 0.5,
         "all_dates_hit": False, "has_revision_wording": True,
         "has_staleness_wording": False, "declares_uncertainty": False,
         "ref_revision_coverage": 1.0, "answer_chars": 100},
        # selfrate 题，没有 date_hit_rate
        {"kind": "selfrate", "date_targets": 0, "date_hit_rate": None,
         "all_dates_hit": False, "has_revision_wording": False,
         "has_staleness_wording": True, "declares_uncertainty": True,
         "ref_revision_coverage": 0.0, "answer_chars": 200},
    ]
    agg = aggregate_rows(rows)
    assert agg["n"] == 2
    assert agg["n_dated"] == 1
    assert agg["n_selfrate"] == 1
    # date 指标只算 1 条；不确定率只算 1 条；措辞率算 2 条
    assert agg["page_level_date_hit_rate"] == 0.5
    assert agg["case_level_all_hit_rate"] == 0.0
    assert agg["has_revision_wording_rate"] == 0.5
    assert agg["has_staleness_wording_rate"] == 0.5
    assert agg["declares_uncertainty_rate"] == 1.0
    assert agg["avg_answer_chars"] == 150


def test_aggregate_of_empty_rows_does_not_zero_divide():
    agg = aggregate_rows([])
    assert agg["n"] == 0
    assert agg["page_level_date_hit_rate"] is None
    assert agg["declares_uncertainty_rate"] is None
    assert agg["avg_answer_chars"] == 0


# --------------------------------------------------------------------------- #
# 7. 题集本身的约束
# --------------------------------------------------------------------------- #
def test_case_ids_are_unique():
    ids = [c.case_id for c in PROVENANCE_CASES]
    assert len(ids) == len(set(ids)), "case_id 重复会让按 id 回填顺序取到别人的行"


def test_every_target_page_names_something():
    """target_pages 为空的只允许是 selfrate 题。

    反过来更危险：一条 restate/multisource 题如果 target_pages 是空的，
    它既不会被 date 指标算分、也不会被判失败，
    就是一条**永远满分且不贡献任何信息**的题。
    """
    for c in PROVENANCE_CASES:
        if c.kind == "selfrate":
            assert c.target_pages == [], f"{c.case_id} 是 selfrate 却带了 target_pages"
        else:
            assert c.target_pages, f"{c.case_id} 没有任何 target_pages，会变成永远满分的空题"


def test_selfrate_question_asks_for_uncertainty():
    """selfrate 题面必须真的在问「有多确定」。

    没有这句，模型就没有理由说出「不确定」，
    于是 `declares_uncertainty_rate` 量到的是
    「模型会不会主动说」而不是「被问到了会不会答」——
    这两件事不能混。
    """
    for c in PROVENANCE_CASES:
        if c.kind == "selfrate":
            assert "确定" in c.question, f"{c.case_id} 题面没在问确定性"


def test_cases_for_filters_by_game():
    assert len(cases_for(["sr"])) == 2
    assert len(cases_for(["sr", "ys"])) == 5
    assert len(cases_for(None)) == len(PROVENANCE_CASES)
    assert kind_composition(cases_for(None))["restate"] == 3


def test_provenance_cases_do_not_leak_into_the_main_eval_set():
    """溯源题不得混进 `cases.py`。

    那是 A/B/C/D/E 历史实验的题源，加题会改分母，
    让新旧实验不再可比 —— 而「新旧用了同一批题」是一切对比的前提。
    """
    from app.eval.cases import ALL_CASES

    prov_ids = {c.case_id for c in PROVENANCE_CASES}
    main_ids = {c.case_id for cs in ALL_CASES.values() for c in cs}
    assert not (prov_ids & main_ids), (
        f"溯源题 id 撞进了主评测集：{sorted(prov_ids & main_ids)}"
    )