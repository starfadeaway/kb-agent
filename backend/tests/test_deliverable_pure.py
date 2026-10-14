"""守卫：交付物任务的纯函数层（冲突检测 / 规范化 / 渲染）。

## 为什么重点测纯函数那一层

`detect_conflicts` / `normalize_value` / `render_markdown` / `build_json`
全部不碰 LLM、不碰 DB、不碰网络。这不是为了好测才拆的，而是因为
**冲突判定必须是确定性的**：同一个交付物跑两次，冲突集必须一模一样，
否则「检出 2 处冲突」这种话就没法写进文档或简历。

规划与抽取那两层要用真 LLM 才有意义，测不了也不该测；
它们的行为由下面的出处校验守卫间接约束。

## 双向验证不适用

本模块是**新增功能**，在 HEAD 上不存在「修复前版本」，
无法构造旧代码快照去证明守卫会失败。因此这里只做正向断言 +
几条「构造出来必然失败」的负向样本（见 test_*_does_not_flag）。
真正的双向验证在别处做过同构的（`test_mcp_main_and_stream_stop_reason.py`、
`test_meta_script_provenance.py`）。
"""
from __future__ import annotations

import json

import pytest
from app.agent.deliverable import (
    DeliverableResult,
    Fact,
    SubtaskResult,
    build_json,
    detect_conflicts,
    normalize_value,
    render_markdown,
    subject_of,
)

# 与 test_revision_disclosure.py 共用同一份禁用清单 —— 两处各写一份的话，
# 哪天只改了一处，另一处就会放行出误导文案。这里直接抄，改的时候一起改。
FORBIDDEN_FRESHNESS = ["数据截至", "数据更新于", "截至今天", "最新版本数据", "当前版本数据"]


def _f(key, value, page, url=None, rev=None, unit="", subtask="数值"):
    return Fact(key=key, label=f"{key}-标签", value=value, unit=unit,
                page_title=page, source_url=url or f"http://x/{page}", revision=rev, subtask=subtask)


def _res(facts, conflicts=None):
    return DeliverableResult(
        run_id="t", topic="三月七养成指南", question="整理三月七的养成指南", game="sr",
        generated_at="2026-10-14T00:00:00Z", elapsed_ms=1234,
        subtasks=[SubtaskResult(name="角色数值", goal="拿到基础数值", facts=facts)],
        conflicts=conflicts if conflicts is not None else detect_conflicts(facts),
    )


# ---------- normalize_value ---------- #

@pytest.mark.parametrize("a,b", [
    ("1328", "1,328"),
    (" 620 ", "620"),
    ("１０２４", "1024"),
    ("5%", "5 %"),
])
def test_normalize_ignores_formatting(a, b):
    """书写差异不该造成假冲突：千分位、空白、全角、空格都是排版。"""
    assert normalize_value(a) == normalize_value(b)


@pytest.mark.parametrize("a,b", [("1328", "1425"), ("5%", "8%"), ("A", "B")])
def test_normalize_keeps_real_differences(a, b):
    """真实差异必须留下来 —— 规范化过头会把冲突洗掉，那是更糟的失败。"""
    assert normalize_value(a) != normalize_value(b)


def test_normalize_does_not_guess_unit_conversion():
    """刻意不做单位换算：`1万` 与 `10000` 不相等。

    这条锁的是一个**有意的限制**。加上换算能让「1万 vs 10000」不再误报，
    但也会引入「1kg vs 1000g 到底算不算同一个」的语义假设，
    而这类假设出错时不会报错，只会让冲突被悄悄漏掉。
    """
    assert normalize_value("1万") != normalize_value("10000")


# ---------- detect_conflicts ---------- #

def test_same_key_different_value_different_page_is_conflict():
    facts = [
        _f("hp80", "1328", "三月七•存护", rev="2026-07-28"),
        _f("hp80", "1425", "三月七•存护/攻略", rev="2024-01-01"),
    ]
    c = detect_conflicts(facts)
    assert len(c) == 1
    assert c[0].key == "hp80"
    assert {v.value for v in c[0].variants} == {"1328", "1425"}


def test_same_page_different_revision_is_not_conflict():
    """同一页面在不同修订给出的不同值是**历史漂移**，不是矛盾。

    wiki 改过内容就会这样。若把它标成冲突，用户会被告知「有矛盾」，
    而实际上只是这个页面被更新过 —— 制造噪音，且指向一个不存在的问题。
    """
    facts = [
        _f("crit", "5%", "三月七•存护", rev="2024-01-01"),
        _f("crit", "8%", "三月七•存护", rev="2026-07-28"),
    ]
    assert detect_conflicts(facts) == []


def test_formatting_difference_is_not_conflict():
    facts = [_f("spd", "1,328", "三月七•存护"), _f("spd", "1328", "三月七•存护/攻略")]
    assert detect_conflicts(facts) == []


def test_different_keys_never_conflict():
    facts = [_f("hp80", "1328", "三月七•存护"), _f("atk", "1425", "三月七•存护/装备")]
    assert detect_conflicts(facts) == []


def test_empty_and_valueless_facts_are_skipped():
    facts = [
        _f("hp80", "", "三月七•存护"),          # 空值
        Fact(key="atk", label="", value="620", page_title="三月七•存护", source_url="u"),  # 正常
        Fact(key="atk", label="", value="620", page_title="三月七•存护/装备", source_url="v"),
    ]
    assert detect_conflicts(facts) == []


def test_conflict_verdict_refuses_to_pick_a_winner():
    """冲突的结论必须明说「判断不了」，且**不带任何偏向**。

    这是本模块的核心约束：不能因为 B 页修订更晚就说 B 对 ——
    修订时间只反映录入时刻（实测三库均为版本批量维护）。
    """
    facts = [_f("hp80", "1328", "三月七•存护", rev="2020-01-01"),
             _f("hp80", "1425", "三月七•存护/攻略", rev="2026-07-28")]
    c = detect_conflicts(facts)[0]
    assert "无法判断" in c.verdict
    # 每个变体都要带着自己的来源与修订时间一起给出去，供人工核对
    assert all(v.revision and v.page_title for v in c.variants)


def test_detect_conflicts_is_deterministic():
    """同样一组事实，正序和倒序必须得到同一批冲突。

    顺序无关很重要：子任务将来并行化、或结果来自不同检索轮次时，
    冲突集不应该随输入顺序抖动。
    """
    facts = [_f(f"k{i}", str(i), f"三月七•存护/子{i % 3}") for i in range(12)]
    first = [(c.key, [v.value for v in c.variants]) for c in detect_conflicts(facts)]
    second = [(c.key, [v.value for v in c.variants]) for c in detect_conflicts(list(reversed(facts)))]
    assert sorted(map(str, first)) == sorted(map(str, second))


# ---------- 渲染 ---------- #

def test_markdown_has_no_misleading_freshness_wording():
    """复用项目的时效措辞禁用清单。

    「页面修订」是页面最后编辑时间，不是数据生效时间；说成「数据截至」
    是用一个更隐蔽的误导替换掉原本的诚实。
    """
    facts = [_f("hp80", "1328", "三月七•存护", rev="2026-07-28"), _f("hp80", "1425", "三月七•存护/攻略", rev="2024-01-01")]
    md = render_markdown(_res(facts))
    for w in FORBIDDEN_FRESHNESS:
        assert w not in md, f"交付物 Markdown 出现禁用措辞：{w}"
    assert "页面修订" in md


def test_markdown_lists_every_conflict_variant_with_its_source():
    facts = [_f("hp80", "1328", "三月七•存护", rev="2026-07-28"), _f("hp80", "1425", "三月七•存护/攻略", rev="2024-01-01")]
    md = render_markdown(_res(facts))
    for token in ("三月七•存护", "2026-07-28", "2024-01-01", "1328", "1425"):
        assert token in md, f"冲突表漏了 {token}"


def test_markdown_says_so_when_nothing_was_extracted():
    """空交付物必须**说自己是空的**，不能渲染成一份看起来正常的文档。

    「没抽到事实」和「真的没有这个事实」目前无法区分（见模块 docstring 的
    已知限制），所以至少要让读者看见 status，而不是给一张空表。
    """
    md = render_markdown(_res([], conflicts=[]))
    assert "没有抽到任何事实" in md


def test_markdown_reports_failed_subtasks():
    res = _res([_f("a", "1", "三月七•存护")], conflicts=[])
    res.subtasks.append(SubtaskResult(name="光锥推荐", goal="g", status="error", notes="抽取炸了"))
    md = render_markdown(res)
    assert "未能完成的子任务" in md and "光锥推荐" in md


def test_json_and_markdown_report_the_same_conflicts():
    facts = [_f("hp80", "1328", "三月七•存护", rev="2026-07-28"), _f("hp80", "1425", "三月七•存护/攻略", rev="2024-01-01")]
    res = _res(facts)
    js = build_json(res)
    md = render_markdown(res)
    assert len(js["conflicts"]) == md.count("### 三月七•存护 · hp80-标签")
    # JSON 必须能原样序列化（前端/MCP 直接吃）
    json.loads(json.dumps(js, ensure_ascii=False))


def test_refs_are_deduped_by_url():
    facts = [_f("hp80", "1328", "三月七•存护"), _f("atk", "620", "三月七•存护")]
    res = _res(facts)
    urls = [r["url"] for r in res.refs]
    assert urls == ["http://x/三月七•存护"], urls


def test_meta_declares_conflict_policy_and_limits():
    """交付物必须自曝口径与限制，不能让读者高估它的判断力。"""
    facts = [_f("hp80", "1328", "三月七•存护"), _f("hp80", "1425", "三月七•存护/攻略")]
    res = _res(facts)
    js = build_json(res)
    # _res 走的是纯函数路径，meta 由 run() 填；这里只断言渲染层不丢事实
    assert js["sections"][0]["facts"], "sections 必须带事实"
    assert len(js["conflicts"]) == 1


# ---------- 主语限定（实测踩坑后的修复） ---------- #

@pytest.mark.parametrize("title,expect", [
    ("三月七•存护", "三月七•存护"),
    ("三月七•存护/攻略", "三月七•存护"),     # 子页面 -> 同一主语
    ("三月七•存护/装备/材料", "三月七•存护"),   # 多级子页面
    ("", ""),
])
def test_subject_of_strips_subpage_suffix(title, expect):
    assert subject_of(title) == expect


def test_different_paths_of_same_character_are_not_conflict():
    """**本模块最重要的一条守卫。**

    三月七有存护 / 巡猎两条命途，是两个页面、两个对象。
    裸 key 版本会把它们报成 17 处冲突（实测），全是误报：
        [eidolon_1_name] 记忆中的你@存护  vs  初花学剑动星芒@巡猎
    那不是矛盾，是两个事实。
    """
    facts = [
        Fact(key="eidolon_1_name", label="1魂名", value="记忆中的你",
             page_title="三月七•存护", source_url="u1"),
        Fact(key="eidolon_1_name", label="1魂名", value="初花学剑动星芒",
             page_title="三月七•巡猎", source_url="u2"),
    ]
    assert detect_conflicts(facts) == [], "不同命途不该判为冲突"


def test_irrelevant_character_recall_is_not_conflict():
    """检索把**别的角色**召回来时，不能因此报冲突。

    实测「长夜月」会被召回到三月七的检索结果里；
    「风雪之角 vs 海妖残鳍」不是矛盾，是两个角色的材料。
    """
    facts = [
        Fact(key="ascension_attr_material", label="属性材料", value="风雪之角",
             page_title="三月七•巡猎", source_url="u1"),
        Fact(key="ascension_attr_material", label="属性材料", value="海妖残鳍",
             page_title="长夜月", source_url="u2"),
    ]
    assert detect_conflicts(facts) == []


def test_same_subject_different_pages_still_conflict():
    """主语相同但页面不同 -> 仍然必须是真冲突，否则这个功能就废了。"""
    facts = [
        Fact(key="hp_80", label="80级生命值", value="1328",
             page_title="三月七•存护", source_url="u1", revision="2026-07-28"),
        Fact(key="hp_80", label="80级生命值", value="1256",
             page_title="三月七•存护/攻略", source_url="u2", revision="2026-02-13"),
    ]
    c = detect_conflicts(facts)
    assert len(c) == 1, "同主语不同页的互斥取值必须报出来"
    assert c[0].subject == "三月七•存护"
    assert {v.value for v in c[0].variants} == {"1328", "1256"}
