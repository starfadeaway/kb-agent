"""评测集的**可验证性**测试。

EVALUATION.md 里记着一条教训：早期一条算术题的 gold 本身前提就不成立，
模型答不出来反而是对的 —— **一个自己都没验证过的 gold，会把模型的正确行为判成错误。**

所以这里做两件事：
1. 结构检查：case_id 不重复、game 字段和所在表一致、gold_pages 非空时页面真实存在；
2. **溯源检查**：对能从 gold 断言出关键数值的题，去本地语料里核对那个值真的存在。

第 2 条是重点：它把"gold 对不对"从人工信任变成可重复验证。
需要本地语料，没有就 skip（CI 上语料不入库）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.eval.cases import ALL_CASES, all_cases, cases_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def page_titles() -> dict[str, set[str]]:
    """game → 该库真实存在的页面标题集合。"""
    out: dict[str, set[str]] = {}
    for game in ALL_CASES:
        d = RAW / game
        titles: set[str] = set()
        if d.is_dir():
            for meta_p in d.glob("*.meta.json"):
                try:
                    titles.add(json.loads(meta_p.read_text(encoding="utf-8")).get("title", ""))
                except ValueError:
                    continue
        out[game] = titles
    return out


def test_all_three_games_have_cases():
    """之前只有 sr 有完整评测集，zzz 1 条、ys 0 条 —— 所有量化结论都只来自一个库。"""
    for game in ("sr", "zzz", "ys"):
        assert len(ALL_CASES.get(game, [])) >= 10, f"{game} 只有 {len(ALL_CASES.get(game, []))} 条"


def test_case_ids_are_unique_globally():
    ids = [c.case_id for c in all_cases()]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"重复的 case_id：{sorted(dupes)}"


def test_case_id_prefix_matches_game():
    for c in all_cases():
        assert c.case_id.startswith(f"{c.game}-"), f"{c.case_id} 的 game 字段是 {c.game}"


def test_game_field_matches_table():
    for game, cases in ALL_CASES.items():
        for c in cases:
            assert c.game == game, f"{c.case_id} 在 {game} 表里但 game={c.game}"


def test_categories_are_from_known_set():
    known = {"single_hop", "multi_hop", "semantic", "numerical", "routing", "unanswerable"}
    for c in all_cases():
        assert c.category in known, f"{c.case_id} 的 category={c.category} 不在已知集合里"


def test_cases_for_unknown_game_returns_empty():
    assert cases_for("nonexistent") == []


def test_gold_answer_never_defers_to_wiki():
    """"以 wiki 为准"这种 gold 等于没有 gold —— 无法判定对错。"""
    banned = ("以 wiki 为准", "以wiki为准", "见 wiki", "参见 wiki")
    for c in all_cases():
        for b in banned:
            assert b not in c.gold_answer, f"{c.case_id} 的 gold 是废话：{c.gold_answer}"


def test_gold_pages_exist_in_corpus(page_titles):
    """gold_pages 指向的页面必须真的被抓进来了，否则这条题永远判失败。"""
    # 没有语料时必须 **skip**，不能 `continue`。
    #
    # `continue` 的话整条用例等于什么都没跑却通过 —— 给的是覆盖率的假象。
    # 而且 `data/raw/` 是 gitignore 的，CI 上必然没有语料，
    # 于是这条守卫在 CI 上恒真，从来没起过作用。skip 至少在报告里是诚实的。
    if not any(page_titles.get(g) for g in ("sr", "ys", "zzz")):
        pytest.skip("本地无原始语料（data/raw 未入库），无法核对 gold 页面")

    missing: list[str] = []
    for c in all_cases():
        titles = page_titles.get(c.game) or set()
        for gp in c.gold_pages:
            if gp not in titles:
                missing.append(f"{c.case_id} -> 「{gp}」")
    assert not missing, f"gold_pages 在语料里不存在：{missing}"


def test_answerable_cases_have_gold_pages():
    """能答的题必须指明证据页；只有 unanswerable / 纯路由题可以不指。"""
    for c in all_cases():
        if c.category in ("unanswerable", "routing"):
            continue
        assert c.gold_pages, f"{c.case_id}（{c.category}）没有 gold_pages，无法判定召回"


@pytest.mark.parametrize(
    ("case_id", "expected_in_page"),
    [
        # 下面每个值都是从 data/raw 的原始页面里抄出来的，
        # 用 scripts/dump_fields.py 可以复现核对过程。
        ("zzz-002", ["维多利亚家政", "冰"]),
        ("zzz-003", ["S", "是"]),
        ("zzz-004", ["电", "击破", "狡兔屋"]),
        ("zzz-005", ["怪啖屋", "S"]),
        ("zzz-006", ["B", "32"]),
        ("zzz-007", ["调频", "玩具店"]),
        ("zzz-008", ["充盈电鸣"]),
        ("zzz-011", ["617", "135", "49"]),
        ("ys-001", ["冒险道具", "须弥"]),
        ("ys-002", ["地图探索"]),
        ("ys-003", ["3", "七国徽印"]),
        ("ys-004", ["明星斋"]),
        ("ys-005", ["1", "3"]),
        ("ys-006", ["1000"]),
        ("ys-007", ["30%"]),
        ("ys-008", ["仲夏庭园"]),
    ],
)
def test_gold_values_actually_appear_in_source_pages(case_id, expected_in_page, page_titles):
    """溯源检查：gold 里断言的值，必须在对应页面原文里找得到。

    这条测试失败意味着两种可能：gold 写错了，或者语料被重建后内容变了。
    两种都必须处理 —— 前者会让评测集失真，后者会让所有历史实验数字失效。
    """
    case = next(c for c in all_cases() if c.case_id == case_id)
    d = RAW / case.game
    if not d.is_dir():
        pytest.skip("没有本地语料")

    from app.ingest.template_parser import clean_wikitext

    for title in case.gold_pages:
        meta_hits = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in d.glob("*.meta.json")
            if json.loads(p.read_text(encoding="utf-8")).get("title") == title
        ]
        assert meta_hits, f"{case_id} 的 gold 页面「{title}」在 {case.game} 语料里不存在"
        txt = list(d.glob(f"{meta_hits[0]['page_id']}_*.txt"))[0].read_text(
            encoding="utf-8", errors="ignore"
        )
        text = clean_wikitext(txt)
        for value in expected_in_page:
            assert value in text, (
                f"{case_id} 的 gold 断言「{value}」在页面「{title}」里找不到。"
                f"页面实际开头：{text[:200]}"
            )