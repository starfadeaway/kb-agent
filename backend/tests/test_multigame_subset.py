"""守卫：三库合并评测后，case 的标识必须仍然可唯一定位。

## 为什么需要这条

`exp_c.run_one` 用 `as_completed` 跑完再按 `case_id` 回填顺序：

```python
rows = [next(x for x in ordered if x["case_id"] == c.case_id) for c in case_list]
```

一旦有**重复的 case_id**，`next()` 会静默取到**另一个 case 的行**：
不报错、不抛异常，只是每条题的分数都张冠李戴。
更糟的是聚合与配对检验照样能算出漂亮的 p 值 ——
这是最难事后发现的一类错误，因为所有数字都"正常"。

跨库合并让这个风险从理论变成现实：三库各有一套 `sr-001` / `ys-001` / `zzz-001`
的独立编号，合并时一旦前缀丢失或重复，统计就全错了。
"""
from __future__ import annotations

import sys
from pathlib import Path

# exp_common / exp_c / exp_e 都在 scripts/ 下，不在 backend 的包路径里。
# 不加这一行，`from exp_common import load_cases` 会 ModuleNotFoundError，
# 而报错完全指不到「只是路径没注册」。
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_case_ids_are_unique_across_games():
    """每个库、以及三库合并后，case_id 都必须唯一。"""
    from exp_common import load_cases

    for game in ("sr", "ys", "zzz", "all"):
        ids = [c.case_id for c in load_cases(game=game)]
        assert len(ids) == len(set(ids)), (
            f"--game {game} 的 case_id 有重复，按 id 回填结果会张冠李戴："
            f"{sorted(i for i in set(ids) if ids.count(i) > 1)}"
        )


def test_case_id_prefix_matches_game_field():
    """case_id 前缀必须与 `case.game` 一致。

    跨库统计按 `case.game` 分组，而不是解析 case_id 前缀。
    两者不一致时，某几行会静默归到错误的一组里 ——
    表现为「某个库的题数莫名不对」，但不会报错。
    """
    from app.eval.cases import ALL_CASES

    for game, cases in ALL_CASES.items():
        for c in cases:
            assert c.case_id.startswith(f"{game}-"), (
                f"{c.case_id} 的 game 字段是 {c.game}，前缀对不上 —— "
                "按 game 分组时它会被算进另一个库"
            )


def test_multigame_subset_actually_spans_all_games():
    """`--game all` 必须真的返回三库的题。

    `load_cases` 里如果有一处库名写错（比如 `ALL_CASES.get('sr')` 写两遍），
    多库模式会静默退化成「只跑了 sr 一次」——
    而产物文件名仍然叫 exp_C_sr_ys_zzz.json，看不出来。
    """
    from exp_common import load_cases, parse_games

    all_cases = load_cases(game="all")
    games_seen = {c.game for c in all_cases}
    assert games_seen == {"sr", "ys", "zzz"}, (
        f"--game all 实际只返回了 {games_seen}，但 parse_games 说有 {set(parse_games('all'))}"
    )


def test_subset_composition_is_reported():
    """题型构成必须能被取出来。

    三库题型分布差很多（ys 的可用题里 7/9 是 single_hop），
    不看构成就比三库均值 = 把「题型构成差」读成「Agent 能力差」。
    """
    from exp_common import load_cases, subset_composition

    comp = subset_composition(load_cases(game="all"))
    assert comp, "题型构成不应为空"
    # 六种题型在全部题里都出现过
    for cat in ("single_hop", "multi_hop", "semantic", "numerical",
                "unanswerable", "routing"):
        assert cat in comp, f"全部题的题型构成里缺 {cat}：{comp}"