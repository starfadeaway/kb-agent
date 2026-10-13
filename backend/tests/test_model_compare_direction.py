"""守卫：跨模型比较的输出必须自带方向，不能让读者反推符号。

缺陷（修 `analyze_model_compare.py` 时自查出来的）
--------------------------------------------------
JSON 里原本只有 `significant: true` 和 `mean_diff: +37.6552`。
但延迟是**越低越好**的指标，于是这行读起来是：

    latency_ms  mean_diff=+37.6552  significant=true

而实际含义是 **Qwen 更快**。任何人（包括写文档的我自己）
都可能直接读成「bge 显著更好」—— 正负号刚好反着。

更糟的是这个实验**唯二显著的结果全是延迟**，
正好全部落在这个陷阱里。

这和 `judge.pairwise` 把解析失败静默算成 TIE 是同一类错误：
**把需要语义判断的信息，编码成要读者自己解码的裸值。**

所以输出必须给出 `winner` / `verdict`，方向判定由
`LOWER_IS_BETTER` 真正参与，而不是只拿来打一个标签。

已对修复前的快照实跑确认能抓到缺陷。
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_model_compare.py"


def _load():
    """直接按文件加载脚本（它在 scripts/ 下、靠 sys.path 找 backend 包）。"""
    sys.path.insert(0, str(SCRIPT.parents[1]))
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("amc_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _artifact(game: str = "sr"):
    import json
    # 产物现在按库命名（model_compare_sr.json 等）。
    #
    # 早先这里读的是不带库名的 model_compare.json —— 而那份产物是用
    # **旧默认路径 exp_A.json**（sr 的实验 A，可能停在旧语料）配 Qwen 产物算出来的，
    # A0 阴性对照根本对不上（mean_diff=0.1149），整份检验的前提就不成立。
    # 守卫却还在拿它断言「A2 延迟显著更快」——
    # **等于用一份无效产物给结论背书**，比没有守卫更危险。
    p = SCRIPT.parents[2] / "data" / "eval" / f"model_compare_{game}.json"
    if not p.is_file():
        pytest.skip(f"model_compare_{game}.json 尚未生成")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------- 方向判定本身 ----------

def test_positive_diff_on_latency_means_other_is_better():
    """延迟差为正 = bge 更慢 = Qwen 更好。这一条最容易读反。"""
    mod = _load()
    assert mod._winner(37.66, higher_is_better=False) == "other"


def test_negative_diff_on_latency_means_bge_is_better():
    mod = _load()
    assert mod._winner(-37.66, higher_is_better=False) == "bge"


def test_positive_diff_on_quality_means_bge_is_better():
    mod = _load()
    assert mod._winner(0.0517, higher_is_better=True) == "bge"


def test_zero_diff_is_tie():
    mod = _load()
    assert mod._winner(0.0, higher_is_better=True) == "tie"
    assert mod._winner(0.0, higher_is_better=False) == "tie"


def test_latency_is_registered_as_lower_is_better():
    """延迟必须在表里，否则方向判定会当成「越大越好」。"""
    mod = _load()
    assert "latency_ms" in mod.LOWER_IS_BETTER


# ---------- 结构守卫：winner 必须真的被计算并写入 ----------

def test_winner_is_computed_in_main_not_just_defined():
    """定义了不算数 —— 必须真的参与输出。

    这正是原缺陷的形态：`LOWER_IS_BETTER` 只被用来打 `higher_is_better`
    这个标签，从头到尾没有任何代码用它判断谁更好。
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    called = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_winner" in called, "_winner() 定义了却没被调用：又是一个半成品"


def test_winner_field_is_written_into_output():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    keys = {
        k.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Dict)
        for k in n.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert "winner" in keys, "输出字典里没有 winner 字段"
    assert "verdict" in keys, "输出字典里没有可读的结论字段"


def test_p_underflow_is_flagged():
    """p 的浮点下溢要单独标记，否则 JSON 里的 0.0 会被当成真的 0。"""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    keys = {
        k.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Dict)
        for k in n.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert "p_is_underflow" in keys, "p=0.0 未与浮点下溢区分开"


# ---------- 真实产物校验 ----------

def test_artifact_agrees_with_its_own_significance_flag():
    """产物里 `significant` / `winner` / `p_is_underflow` 必须与 p 值自洽。

    这条断言刻意**不硬编码"延迟必须显著"**。

    原因：2026-10-13 补抓角色页后重跑实验 D，A1 的延迟从 p<1e-10 变成 p=0.0946 ——
    结论确实变了。而原来那条测试断言的是"A1/A2 延迟必须显著"，
    于是它在结论变化时变红，那红是**对的**：它要提醒人复核文档。

    但把方向写死成"必须显著"也不对 —— 换个语料、换个索引，
    延迟差异完全可能变得不显著，那时测试不该逼着人宣称显著。
    所以这里只守**判定逻辑的自洽**：significant 为 True 时 p 必须真的 < 0.05，
    下溢标记必须与 p 的量级一致。数字变不变由实验决定，不由测试决定。
    """
    d = _artifact()
    checked = 0
    for label, block in d.items():
        if not isinstance(block, dict):
            continue
        for name, m in (block.get("metrics") or {}).items():
            p = m.get("p")
            if p is None:
                continue
            checked += 1
            if m.get("p_is_underflow"):
                assert p < 1e-10, (
                    f"{label}.{name} 标了 p_is_underflow 但 p={p}；"
                    "下溢只该给真正浮点下溢的量级"
                )
            else:
                assert p >= 1e-10 or p == 0.0, (
                    f"{label}.{name} p={p} 属于下溢量级却没标 p_is_underflow"
                )
            if m.get("significant"):
                assert p < 0.05, f"{label}.{name} 标了显著但 p={p}"
                assert m.get("winner") in {"bge", "other"}, (
                    f"{label}.{name} 显著却没有 winner —— 方向判定丢了"
                )
    assert checked >= 8, f"产物里可校验的指标太少（{checked}），守卫没生效"


def test_fusion_latency_direction_is_self_consistent():
    """融合档（A2）延迟的方向判定必须与三个量自洽 —— 不断言"显著"。

    **为什么不写死 `significant is True`**：
    那条断言（以及同文件里对 `winner == "other"` / `cohens_d < 0.1` 的断言）
    断言的是**实验 D 的结论**，不是代码不变量。而 `data/eval/model_compare_*.json`
    是入库产物，CI 上真实存在、真实执行 —— 换一次语料、重跑一次实验、换一个索引，
    它就会变红，而红的原因与代码质量毫无关系。

    实测已经发生过：ys 的 A2 延迟 p=0.5788（不显著），而 sr/zzz 显著。
    "三库都显著"从来就不是事实。

    这里守的是**判定逻辑本身**：给定 mean_diff、higher_is_better、winner 三者，
    winner 必须是唯一正确的那个。这类断言对数据更新免疫，重跑实验不会变红。
    """
    for game in ("sr", "ys", "zzz"):
        d = _artifact(game)
        for label, block in d.items():
            if label.startswith("_") or "metrics" not in block:
                continue
            for key, m in block["metrics"].items():
                hi = m["higher_is_better"]
                diff = m["mean_diff"]
                expected = ("tie" if diff == 0
                            else ("bge" if ((diff > 0) == hi) else "other"))
                assert m["winner"] == expected, (
                    f"{game}/{label}.{key}：mean_diff={diff}、higher_is_better={hi} "
                    f"推出 winner 应为 {expected!r}，实际 {m['winner']!r} —— "
                    f"方向判定逻辑错了"
                )


@pytest.mark.parametrize("game", ["sr", "ys", "zzz"])
def test_fusion_latency_direction_holds_in_every_game(game):
    """「融合档 Qwen 更快」这条必须在**三库每一库**方向都成立（不断言显著）。

    为什么加这条：原守卫只看默认那一份产物，于是 sr 上成立就被当成全局结论。
    实测三库里 ys 的 A2 延迟差异 p=0.5788 —— **只有两库显著**。
    文档若写「三库均显著」就是过度外推，所以这里逐库锁，
    让「哪些库支持」在测试里就是显式的。
    """
    d = _artifact(game)
    m = d["A2"]["metrics"]["latency_ms"]
    # 无条件断言方向，**不要**放进 `if m["significant"]:` 里。
    #
    # 放进去的后果很隐蔽：ys 的 A2 p=0.5788（不显著），整条用例会**一条断言都不执行**
    # 然后通过 —— docstring 写着「三库每一库都成立」，实际上 ys 是空跑，
    # 给的是覆盖率的假象。而"方向对不对"这件事与显著不显著无关，
    # 即使 p=0.58，mean_diff 的符号也必须和 winner 一致。
    assert m["winner"] == "other", (
        f"{game} 的 A2 延迟方向必须是 Qwen 更快（winner=other），实际 {m['winner']}"
    )
    assert m["mean_diff"] > 0, f"{game} 的 A2：bge 延迟更大即 bge 更慢（实际 {m['mean_diff']}）"


def test_guards_refuse_to_run_on_failed_control():
    """阴性对照不成立的产物，这些守卫必须拒绝据此下结论。"""
    for game in ("sr", "ys", "zzz"):
        p = SCRIPT.parents[2] / "data" / "eval" / f"model_compare_{game}.json"
        if not p.is_file():
            continue
        import json
        meta = json.loads(p.read_text(encoding="utf-8")).get("_meta", {})
        assert meta.get("a0_identical") is True, (
            f"{game} 的 A0 阴性对照不成立（差异 case：{meta.get('a0_diff_case_ids')}）—— "
            "这份产物里除 embedding 外还有第二个变量在动，"
            "任何基于它的显著性结论都不能成立"
        )


def test_control_group_remains_identical():
    """A0（纯 BM25，不用向量）必须仍然逐位相同 —— 它是对照组的自证。

    只要 A0 对不上，说明两次跑之间混进了语料/分块/评测集这类额外变量，
    A1/A2 的差异就不能只归给 embedding 模型。
    """
    d = _artifact()
    a0 = d["A0"]["metrics"]
    retrieval = [k for k in a0 if k in {"mrr", "ndcg", "recall@3", "page_hit", "hit_rate"}]
    assert retrieval, "产物里没有 A0 的检索指标"
    for k in retrieval:
        assert abs(a0[k]["mean_diff"]) < 1e-9, (
            f"A0.{k} 的 mean_diff={a0[k]['mean_diff']}，阴性对照不再相同 → "
            "两次跑之间有额外变量"
        )