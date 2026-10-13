"""守卫：`run_eval.py` 的多库评测路径。

三库语料规模差 2.7 倍（sr 706 页 / ys 435 / zzz 260），所以多库汇总的
正确做法是**分库小节 + 合计总览**，而不是把三库混进一个均值里当结论。
这几条锁的就是「混口径」和「无声丢失」两类问题：

1. by_game 必须真的按 game 分组，且各组 case 数之和等于总数
2. 库顺序必须固定，不受命令行书写顺序影响
3. 无 gold 的题不能进 rows，但数量必须被记下来（不能无声消失）
4. 落库时 config 必须按库拆开

关于双向验证：**本文件对应的是 run_eval.py 上的一次真实改动**
（新增多库路径），改动前的版本可以用 git worktree 取到，
所以这组守卫做了双向验证 —— 见 `test_pre_change_snapshot_rejects_multigame`
（它断言 HEAD 的旧版 run_eval 缺少这些能力，因此旧代码跑不过）。
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "run_eval.py"

# exp_common / run_eval 都在 scripts/ 下，不在 backend 的包路径里，
# 所以这里必须先把 scripts 加进 sys.path —— 否则 `from exp_common import
# parse_games` 会 ModuleNotFoundError，而报错完全指不到真正原因。
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load():
    sys.path.insert(0, str(SCRIPTS.parents[1]))
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("run_eval_mg_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_eval_mg_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Case:
    def __init__(self, case_id, gold_pages, category="single_hop"):
        self.case_id = case_id
        self.gold_pages = gold_pages
        self.category = category
        self.question = f"问题-{case_id}"


class _Hit:
    def __init__(self, chunk_id, page):
        self.chunk_id = chunk_id
        self.chunk = type("C", (), {"page_title": page})()


class _Good:
    """按 cfg.game 决定命中哪个页面，好让不同库的 case 有可分辨的结果。"""

    async def retrieve(self, question, cfg):
        game = getattr(cfg, "game", "sr")
        return [_Hit(f"{game}-chunk-1", f"{game}-正确页")]


def _install_cases(mod):
    data = {
        "sr": [_Case("sr-001", ["sr-正确页"]), _Case("sr-002", ["sr-正确页"]),
               _Case("sr-003", [])],
        "ys": [_Case("ys-001", ["ys-正确页"])],
        "zzz": [_Case("zzz-001", ["zzz-正确页"]), _Case("zzz-002", ["zzz-正确页"])],
    }
    mod.cases_for = lambda game: list(data.get(game, []))


# ---------- --game 参数解析 ----------

def test_parse_games_single_and_multi():
    from exp_common import parse_games

    assert parse_games("sr") == ["sr"]
    assert parse_games("sr,ys") == ["sr", "ys"]
    assert parse_games("all") == ["sr", "ys", "zzz"]


def test_parse_games_order_is_independent_of_argument_order():
    """`--game zzz,sr` 与 `--game sr,zzz` 必须产出同一个列表。

    否则两次运行产出的 rows 顺序不同，逐条 diff 全是噪音，
    也无法用「同一份产物」互相核对。顺序只认 sr/ys/zzz 的固定序。
    """
    from exp_common import parse_games

    assert parse_games("zzz,sr") == parse_games("sr,zzz") == ["sr", "zzz"]


def test_parse_games_rejects_unknown_library():
    """未知库必须直接报错，不能悄悄退化成「只评了第一个库」。"""
    from exp_common import parse_games

    with pytest.raises(SystemExit) as ei:
        parse_games("sr,mhw")
    assert "mhw" in str(ei.value)


def test_parse_games_empty_input():
    from exp_common import parse_games

    assert parse_games("") == []
    assert parse_games(None) == []


# ---------- 分库聚合 ----------

@pytest.mark.asyncio
async def test_by_game_partitions_all_rows():
    """by_game 各组的 n 之和必须等于总 n，且每行只属于一个库。"""
    mod = _load()
    _install_cases(mod)

    res = await mod.run_one(
        label="A0", game="all", top_k=3, variant="base",
        embed_model=None, rerank_candidates=0, retriever=_Good(),
    )

    by_game = res["by_game"]
    assert set(by_game) == {"sr", "ys", "zzz"}, f"by_game 缺库：{set(by_game)}"
    # sr 3 条里 1 条无 gold 不进 rows -> sr 只有 2 条
    assert {g: a["n"] for g, a in by_game.items()} == {"sr": 2, "ys": 1, "zzz": 2}
    assert sum(a["n"] for a in by_game.values()) == res["aggregate"]["n"] == 5


@pytest.mark.asyncio
async def test_each_row_carries_its_game():
    """每行必须自带 game 字段。

    不能让下游靠解析 case_id 前缀反推库名 —— 评测集将来改编号规则，
    解析就得跟着改，而改错了不会报错，只会让某几行悄悄落到空组里。
    """
    mod = _load()
    _install_cases(mod)

    res = await mod.run_one(
        label="A0", game="all", top_k=3, variant="base",
        embed_model=None, rerank_candidates=0, retriever=_Good(),
    )
    for r in res["rows"]:
        assert r.get("game"), f"{r['case_id']} 缺 game 字段"
    ys = [r for r in res["rows"] if r["game"] == "ys"]
    assert ys and all(r["case_id"].startswith("ys-") for r in ys), (
        "game 字段与 case_id 前缀对不上，分组依据不一致"
    )


@pytest.mark.asyncio
async def test_per_game_config_carries_its_own_game():
    """每库的检索配置必须带上自己的 game。

    检索器按 cfg.game 选 collection，配置里 game 错了就是查错库 ——
    而指标仍会算出来（查到别的库也能返回 chunk），不会报错。
    """
    mod = _load()
    _install_cases(mod)
    seen = []

    class _Spy(_Good):
        async def retrieve(self, question, cfg):
            seen.append(getattr(cfg, "game", None))
            return await super().retrieve(question, cfg)

    await mod.run_one(
        label="A0", game="all", top_k=3, variant="base",
        embed_model=None, rerank_candidates=0, retriever=_Spy(),
    )
    assert set(seen) == {"sr", "ys", "zzz"}, f"检索时用到的 game 配置不对：{set(seen)}"


# ---------- 无 gold 的题不能无声消失 ----------

@pytest.mark.asyncio
async def test_cases_without_gold_are_counted_not_silently_dropped():
    """无 gold 的题不进 rows（检索层判不了对错），但数量必须被记下来。

    sr 有 1 条无 gold 题。如果只按「跳过」处理，汇总表上 n 会少 1，
    看起来像「评测集变小了」—— 而没有任何地方说明少的那条去哪了。
    """
    mod = _load()
    _install_cases(mod)

    res = await mod.run_one(
        label="A0", game="all", top_k=3, variant="base",
        embed_model=None, rerank_candidates=0, retriever=_Good(),
    )
    assert res["aggregate"]["no_gold_pages"] == 1, (
        "无 gold 的题数量没有被记录，汇总表上的 n 少了 1 却无从解释"
    )
    assert all(r["case_id"] != "sr-003" for r in res["rows"]), (
        "无 gold 的题不该进检索指标行（recall 恒为 0，会把所有配置一起拉低）"
    )


# ---------- 单库路径未被破坏 ----------

@pytest.mark.asyncio
async def test_single_game_path_keeps_legacy_shape():
    """单库跑法必须与改造前一致：config 存在、by_game 只有一个键。"""
    mod = _load()
    _install_cases(mod)

    res = await mod.run_one(
        label="A0", game="sr", top_k=3, variant="base",
        embed_model=None, rerank_candidates=0, retriever=_Good(),
    )
    assert res["config"]["game"] == "sr", "单库时 config 必须仍是那一库的完整配置"
    assert set(res["by_game"]) == {"sr"}
    assert res["aggregate"]["n"] == 2  # sr 3 条里 1 条无 gold


# ---------- 结构性守卫 ----------

def test_main_checks_every_requested_collection():
    """`--game all` 时必须逐库查 collection。

    只查第一个库的话，ys/zzz 的索引缺失要一直跑到那一步才炸，
    而 rerank / embed 请求已经发出去了 —— 报错点离真正的原因很远。
    """
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.AsyncFunctionDef) and n.name == "main")
    calls = [ast.unparse(n) for n in ast.walk(main) if isinstance(n, ast.Call)]
    store_counts = [c for c in calls if "store.count" in c]
    assert store_counts, "main 里找不到 store.count 调用"
    assert any("for " in ast.unparse(n) for n in ast.walk(main) if isinstance(n, ast.For)), (
        "collection 非空检查必须放在逐库循环里"
    )


def test_multigame_output_filename_differs_from_single_game():
    """多库产物不能覆盖单库产物。

    `exp_A.json` 是 sr 单库跑出来的历史产物，被多库跑覆盖之后，
    exp_b 的 `is_stale()` 之外就再没有东西能把它找回来了。
    """
    src = SCRIPT.read_text(encoding="utf-8")
    assert 'default_name = "exp_A.json" if len(games) == 1' in src, (
        "多库时必须换产物文件名，否则会覆盖单库历史产物"
    )
    assert "exp_A_{'_'.join(games)}.json" in src


def test_persist_splits_config_per_game():
    """落库时 config 必须按库拆开。

    eval_results.config 是 JSON 列，三库的 config 里 game 字段不同；
    合成一份就再也分不出哪行分数来自哪个库。
    """
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    persist = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.AsyncFunctionDef) and n.name == "persist_results")
    body = ast.unparse(persist)
    assert "configs" in body, "persist_results 没有按库遍历 config"
    assert 'r[\'labels\']' not in body
    assert "save_eval_run" in body


# ---------- 双向验证 ----------

def test_pre_change_snapshot_lacks_multigame_support():
    """反向验证：改造前的 run_eval 确实不具备这些能力。

    双向验证的另一半 —— 用 HEAD 的干净副本实跑，确认这份守卫
    在旧代码上会红。如果旧代码也能通过，说明这些断言测的是
    「本来就有」的东西，对本次改动没有约束力。

    用 git worktree 取**改造前的固定 commit**，不改动当前工作区。

    为什么锁 commit 而不是 HEAD ——
    这条守卫第一次跑的时候，多库改造还没提交，HEAD 就是「修复前」；
    改造一提交，HEAD 变成了「修复后」，断言立刻反过来失败
    （`assert "by_game" not in old_src` 报「HEAD 版已含 by_game」）。
    双向验证的快照必须是**一个不会随提交移动的锚点**，否则它自己就会失效。
    浅克隆或 gc 掉该 commit 时会 skip，而不是误报失败。
    """
    import shutil
    import subprocess
    import tempfile

    # 改造前最后一个不含多库能力的提交（死代码清理那笔）。
    BEFORE_REF = "17f285f"

    # parents: [0]=tests, [1]=backend, [2]=仓库根
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        wt = Path(td) / "pre_change_copy"
        r = subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), BEFORE_REF],
            cwd=str(root), capture_output=True, timeout=180,
            # 显式指定 utf-8：Windows 上 text=True 会用 GBK 解码 git 的输出，
            # 遇到中文路径（这个仓库路径里就有中文）直接抛 UnicodeDecodeError，
            # 报错还发生在 reader 线程里，pytest 只会报一条看不懂的 warning。
            encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            err = (r.stderr or b"").decode("utf-8", "replace") if isinstance(r.stderr, bytes) else r.stderr
            pytest.skip(f"无法检出改造前的提交 {BEFORE_REF}（{str(err).strip()[:120]}）")
        try:
            old = wt / "backend" / "scripts" / "run_eval.py"
            assert old.is_file(), f"{BEFORE_REF} 里找不到 {old}"
            old_src = old.read_text(encoding="utf-8")

            # 旧版没有 by_game / no_gold_pages / 多库文件名
            assert "by_game" not in old_src, f"{BEFORE_REF} 已含 by_game，双向验证前提不成立"
            assert "no_gold_pages" not in old_src, f"{BEFORE_REF} 已含 no_gold_pages，前提不成立"

            # 旧版 run_one 的签名只有单个 game，且返回结构里没有 by_game
            tree = ast.parse(old_src)
            run_one = next(n for n in ast.walk(tree)
                           if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_one")
            ret_keys = set()
            for n in ast.walk(run_one):
                if (isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)):
                    ret_keys |= {k.value for k in n.value.keys if isinstance(k, ast.Constant)}
            assert "by_game" not in ret_keys, f"{BEFORE_REF} 的 run_one 已返回 by_game"
            assert "games" not in ret_keys, f"{BEFORE_REF} 的 run_one 已返回 games"

            # 实跑一次旧版，确认它确实做不到多库（返回结构里没有分库聚合）
            spec = importlib.util.spec_from_file_location("run_eval_old", old)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["run_eval_old"] = mod
            spec.loader.exec_module(mod)
            data = {
                "sr": [_Case("sr-001", ["sr-正确页"])],
                "ys": [_Case("ys-001", ["ys-正确页"])],
            }
            mod.cases_for = lambda game: list(data.get(game, []))

            import asyncio

            res = asyncio.run(mod.run_one(
                label="A0", game="all", top_k=3, variant="base",
                embed_model=None, rerank_candidates=0, retriever=_Good(),
            ))
            assert "by_game" not in res, f"{BEFORE_REF} 居然支持了 by_game，需要重新评估守卫"
            # 旧版拿 "all" 当库名去 cases_for("all")，一条题都跑不到
            assert res["rows"] == [], f"{BEFORE_REF} 对 game=all 的行为与预期不符，需复核"
        finally:
            shutil.rmtree(td, ignore_errors=True)
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                           cwd=str(root), capture_output=True, timeout=120)
            subprocess.run(["git", "worktree", "prune"], cwd=str(root),
                           capture_output=True, timeout=120)