"""守卫：实验 D 必须区分「指标差」与「没跑成」。

2026-10-13 实测踩到的坑：`EMBED_MODEL_ALT=embo-01` 已被 SiliconFlow 下线，
索引重建时接口返回 `400 Model does not exist`，`rebuild_index.py` 非零退出。
而 `exp_d.py` 的汇总表对「没有产物文件」只打一行 `（无结果）` ——
这一行和「这个模型指标很差」在表里长得一模一样，
任何人（包括我自己）看汇总表都会读成后者。

这类失败比指标难看得多，也更容易被当成结论写进简历，所以必须单列。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "exp_d.py"


def _main_tree():
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def _main_src() -> str:
    tree = _main_tree()
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    return ast.unparse(main)


def test_missing_artifact_is_reported_as_not_run():
    """产物缺失必须标成「未跑成」，不能只打「无结果」。"""
    src = _main_src()
    assert "（无结果）" not in src, (
        "exp_d 不允许再出现「（无结果）」这种两义表述 —— "
        "它和「该模型指标差」在汇总表里无法区分"
    )
    assert "未跑成" in src, "缺产物的那一行必须显式说明是没跑成"


def test_missing_models_are_listed_with_a_reason():
    """没跑成的模型要单独汇总，并给出可能原因。"""
    src = _main_src()
    assert "missing" in src, "exp_d 必须收集本轮没跑成的模型"
    assert "没有出数" in src, "汇总末尾要显式列出本轮没出数的模型"
    assert "不参与任何对比结论" in src, (
        "没跑成的模型不能参与对比结论 —— 单变量对比要求两档都真的跑过"
    )


def test_rebuild_failure_names_the_likely_cause():
    """索引重建失败要说清最常见原因（模型被下线），而不是只打一个 rc。"""
    src = _main_src()
    assert "Model does not exist" in src, (
        "重建失败时必须提到「模型已下线」这个最常见原因"
    )


def test_rebuild_failure_does_not_silently_continue():
    """重建失败分支不能只写一行普通日志就 continue。"""
    tree = _main_tree()
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    texts = [ast.unparse(n) for n in ast.walk(main) if isinstance(n, ast.If)]
    fail_branch = [t for t in texts if "continue" in t and "rc" in t]
    assert fail_branch, "找不到重建失败的分支"
    for t in fail_branch:
        assert "未跑成" in t, f"重建失败分支必须说明本档未跑成，实际是：{t[:120]}"


def test_pre_change_snapshot_uses_the_ambiguous_wording():
    """双向验证：改动前的 exp_d.py 确实用的是两义表述。

    不做这一步的话，这几条断言测的可能是「本来就有」的东西，
    对本次改动就没有约束力。用 git worktree 取**改造前的固定 commit** 实跑：
    旧版应当 `（无结果）` 在、而「未跑成」/「Model does not exist」都不在。

    为什么锁 commit 而不是 HEAD：exp_d.py 一提交，HEAD 就变成了「修复后」，
    `assert "（无结果）" in src` 会当场反过来失败，守卫自己先失效。
    双向验证的快照必须是**不会随提交移动的锚点**。
    浅克隆或 gc 掉该 commit 时会 skip，而不是误报失败。
    """
    import shutil
    import subprocess
    import tempfile

    # exp_d.py 措辞改造之前最后一个提交。
    BEFORE_REF = "acbb64f"

    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        wt = Path(td) / "pre_change_copy"
        r = subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), BEFORE_REF],
            cwd=str(root), capture_output=True, timeout=180,
            encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            pytest.skip(f"无法检出改造前的提交 {BEFORE_REF}（{str(r.stderr or '')[:120]}）")
        try:
            old = wt / "backend" / "scripts" / "exp_d.py"
            assert old.is_file(), f"{BEFORE_REF} 里找不到 {old}"
            src = old.read_text(encoding="utf-8")
            assert "（无结果）" in src, (
                f"{BEFORE_REF} 已不再用两义表述，双向验证前提不成立，需要重新评估守卫"
            )
            for token in ("未跑成", "Model does not exist", "没有出数"):
                assert token not in src, (
                    f"{BEFORE_REF} 已含 {token!r}，双向验证前提不成立"
                )
        finally:
            shutil.rmtree(td, ignore_errors=True)
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                           cwd=str(root), capture_output=True, timeout=120)
            subprocess.run(["git", "worktree", "prune"], cwd=str(root),
                           capture_output=True, timeout=120)