"""守卫：跨模型检验的阴性对照不成立时必须显式失败。

`analyze_model_compare.py` 的全部前提是「除 embedding 模型外没有第二个变量在动」。
A0（纯 BM25，完全不用向量）逐位相同就是这件事的唯一自证。

一旦 A0 对不上，说明语料版本 / BM25 参数 / top_k / 评测集样本里有东西没冻住 ——
此时 A1/A2 的 p 值都不能读作 embedding 造成的差异。但表格照样会把它们打出来，
看起来和正常结果一模一样。

2026-10-13 实测踩到：三库版跑完后脚本仍默认读 sr 的**旧产物** `exp_A.json`，
A0 对不上（False）却照样打出了「bge 显著更优 p=0.0142」。
两个问题一起修：
1. 阴性对照不成立时显式报错，而不是只打一行 `False`；
2. 默认产物按 `--game` 推导，且默认输出按库命名，不互相覆盖。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_model_compare.py"


def _src() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _main_src() -> str:
    tree = ast.parse(_src())
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    return ast.unparse(main)


def test_control_failure_is_explicit_not_a_bare_bool():
    """A0 对不上时必须报错，不能只打一行 False。"""
    src = _main_src()
    assert "阴性对照 A0 不成立" in src, (
        "A0 阴性对照不成立时必须显式报错 —— 只打一行 False 会被当成正常输出略过"
    )
    assert "第二个隐藏变量" in src, "必须点明失败意味着还有未冻住的变量"


def test_control_failure_invalidates_downstream_reading():
    """阴性对照失败时要声明上面表格的 p 值不可用。"""
    src = _main_src()
    assert "不能" in src and "读作 embedding 模型造成的差异" in src, (
        "A0 失败时必须声明 A1/A2 的 p 值不能读作 embedding 差异，"
        "否则表格照样会被当正常结果引用"
    )


def test_control_result_is_persisted_with_case_ids():
    """不成立时要记下是哪几条对不上，只留一个 False 无法排查。"""
    src = _main_src()
    assert "a0_identical" in src, "阴性对照结论必须写进产物"
    assert "a0_diff_case_ids" in src, "必须记录逐条不同的 case_id，否则无从排查"


def test_files_are_derived_per_game():
    """默认产物必须按 --game 推导，不能写死某一个库的路径。"""
    src = _src()
    assert 'f"exp_D_BAAI_bge-m3_{game}_k3.json"' in src, (
        "bge 侧产物路径必须由 --game 推导；写死会让三库跑出同一份对比"
    )
    assert 'f"model_compare_{game}.json"' in src, (
        "输出必须按库命名，否则后跑的库会覆盖先跑的库的结果"
    )
    # 旧版写死的路径不能再出现
    assert '"exp_A.json"' not in src, (
        "exp_A.json 是 sr 的实验 A 产物（可能是旧语料），不能再当 bge 侧输入"
    )


def test_game_argument_exists():
    """必须有 --game 参数，否则三库无法分别跑。"""
    src = _src()
    assert '"--game"' in src, "缺少 --game 参数"
    assert '"--game", default="sr"' in src


def test_no_stale_module_constants():
    """写死路径的模块常量在改成按 game 推导后就是死代码。"""
    src = _src()
    assert "BGE_FILE" not in src, "BGE_FILE 是写死 sr 的路径常量，已无调用方，应删除"
    assert "QWEN_GLOB" not in src, "QWEN_GLOB 同上"


def test_pre_change_snapshot_soft_failed_on_control():
    """双向验证：改造前对 A0 失败只打一行 False，且产物路径写死。

    旧版 exp_d 跑完后这个脚本默认拿 sr 的旧 exp_A.json 去比，
    A0 对不上仍照常输出显著结论 —— 守卫必须能抓住这一点。
    """
    import shutil
    import subprocess
    import tempfile

    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        wt = Path(td) / "pre_change_copy"
        r = subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), "acbb64f"],
            cwd=str(root), capture_output=True, timeout=180,
            encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            pytest.skip(f"无法检出改造前的提交（{str(r.stderr or '')[:120]}）")
        try:
            old = wt / "backend" / "scripts" / "analyze_model_compare.py"
            assert old.is_file(), f"改造前找不到 {old}"
            src = old.read_text(encoding="utf-8")
            assert "阴性对照 A0 不成立" not in src, (
                "改造前已含显式失败分支，双向验证前提不成立"
            )
            assert "a0_diff_case_ids" not in src, "改造前已记录差异 case_id，前提不成立"
            assert '"--game"' not in src, "改造前已有 --game，前提不成立"
            assert "BGE_FILE" in src, "改造前的写死路径常量应存在"
        finally:
            shutil.rmtree(td, ignore_errors=True)
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                           cwd=str(root), capture_output=True, timeout=120)
            subprocess.run(["git", "worktree", "prune"], cwd=str(root),
                           capture_output=True, timeout=120)