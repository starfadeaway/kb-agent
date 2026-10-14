"""守卫：`_meta.script` 必须记调用方的脚本名，而不是 `exp_common.py`。

## 背景

`exp_common.build_meta` 原先写的是 `"script": Path(__file__).name`，
而 `__file__` 恒等于本文件自己，于是**所有实验产物的出处字段都指向
`exp_common.py`**。靠 script 判来源只会得到「所有实验都是 exp_common」，
出处溯源功能等于失效。

这个缺陷藏得深：`experiment` / `variant` / `game` / `labels` 仍然能区分产物，
所以数字一直是对的，只有出处字段错 —— 没有任何校验会报错。

## 双向验证

「修复前快照」用 `git show HEAD:backend/scripts/exp_common.py` 取出，
importlib 加载成独立模块后跑同一套断言 —— 必须失败且拿到
`script == "exp_common.py"`。正向用当前工作区，必须拿到测试文件自己的名字。
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import exp_common  # noqa: E402

# **修复前的固定 commit**，不是 HEAD。
# 用 HEAD 取快照是错的：修复一旦提交，HEAD 就是修好的版本，
# 守卫会安静地失去意义（实测踩过 —— 提交修复后这条用例立刻红了）。
# 这个常量必须指向 build_meta 修复之前的那个提交，改代码时不要动它。
PRE_FIX_COMMIT = "978e835"


def _args() -> argparse.Namespace:
    return argparse.Namespace(experiment="T", game="sr", variant=None, top_k=None, labels=None)


def test_caller_script_name_returns_this_test_file():
    """直接测辅助函数：它必须返回**调用者**的文件名。

    用测试文件自己当调用者，因为它的名字既确定又不与被测代码重合 ——
    断言成 `"exp_common.py"` 那种「碰巧相等」的情况不可能发生。
    """
    assert exp_common._caller_script_name() == Path(__file__).name


def test_build_meta_script_is_the_caller_not_exp_common():
    """从本测试文件调 build_meta，script 必须落在本文件上。"""
    meta = exp_common.build_meta(_args())
    assert meta["script"] == Path(__file__).name, f"script 记成了 {meta['script']!r}"
    assert meta["script"] != "exp_common.py", "script 仍是公共底座的名字，溯源仍然失效"


def test_build_meta_script_varies_with_caller(tmp_path):
    """换一个调用者，script 必须跟着变。

    这条锁的是「不是碰巧对」：如果实现退化成硬编码某个常量，
    前两条仍可能通过，这条会挂。

    探针模块写在 tmp_path 而不是 `scripts/`：它是测试脚手架，
    放进仓库目录会被当成真脚本，污染 scripts/ 的可读性。
    """
    probe = tmp_path / "_probe_caller.py"
    probe.write_text(
        "import argparse\n"
        "import sys\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "from exp_common import build_meta\n"
        "EMITTED = build_meta(argparse.Namespace(experiment='T', game='sr',\n"
        "                                    variant=None, top_k=None, labels=None))\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("_fake_caller_probe", probe)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        assert mod.EMITTED["script"] == "_probe_caller.py", mod.EMITTED
    finally:
        sys.modules.pop("_fake_caller_probe", None)


@pytest.fixture
def _head_snapshot(tmp_path):
    """把 HEAD 版本的 exp_common 加载成独立模块，验证修复前确实会失败。"""
    src = subprocess.run(
        ["git", "show", f"{PRE_FIX_COMMIT}:backend/scripts/exp_common.py"],
        cwd=str(SCRIPTS.parents[1]), capture_output=True,
        # 必须显式 encoding：Windows 上 text=True 会按 GBK 解码，
        # 而 git 输出是 UTF-8，中文注释直接触发 UnicodeDecodeError。
        encoding="utf-8", errors="replace", timeout=30,
    )
    if src.returncode != 0:
        pytest.fail(
            f"取不到 {PRE_FIX_COMMIT} 的 exp_common.py（{((src.stderr or '')[:120])}）—— "
            "该 commit 被 rebase 掉了，PRE_FIX_COMMIT 需要更新到新的修复前位置"
        )
    f = tmp_path / "exp_common_head.py"
    f.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("exp_common_head_snapshot", f)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_head_version_fails_this_guard(_head_snapshot):
    """修复前的代码在本守卫上必须失败，且失败方式正是缺陷本身。

    修复前 build_meta 写的是 `Path(__file__).name` —— `__file__` 指向
    **定义它的那个模块文件**，而不是调用它的脚本。所以快照被写成
    `exp_common_head.py`，它记下的就是自己这个文件名。

    这恰好是最干净的证明：把源码换个文件名再加载，script 就跟着变了 ——
    证明它记的是「模块自己」而不是「产物来源」。
    生产里该文件一直叫 exp_common.py，所以这个缺陷表现为
    **所有产物的 script 恒等于同一个值**，不报错、不变化、只是错的。

    注意断言方向：这里断言 HEAD 侧（修复前）**抓得到缺陷**，
    不是断言它通过。哪天这个用例开始失败，说明 PRE_FIX_COMMIT 指错了地方。
    """
    old = _head_snapshot.build_meta(_args())
    assert old["script"] == "exp_common_head.py", (
        f"HEAD 版本本该记快照自己的名字，实际 {old['script']!r} —— "
        "守卫可能已失去意义，请重新确认"
    )
    assert old["script"] != Path(__file__).name, (
        "HEAD 版本居然记到了调用者，说明 git 历史已变或守卫失效"
    )
