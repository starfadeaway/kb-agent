"""跑消融实验时的公共工具：解释器探测 + 子进程执行。

`exp_b` / `exp_d` 这类脚本的形态是一样的：**用项目 venv 起子进程**去重建索引、
跑评测，自己做编排。这段编排逻辑此前在两个文件里逐字相同（连警告文案都一样），
于是「解释器探测写错」这种缺陷要修两遍。

只放与实验语义无关的工具层。**迭代语义不合**：exp_b 按 `STEPS` 迭代、
exp_d 按 embedding 模型迭代，那部分各自演化。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent


def find_python() -> Path:
    """解释器路径：先按 Windows 布局找，再试 POSIX 布局，都没有才退回当前解释器。

    退回时**必须提示** —— 否则可能在一个没装项目依赖的解释器里跑完整套实验，
    最后炸在一个看不出根因的 ImportError 上。
    """
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if py.is_file():
        return py
    posix = ROOT / ".venv" / "bin" / "python"
    if posix.is_file():
        return posix
    fallback = Path(sys.executable)
    print(f"[警告] 没找到项目 venv（{ROOT / '.venv'}），改用当前解释器：{fallback}")
    return fallback


#: 项目 venv 的解释器。模块级求值 —— 与它原来的位置（各自脚本的模块级）一致，
#: 所以警告仍然在脚本启动时打印，而不是等到第一次 sh()。
PY = find_python()


def sh(cmd: list[str], label: str) -> int:
    """在 `backend/` 下跑一条子进程命令，返回退出码。

    cwd 固定为 `backend/`：脚本里传进来的都是 `scripts/xxx.py` 这种相对路径。
    """
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)
    p = subprocess.run(cmd, cwd=str(BACKEND),
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return p.returncode
