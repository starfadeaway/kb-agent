"""探针 / 压测脚本的**落盘出口**。

为什么单独一个模块：
`exp_common.py` 也有 `save()`，但它给 A–E 评测脚本用，导入它要先走一遍
sys.path 注入、还得拖上 `app.*` 依赖。探针脚本（bench_* / probe_* / check_*）
恰恰是**最需要落盘**的一批 —— 它们的数字会被写进简历和面试稿 ——
却偏偏最轻量，有的连 `app` 都不 import。所以这里另起一个零依赖的出口，
谁都能一行 `from probe_io import save_probe` 接上。

它解决的真问题：
文档里所有延迟 / 并发数字（778ms、3.95x、TTFT 721ms、7 步→4 步…）
原先只 `print` 到 stdout，**没有任何文件留存**。后果有三条：
1. 面试被问"这张表哪来的"只能复述，出示不了原始记录；
2. 数字无法回归验证 —— 上一轮就因为没有底稿，
   把"实验 B p=0.0074 复现不出"误判成数据有问题，实际是对照组比错了；
3. 改完代码没有基线可比，不知道结论是否还成立。

所以落盘时**必须带出处**：谁跑的、什么时候跑的、什么配置下跑的、
对应哪个 git commit。否则半年后看到一份 JSON，仍然不知道它对应文档里哪句话。

用法：
    from probe_io import save_probe
    save_probe("bench_concurrency_retrieve", payload,
               context={"levels": [1, 2, 4, 8], "concurrency_limit": 4})
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: 与 A–E 实验结果同一个目录（`data/eval/`），避免出现第二个"结果目录"。
PROBE_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"


def _git_commit() -> str | None:
    """当前 commit。拿不到就返回 None —— 绝不因为它把整个保存搞失败。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def build_meta(script_file: str, context: dict[str, Any] | None = None) -> dict:
    return {
        "script": script_file,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "context": context or {},
    }


def save_probe(name: str, payload: Any, *, context: dict[str, Any] | None = None,
               quiet: bool = False) -> Path:
    """把探针结果写成 `data/eval/<name>.json`，并注入 `_meta` 出处块。

    `context` 放**影响这次测量的配置**：并发档位、模型名、开关状态等。
    这一栏比结果本身更重要 —— 它是以后判断"数字还对不对"的唯一依据。
    """
    if isinstance(payload, dict):
        body = dict(payload)
        body["_meta"] = build_meta(Path(sys.argv[0]).name or "<stdin>", context)
    else:
        body = {"_meta": build_meta(Path(sys.argv[0]).name or "<stdin>", context),
                "result": payload}

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    out = PROBE_DIR / f"{name}.json"
    out.write_text(json.dumps(body, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    if not quiet:
        print(f"原始结果已写入 {out}")
    return out