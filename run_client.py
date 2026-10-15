"""启动前端 Vite 开发服务器。

用法（PyCharm 里直接右键 Run）：
    node\\run_client.py

等价于 `cd frontend && npm run dev`，但不用切目录、不受 PowerShell 执行策略影响。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


def find_npm() -> str:
    """找到可用的 npm。

    Windows 上必须用 npm.cmd：npm 会被解析成 npm.ps1，
    而 PowerShell 默认禁止执行 .ps1 脚本，会直接报 SecurityError。
    """
    exe = "npm.cmd" if os.name == "nt" else "npm"
    path = shutil.which(exe)
    if path:
        return path
    found = shutil.which("npm")
    if found:
        return found
    raise SystemExit("找不到 npm，请确认已安装 Node.js 并加入 PATH")


def main() -> None:
    ap = argparse.ArgumentParser(description="启动 kb-agent 前端")
    ap.add_argument("--install", action="store_true", help="先执行 npm install")
    args = ap.parse_args()

    npm = find_npm()
    if args.install or not (FRONTEND / "node_modules").is_dir():
        print("[client] 安装依赖…")
        subprocess.check_call([npm, "install", "--no-audit", "--no-fund"], cwd=FRONTEND)

    print("[client] http://localhost:5173  (Vite 会把 /api 代理到 127.0.0.1:8000)")
    try:
        subprocess.check_call([npm, "run", "dev"], cwd=FRONTEND)
    except KeyboardInterrupt:
        print("\n[client] 已停止")


if __name__ == "__main__":
    main()
