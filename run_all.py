"""一条命令同时起后端 + 前端（等价于 npm 的 concurrently）。

用法：
    .venv\\Scripts\\python.exe run_all.py
    .venv\\Scripts\\python.exe run_all.py --only server     # 只起后端
    .venv\\Scripts\\python.exe run_all.py --only client     # 只起前端

Ctrl+C 一次会把两个进程都收掉，不会留下占端口的孤儿进程。
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.is_file():  # 非 Windows
    PY = Path(sys.executable)


def npm_cmd() -> str:
    import shutil

    exe = "npm.cmd" if os.name == "nt" else "npm"
    return shutil.which(exe) or "npm"


def spawn(label: str, args: list[str], cwd: Path) -> subprocess.Popen:
    print(f"[{label}] 启动: {' '.join(args[:3])} …")
    return subprocess.Popen(
        args, cwd=str(cwd), env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="同时启动前后端")
    ap.add_argument("--only", choices=["server", "client"], help="只启动其中一个")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    procs: list[tuple[str, subprocess.Popen]] = []
    if args.only != "client":
        procs.append(("server", spawn("server", [str(PY), "run_server.py", "--port", str(args.port)], ROOT)))
    if args.only != "server":
        procs.append(("client", spawn("client", [npm_cmd(), "run", "dev"], ROOT / "frontend")))

    print("\n" + "=" * 56)
    print("  后端 API : http://127.0.0.1:%d/api/health" % args.port)
    print("  接口文档 : http://127.0.0.1:%d/docs" % args.port)
    print("  前端界面 : http://localhost:5173")
    print("  Ctrl+C 同时停止两者")
    print("=" * 56 + "\n")

    try:
        while True:
            time.sleep(1)
            for label, p in procs:
                if p.poll() is not None:
                    print(f"[{label}] 已退出(code={p.returncode})，正在停止其余进程…")
                    raise SystemExit(p.returncode or 0)
    except KeyboardInterrupt:
        pass
    finally:
        for label, p in procs:
            if p.poll() is None:
                print(f"[{label}] 停止中…")
                # Windows 下 Popen 树要用 taskkill 才能连子进程一起收
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                                   capture_output=True, check=False)
                else:
                    p.terminate()
        print("全部停止。")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"启动失败：{type(exc).__name__}: {exc}")
        sys.exit(1)
