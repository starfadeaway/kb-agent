"""启动后端 API 服务。

用法（PyCharm 里直接右键 Run 就行，不需要配 Module name / Working directory）：
    .venv\\Scripts\\python.exe run_server.py
    .venv\\Scripts\\python.exe run_server.py --reload --port 8080

为什么不用 `uvicorn app.main:app`：
那条命令依赖「当前工作目录是 backend/」，PyCharm 默认工作目录是项目根，
一改就报 ModuleNotFoundError: app。这个脚本自己把 backend 加进 sys.path，
所以**在任何目录、任何 Working directory 下都能跑**。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

# 必须在 import app.* 之前，否则 PyCharm 的默认工作目录会让导入失败
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def main() -> None:
    ap = argparse.ArgumentParser(description="启动 kb-agent 后端")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="改代码自动重启，开发用")
    args = ap.parse_args()

    import uvicorn

    print(f"[server] http://{args.host}:{args.port}/docs   (接口文档)")
    print(f"[server] health: http://{args.host}:{args.port}/api/health")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # reload 需要从 backend 目录启动才能发现 app 包
        app_dir=str(BACKEND) if args.reload else None,
        log_level="info",
    )


if __name__ == "__main__":
    main()
