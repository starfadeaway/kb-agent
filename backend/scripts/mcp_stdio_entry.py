"""MCP server 的 stdio 入口 —— 独立进程入口。

## 为什么必须单独一个文件

stdio 传输下 **stdout 是 JSON-RPC 的协议通道**，
任何 `print()` / 日志输出写进 stdout 都会让客户端解析失败。
而 `app/agent/mcp_server.py` 的 `main()` 里有一行启动提示写的是 stderr ——
直接用 `python -m app.agent.mcp_server` 也能跑，但把它拆成独立入口有两个好处：

1. **可测**：探针 `probe_mcp_stdio.py` 以子进程方式拉起它，
   验证的是真实的跨进程 stdio 链路，而不是进程内 in-memory 传输。
2. **不污染应用模块**：server 逻辑住在 `app/agent/mcp_server.py`（可被守卫导入），
   这个文件只做「把 sys.path 摆好然后启动」，职责单一。

启动：
    python backend/scripts/mcp_stdio_entry.py [sr|ys|zzz]
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.agent.tools import ToolContext  # noqa: E402


async def _run() -> None:
    # ⚠️ stdout 只能留给协议。这里任何 print 都会破坏 JSON-RPC 帧。
    game = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"[kb-agent-mcp] 启动 game={game}", file=sys.stderr, flush=True)

    from app.agent.mcp_server import MCP_TOOLS, serve_stdio
    from app.core.config import get_settings

    settings = get_settings()
    ctx = ToolContext(
        settings=settings,
        retriever=None,
        store=None,
        game=game or settings.game_primary,
    )
    print(f"[kb-agent-mcp] 工具：{list(MCP_TOOLS)}", file=sys.stderr, flush=True)
    await serve_stdio(ctx)


if __name__ == "__main__":
    asyncio.run(_run())