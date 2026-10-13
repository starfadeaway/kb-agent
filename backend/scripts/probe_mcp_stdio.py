"""手动验证 MCP server 的 stdio 传输能真正起来。

不是单测（要 spawn 子进程、耗时），是一次性核验脚本。
用法：
    .venv\\Scripts\\python.exe backend\\scripts\\probe_mcp_stdio.py

为什么要有这个探针而不是只靠 in-memory 守卫：
守卫跑的是进程内传输，**跨进程的 stdio 是另一条链路** ——
stdout 被别的输出污染、子进程的 sys.path 不对、启动时序问题，
这三种故障在 in-memory 下全都看不见。
而 stdio 恰恰是 MCP host（Claude Code / Cursor）实际用的那种。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "backend" / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mcp import Client, StdioServerParameters  # noqa: E402


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-X", "utf8", "backend/scripts/mcp_stdio_entry.py", "sr"],
        cwd=str(ROOT),
    )
    # mcp 2.3.0 的 Client 可以直接吃 StdioServerParameters，
    # 自己负责拉子进程与流管道（比手工传 read/write 稳）。
    async with Client(params) as c:
        res = await c.list_tools()
        print("工具列表（stdio 子进程）:")
        for t in res.tools:
            print(f"  {t.name}: {t.description}")
            print(f"    schema={json.dumps(t.input_schema, ensure_ascii=False)}")

        out = await c.call_tool("calculator", {"expression": "(0.8+0.2)*1000"})
        print("\n调用 calculator(0.8+0.2)*1000 ->", out.content[0].text)

        out2 = await c.call_tool("list_games", {})
        print("调用 list_games ->", out2.content[0].text.splitlines()[0])

    print("\nOK: stdio 传输下的 MCP server 可用")


if __name__ == "__main__":
    asyncio.run(main())