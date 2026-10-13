"""MCP（Model Context Protocol）工具层 —— 把现有工具暴露成标准 MCP server。

## 为什么是「薄适配」而不是重写

项目里的 4 个工具（`search_kb` / `get_page` / `calculator` / `list_games`）
本来就已经是一个干净的注册表：`app/agent/tools.py` 里有 `REGISTRY` 与
`@tool` 装饰器，`call_tool(name, args, ctx, timeout)` 统一封装了超时与错误语义。
MCP 只是把这套东西再包一层协议壳，所以**这里一行工具逻辑都没有**。

**这一点是可验证的，不是嘴上说说** —— `backend/tests/test_mcp_parity.py`
里有一条守卫，对同一批输入分别走「内置函数路径」和「MCP 路径」，
断言返回值逐字符相同。薄适配的证明方式是实测一致，
不是看代码只有几行。

## 默认不启用

`TOOL_TRANSPORT=mcp` 才走 MCP；默认 `builtin` 走原来的进程内调用。
理由：MCP 是给**外部 MCP host**（Claude Code / Cursor 等）用的，
本项目的 Agent 自己不需要绕一圈网络/子进程回到自己的函数。
把它做成默认只会给主链路凭空加一层延迟和故障面。

## 刻意不做的事

- **不改 `tools.py`**。工具定义是实验 C/E 的被测对象，
  边跑实验边改它会让数字失去意义（单变量前提）。
- **不做 MCP client**。本项目是工具的**提供方**，不是消费方。
- **不自动注册**。4 个工具逐个显式注册，避免以后新增工具时
  MCP 侧漏注册、Agent 侧却以为可用的静默不一致。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# ---- sys.path 引导（与 scripts/ 下其他入口保持一致）----
BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.agent.tools import REGISTRY, ToolContext, call_tool  # noqa: E402

#: MCP 侧暴露的工具名 → 超时秒数。
#:
#: 显式列出来而不是遍历 REGISTRY：`call_tool` 的超时是**逐工具**的，
#: 而 MCP 适配层拿不到「这个工具原本配了多少超时」，
#: 只能自己声明。少一个就会出现「某个工具在 MCP 下超时行为与内置路径不一致」，
#: 而这种不一致在结果里表现得和「工具坏了」一模一样。
MCP_TOOLS: dict[str, float] = {
    "search_kb": 15.0,
    "get_page": 15.0,
    "calculator": 5.0,
    "list_games": 5.0,
}

#: 每个工具的参数声明，与 `app/agent/tools.py` 的 @tool 声明**刻意保持一致**。
#:
#: 为什么要重复写一遍而不是从 SCHEMAS 里转：MCP 从**类型注解**生成 schema，
#: 而本项目的工具签名是 `(args: dict, ctx: ToolContext) -> str`，
#: 从这里转不出逐参数的类型。写两遍的风险是两边漂移，
#: 所以 `test_mcp_parity.py` 里有一条守卫断言两边 schema 的字段名与 required 一致。
#: 每个工具的参数声明。
#:
#: **必须与 `app/agent/tools.py` 的 `@tool` 声明逐字段一致** ——
#: 参数名、required、类型全部要对齐，否则 MCP host 会用错误的参数名调用。
#: 这里的值是从内置 schema 抄来的（2026-10-13 实测 `get_page` 少了 `game`、
#: `search_kb` 少了 `top_k`，两边一漂移就是每个工具调用都报
#: `validation error: Field required`）。
#:
#: 手抄必然有漂移风险，所以 `test_mcp_parity.py::test_mcp_params_match_tool_schemas`
#: 逐字段比对 —— 那条守卫在本模块写完后立刻就抓到了这两处漏项。
MCP_PARAMS: dict[str, tuple[dict[str, str], list[str]]] = {
    "search_kb": (
        {
            "query": "检索问题，用完整句子描述需要的信息",
            "game": "可选，星铁填 sr，绝区零填 zzz。默认用当前会话的游戏",
            "top_k": "可选，返回条数，默认 5，最大 10",
        },
        ["query"],
    ),
    "get_page": (
        {"title": "wiki 页面标题，如 三月七 / 光锥", "game": "可选游戏 slug"},
        ["title"],
    ),
    "calculator": ({"expression": "算式，例如 (0.8 + 0.2) * 1000"}, ["expression"]),
    "list_games": ({}, []),
}

SERVER_NAME = "kb-agent"


#: MCP host 看到的工具描述。刻意比 Agent 内部 prompt 更短 ——
#: host 侧的工具描述要和它自己的 system prompt 抢注意力预算。
_DESCRIPTIONS: dict[str, str] = {
    "search_kb": "在游戏 wiki 知识库中检索（星穹铁道 / 原神 / 绝区零）",
    "get_page": "按标题取回知识库中的页面全文",
    "calculator": "计算算术表达式",
    "list_games": "列出知识库覆盖的游戏",
}


def _make_no_arg(name: str, ctx: ToolContext, timeout: float):
    """构造一个零参数工具的 MCP 适配函数。

    **为什么用工厂函数而不是在循环里 `async def`**：
    循环内的闭包直接引用循环变量（`name` / `timeout`）时，函数体看到的
    是**循环结束后的最终值** —— 四个工具的函数体全部指向 `list_games`。
    表现为 MCP host 调 `search_kb` 却拿到游戏列表，**且不抛任何异常**。
    （ruff 的 B023 就是在报这个。）

    工厂函数把 `name` / `timeout` 变成**真参数**（真绑定），根上不存在这问题。

    **返回值必须解包成 str**：MCP 用返回注解生成 pydantic 输出模型并校验，
    而 `call_tool` 返回的是 `(结果文本, 是否成功)` 二元组，直接返回会得到
    `ValidationError: result Input should be a valid string`。
    """
    async def _f() -> str:
        text, _ok = await call_tool(name, {}, ctx, timeout)
        return text

    _f.__name__ = name
    _f.__doc__ = _DESCRIPTIONS.get(name, name)
    return _f


def _make_one_arg(name: str, key: str, ctx: ToolContext, timeout: float):
    """构造一个单参数工具的 MCP 适配函数。

    **形参名必须等于真实业务参数名**（`expression` / `title`），不能图省事
    统一叫 `value`。MCP 是拿 `inspect.signature` 生成的 pydantic 模型
    来校验入参的，host 按 schema 传 `{"expression": "1+1"}`，
    而模型里只有 `value` —— 结果是
    `1 validation error ... value Field required`，
    **每个工具调用都失败**，而且错误信息里完全没有「参数名不对」这层提示。

    这里用 `exec` 动态生成具名形参：`async def _f(expression: str) -> str: ...`。
    `key` 取自本模块的 `MCP_PARAMS` 常量，不是外部输入；
    `test_mcp_params_match_tool_schemas` 守卫它与内置 schema 的参数名一致。
    """
    async def _invoke(value: str) -> str:
        text, _ok = await call_tool(name, {key: value}, ctx, timeout)
        return text

    src = f"async def _f({key}: str) -> str:\n    return await _invoke({key})"
    ns = {"_invoke": _invoke}
    exec(src, ns)  # noqa: S102 - key 来自本模块常量，不是外部输入
    _f = ns["_f"]
    _f.__name__ = name
    _f.__doc__ = _DESCRIPTIONS.get(name, name)
    return _f


def _make_optional_arg(name: str, required: str, optional: tuple[str, ...],
                       ctx: ToolContext, timeout: float):
    """构造「一个必填 + 若干可选」工具的 MCP 适配函数。

    当前用于 `get_page(title, game=)` 与 `search_kb(query, game=, top_k=)`。

    **可选参数一律声明为带默认值的 `str`**，且默认值是空串而不是 `None`：
    MCP 会把 schema 里的 optional 字段原样传给模型，
    host 不传时 Python 侧的默认值就是 ``，于是 payload 里不出现这个键，
    工具内部 `args.get("game") or ctx.game` 的回退逻辑照常生效。

    `top_k` 的 int 转换也**刻意放在工具内部**（`search_kb` 里自己
    `int(args.get("top_k") or s.retrieve_top_k)`）：在适配层转的话，
    host 传 "abc" 会让适配层抛异常，而内置路径是工具自己兜底 ——
    **错误语义会与内置路径不一致**，而错误语义不一致在 Agent 侧
    表现为「工具失败率」这个指标失真。
    """
    params = [f"{required}: str"]
    params += [f"{o}: str = ''" for o in optional]
    body = (
        f"async def _f({', '.join(params)}) -> str:\n"
        f"    _p = {{'{required}': {required}}}\n"
        + "".join(f"    if {o}:\n        _p['{o}'] = {o}\n" for o in optional)
        + "    return await _invoke(_p)\n"
    )

    async def _invoke(payload: dict) -> str:
        text, _ok = await call_tool(name, payload, ctx, timeout)
        return text

    ns = {"_invoke": _invoke}
    exec(body, ns)  # noqa: S102 - 参数名来自本模块常量，不是外部输入
    _f = ns["_f"]
    _f.__name__ = name
    _f.__doc__ = _DESCRIPTIONS.get(name, name)
    return _f


def build_server(ctx: ToolContext, *, only: list[str] | None = None):
    """构造一个绑定了 ctx 的 MCPServer。

    `only` 用于与 `Settings.agent_tool_names` 对齐 ——
    MCP host 能看到的工具集合必须与 Agent 启用的是同一个，
    否则「生产默认只给 search_kb」这条配置在 MCP 侧就失效了。
    """
    from mcp.server import MCPServer

    names = list(only) if only else list(MCP_TOOLS)
    missing = [n for n in names if n not in REGISTRY]
    if missing:
        raise ValueError(f"请求的工具未注册：{missing}；已注册：{list(REGISTRY)}")

    server = MCPServer(SERVER_NAME)
    for name in names:
        props, _required = MCP_PARAMS[name]
        timeout = MCP_TOOLS[name]
        if not props:
            fn = _make_no_arg(name, ctx, timeout)
        elif len(props) == 1:
            fn = _make_one_arg(name, next(iter(props)), ctx, timeout)
        else:
            _req, required = MCP_PARAMS[name]
            if len(required) != 1:
                raise ValueError(
                    f"{name} 有 {len(props)} 个参数但 required 是 {required}；"
                    f"带多个必填参数的工具需要专门的适配函数"
                )
            optional = tuple(k for k in props if k != required[0])
            fn = _make_optional_arg(name, required[0], optional, ctx, timeout)
        server.tool()(fn)

    return server


async def serve_stdio(ctx: ToolContext, *, only: list[str] | None = None) -> None:
    """以 stdio 传输启动 MCP server（阻塞直到 host 断开）。"""
    server = build_server(ctx, only=only)
    await server.run_stdio_async()


def main() -> None:
    """命令行入口：`python -m app.agent.mcp_server [game]`"""
    from app.agent.tools import ToolContext as _TC

    game = sys.argv[1] if len(sys.argv) > 1 else None
    ctx = _TC.create(game)
    print(f"kb-agent MCP server 启动（stdio），工具：{list(MCP_TOOLS)}", file=sys.stderr)
    asyncio.run(serve_stdio(ctx))


if __name__ == "__main__":
    main()