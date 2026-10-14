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

## 与 Agent 主链路的关系：**没有运行时开关**

这里曾经有一个 `TOOL_TRANSPORT` 配置项（`builtin` / `mcp`），并在 config 注释
与 `docs/mcp.md` 里描述成「设成 mcp 就走 MCP 协议」。核实结果是：
**`app/` 与 `scripts/` 里没有任何一处代码分支读它** —— 改 `.env` 完全不生效，
而且不报任何错。一个宣称能用、实际不能的开关比没有开关更糟：
它经不起一次追问，而当时的守卫只断言了那个**配置值**，没断言任何行为。

所以字段与相关描述都已删除。现在的事实是：

- 本项目的 Agent **始终**走 `tools.py` 的进程内直调（`call_tool`），没有分支。
- MCP 是一条**独立的 server 入口**（`backend/scripts/mcp_stdio_entry.py`），
  供**外部 MCP host**（Claude Code / Cursor 等）连入。
- 两者**不在同一条调用链上**，所以也不需要开关。

`test_mcp_parity.py` 的比对仍然有效，但含义要说准：那是**在测试里分别直调
两条路径**（真 `mcp.Client` 连真 server vs 直接调函数）断言逐字符一致，
用来证明「适配层没有偷偷改逻辑」—— **不是**证明「运行时可切换」。

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
# 消费方：`_make_*` 把它透传给 `call_tool` 的 asyncio.wait_for，是 MCP 侧
# **唯一**的超时控制点。
#
# 与内置路径的差别值得记住：内置路径（`ReActAgent._exec_call`）对所有工具统一用
# Settings.tool_timeout（默认 15s）。默认配置下检索类两边都是 15s、一致；
# 但 calculator / list_games 在 MCP 侧被收窄到 5s，属于**有意的差异**：
# 它们不做 IO，多等没有意义，而 5s 足够覆盖一个超大表达式。
# 也就是说 `test_mcp_parity` 断言的是"返回值逐字符一致"，**不含超时行为一致** ——
# 引用这个守卫时不要过度声称。

#: 每个工具的参数声明，与 `app/agent/tools.py` 的 @tool 声明**刻意保持一致**。
#:
#: 为什么要重复写一遍而不是从 SCHEMAS 里转：MCP 从**类型注解**生成 schema，
#: 而本项目的工具签名是 `(args: dict, ctx: ToolContext) -> str`，
#: 从这里转不出逐参数的类型。写两遍的风险就是两边漂移。
#:
#: **必须逐字段一致** —— 参数名、required、类型全部要对齐，
#: 否则 MCP host 会用错误的参数名调用，表现为
#: `validation error: Field required`，且错误信息里完全没有"参数名不对"这层提示。
#: 这不是假设：2026-10-13 实测就抓到过 `get_page` 少 `game`、`search_kb` 少 `top_k`
#: 两处漏项，两边一漂移就是**每个工具调用都报错**。
#:
#: 手抄必然有漂移风险，所以 `test_mcp_parity.py::test_mcp_params_match_tool_schemas`
#: 逐字段比对 —— 上面那两处就是那条守卫在本模块写完后立刻抓到的。
#: 也就是说：这份常量**是被守卫看着的**，改它之前先想清楚会不会动到实验 C 的被测变量。
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
        # `ok` 在这里被丢弃是**安全**的，但理由要说准：
        # 不是"MCP 不关心成败"，而是失败已经被 call_tool 写进 text 本身了
        # （`[超时] …` / `[检索失败] …`），host 拿到的就是这段可读文案。
        # 真正丢掉的信息是**结构化的成败标记** —— MCP 这条路上没有地方能上报它，
        # 所以两个路径的失败率指标本来就不可比，不要拿它们对齐。
        text, _ok = await call_tool(name, {}, ctx, timeout)
        return text

    # __name__ / __doc__ 不是装饰：MCP 把它当工具名和描述发给 host。
    # 不设的话 host 侧会看到 `_f`，四个工具重名，列表根本没法区分。
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

    # exec 的两个前提（都在下面注释里）：生成的源码形参名必须等于 key，
    # 否则 pydantic 模型与 host 的 payload 对不上；key 来自本模块常量而非外部输入。
    # 这段 src 是**逐字符**可控的 —— 一旦 key 变成外部输入，这里就是注入面。
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
    # 生成体里 `if {o}:` 这一层是**语义**而不是风格问题：
    # 空串表示"host 没传"，而工具内部用 `args.get("game") or ctx.game` 兜底。
    # 如果把空串原样塞进 payload，工具会收到 game=""，
    # `or` 虽然仍能兜住，但像 search_kb 那种"先 .strip() 再判空"的地方
    # 行为就和"键不存在"不完全一致了 —— 两个路径不该有这种细微差别。
    # 代价：host 显式传空串与不传在这里被当成同一件事，这正是想要的。
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
    # 下面这个循环的依据只有一条：**MCP 是拿函数签名生成入参模型的**。
    # 所以适配函数的形状必须与 MCP_PARAMS 里声明的形状严格对应，
    # 分派却按参数**个数**（0 / 1 / 多个）而不是按工具名 ——
    # 按名字分派的话，新增一个三参数工具就会静默走错分支，
    # 而"三参数"这件事 MCP_PARAMS 已经如实描述了，不需要额外的名字白名单。
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
                # 注册期就炸，而不是等到 host 调用时才发现。
                # 这里抛出的是**编程错误**（适配层覆盖不全），不是用户输入错误，
                # 所以用 raise 而不是返回一段文案 —— 它永远不该被模型看到。
                raise ValueError(
                    f"{name} 有 {len(props)} 个参数但 required 是 {required}；"
                    f"带多个必填参数的工具需要专门的适配函数"
                )
            optional = tuple(k for k in props if k != required[0])
            fn = _make_optional_arg(name, required[0], optional, ctx, timeout)
        # fn 的签名就是 MCP 对外承诺的入参模型；
        # ctx 被闭包捕获，所以下面所有工具共享**同一个** ToolContext 实例。
        server.tool()(fn)

    # 出口是一个已注册完毕的 MCPServer，消费方只有 serve_stdio。
    # 它在这里就被构造好（而不是在 serve_stdio 里），是为了让
    # `test_mcp_parity` 能不启动 stdio 就直接拿到 server 对象做对照。
    return server


async def serve_stdio(ctx: ToolContext, *, only: list[str] | None = None) -> None:
    """以 stdio 传输启动 MCP server（阻塞直到 host 断开）。"""
    # 前置：ctx 必须**已经**是带好 retriever 的实例。
    # 和 /chat 每请求 `ToolContext.build()` 的差别在于这里 ctx 只在启动时构造一次、
    # 被所有工具闭包共享，所以 settings 是启动那一刻的快照 ——
    # 运行期改 .env 不会影响已经起来的 server（要改必须重启 host 连接）。
    # 好处是 BM25 索引、embedding 缓存只用得上一次，不必每个工具调用都重新预热。
    server = build_server(ctx, only=only)
    await server.run_stdio_async()


def main() -> None:
    """命令行入口：`python -m app.agent.mcp_server [game]`

    这里原先调的是 `_TC.create(game)`，而 `ToolContext` 只有 `@classmethod build()`
    —— 只要有人真的敲这条命令就立刻 AttributeError，而 docstring 恰恰宣称它可用。
    真实入口是 `scripts/mcp_stdio_entry.py`（走 `serve_stdio`），平时测不出来，
    属于「没人走 + 一走就崩 + 文档说它能走」。

    `game or "sr"`：命令行不给参数时 `sys.argv[1]` 不存在，不能把 None 透给 build()。
    """
    from app.agent.tools import ToolContext as _TC

    game = sys.argv[1] if len(sys.argv) > 1 else "sr"
    ctx = _TC.build(game)
    print(f"kb-agent MCP server 启动（stdio），工具：{list(MCP_TOOLS)}", file=sys.stderr)
    asyncio.run(serve_stdio(ctx))


if __name__ == "__main__":
    main()