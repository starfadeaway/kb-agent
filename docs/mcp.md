# MCP 接入说明

本文档说明 kb-agent 的 MCP（Model Context Protocol）支持：它解决什么问题、
怎么启动、两条路径如何保证一致、以及踩过的坑。

## 一句话定位

**MCP 是给外部 MCP host 用的出口，不是给本项目 Agent 用的入口。**

`kb-agent` 自己的 Agent 调自己的工具时走进程内直调（`builtin`），
不绕 MCP —— 绕一圈只增加延迟和故障面。
MCP 的价值是让 **Claude Code / Cursor 等外部 host** 能直接用这个知识库。

## 启动

```powershell
# 先装依赖（注意顺序，见下方「坑」）
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

# 启动 stdio server（可指定库，默认 sr）
.\.venv\Scripts\python.exe -m app.agent.mcp_server sr
```

以 stdio 传输启动，暴露 4 个工具：

| 工具 | 参数 | 说明 |
|---|---|---|
| `search_kb` | `query`(必填), `game`, `top_k` | 三库混合检索 |
| `get_page` | `title`(必填), `game` | 取页面全文 |
| `calculator` | `expression`(必填) | 算术 |
| `list_games` | — | 列出覆盖的游戏 |

在 MCP host 的配置里这样写：

```json
{
  "mcpServers": {
    "kb-agent": {
      "command": "python",
      "args": ["-m", "app.agent.mcp_server", "sr"],
      "cwd": "/path/to/kb-agent/backend"
    }
  }
}
```

## 两条路径的一致性是怎么保证的

这是本项目认为**唯一值得强调**的部分。

「MCP 只是薄适配」如果只靠嘴说，等于没有证据。所以有一条守卫
（`backend/tests/test_mcp_parity.py`，16 项）**实跑两条路径并比对输出**：

```python
# 内置路径
builtin_text, builtin_ok = await call_tool(name, args, ctx, timeout)

# MCP 路径：真连一次 server，不用 mock
server = build_server(ctx, only=[name])
async with Client(server) as client:
    result = await client.call_tool(name, args)
    mcp_text = result.content[0].text
```

断言 `mcp_text == builtin_text` **逐字符相同**，以及
错误前缀判定一致（错误语义不一致会让 Agent 侧的熔断与失败率指标失真）。

**为什么不用 mock**：MCP 的价值恰恰在于「协议真的能通」。
用 mock 测协议适配，测的是 mock 自己。

守卫还锁住了：schema 参数名与内置一致、每个工具注册独立
（专治闭包捕获循环变量）、MCP 不是默认传输方式、starlette 版本兼容。

## 开关

`TOOL_TRANSPORT`：

- `builtin`（默认）—— 进程内直调，本项目 Agent 走这条
- `mcp` —— 走 MCP 协议

默认 `builtin` 是刻意的，守卫 `test_mcp_is_not_default_transport` 锁住它。

## 踩过的坑（都留了守卫）

### 1. mcp 会把 starlette 顶到 FastAPI 不兼容的版本

```
mcp 2.3.0 声明 starlette>=0.48
fastapi 0.115.6 要求 starlette<0.42
```

直接 `pip install mcp` 会把 starlette 顶到 1.8.0，结果：

```
TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'
```

**整个后端起不来，而报错完全指不到 mcp。**
当时正在跑的实验 E 因此被污染，及时停掉了。

修法：两者可以共存（`pip check` 干净），但必须**先装 starlette 再装 mcp**：

```powershell
.\.venv\Scripts\python.exe -m pip install starlette==0.41.3
.\.venv\Scripts\python.exe -m pip install mcp==2.3.0
```

守卫 `test_starlette_is_pinned_below_fastapi_limit` 锁住这条。

### 2. 循环里的闭包会捕获循环变量

```python
for name in names:
    async def _f(value: str) -> str:
        return await call_tool(name, ...)   # ← 这里的 name 是循环结束后的值
```

四个工具的函数体全部指向 `list_games`。表现是 host 调 `search_kb`
拿到游戏列表，**且不抛任何异常**。

ruff 的 `B023` 抓到了它。修法：工厂函数把变量变成真参数。

### 3. 形参名不能图省事统一叫 `value`

MCP 拿 `inspect.signature` 生成 pydantic 模型校验入参。
如果 schema 里是 `expression` 而函数形参叫 `value`，host 传
`{"expression": "1+1"}` 会得到：

```
1 validation error for calculatorArguments
value  Field required
```

**每个工具调用都失败，而错误信息里完全没有「参数名不对」这层提示。**

修法：用 `exec` 动态生成具名形参（形参名取自本模块常量，不是外部输入）。

### 4. 返回值必须解包成 str

`call_tool` 返回 `(结果文本, 是否成功)` 二元组，直接返回给 MCP 会得到：

```
ValidationError: result Input should be a valid string
input_value=('1+1 = 2', True)
```

MCP 用返回注解生成输出模型并校验，所以必须 `text, _ok = await call_tool(...)`。

## 双向验证为什么不适用

本模块是**新增**的，MCP 适配层在引入之前不存在 ——
没有「改造前的代码版本」，也就没有可注入的旧实现，
套 worktree 式双向验证是自欺欺人。

替代动作（都已做，且都真的抓到过东西）：

1. 实跑一致性（真 `mcp.Client` 连真 server）→ 抓到坑 3、4
2. ruff `B023` → 抓到坑 2
3. starlette 兼容性实测 → 抓到坑 1（真的炸过一次）

守卫 `test_this_module_is_new_so_bidirectional_verification_does_not_apply`
显式记录这个事实，而不是假装能做双向验证。

## 参数声明为什么写两遍

`MCP_PARAMS` 与 `tools.py` 的 `@tool` 声明各写一份，原因是：
MCP 从**类型注解**生成 schema，而本项目的工具签名是
`(args: dict, ctx: ToolContext) -> str` —— 从这里转不出逐参数的类型。

手抄必然有漂移风险，所以 `test_mcp_params_match_tool_schemas` 逐字段比对。
**这条守卫在本模块写完后立刻抓到了两处漏项**：`search_kb` 少了 `top_k`、
`get_page` 少了 `game` —— 漏一个就是该工具必挂。

## 版本信息

| 项 | 值 |
|---|---|
| MCP Python SDK | `2.3.0`（2026-10-02，`Production/Stable`） |
| Server API | `from mcp.server import MCPServer` + `@mcp.tool()` |
| Client API | `from mcp import Client`（in-process / stdio / streamable-http） |
| transport | stdio（本项目只实现这个） |

> 早先记的「MCP v1.x 稳定、v2 还在 pre-alpha」**已过时**。
> 这类会变的外部依赖信息必须现查 PyPI，不能凭记忆。