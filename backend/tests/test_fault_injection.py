"""故障注入测试 —— `docs/failure_analysis.md` 第六节自认的那个盲区。

## 为什么这个文件之前不存在

`failure_analysis.md` 第七节结论 3 当时写着下面这段
（**这是历史原文，该盲区已由本文件补上**，保留是为了说明这个文件为什么被写出来）：

> n=15 看不见的东西不等于不存在。工具选错与循环不终止属于低频故障，
> 15 条题碰不到。要覆盖它们，需要的是**故障注入测试**
> （plan.md 的 DoD 之一，**至今未实现**）。

当时已有的测试覆盖是**单元级**的：
`test_search_guard` 测 SearchGuard 这个类本身、`test_agent_loop` 测步数与预算上限、
`test_llm_fallback_chain` 测降级链的分类。中间那层 —— **故障穿过整条 ReAct 循环
之后会怎样** —— 没人测过。所以熔断路径「只有探针级验证」这句话是准确的，
而且准确了这么久。

## 这一批测的正是那一层

全部用 FakeLLM + 注入式 `call_tool`，**不打网络、不打索引、不花 token**。
故障从工具层注入，断言的是**循环的最终状态**：步数、stop_reason、
tool_failures、guard_stats、消息链完整性。

## 一条贯穿全文件的判断

每一组测试末尾都有一条断言在检查**消息链是否完整**：
`role=tool` 必须紧跟在带 `tool_calls` 的 assistant 消息之后。

原因是故障路径最容易坏在这里 —— 一旦某处 `continue`/`break` 打断了
tool_call 与 tool 消息的配对，循环下一轮直接被 API 判成非法消息链，
而**本地测试全绿**（因为 FakeLLM 不校验消息链）。这是本项目踩过两次的坑，
所以每一组都必须自己盯住。
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest
from app.agent import react as react_mod
from app.agent.react import ReActAgent
from app.core.config import get_settings
from fakes import FakeLLM

# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class ToolInjector:
    """替换 react 模块里的 `call_tool`，按脚本决定每次调用的结果。

    注入的是 **react 命名空间里的那个名字**，而不是 `app.agent.tools` 里的原函数 ——
    `_exec_call` 是 `from ... import call_tool` 引进来的，只改 tools 模块不起作用。
    （改错了的话这个文件会「全绿但一条故障都没注入」，所以下面每条断言
    都同时检查注入计数。）
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[tuple[str, dict, object]] = []
        self._n = 0
        self._orig = react_mod.call_tool

    def __enter__(self):
        react_mod.call_tool = self._fake  # type: ignore[assignment]
        return self

    def __exit__(self, *exc):
        react_mod.call_tool = self._orig  # type: ignore[assignment]
        return False

    async def _fake(self, name, args, ctx, timeout=None):
        self.calls.append((name, dict(args), timeout))
        idx = min(self._n, len(self.script) - 1)
        self._n += 1
        item = self.script[idx]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return await item(name, args)
        return item


def run(coro):
    return asyncio.run(coro)


def make_agent(script, settings=None, tools=("search_kb",)):
    llm = FakeLLM(script)
    agent = ReActAgent(
        settings=settings or get_settings(),
        llm=llm,
        tools=[{"type": "function", "function": {"name": t}} for t in tools],
    )
    agent._fake = llm
    return agent


#: 一段格式合法的 search_kb 成功输出（会被引用校验当成真实证据）
OK_HIT = (
    "status=ok hits=1 top_score=0.99\n"
    "[1] 分数 0.9900 | 太空封印站 > 正文\n"
    "太空封印站（正文）两件套效果：使装备者的攻击力提高12%。\n"
    "出处：https://wiki.biligame.com/sr/太空封印站 | 修订 2025-04-13"
)
EMPTY_HIT = "status=empty hits=0 top_score=0.0000\n（没有命中）"
LOW_HIT = (
    "status=ok hits=1 top_score=0.10\n"
    "[1] 分数 0.1000 | 某个不相关的页面 > 正文\n"
    "正文\n出处：https://wiki.biligame.com/sr/x | 修订 2025-01-01"
)
#: 基础设施失败。前缀取自 tools.py 的 `_INFRA_FAILURE_PREFIXES` 实测值
#: （('[超时]', '[异常]', '[检索失败]', '[抓取失败]')）——
#: 这里**照抄**而不是自己编一个：编错前缀的话，SearchGuard 认不出来、
#: 失败会落到兜底分支把 misses 清零，熔断自我解除，
#: 而测试会「因为循环最终还是停了」而看起来是绿的。
INFRA_HIT = "[检索失败] 向量服务连接失败，请稍后重试"


def assert_message_chain_intact(messages):
    """每个带 tool_calls 的 assistant 后面必须紧跟等量的 tool 消息。

    这是整个文件最重要的一条公共断言。FakeLLM 不校验消息链，
    所以一旦某条故障路径把配对打断，**没有任何东西会报错** ——
    真实 API 会直接 400，而本地测试照样绿。
    """
    i = 0
    while i < len(messages):
        m = messages[i]
        calls = m.get("tool_calls") or []
        if m.get("role") == "assistant" and calls:
            following = messages[i + 1 : i + 1 + len(calls)]
            assert len(following) == len(calls), (
                f"assistant 有 {len(calls)} 个 tool_call，"
                f"后面却只有 {len(following)} 条消息 —— 消息链断了"
            )
            for k in range(len(calls)):
                assert following[k].get("role") == "tool", (
                    f"tool_call #{k} 后面跟的是 {following[k].get('role')} 而不是 tool"
                )
            i += len(calls)
        i += 1


# --------------------------------------------------------------------------- #
# 1. 工具抛异常
# --------------------------------------------------------------------------- #
def test_工具抛异常不会让整轮崩掉():
    """工具层异常必须被转成可读文本回填给模型，循环继续。"""
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "知识库服务暂时不可用。"},
    ])
    with ToolInjector([RuntimeError("向量库连接中断")]) as inj:
        state = run(agent.run("太空封印站两件套效果是什么？"))

    assert len(inj.calls) == 1, "注入没有被调用 —— 下面的异常断言会变成假绿"
    assert state.stop_reason == "answer"
    assert "知识库服务暂时不可用" in state.answer
    assert state.tool_failures == 1, "抛异常的调用必须计入 tool_failures"
    assert len(state.steps) == 1 and state.steps[0].ok is False
    assert_message_chain_intact(state.messages)


def test_工具异常文本确实喂回了模型():
    """异常要变成模型能读到的文本，而不是被吞掉让模型看到空结果。"""
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "查不到。"},
    ])
    with ToolInjector([RuntimeError("向量库连接中断")]):
        state = run(agent.run("太空封印站两件套效果是什么？"))

    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    assert tool_msgs, "工具结果没有回到消息链里"
    assert "向量库连接中断" in tool_msgs[0]["content"]


def test_工具成功时不算失败():
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "答", },
    ])
    with ToolInjector([(OK_HIT, True)]):
        state = run(agent.run("太空封印站两件套效果是什么？"))
    assert state.tool_failures == 0
    assert state.steps[0].ok is True


# --------------------------------------------------------------------------- #
# 2. 超时参数真的传下去了
# --------------------------------------------------------------------------- #
def test_工具超时参数来自配置():
    """单工具超时是第四道安全阀。参数没传下去 = 慢工具能拖死整条链。

    断言的是**传下去的值**，而不是「有没有超时」—— 后者要靠真的 sleep 才能测，
    而那会让 CI 变慢且不稳定。
    """
    s = dataclasses.replace(get_settings(), tool_timeout=7)
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "答"},
    ], settings=s)
    with ToolInjector([(OK_HIT, True)]) as inj:
        run(agent.run("太空封印站两件套效果是什么？"))

    assert inj.calls, "注入没有被调用 —— 下面的超时断言会变成假绿"
    assert inj.calls[0][2] == 7, f"传下去的 timeout 是 {inj.calls[0][2]}，不是配置的 7"


# --------------------------------------------------------------------------- #
# 3. 基础设施失败会累计 miss 并最终熔断（文档里的那个盲区）
# --------------------------------------------------------------------------- #
def test_连续基础设施失败触发熔断并强制收尾():
    """这是 `failure_analysis.md` 第六节说的「只有探针级验证」的那条路径。

    模型在基础设施故障期间反复检索 → SearchGuard 计数到 MISS_LIMIT →
    下一次检索被熔断 → 模型读到熔断文案后收尾。
    """
    s = dataclasses.replace(get_settings(), agent_max_steps=8)
    agent = make_agent([
        {"tool": ("search_kb", {"query": "问题一"})},
        {"tool": ("search_kb", {"query": "问题二"})},
        {"tool": ("search_kb", {"query": "问题三"})},
        {"tool": ("search_kb", {"query": "问题四"})},
        {"answer": "知识库暂时不可用，无法回答。"},
    ], settings=s)
    # 前两次真失败（计入 miss），第三次被熔断拦下（不执行）
    with ToolInjector([(INFRA_HIT, False), (INFRA_HIT, False)]) as inj:
        state = run(agent.run("一个会一直检索的问题"))

    stats = state.guard_stats()
    assert stats["forced_finishes"] >= 1, f"熔断没有触发：{stats}"
    assert stats["consecutive_misses"] >= 2, f"基础设施失败没计入 miss：{stats}"
    assert state.stop_reason in {"answer", "max_steps"}
    assert "知识库暂时不可用" in state.answer
    assert_message_chain_intact(state.messages)

    # 熔断之后不该再真的执行工具：注入次数应当少于模型的检索次数
    assert len(inj.calls) < 4, (
        f"熔断后仍在执行工具（注入了 {len(inj.calls)} 次）—— "
        f"熔断只是文案的话，Agent 会继续空转"
    )


def test_空结果会累计miss并熔断():
    """检索得到 0 条 ≠ 故障，但同样说明「查不到」，同样该计入 miss。"""
    s = dataclasses.replace(get_settings(), agent_max_steps=8)
    agent = make_agent([
        {"tool": ("search_kb", {"query": "问题一"})},
        {"tool": ("search_kb", {"query": "问题二"})},
        {"tool": ("search_kb", {"query": "问题三"})},
        {"tool": ("search_kb", {"query": "问题四"})},
        {"answer": "知识库没有收录这个内容。"},
    ], settings=s)
    with ToolInjector([(EMPTY_HIT, True)]):
        state = run(agent.run("一个查不到的问题"))

    stats = state.guard_stats()
    assert stats["forced_finishes"] >= 1, f"空结果没有触发熔断：{stats}"
    assert_message_chain_intact(state.messages)


def test_低分命中也会累计miss():
    """status=ok 但 top_score 很低 —— 查到了但都不相关，和查不到是一回事。"""
    s = dataclasses.replace(get_settings(), agent_max_steps=8)
    agent = make_agent([
        {"tool": ("search_kb", {"query": "问题一"})},
        {"tool": ("search_kb", {"query": "问题二"})},
        {"tool": ("search_kb", {"query": "问题三"})},
        {"tool": ("search_kb", {"query": "问题四"})},
        {"answer": "没有找到可信内容。"},
    ], settings=s)
    with ToolInjector([(LOW_HIT, True)]):
        state = run(agent.run("一个查不到的问题"))
    assert state.guard_stats()["forced_finishes"] >= 1


def test_熔断后唯一查询数不再增长():
    """熔断的语义是「不再执行新查询」，unique_queries 必须停住。

    这条比「forced_finishes > 0」更强：它验证熔断**真的省下了检索**，
    而不只是给模型递了句文案。
    """
    s = dataclasses.replace(get_settings(), agent_max_steps=8)
    agent = make_agent([
        {"tool": ("search_kb", {"query": f"问题{i}"})} for i in range(1, 7)
    ] + [{"answer": "没有找到。"}], settings=s)
    with ToolInjector([(INFRA_HIT, False)]):
        state = run(agent.run("查不到的问题"))

    stats = state.guard_stats()
    # MISS_LIMIT=2 → 最多 2 次真实检索被登记，第 3 次起全部被熔断
    assert stats["unique_queries"] <= 3, (
        f"熔断没有限制住检索：unique_queries={stats['unique_queries']}，"
        f"模型跑了 {len(state.steps)} 步"
    )


# --------------------------------------------------------------------------- #
# 4. 重复查询去重（不算故障，但和熔断共享 seen 状态）
# --------------------------------------------------------------------------- #
def test_重复查询被拦截且不计入miss():
    """去重与熔断共用同一个 guard。去重被拦下**不能**算 miss ——
    否则重复查询会把熔断计数推上去，在查得到的情况下误熔断。"""
    s = dataclasses.replace(get_settings(), agent_max_steps=8)
    agent = make_agent([
        {"tool": ("search_kb", {"query": "同一个问题"})},
        {"tool": ("search_kb", {"query": "同一个问题"})},
        {"answer": "答案"},
    ], settings=s)
    with ToolInjector([(OK_HIT, True)]) as inj:
        state = run(agent.run("同一个问题"))

    stats = state.guard_stats()
    assert stats["dedup_blocked"] >= 1, f"重复查询没有被拦截：{stats}"
    assert stats["consecutive_misses"] == 0, f"去重被误计成 miss：{stats}"
    assert len(inj.calls) == 1, f"被去重的查询仍然执行了工具，共 {len(inj.calls)} 次"


# --------------------------------------------------------------------------- #
# 5. 模型侧故障
# --------------------------------------------------------------------------- #
def test_模型调用失败如实报错而不是静默空转():
    agent = make_agent([{"error": "上游 503"}])
    state = run(agent.run("随便问点什么"))

    assert state.stop_reason == "error"
    assert "模型调用失败" in state.answer
    assert "503" in state.answer, "错误原因必须如实带出来，不能只说'失败了'"


def test_熔断后强制收尾本身失败也如实报错():
    """收尾那一次调用再失败时，答案必须是**错误说明**而不是空串。

    空串会让前端显示一个空白回答框，用户以为「它就是没答上来」，
    而真相是系统内部炸了 —— 这种静默比报错更难查。
    """
    s = dataclasses.replace(get_settings(), agent_max_steps=1)
    agent = make_agent([
        {"tool": ("search_kb", {"query": "问题一"})},
        {"error": "收尾时上游挂了"},
    ], settings=s)
    with ToolInjector([(OK_HIT, True)]):
        state = run(agent.run("一个问题"))

    assert state.stop_reason == "max_steps"
    assert state.answer.strip(), "收尾失败后答案不能是空串"
    assert "安全阀上限" in state.answer


# --------------------------------------------------------------------------- #
# 6. 故障与引用校验的交互（本轮新增的一处风险）
# --------------------------------------------------------------------------- #
def test_工具全挂时不会把答案里的引用全剥掉():
    """**这是加引用校验后最容易造成大范围误伤的组合**。

    工具全挂 → `retrieved_pages` 是空的 → 如果校验器在判据缺失时照样动手，
    一次纯粹的**基础设施故障**会被当成模型大面积编造引用，把答案削成满身弹孔。

    正确行为：一个引用都不剥，`checked=False`，并且把故障如实报出去。
    """
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "两件套效果是攻击力提高 12%。[太空封印站]"},
    ])
    with ToolInjector([(INFRA_HIT, False)]):
        state = run(agent.run("太空封印站两件套效果是什么？"))

    assert "[太空封印站]" in state.answer, "判据缺失时把引用剥掉了 —— 这是误伤"
    assert state.citation_audit is not None
    assert state.citation_audit.checked is False, "工具全挂时不该声称校验过"
    assert state.citation_audit.fabricated == []
    assert state.citation_audit.ok is False, "没校验过不能报成 ok"
    assert state.tool_failures == 1, "同时必须如实记下工具确实失败了"


def test_工具正常时校验会照常执行():
    """与上一条成对：判据可用时校验必须真的跑，不能被上一条的分支吞掉。"""
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "两件套效果是攻击力提高 12%。[太空封印站]"},
    ])
    with ToolInjector([(OK_HIT, True)]):
        state = run(agent.run("太空封印站两件套效果是什么？"))

    assert state.citation_audit is not None
    assert state.citation_audit.checked is True
    assert state.citation_audit.ok is True
    assert "[太空封印站]" in state.answer


def test_越界拒答不跑引用校验():
    """没调过工具、正文也没有引用，硬跑校验只会得到全假阳性。"""
    agent = make_agent([
        {"answer": "本知识库未收录崩坏3。"},
        {"answer": "不该被调用第二次"},
    ])
    with ToolInjector([]) as inj:
        state = run(agent.run("崩坏3的雷电女王怎么配装？"))

    assert state.stop_reason == "out_of_scope"
    assert state.citation_audit is None, "越界分支不该产生校验结果"
    assert not inj.calls, "越界分支不该调用任何工具"


# --------------------------------------------------------------------------- #
# 7. 流式路径的故障
# --------------------------------------------------------------------------- #
def stream_events(agent, question):
    """跑一次流式并收齐全部事件。

    `run()` 是 async def，`stream=True` 时它先返回一个**协程**，
    必须先 await 一次才拿到 `_run_stream` 这个异步生成器 ——
    直接把 `agent.run(...)` 丢进列表推导会得到
    "coroutine was never awaited"，而且事件列表是空的，
    后续所有断言都会以莫名其妙的方式失败。
    """
    async def drive():
        it = await agent.run(question, stream=True)
        return [ev async for ev in it]

    return run(drive())


def test_流式下工具异常仍会发出done事件():
    """流式路径的 SSE 连接不能裸断 —— 前端会一直转圈到超时。"""
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "工具挂了，但我还在。"},
    ])
    with ToolInjector([RuntimeError("向量库炸了")]):
        events = stream_events(agent, "太空封印站效果")

    names = [e["event"] for e in events]
    assert "done" in names, f"没有 done 事件，事件序列：{names}"
    assert "error" not in names, "工具异常不该被当成整轮失败"


def test_流式路径的done里带上引用校验结果():
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "两件套提高 12% 攻击力。[太空封印站]"},
    ])
    with ToolInjector([(OK_HIT, True)]):
        events = stream_events(agent, "太空封印站效果")

    done = next(e for e in events if e["event"] == "done")
    audit = done["data"]["citation_audit"]
    assert audit is not None and audit["checked"] is True, (
        f"done 事件没带可用的校验结果：{done['data'].get('citation_audit')}"
    )


def test_流式路径不改写答案():
    """token 已经推给用户了，再改 state.answer 就是「显示的」与「存下的」两份文本。"""
    fabricated_answer = "两件套提高 12% 攻击力。[太空封印站]，也可参考[不老者的仙舟]。"
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": fabricated_answer},
    ])
    with ToolInjector([(OK_HIT, True)]):
        events = stream_events(agent, "太空封印站效果")

    done = next(e for e in events if e["event"] == "done")
    assert "[不老者的仙舟]" in done["data"]["answer"], (
        "流式路径把答案改写了 —— 屏幕上推出去的 token 与 done.answer 会不一致"
    )
    # 但必须**发现**它，否则前端无从提示
    assert done["data"]["citation_audit"]["fabricated"] == ["不老者的仙舟"]
    streamed = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    assert "[不老者的仙舟]" in streamed, "token 事件与 done.answer 应当是同一段文字"


def test_非流式路径会改写答案():
    """与上一条成对：非流式时用户还没看到整段答案，剥掉是安全的。"""
    agent = make_agent([
        {"tool": ("search_kb", {"query": "太空封印站"})},
        {"answer": "两件套提高 12% 攻击力。[太空封印站]，也可参考[不老者的仙舟]。"},
    ])
    with ToolInjector([(OK_HIT, True)]):
        state = run(agent.run("太空封印站效果"))

    assert "[不老者的仙舟]" not in state.answer
    assert "[太空封印站]" in state.answer
    assert state.citation_audit.fabricated == ["不老者的仙舟"]


# --------------------------------------------------------------------------- #
# 8. 守卫自身不能假绿
# --------------------------------------------------------------------------- #
def test_双向验证_摘掉注入点则本文件会失去意义():
    """反证：如果注入根本没生效（`call_tool` 没被替换），上面所有断言都会
    因为「工具真的跑了、结果碰巧符合」而通过。

    这条直接验注入机制本身：把 `_exec_call` 里的 `call_tool` 改名后，
    ToolInjector 必须**一个调用都收不到**。
    """
    assert react_mod.call_tool.__module__.startswith("app.agent"), (
        f"react 里的 call_tool 来自 {react_mod.call_tool.__module__}，"
        f"替换它未必替换得到真实调用点"
    )
    agent = make_agent([
        {"tool": ("search_kb", {"query": "a"})},
        {"answer": "答"},
    ])
    with ToolInjector([(OK_HIT, True)]) as inj:
        run(agent.run("问题"))
    assert len(inj.calls) == 1, "注入没有拦到真实调用"


def test_双向验证_故障脚本被误当成正常时测试必须失败():
    """把「注入抛异常」换成「注入正常返回」，故障测试的前提就消失了。

    这里只验机制：`ToolInjector` 对异常与返回值两种形态的处理是分开的，
    混淆会让故障测试悄悄退化成正常路径测试。

    注意这两个辅助函数是**同步**的：写成 `async def` 再在里面调
    `asyncio.run`，等于在运行中的事件循环里再起一个循环，
    直接抛 "asyncio.run() cannot be called from a running event loop"。
    """
    def go_raise():
        with ToolInjector([RuntimeError("x")]):
            agent = make_agent([
                {"tool": ("search_kb", {"query": "a"})},
                {"answer": "答"},
            ])
            return run(agent.run("问题"))

    def go_ok():
        with ToolInjector([(OK_HIT, True)]):
            agent = make_agent([
                {"tool": ("search_kb", {"query": "a"})},
                {"answer": "答"},
            ])
            return run(agent.run("问题"))

    bad = go_raise()
    good = go_ok()
    assert bad.tool_failures == 1, "注入异常没有变成失败"
    assert good.tool_failures == 0, "注入正常返回却算成了失败"


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_步数上限在任何配置下都兜得住(limit):
    """模型死循环时，无论上限设成几，都必须停下来而不是把 FakeLLM 脚本耗穿。

    FakeLLM 脚本耗穿会抛 AssertionError，所以「测试失败」在这里就是
    「循环没停」的正确信号 —— 但必须让三个取值都试，因为熔断与步数上限
    是两条独立的路径，只有其中一个生效时另一个的参数改动不影响结果。
    """
    s = dataclasses.replace(get_settings(), agent_max_steps=limit)
    agent = make_agent(
        [{"tool": ("search_kb", {"query": f"q{i}"})} for i in range(limit + 2)]
        + [{"answer": "（强制收尾）"}],
        settings=s,
    )
    with ToolInjector([(OK_HIT, True)]):
        state = run(agent.run("会一直调用工具的问题"))

    assert state.stop_reason == "max_steps"
    assert len(state.steps) <= limit + 1
    assert_message_chain_intact(state.messages)
