"""重构后的自检脚本（不依赖外部 API 也能跑完前 6 项）。

覆盖的都是这轮改动真实可能踩到的点：
  1. 模块全部可导入（补完 llm/client.py 后最容易漏的地方）
  2. strip_think：完整/多段/未闭合三种输入
  3. parse_inline_tool_calls：三种内联写法 + 只认白名单里的名字
  4. 工具超时返回可读文案（**修复前这里会 NameError**）
  5. 熔断在两条路径上行为一致（_exec_call 抽取后的回归点）
  6. /api/health 能列出可用基座且不泄露密钥
  7. 未知 backend 在入口处被 400 拦下，而不是跑起来才 500
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.react import ReActAgent, SearchGuard, ToolContext
from app.agent.tools import call_tool
from app.llm.client import LLMClient, parse_inline_tool_calls, strip_think

OK, FAIL = "✅", "❌"
results: list[tuple[bool, str]] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    results.append((cond, name))
    print(f"  {OK if cond else FAIL} {name}{('  ' + extra) if extra else ''}")


async def main() -> None:
    print("=" * 74)
    print("  1. 导入")
    print("=" * 74)
    llm = LLMClient()
    check("llm/client.py 可导入", True)
    check("CHAT_BACKENDS 覆盖两个基座", set(llm.CHAT_BACKENDS) == {"deepseek", "minimax"})

    print("\n" + "=" * 74)
    print("  2. strip_think")
    print("=" * 74)
    check("完整标签被剥除", strip_think("<think>abc</think>答案") == "答案")
    check("两段都剥掉", strip_think("<think>a</think>甲<think>b</think>乙") == "甲乙")
    check("未闭合标签丢弃其后全部", strip_think("正文<think>半截推理") == "正文")
    check("无标签原样返回", strip_think("直接回答") == "直接回答")

    print("\n" + "=" * 74)
    print("  3. parse_inline_tool_calls")
    print("=" * 74)
    allowed = {"search_kb", "calculator"}
    cases = [
        ('{"name": "search_kb", "arguments": {"query": "刻晴"}}', "name/arguments"),
        ('{"tool": "calculator", "parameters": {"expression": "1+1"}}', "tool/parameters"),
        ('{"function": {"name": "search_kb", "arguments": "{\\"query\\": \\"x\\"}"}}', "嵌套 function"),
    ]
    for blob, label in cases:
        calls = parse_inline_tool_calls(blob, allowed)
        check(f"内联解析：{label}", len(calls) == 1, f"-> {calls[0]['name'] if calls else '无'}")
    check("白名单外的名字被拒", parse_inline_tool_calls('{"name":"rm_rf","arguments":{}}', allowed) == [])
    check("普通文本不误判", parse_inline_tool_calls("答案是 3，不含任何 JSON", allowed) == [])
    check("非 JSON 花括号不崩", parse_inline_tool_calls("例如 f(x) = {a: 1} 这样", allowed) == [])

    print("\n" + "=" * 74)
    print("  4. 工具超时（修复前 asyncio.TimeoutError 分支会 NameError）")
    print("=" * 74)
    ctx = ToolContext.build("sr")
    text, ok = await call_tool("calculator", {"expression": "1/0"}, ctx, timeout=5)
    check("除零返回可读文案且 ok=False", not ok and "除零" in text, f"-> {text[:30]}")
    text, ok = await call_tool("no_such_tool", {}, ctx, timeout=5)
    check("未知工具返回可读文案", not ok and "未知工具" in text)
    # 真超时：注册一个永远不返回的工具
    import app.agent.tools as tools_mod

    async def _hang(args, c):
        await asyncio.sleep(99)

    tools_mod.REGISTRY["__hang__"] = _hang
    try:
        text, ok = await call_tool("__hang__", {}, ctx, timeout=0.2)
        check("超时返回 [超时] 且 ok=False", (not ok) and text.startswith("[超时]"), f"-> {text[:40]}")
    finally:
        tools_mod.REGISTRY.pop("__hang__", None)

    print("\n" + "=" * 74)
    print("  5. _exec_call 抽取后：熔断行为一致")
    print("=" * 74)
    agent = ReActAgent()
    guard = SearchGuard()
    from app.agent.react import AgentState

    st = AgentState()
    t1, r1 = await agent._exec_call({"name": "search_kb", "args": {"query": "刻晴 突破"}}, ctx, guard, st)
    t2, r2 = await agent._exec_call({"name": "search_kb", "args": {"query": "刻晴 突破"}}, ctx, guard, st)
    check("首次执行真实检索", r1.ok or r1.error is not None, f"ok={r1.ok} {r1.latency_ms}ms")
    check("等价查询被熔断", t2.startswith("[重复查询已拦截]"), f"-> {t2[:34]}")
    check("熔断计入失败数", st.tool_failures >= 1, f"-> tool_failures={st.tool_failures}")
    check("StepRecord 已登记", len(st.steps) == 2 and st.steps[1].ok is False)

    print("\n" + "=" * 74)
    print("  6. /api/health 的基座列表 + 单例一致性")
    print("=" * 74)
    backs = llm.available_backends()
    check("至少列出 deepseek", any(b["id"] == "deepseek" for b in backs))
    check("不含任何密钥字段", all("key" not in b and "token" not in b for b in backs), f"-> {backs}")

    # 检索器如果自己 new 一个 LLMClient，缓存就分裂到两个池子里：
    # /api/health 看的那份永远是空的，而命中率展示就变成了假数据。
    from app.llm.client import get_llm
    from app.retrieval.hybrid import get_retriever

    r = get_retriever()
    check("检索器复用 get_llm() 单例", r.llm is get_llm(), f"-> id={id(r.llm)} vs {id(get_llm())}")
    # /api/health 里的 cache 字段来自 get_llm().cache_stats()，
    # 检索也走同一个实例，所以那个命中率才是真的。
    check("缓存统计可观测", isinstance(get_llm().cache_stats()["embed"], dict))

    print("\n" + "=" * 74)
    print("  7. 未知 backend 的拦截")
    print("=" * 74)
    try:
        llm.chat_backend_name("gpt4")
        check("未知基座抛 ValueError", False)
    except ValueError as e:
        check("未知基座抛 ValueError", True, f"-> {str(e)[:48]}")
    from app.main import ChatRequest, _build_agent
    from fastapi import HTTPException

    try:
        _build_agent(ChatRequest(question="测试", backend="gpt4"))
        check("入口处拦截未知基座", False)
    except HTTPException as e:
        check("入口处拦截未知基座", e.status_code == 400, f"-> {e.status_code}")

    print("\n" + "=" * 74)
    print("  8. 多游戏：路由三态（库内 / 越界 / 未提及）")
    print("=" * 74)
    from app.agent.react import UNCOVERED_GAMES, game_display_name, route_game
    from app.core.config import get_settings

    s = get_settings()
    covered = s.games
    check("settings.games 是 N 游戏而非写死两个", len(covered) >= 3, f"-> {list(covered)}")
    check("原神已入库", "ys" in covered)

    # 三种情形必须能区分：这是修复"假阴性"的核心
    in_scope = route_game("原神里圣遗物套装有哪些效果", covered)
    check("库内游戏能被路由", in_scope == "ys", f"-> {in_scope}")
    out_scope = route_game("崩坏3的雷电女王怎么配装", covered)
    check("未收录游戏能被识别为越界", out_scope == "hhzr", f"-> {out_scope}")
    check("未收录游戏确实不在覆盖范围", out_scope not in covered)
    unmentioned = route_game("请计算 (0.8+0.2)*1000", covered)
    check("未提及游戏返回 None（交由调用方兜底）", unmentioned is None, f"-> {unmentioned}")

    check(
        "显示名可解析",
        game_display_name("ys", covered) == "原神"
        and game_display_name("hhzr", covered) == "崩坏3",
    )
    check("越界表与 settings.games 不重叠", not (set(UNCOVERED_GAMES) & set(covered)))

    # 未收录游戏必须走拒答，而不是硬去搜一个错的库
    from app.agent.react import AgentState
    from app.agent.react import ReActAgent as _RA

    agent = _RA()
    ctx2 = ToolContext.build("sr")
    _, _, st = agent._prepare("崩坏3的雷电女王怎么配装", None, None, ctx2)
    check("越界问题被标记 out_of_scope", st.uncovered_game == "hhzr", f"-> {st.uncovered_game}")
    _, _, st2 = agent._prepare("太空封印站的两件套效果是什么", None, None, ctx2)
    check("库内问题不被误标越界", st2.uncovered_game is None)

    # 回归：系统提示词必须覆盖**每一个**已收录游戏。
    # 之前 SYSTEM_PROMPT 是写死"星铁与绝区零"的常量，加了第三个库之后
    # 检索层能召回原神、Agent 却拿着"只覆盖两款"的提示词把问题自己拒了。
    from app.agent.react import build_system_prompt

    prompt = build_system_prompt(covered, s.game_terms)
    missing_names = [n for n in covered.values() if n not in prompt]
    check(
        "系统提示词覆盖所有已收录游戏",
        not missing_names,
        f"-> 缺 {missing_names}" if missing_names else f"-> {list(covered.values())}",
    )
    check(
        "提示词里的游戏名与 settings.games 一一对应",
        sum(n in prompt for n in covered.values()) == len(covered),
    )
    # 越界提示词同样要列全，且不能出现"只覆盖 N 款"这种硬编码措辞
    _, _, st3 = agent._prepare("崩坏3的雷电女王怎么配装", None, None, ctx2)
    scope_prompt = st3.messages[0]["content"]
    check(
        "越界提示词列出全部已收录游戏",
        all(n in scope_prompt for n in covered.values()),
        f"-> 缺 {[n for n in covered.values() if n not in scope_prompt]}",
    )

    # 回归：索引的 .meta.json 必须能被读回来。
    # list_collections 之前直接 new 一个空 collection，元数据"存了但没人读"，
    # 于是 game/variant/embed_model 全空 → 前端只能退回正则解析 collection 名，
    # 把实验变体库也加进"生产语料"，界面显示星铁 9986 片段（实际 2293，虚高 4.4 倍）。
    from app.store.vector_store import get_vector_store

    cols = get_vector_store(s).list_collections()
    named = [c for c in cols if c.get("game") and c.get("variant") and c.get("embed_model")]
    check(
        "list_collections 能读回索引元数据",
        cols and len(named) == len(cols),
        f"-> {len(named)}/{len(cols)} 带完整 meta",
    )
    if named:
        # 生产索引 = base 变体 + 生产 embedding 模型，按游戏合计必须等于真实语料量
        prod = [
            c for c in named
            if c["variant"] == "base" and c["embed_model"] == s.embed_model_main
        ]
        per_game = {c["game"]: c["count"] for c in prod}
        check(
            "生产索引覆盖全部已收录游戏",
            set(per_game) == set(covered),
            f"-> 索引有 {sorted(per_game)}，配置声明 {sorted(covered)}",
        )
        check(
            "实验变体索引不会混入生产合计",
            sum(per_game.values()) < sum(c["count"] for c in named),
            f"-> 生产 {sum(per_game.values())} / 全部 {sum(c['count'] for c in named)}",
        )

    print("\n" + "=" * 74)
    bad = [n for okk, n in results if not okk]
    print(f"  合计 {len(results)} 项，通过 {len(results) - len(bad)}，失败 {len(bad)}")
    if bad:
        for n in bad:
            print(f"    {FAIL} {n}")
    print("=" * 74)
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
