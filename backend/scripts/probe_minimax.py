"""MiniMax-M2.5 工具调用协议探测（实验 E 的前置条件）。

背景：实验 E 要对比 DeepSeek vs MiniMax 做 Chat 基座。
但 ReAct Agent 依赖 OpenAI 的 tool_calls 协议，如果 MiniMax
的响应格式不一样（比如把工具调用写在文本里、或输出 <think> 标签），
Agent 的消息链会直接断掉。

所以**先探协议，再谈对比** —— 否则实验 E 会变成"哪个模型不报错"
而不是"哪个模型答得更好"，变量完全没控制住。

这个脚本回答四个问题：
  1. MiniMax 能不能正常对话？
  2. 它认不认 OpenAI 的 tools schema？
  3. 它返回的是标准 tool_calls 还是塞在 content 里？
  4. <think> 标签会不会污染后续轮次的消息链？
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 复用生产代码里的实现，而不是在脚本里抄一份 ——
# 探针验的就是线上真正在用的那个行为，抄一份就变成验了个不存在的东西。
from app.agent.tools import get_schemas
from app.core.config import get_settings
from app.llm.client import strip_think
from openai import AsyncOpenAI


async def main() -> None:
    s = get_settings()
    if not s.minimax_api_key:
        print("MINIMAX_API_KEY 未配置，实验 E 无法进行。")
        return

    client = AsyncOpenAI(api_key=s.minimax_api_key, base_url=s.minimax_base_url)
    print(f"模型：{s.minimax_chat_model}")
    print(f"端点：{s.minimax_base_url}\n")

    # ---- 1. 纯对话 ----
    print("=" * 76)
    print("  1. 基础对话")
    print("=" * 76)
    try:
        r = await client.chat.completions.create(
            model=s.minimax_chat_model,
            messages=[{"role": "user", "content": "用一句话回答：1+1等于几"}],
            max_tokens=64, temperature=0.0,
        )
        msg = r.choices[0].message
        raw = msg.content or ""
        print(f"  content     : {raw[:160]!r}")
        print(f"  finish      : {r.choices[0].finish_reason}")
        print(f"  usage       : {r.usage.prompt_tokens}/{r.usage.completion_tokens}")
        has_think = "<think>" in raw or "</think>" in raw
        print(f"  含 <think>  : {has_think}")
        if has_think:
            print(f"  剥除后      : {strip_think(raw)[:160]!r}")
    except Exception as exc:
        print(f"  [FAIL] {type(exc).__name__}: {str(exc)[:200]}")
        return

    # ---- 2. 工具调用 ----
    print("\n" + "=" * 76)
    print("  2. 工具调用（用项目真实的 search_kb schema）")
    print("=" * 76)
    tools = get_schemas()
    try:
        r = await client.chat.completions.create(
            model=s.minimax_chat_model,
            messages=[{"role": "system", "content": "你是星穹铁道知识库助手，涉及游戏事实必须先调 search_kb。"},
                      {"role": "user", "content": "太空封印站的两件套效果是什么？"}],
            tools=tools, tool_choice="auto", max_tokens=512, temperature=0.0,
        )
        msg = r.choices[0].message
        tcs = msg.tool_calls or []
        print(f"  标准 tool_calls : {len(tcs)} 个")
        for tc in tcs:
            print(f"    → {tc.function.name}({tc.function.arguments})")
        print(f"  content 前 200 : {(msg.content or '')[:200]!r}")

        if not tcs:
            print("\n  ⚠️ 没有返回标准 tool_calls。检查 content 里是否有 JSON 工具调用：")
            c = strip_think(msg.content or "")
            print(f"    剥除 <think> 后: {c[:400]!r}")
            brace = c.find("{")
            if brace != -1:
                try:
                    parsed = json.loads(c[brace:])
                    print(f"    能解析出 JSON : {json.dumps(parsed, ensure_ascii=False)[:200]}")
                except Exception as pe:
                    print(f"    JSON 解析失败 : {pe}")
    except Exception as exc:
        print(f"  [FAIL] {type(exc).__name__}: {str(exc)[:300]}")
        return

    # ---- 3. 多轮消息链（ReAct 真正依赖的东西）----
    print("\n" + "=" * 76)
    print("  3. 多轮消息链：assistant(tool_calls) + tool 能否被接受")
    print("=" * 76)
    if not tcs:
        print("  跳过（没有 tool_calls 可回传）")
        return
    try:
        messages = [
            {"role": "system", "content": "你是星穹铁道知识库助手。"},
            {"role": "user", "content": "太空封印站的两件套效果是什么？"},
            {"role": "assistant", "content": "" or (msg.content or ""),
             "tool_calls": [{"id": tcs[0].id, "type": "function",
                             "function": {"name": tcs[0].function.name,
                                          "arguments": tcs[0].function.arguments}}]},
            {"role": "tool", "tool_call_id": tcs[0].id, "name": tcs[0].function.name,
             "content": "[search_kb] status=ok hits=2 top_score=0.93\n[1] 太空封印站 > 两件套效果\n"
                        "使装备者攻击力提高12%，速度≥120时额外提高12%。\n出处：https://wiki.biligame.com/sr/太空封印站"},
        ]
        r2 = await client.chat.completions.create(
            model=s.minimax_chat_model, messages=messages,
            tools=tools, tool_choice="auto", max_tokens=512, temperature=0.0,
        )
        m2 = r2.choices[0].message
        print("  ✅ 消息链被接受")
        print(f"  第二轮 tool_calls: {len(m2.tool_calls or [])} 个")
        print(f"  第二轮 content   : {strip_think(m2.content or '')[:260]!r}")
        print(f"  finish_reason   : {r2.choices[0].finish_reason}")
        print("\n  → 若以上都通过，MiniMax 可以直接替换 DeepSeek 做实验 E。")
    except Exception as exc:
        print(f"  [FAIL] {type(exc).__name__}: {str(exc)[:300]}")
        print("  → MiniMax 不接受标准 OpenAI 消息链，需要在 LLMClient 里加适配层。")


if __name__ == "__main__":
    asyncio.run(main())