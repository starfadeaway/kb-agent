"""测试替身：多处复用的假对象。

**为什么单独有这么一个模块**：`FakeLLM` 与 `FakeCtx` 此前各有两份几乎逐字
相同的副本 —— `test_agent_loop` / `test_fault_injection` 各一份 FakeLLM，
`test_mcp_parity` / `test_mcp_main_and_stream_stop_reason` 各一份 FakeCtx
（后者的注释里直接写着「与 test_mcp_parity 同法」）。

复制出来的那一份会在 `ToolContext` 加字段、或 `chat` 换签名时**静默失配** ——
测试照样绿，而假对象已经不像真的了。这正是本项目最在意的那类失败：
不报错、只是让断言失去意义。

**只放契约确实相同的替身。** 行为有差异的故意留原地：
`test_context_budget` 的 `_FakeRetriever` 收一段 text、
`test_revision_disclosure` 的收 hits 列表，强行统一要加开关参数 ——
那比重复更贵。
"""
from __future__ import annotations

import json

from app.agent import tools as T


class FakeLLM:
    """按预设脚本逐次返回 chat 结果。

    每项可以是：

      ``{"tool": ("search_kb", {...})}`` → 让 Agent 去调工具
      ``{"answer": "..."}``             → 直接作答，循环终止
      ``{"error": "boom"}``             → 模拟模型调用抛异常

    **脚本用完再调就 AssertionError**：它把「循环没有在预期的地方终止」变成
    硬失败，而不是让循环悄悄多跑几步、测试看着还是绿的。
    """

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls = 0
        self.seen_messages: list[list[dict]] = []

    async def chat(self, messages, **kw):  # noqa: D102 - 满足 LLMClient 形状
        self.calls += 1
        self.seen_messages.append(list(messages))
        if self.calls > len(self.script):
            raise AssertionError(
                f"FakeLLM 脚本只有 {len(self.script)} 步，却被调用了 {self.calls} 次 —— "
                f"说明循环没有在预期的地方终止"
            )
        step = self.script[self.calls - 1]
        if "error" in step:
            raise RuntimeError(step["error"])
        usage = step.get("usage", {"prompt_tokens": 100, "completion_tokens": 20})
        if "tool" in step:
            name, args = step["tool"]
            # id 不能省：_tool_result_message 要用 call["id"] 去匹配 assistant
            # 消息里的 tool_call_id，两边对不上消息链就断了。
            return {"content": "",
                    "tool_calls": [{"id": f"call_{self.calls}", "name": name, "args": args}],
                    "usage": usage}
        return {"content": step.get("answer", ""), "tool_calls": [], "usage": usage}

    async def rerank(self, *a, **kw):  # pragma: no cover - 走到就是用例写错了
        raise AssertionError("不该走到 rerank")

    async def embed(self, *a, **kw):  # pragma: no cover - 走到就是用例写错了
        raise AssertionError("不该走到 embed")

    def assistant_tool_message(self, content: str, tool_calls: list[dict]) -> dict:
        """Agent 用它把 tool_calls 包成合法的 assistant 消息。

        必须提供：ReAct 循环在回填 tool 结果前要先造这条消息，
        否则下一轮 API 会因为 role=tool 找不到锚点而报错。
        """
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {"id": tc.get("id", f"call_{i}"), "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                for i, tc in enumerate(tool_calls)
            ],
        }

    async def aclose(self):  # pragma: no cover
        return None


class FakeCtx(T.ToolContext):
    """不碰数据库 / 网络的 ToolContext。

    刻意**绕过父类构造**（那个会去建检索器）。属性集要与 `ToolContext` 的字段
    保持一致 —— 少一个就会在某个工具里 AttributeError，而那种失败看起来像
    「这个工具坏了」，实际是替身过时了。
    """

    def __init__(self):  # noqa: D107 - 刻意绕过父类的检索器构造
        self.settings = None
        self.retriever = None
        self.store = None
        self.game = "sr"
