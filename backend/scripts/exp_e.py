"""实验 E：Chat 基座对比（DeepSeek vs MiniMax）。

这是唯一一个「换大脑」的实验。检索、分块、评测集、Judge 全部冻结，
唯一变量是 Chat 基座 —— 所以差异只能归因到模型本身。

**冻结清单**（不冻结就不是单变量对比）：
  temperature = 0
  tools schema = 同一套（get_schemas()）
  系统提示 = SYSTEM_PROMPT，不改一个字
  评测集 = 实验 C 的同一批 15 条
  Judge = 同一个独立基座（SiliconFlow Qwen3-8B）

**注意 MiniMax 的两个已知风险**（先探测再跑，scripts/probe_minimax.py）：
  1. 会输出 `<think>` 标签 —— 混进 assistant content 会污染后续轮次的消息链
  2. 工具调用协议可能与 OpenAI 不完全一致

所以这里做了两处适配：
  - 剥除 `<think>...</think>` 再放进消息
  - 把 content 里的 JSON 工具调用解析回标准 tool_calls（兜底）

用法：
    .venv\\Scripts\\python.exe scripts\\exp_e.py
    .venv\\Scripts\\python.exe scripts\\exp_e.py --limit 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.agent.react import ReActAgent  # noqa: E402
from app.agent.tools import ToolContext, get_schemas  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.judge import Judge, contexts_from_hits  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from app.retrieval.hybrid import RetrievalConfig  # noqa: E402
from exp_common import (  # noqa: E402
    SUBSET_IDS as _SUBSET_IDS,
)
from exp_common import (
    aggregate_rows,
    load_cases,
    paired_scores,
    print_summary,
    save,
)

MODELS = {
    "E1": None,                       # None = .env 的 deepseek_chat_model
    "E2": "MiniMax-M2.5",
}

# 评测子集（15 条）与实验 C 完全一致 —— 两个实验要能横向对比，
# 子集就必须来自同一个来源，所以复用 exp_common.SUBSET_IDS，不在这里抄一份。
# 抄一份的后果是：哪天给 C 换子集，E 还停在旧子集上，对比结论悄悄失效。
SUBSET_IDS = _SUBSET_IDS

_THINK = re.compile(r"<think>.*?</think>", re.S)
_JSON_CALL = re.compile(r"\{[\s\S]*\}")

from openai import AsyncOpenAI  # noqa: E402


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def repair_tool_calls(content: str) -> list[dict]:
    """兜底：模型把工具调用写成了 content 里的 JSON 而不是标准 tool_calls。

    没有这层，ReAct 的消息链会断 —— assistant 没有 tool_calls，
    后面就挂不上 role='tool' 的消息。
    """
    if not content:
        return []
    m = _JSON_CALL.search(content)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    # 兼容几种常见写法
    for name in ("tool", "tool_name", "function", "name"):
        if isinstance(obj, dict) and name in obj:
            fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
            arg = fn.get("arguments") or fn.get("args") or fn.get("parameters") or {}
            if isinstance(arg, str):
                try:
                    arg = json.loads(arg)
                except json.JSONDecodeError:
                    arg = {"raw": arg}
            return [{"id": "call_repaired_0", "name": fn.get(name) or fn.get("name"),
                     "args": arg if isinstance(arg, dict) else {}}]
    return []


async def chat_with(label: str, model_name: str | None, messages: list[dict],
                    tools: list[dict], timeout: float = 120.0) -> dict:
    """单次调用，带 MiniMax 适配与工具调用修复。"""
    s = get_settings()
    llm = get_llm()

    if label == "E1":
        out = await llm.chat(messages, model=model_name, tools=tools,
                             temperature=0.0, max_tokens=1024)
        return {"content": out["content"], "tool_calls": out["tool_calls"], "usage": out["usage"]}

    # E2：MiniMax 走独立 client（不同 base_url / key）
    if not s.minimax_api_key:
        raise RuntimeError("MINIMAX_API_KEY 未配置")

    client = AsyncOpenAI(api_key=s.minimax_api_key, base_url=s.minimax_base_url)
    kwargs = {"model": model_name, "messages": messages, "temperature": 0.0, "max_tokens": 1024}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    r = await client.chat.completions.create(**kwargs)
    msg = r.choices[0].message
    content = strip_think(msg.content or "")

    calls = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"__raw__": tc.function.arguments}
        calls.append({"id": tc.id, "name": tc.function.name, "args": args})
    if not calls:
        calls = repair_tool_calls(content)

    usage = {
        "prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
        "completion_tokens": r.usage.completion_tokens if r.usage else 0,
    }
    await client.close()
    return {"content": content, "tool_calls": calls, "usage": usage}


async def run_one(label: str, case_list, judge: Judge) -> dict:
    s = get_settings()
    model_name = MODELS[label]
    if label == "E1" and not model_name:
        model_name = s.deepseek_chat_model
    rows = []
    tools = get_schemas()

    for case in case_list:
        ctx = ToolContext.build(case.game)
        # 证据固定：两个模型看到完全一样的检索结果，
        # 否则「谁答得好」可能只是因为「谁检索到的证据更好」
        try:
            ev = await ctx.retriever.retrieve(case.question, RetrievalConfig(game=case.game, final_top_k=5))
            contexts = contexts_from_hits(ev)
        except Exception:  # noqa: BLE001
            contexts = ""

        # 只注入 chat 函数，ReAct 循环、工具执行、安全阀、引用收集
        # 全部走生产代码本身 —— 不在脚本里抄一份副本。
        # 抄一份的话，测出来的就不是线上真正在跑的东西。
        async def chat_fn(messages, **kw):
            return await chat_with(label, model_name, messages, kw.get("tools") or tools)

        agent = ReActAgent(settings=s, llm=get_llm(), tools=tools, chat_fn=chat_fn)

        t0 = time.perf_counter()
        err = None
        try:
            st = await agent.run(case.question, ctx=ctx)
            answer = st.answer
            steps, failures, refs = len(st.steps), st.tool_failures, st.refs
            usage = {"prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens}
        except Exception as exc:  # noqa: BLE001
            answer, usage, steps, failures, refs = "", {}, 0, 0, []
            err = f"{type(exc).__name__}: {str(exc)[:160]}"
        dt = int((time.perf_counter() - t0) * 1000)

        js = await judge.score(question=case.question, gold=case.gold_answer,
                               answer=answer, contexts=contexts)
        d = js.as_dict()
        # 报错的 case 不能和"答得差"混为一谈 —— judge 对空回答给 0 分，
        # 但那是链路失败不是能力差，不加标记的话均值会被悄悄拉低，
        # 看起来像"这个模型更差"，实际是它根本没跑起来。
        if err:
            d["errored"] = True
        rows.append({
            "case_id": case.case_id, "category": case.category,
            "correctness": d["correctness"], "faithfulness": d["faithfulness"],
            "completeness": d["completeness"], "judge_score": d["judge_score"],
            "judge_reason": d["reason"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "steps": steps, "tool_failures": failures, "has_refs": bool(refs),
            "latency_ms": dt, "error": err, "answer": answer[:500],
        })
        print(f"  [{label}] {case.case_id:<8} score={d['judge_score']:<6} "
              f"steps={steps} tok={usage.get('prompt_tokens', 0):>6} {dt:>6}ms"
              + (f"  ⚠ ERR {str(err)[:70]}" if err else ""), flush=True)

    # 聚合口径统一在 exp_common.aggregate_rows —— 与实验 C 共用同一个函数
    return {"label": label, "model": model_name, "aggregate": aggregate_rows(rows), "rows": rows}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="E1,E2")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()


    case_list = load_cases(args.limit)
    print(f"评测子集：{len(case_list)} 条（与实验 C 同一份 SUBSET_IDS）\n")

    # MODELS[E1] 是 None（= 用 .env 默认值），直接打印会显示 "None"，
    # 看起来像脚本坏了；这里显示实际会用的模型名。
    def _label_of(label: str) -> str:
        return MODELS.get(label) or get_settings().deepseek_chat_model

    results = []
    judge = Judge(get_llm())
    for label in args.labels.split(","):
        label = label.strip().upper()
        print(f"=== 运行 {label}（{_label_of(label)}）===", flush=True)
        results.append(await run_one(label, case_list, judge))

    save(results, "exp_E.json")
    print_summary(results, f"实验 E：Chat 基座对比（冻结 temperature/tools/prompt/评测集，n={len(case_list)}）")

    print("--- 各档 Chat 基座 ---")
    for r in results:
        print(f"  {r['label']:<5}{r.get('model') or ''}")

    t = paired_scores(results)
    if t:
        print(f"\n--- 配对检验 E1 → E2（metric=judge_score，n={t['n']}）---")
        print(f"  Δ={t['mean_diff']:+.4f}  t={t['t']}  p={t['p']}  Cohen's d={t['cohens_d']}")


if __name__ == "__main__":
    asyncio.run(main())

