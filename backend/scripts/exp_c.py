"""实验 C：Agent 能力消融（生成侧单变量阶梯）。

这是唯一一个**回答"RAG 到底有没有必要"**的实验。
实验 A/B/D 都是"有了 RAG 之后怎么调优"，但没人证明过
「如果不上 RAG，模型会答成什么样」。

阶梯（每一步只放开一个能力）：

  C0  无工具      不给 tools，模型只能靠训练数据记忆回答
                 → 预期：数值类问题会编造（这就是幻觉的对照组）
  C1  单工具      只给 search_kb
                 → 预期：事实类问题开始可靠，但算数类不行
  C2  多工具循环  全部 4 个工具（search_kb / get_page / calculator / list_games）
                 → 预期：数值题要靠 calculator 串起来

**为什么 C0 是关键**：它不是"一个更差的系统"，它是
「通用大模型的知识截止 + 幻觉」的原样基线。
C0 vs C2 的差距就是这个项目存在的理由。

评分用独立 Judge（SiliconFlow Qwen3-8B），绝对打分不成对 ——
32 条 × 3 配置的成对比较要 192 次 Judge 调用，单次 ~15s，跑一整天。
这里用 15 条代表性子集 + 绝对打分，45 次调用约 11 分钟。

用法：
    .venv\\Scripts\\python.exe scripts\\exp_c.py
    .venv\\Scripts\\python.exe scripts\\exp_c.py --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.agent.react import SYSTEM_PROMPT, ReActAgent  # noqa: E402
from app.agent.tools import ToolContext, get_schemas  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.judge import Judge, contexts_from_hits  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from app.retrieval.hybrid import RetrievalConfig  # noqa: E402
from exp_common import (  # noqa: E402
    aggregate_rows,
    load_cases,
    paired_scores,
    print_summary,
    save,
)

# 每级开放哪些工具 —— 唯一的变量
PRESETS = {
    "C0": [],                                  # 无工具
    "C1": ["search_kb"],                       # 单工具
    "C2": ["search_kb", "get_page", "calculator", "list_games"],
}



async def run_one(label: str, case_list, judge: Judge) -> dict:
    s = get_settings()
    llm = get_llm()
    rows = []

    allowed = set(PRESETS[label])
    schemas = [sc for sc in get_schemas() if sc["function"]["name"] in allowed]
    # 没给工具时，直接走「单轮问答」而不是 ReAct 循环 ——
    # 否则 Agent 收不到 tool_calls 声明会反复尝试调用。
    single_shot = label == "C0"

    for case in case_list:
        ctx = ToolContext.build(case.game)
        agent = ReActAgent(settings=s, llm=llm, tools=schemas)
        # 证据单独取一份喂给 Judge。C0 没调工具，所以「它看到的证据」就是空 ——
        # 这正是要测的：没证据时它会编。
        cfg = RetrievalConfig(game=case.game, final_top_k=5)
        try:
            ev = await ctx.retriever.retrieve(case.question, cfg)
            contexts = contexts_from_hits(ev)
        except Exception:  # noqa: BLE001
            contexts = ""

        t0 = time.perf_counter()
        try:
            if single_shot:
                out = await llm.chat(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": case.question}],
                    max_tokens=800, temperature=0.0,
                )
                answer = out["content"]
                usage = out["usage"]
                steps, failures, refs = 0, 0, []
            else:
                st = await agent.run(case.question, ctx=ctx)
                answer, steps, failures, refs = st.answer, len(st.steps), st.tool_failures, st.refs
                usage = {"prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens}
            err = None
        except Exception as exc:  # noqa: BLE001
            answer, usage, steps, failures, refs = "", {}, 0, 0, []
            err = f"{type(exc).__name__}: {exc}"
        dt = int((time.perf_counter() - t0) * 1000)

        js = await judge.score(
            question=case.question, gold=case.gold_answer,
            answer=answer, contexts=contexts if label != "C0" else "",
        )
        d = js.as_dict()
        rows.append({
            "case_id": case.case_id, "category": case.category,
            "correctness": d["correctness"], "faithfulness": d["faithfulness"],
            "completeness": d["completeness"], "judge_score": d["judge_score"],
            "judge_reason": d["reason"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "steps": steps, "tool_failures": failures,
            "has_refs": bool(refs), "latency_ms": dt,
            "error": err, "answer": answer[:500],
        })
        print(f"  [{label}] {case.case_id:<8} score={d['judge_score']:<6} "
              f"corr={d['correctness']:<4} faith={d['faithfulness']:<4} "
              f"steps={steps} tok={usage.get('prompt_tokens', 0):>6} {dt:>6}ms", flush=True)

    # 聚合口径统一在 exp_common.aggregate_rows —— 三个实验必须用同一份，否则数字不可比
    return {"label": label, "tools": PRESETS[label], "aggregate": aggregate_rows(rows), "rows": rows}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="C0,C1,C2")
    ap.add_argument("--limit", type=int, default=0, help="0 = 用 15 条子集")
    args = ap.parse_args()

    case_list = load_cases(args.limit)
    print(f"评测子集：{len(case_list)} 条，标签 {args.labels}\n")

    results = []
    judge = Judge(get_llm())
    for label in args.labels.split(","):
        print(f"=== 运行 {label}（工具：{PRESETS.get(label, '?')}）===", flush=True)
        results.append(await run_one(label.strip().upper(), case_list, judge))

    save(results, "exp_C.json")
    print_summary(results, f"实验 C：Agent 能力阶梯消融（生成侧，n={len(case_list)}）")

    print("\n--- 各档开放了哪些工具 ---")
    for r in results:
        print(f"  {r['label']:<5}{','.join(r['tools']) or '（无）'}")

    t = paired_scores(results)
    if t:
        print(f"\n--- 配对检验 C0 → C2（metric=judge_score，n={t['n']}）---")
        print(f"  Δ={t['mean_diff']:+.4f}  t={t['t']}  p={t['p']}  Cohen's d={t['cohens_d']}")


if __name__ == "__main__":
    asyncio.run(main())
