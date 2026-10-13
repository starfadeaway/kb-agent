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
    parse_games,
    print_summary,
    save,
    subset_composition,
)

# 每级开放哪些工具 —— 唯一的变量
PRESETS = {
    "C0": [],                                  # 无工具
    "C1": ["search_kb"],                       # 单工具
    "C2": ["search_kb", "get_page", "calculator", "list_games"],
}



async def run_case(case, *, label: str, judge: Judge, llm, schemas, ctx,
                   single_shot: bool, sem: asyncio.Semaphore) -> dict:
    """跑单条 case。并发安全：只读共享的 llm/retriever，不写任何共享可变状态。"""
    s = get_settings()
    agent = ReActAgent(settings=s, llm=llm, tools=schemas)
    # 证据单独取一份喂给 Judge。C0 没调工具，所以「它看到的证据」就是空 ——
    # 这正是要测的：没证据时它会编。
    cfg = RetrievalConfig(game=case.game, final_top_k=5)
    async with sem:
        t0 = time.perf_counter()
        try:
            ev = await ctx.retriever.retrieve(case.question, cfg)
            contexts = contexts_from_hits(ev)
        except Exception:  # noqa: BLE001
            contexts = ""

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
    return {
        "case_id": case.case_id, "category": case.category,
        "correctness": d["correctness"], "faithfulness": d["faithfulness"],
        "completeness": d["completeness"], "judge_score": d["judge_score"],
        "judge_reason": d["reason"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "steps": steps, "tool_failures": failures,
        "has_refs": bool(refs), "latency_ms": dt,
        "error": err, "answer": answer[:500],
    }


async def run_one(label: str, case_list, judge: Judge, *, concurrency: int = 2) -> dict:
    llm = get_llm()
    rows = []

    allowed = set(PRESETS[label])
    schemas = [sc for sc in get_schemas() if sc["function"]["name"] in allowed]
    # 没给工具时，直接走「单轮问答」而不是 ReAct 循环 ——
    # 否则 Agent 收不到 tool_calls 声明会反复尝试调用。
    single_shot = label == "C0"

    # ① 先串行把 ToolContext 建好：第一次会建 BM25 索引（~1.8s），
    #    之后全部命中进程级缓存（~17ms）。放在并发里会让 15 条同时抢着建。
    for case in case_list:
        ToolContext.build(case.game)

    # ② 再并发跑题。并发只改"什么时候算"，不改"算出什么"：
    #    检索是纯读、生成 temperature=0、Judge 输入互不相干。
    # 用 as_completed 边跑边打，而不是 gather 跑完一次性打。
#
# 15 条时无所谓，56 条（三库全量）就是好几分钟的完全静默 ——
# 后台任务看不出是在跑还是卡死了。**可观测性缺失会让「慢」和「挂」无法区分**，
# 而这两者的处置完全不同（等 vs 中止）。
    sem = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        run_case(case, label=label, judge=judge, llm=llm, schemas=schemas,
                 ctx=ToolContext.build(case.game), single_shot=single_shot, sem=sem)
        for case in case_list
    ]
    # 按 case_list 顺序回填，保证 rows 与输入顺序一致
    # （聚合与题型矩阵都依赖这个顺序）
    ordered: list[dict] = []
    for done, coro in enumerate(asyncio.as_completed(tasks), start=1):
        r = await coro
        ordered.append(r)
        print(f"  [{label}] {r['case_id']:<8} score={r['judge_score']:<6} "
              f"corr={r['correctness']:<4} faith={r['faithfulness']:<4} "
              f"steps={r['steps']} tok={r['prompt_tokens']:>6} {r['latency_ms']:>6}ms "
              f"({done}/{len(case_list)})", flush=True)
    # case_id 必须唯一，否则下面按 id 回填顺序时会静默取到别的 case 的行。
    # 守卫在 test_exp_common_counting.py 里。
    rows = [next(x for x in ordered if x["case_id"] == c.case_id) for c in case_list]

    # 聚合口径统一在 exp_common.aggregate_rows —— 三个实验必须用同一份，否则数字不可比
    return {
        "label": label, "tools": PRESETS[label], "concurrency": concurrency,
        "aggregate": aggregate_rows(rows), "rows": rows,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="C0,C1,C2")
    ap.add_argument("--limit", type=int, default=0, help="0 = 用代表性子集")
    ap.add_argument("--game", default="sr",
                    help="目标库：sr / ys / zzz，可逗号分隔或 all")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="同档内并发跑几条题。1=串行。默认 2（plan.md 原设计）。"
                         "注意：并发只影响墙钟，不影响质量指标；但 latency_ms 会因排队变长，"
                         "**不能和串行跑的延迟数字直接比**")
    ap.add_argument("--out", default=None, help="结果文件名；不给则按样本量自动命名")
    args = ap.parse_args()

    case_list = load_cases(args.limit, game=args.game)
    print(f"评测子集：{len(case_list)} 条，标签 {args.labels}，并发 {args.concurrency}")
    # 题型构成必须打出来：三库题型分布差得很远（ys 可用题里 7/9 是 single_hop），
    # 不看构成就比三库 judge 均值，等于把「题型构成差」读成「Agent 能力差」。
    print(f"题型构成：{subset_composition(case_list)}")
    if len(parse_games(args.game)) > 1:
        print("  ⚠ 多库题混跑：各库题型构成不同（见上），**跨库均值不可直接比较**，"
              "只能看同一库内的档位对比。")
    if args.concurrency > 1:
        print("  ⚠ 并发下 latency_ms 含排队时间，只用于本次内部参考，"
              "跨并发度的延迟数字不可比。\n")

    results = []
    judge = Judge(get_llm())
    for label in args.labels.split(","):
        print(f"=== 运行 {label}（工具：{PRESETS.get(label, '?')}）===", flush=True)
        results.append(await run_one(label.strip().upper(), case_list, judge,
                                     concurrency=args.concurrency))

    # 输出名可指定：否则 --limit 4 这种试跑会静默覆盖掉正式的 15 条结果。
    # （和 exp_b 的 B0 读旧文件是同一类病：产物路径写死，跑错一次就没了）
    #
    # 单库时沿用历史命名 exp_C.json（n=15），多库时带库名 ——
    # 用 `len(case_list) == 15` 判断单库在多库下会误判：
    # 恰好 15 条的多库子集会被写成 exp_C.json，直接覆盖单库的历史产物。
    games = parse_games(args.game)
    if args.out:
        out = args.out
    elif len(games) > 1:
        out = f"exp_C_{'_'.join(games)}.json"
    else:
        out = f"exp_C{'' if len(case_list) == 15 else f'_{len(case_list)}'}.json"

    # 题型构成写进产物：三库题型分布差很多，后续汇总必须能看出
    # 「三库均值差」里有多少来自题型构成而不是被测系统。
    results[0]["_meta"] = {
        "script": "exp_c.py",
        "experiment": "C",
        "game": args.game,
        "games": games,
        "n": len(case_list),
        "subset_composition": subset_composition(case_list),
        "concurrency": args.concurrency,
        "note": "concurrency>1 时 latency_ms 含排队时间，跨并发度不可比",
    }
    save(results, out)
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
