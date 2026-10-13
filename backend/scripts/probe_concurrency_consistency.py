"""生成侧并发一致性实证：换一个并发度，评测指标会不会变？

为什么必须实测而不是推断
------------------------
`docs/concurrency.md` 已经证明了**检索侧**在 1/2/4/8 并发下
`top5 chunk_id` 逐条一致。但那只覆盖了检索 —— 生成侧（ReAct 循环 +
LLM 调用 + Judge）没有对应的证据，而所有生成类实验（C/E）都是并发跑的。

「温度已经是 0，所以并发不会影响结果」是一个**看起来合理但未经检验**的假设：
temperature=0 只消除了采样随机性，并不保证浮点非确定性、批处理组合、
缓存命中与否都不会改变输出。评测 Runner 的并发顺序效应这个坑，
本项目已经吃过一次（同一份代码两次跑出相反的符号）。

这个探针直接复用 `exp_e.run_one`（生产路径本身，不抄副本），
在 concurrency=1 与 concurrency=N 下各跑一遍同一批 case，然后逐条对比。

产出 `data/eval/probe_concurrency_consistency.json`。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.eval.judge import Judge  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from exp_common import EVAL_DIR, aggregate_rows  # noqa: E402
from exp_e import (
    is_counted,  # noqa: E402
    load_cases,  # noqa: E402
    run_one,  # noqa: E402
)


def _cmp(a: dict, b: dict) -> dict:
    """逐字段比较同一 case 在两次运行下的差异。"""
    out = {}
    for k in ("judge_score", "correctness", "faithfulness", "steps", "prompt_tokens"):
        va, vb = a.get(k), b.get(k)
        out[k] = {"v1": va, "v2": vb, "same": va == vb}
    out["answer_same"] = a.get("answer", "") == b.get("answer", "")
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="E1", help="跑哪一档（默认 E1=DeepSeek）")
    ap.add_argument("--concurrency", type=int, default=2, help="对照用的并发度")
    ap.add_argument("--limit", type=int, default=0, help="0=全部 15 条")
    ap.add_argument("--out", default="probe_concurrency_consistency.json")
    args = ap.parse_args()

    cases = load_cases(args.limit)
    print(f"评测子集 {len(cases)} 条；对照 concurrency=1 vs {args.concurrency}；档位 {args.label}\n")

    results = {}
    for conc in (1, args.concurrency):
        print(f"=== concurrency={conc} ===", flush=True)
        t0 = time.perf_counter()
        # 每次跑都用全新的 Judge，但复用 exp_e 的生产聚合路径
        judge = Judge(get_llm())
        res = await run_one(args.label, cases, judge, concurrency=conc)
        results[conc] = res
        print(f"  用时 {time.perf_counter() - t0:.1f}s\n", flush=True)

    r1, rn = results[1], results[args.concurrency]
    m1 = {r["case_id"]: r for r in r1["rows"]}
    mn = {r["case_id"]: r for r in rn["rows"]}
    common = sorted(set(m1) & set(mn))

    rows = []
    for cid in common:
        rows.append({"case_id": cid, **_cmp(m1[cid], mn[cid])})

    agg1 = aggregate_rows(r1["rows"], exclude=is_counted)
    aggn = aggregate_rows(rn["rows"], exclude=is_counted)

    def flag(key):
        return sum(1 for r in rows if r[key]["same"])

    report = {
        "_meta": {
            "script": "probe_concurrency_consistency.py",
            "label": args.label,
            "concurrencies": [1, args.concurrency],
            "n_cases": len(common),
            "note": "同一批 case、同一档模型，唯一变量 = 并发度；temperature 冻结为 0",
        },
        "aggregate": {
            "concurrency_1": agg1,
            f"concurrency_{args.concurrency}": aggn,
        },
        "agreement": {
            "judge_score": f"{flag('judge_score')}/{len(rows)}",
            "correctness": f"{flag('correctness')}/{len(rows)}",
            "faithfulness": f"{flag('faithfulness')}/{len(rows)}",
            "steps": f"{flag('steps')}/{len(rows)}",
            "prompt_tokens": f"{flag('prompt_tokens')}/{len(rows)}",
            "answer_identical": f"{sum(1 for r in rows if r['answer_same'])}/{len(rows)}",
        },
        "rows": rows,
    }

    print("=" * 84)
    print(f"  逐条一致性（concurrency=1 vs {args.concurrency}）")
    print("=" * 84)
    for k, v in report["agreement"].items():
        print(f"  {k:<18} {v}")
    print(f"\n  aggregate judge_score: c1={agg1['judge_score']}  "
          f"c{args.concurrency}={aggn['judge_score']}")

    diff = [r for r in rows if not r["judge_score"]["same"]]
    if diff:
        print(f"\n  judge 分不一致的 case（{len(diff)} 条）：")
        for r in diff:
            print(f"    {r['case_id']}  c1={r['judge_score']['v1']}  "
                  f"c{args.concurrency}={r['judge_score']['v2']}  "
                  f"answer_same={r['answer_same']}")
    else:
        print("\n  ✅ judge 分逐条完全一致")

    out = EVAL_DIR / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())