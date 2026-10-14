"""实验 F：披露层消融（页面修订时间 → 模型有没有真的用上）。

## 这个实验能回答什么，不能回答什么

**能回答**：把工具输出里的页面修订日期、提示词里教模型用日期的那一条
全部撤掉之后，模型的行为有什么变化。

**不能回答**（这一点必须写在结论里，否则这个实验会被误引用）：
披露层有没有让回答**更准**。

原因在 `docs/version_staleness.md` 第三、四节：当前语料每页只有最新版快照，
没有历史版本可比对，「这个数值现在还对吗」这个问题在本语料上根本无法验证。
所以本实验**不跑 judge**，全部用客观判据（`app/eval/provenance_score.py`）。

## 为什么 F1 的日期复述率必然接近 0，而这不是结论

F1 关掉披露后，工具输出里**一个日期都没有**，提示词也删掉了那条要求。
于是「答案里没有日期」是机械必然，不是模型能力差。

> F1 的 date 指标只能读作「信息有没有被送到答案里」，
> **不能**读作「加披露层之后模型变强了」。

真正有信息量的是另外三个指标，它们在 F1 档同样有意义：

- `has_revision_wording_rate` —— 模型会不会说出「修订」这个概念；
- `has_staleness_wording_rate` —— 模型会不会主动声明「可能已过时」；
- `declares_uncertainty_rate` —— selfrate 题上模型会不会承认自己没把握。

后两个尤其重要：它们正是 `docs/version_staleness.md` 第五节
聚类实测所指的那条**原理上就不该由页面修订时间推出**的推断。
模型给出的任何这类声明都只能读作「它察觉到了不确定性」，
**不能**读作「它判断对了」。

## 冻结清单（不冻结就不是单变量）

  Chat 基座 / temperature / tools schema / 检索配置 / 评测题集 / gold 日期来源
  全部冻结。唯一变量是 `ToolContext.disclose_revision`，
  它同时驱动两处（`search_kb`+`get_page` 的输出、`build_system_prompt` 的条款）。

用法：
    .venv\\Scripts\\python.exe scripts\\exp_f.py
    .venv\\Scripts\\python.exe scripts\\exp_f.py --game sr --limit 2
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent.react import ReActAgent  # noqa: E402
from app.agent.tools import ToolContext, get_schemas  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.provenance_cases import cases_for, kind_composition  # noqa: E402
from app.eval.provenance_score import aggregate_rows, score_case  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from exp_common import build_meta, parse_games, save  # noqa: E402
from sqlalchemy import text  # noqa: E402

#: F1 = 消融档，F2 = 生产档。名字里的 F 跟实验名一致，
#: 别改成 on/off —— 产物落盘后标签就是唯一的口径线索。
VARIANTS = {"F1": False, "F2": True}


async def load_gold_dates(games: list[str]) -> dict[str, str]:
    """`页面标题 -> 修订日期`，直接从 MySQL 现读。

    **为什么不把 gold 日期写进题集**：页面一旦被重抓，
    手打常量就成了一个早已不存在的日期，而题目看起来完全正常、
    判据照常打分，最后报出来的是「模型日期复述率 0%」——
    实际是 gold 过期了。现读则保证 gold 与工具输出同源，
    测的确实是「有没有忠实复述工具给的那个日期」。
    """
    from app.store.db import session_scope

    out: dict[str, str] = {}
    async with session_scope(get_settings()) as s:
        for g in games:
            rows = (await s.execute(
                text("SELECT title, last_revision_ts FROM pages "
                     "WHERE game = :g AND last_revision_ts <> ''"),
                {"g": g},
            )).all()
            for title, ts in rows:
                # 按标题记，不按库记：多库同标题时后写的覆盖先写的，
                # 但溯源题的目标页都是各库内部唯一的，实际不会撞。
                out[str(title)] = str(ts)[:10]
    return out


async def run_case(case, *, disclose: bool, gold_dates: dict[str, str],
                   tools, sem: asyncio.Semaphore, full_answer: bool = True) -> dict:
    """跑单条 case。

    刻意**不走 judge**：judge 的三个维度里没有时效敏感度，
    用它评只会得到一个与披露层无关的分数，
    再拿那个分数论证「披露层有效」是循环论证。

    两档看到完全一样的检索结果（同一个 `ToolContext` 只换 disclose_revision），
    所以 F1/F2 的差异只能归因到披露层本身。
    """
    ctx = dataclasses.replace(ToolContext.build(case.game), disclose_revision=disclose)

    async with sem:
        t0 = time.perf_counter()
        err = None
        refs: list = []
        answer = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        steps = failures = 0
        try:
            st = await ReActAgent(
                settings=get_settings(), llm=get_llm(), tools=tools
            ).run(case.question, ctx=ctx)
            answer, refs = st.answer, st.refs
            steps, failures = len(st.steps), st.tool_failures
            usage = {"prompt_tokens": st.prompt_tokens,
                     "completion_tokens": st.completion_tokens}
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:160]}"
        dt = int((time.perf_counter() - t0) * 1000)

    row = score_case(case, answer, refs, gold_dates)
    row.update({
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "steps": steps, "tool_failures": failures,
        "latency_ms": dt, "error": err,
        # 答案全文必须落盘：判据只有几条正则，看产物复核时需要原文，
        # 而截断到 500 字很可能刚好把末尾那句时效声明切掉。
        "answer": answer,
    })
    return row


async def run_one(label: str, cases, gold_dates, tools, *, concurrency: int = 2) -> dict:
    disclose = VARIANTS[label]
    # 先串行建好 ToolContext（首次建 BM25 ~1.8s，之后命中进程缓存），
    # 再并发跑题 —— 与 exp_c / exp_e 同一套并发策略。
    for c in cases:
        ToolContext.build(c.game)

    sem = asyncio.Semaphore(max(1, concurrency))
    rows = list(await asyncio.gather(*[
        run_case(c, disclose=disclose, gold_dates=gold_dates,
                 tools=tools, sem=sem)
        for c in cases
    ]))
    # case_id 必须唯一且按题集顺序回填，否则下面 next() 会静默取到别人的行
    rows = [next(r for r in rows if r["case_id"] == c.case_id) for c in cases]

    for r in rows:
        print(f"  [{label}] {r['case_id']:<12} 日期 {r['date_hits']}/{r['date_targets']}"
              f"  修订词={int(r['has_revision_wording'])}"
              f" 陈旧词={int(r['has_staleness_wording'])}"
              f" 存疑={int(r['declares_uncertainty'])}"
              f" 引用修订覆盖={r['ref_revision_coverage']}"
              f" tok={r['prompt_tokens']:>6} {r['latency_ms']:>6}ms"
              + (f"  ⚠ ERR {str(r['error'])[:70]}" if r["error"] else ""), flush=True)

    agg = aggregate_rows(rows)
    print(f"  [{label}] 逐页命中={agg['page_level_date_hit_rate']} "
          f"全对={agg['case_level_all_hit_rate']} "
          f"修订词={agg['has_revision_wording_rate']} "
          f"陈旧词={agg['has_staleness_wording_rate']} "
          f"存疑={agg['declares_uncertainty_rate']} "
          f"引用修订覆盖={agg['ref_revision_coverage']}")
    return {
        "label": label, "disclose_revision": disclose, "concurrency": concurrency,
        "aggregate": agg, "rows": rows,
    }


def print_report(results) -> None:
    print("\n" + "=" * 96)
    print("  实验 F：披露层消融（F1 关 / F2 开）")
    print("=" * 96)
    hdr = (f"{'档':<4}{'逐页日期命中':>14}{'全对率':>9}{'修订词':>9}"
           f"{'陈旧词':>9}{'存疑率':>9}{'引用修订覆盖':>13}{'延迟ms':>9}{'错误':>6}")
    print(hdr)
    for r in results:
        a = r["aggregate"]
        print(f"{r['label']:<4}{str(a['page_level_date_hit_rate']):>14}"
              f"{str(a['case_level_all_hit_rate']):>9}"
              f"{str(a['has_revision_wording_rate']):>9}"
              f"{str(a['has_staleness_wording_rate']):>9}"
              f"{str(a['declares_uncertainty_rate']):>9}"
              f"{a['ref_revision_coverage']:>13}{'':>9}"
              f"{sum(1 for x in r['rows'] if x['error']):>6}")

    print("\n--- 逐条 ---")
    for r in results:
        for x in r["rows"]:
            gold = x["gold_dates"]
            print(f"  {r['label']} {x['case_id']:<12} {x['kind']:<12} "
                  f"命中 {x['date_hits']}/{x['date_targets']}"
                  + (f"  漏={x['missed_pages']}" if x["missed_pages"] else "")
                  + (f"  gold={gold}" if gold else ""))

    print("\n⚠ 读这张表必须先接受三件事，否则会得出错误结论：")
    print("  1) F1 的工具输出里没有任何日期，所以「日期复述率低」是机械必然，")
    print("     测的是信息有没有送到答案，不是模型变强了；")
    print("  2) 本实验不评正确性。语料只有最新版快照，没有历史版本可比对，")
    print("     「这个数值现在还对吗」在本语料上无法验证"
          "（docs/version_staleness.md 第四节）；")
    print("  3) 「陈旧词」「存疑」只说明模型表达了这层意思，"
          "**不代表它的判断是对的** —— 页面修订时间推不出内容时效。")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", default="F1,F2")
    ap.add_argument("--game", default="sr,ys,zzz",
                    help="目标库：sr / ys / zzz，可逗号分隔或 all")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # build_meta 用 getattr 读这些字段，不给的话产物的 `_meta` 里
    # experiment/variant 会是 null —— 而溯源产物与其它实验放同一个目录，
    # 靠文件名以外的字段分辨来源正是 build_meta 存在的理由。
    args.experiment = "F"
    args.variant = "disclosure_ablation"
    args.top_k = None

    games = parse_games(args.game)
    cases = cases_for(games)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit(f"库 {args.game} 下没有溯源题")

    print(f"溯源题：{len(cases)} 条，库 {games}")
    print(f"题型构成：{kind_composition(cases)}")
    print("  ⚠ 逐页日期命中率的分母是**页数**不是 case 数；"
          "存疑率的分母只算 selfrate 题。跨库比均值前先看题型构成。\n")

    gold_dates = await load_gold_dates(games)
    missing = {p for c in cases for p in c.target_pages if not gold_dates.get(p)}
    if missing:
        # 缺 gold 不是「那条题算 0 分」，而是判据失效。
        # 继续跑会报出一个看起来正常的命中率，分母里混着不可判的题。
        raise SystemExit(
            f"以下目标页在 MySQL 里查不到修订日期，实验中止：{sorted(missing)}\n"
            "（gold 日期缺失时继续跑，报出来的是判据失效，不是模型没复述。）"
        )
    print(f"gold 日期已从 MySQL 现读，覆盖 {len(cases)} 条题的目标页。\n")

    tools = get_schemas()
    results = []
    for label in [x.strip().upper() for x in args.labels.split(",") if x.strip()]:
        if label not in VARIANTS:
            raise SystemExit(f"未知档 {label}，可选 {list(VARIANTS)}")
        print(f"=== 运行 {label}（disclose_revision={VARIANTS[label]}）===", flush=True)
        results.append(await run_one(label, cases, gold_dates, tools,
                                     concurrency=args.concurrency))
        if results:
            results[0]["_meta"] = build_meta(args)
            stem = (args.out or "exp_F.json").removesuffix(".json")
            # 中间产物另存：半截结果绝不能占用正式文件名，
            # 否则会被后续读取当成完整实验。
            save(results, f"{stem}.partial.json")

    out = args.out or "exp_F.json"
    results[0]["_meta"] = build_meta(args)
    save(results, out)
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())