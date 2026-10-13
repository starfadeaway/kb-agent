"""检索侧消融实验（实验 A：A0→A3 阶梯）。

用法:
    .venv\\Scripts\\python.exe backend\\scripts\\run_eval.py --experiment A

严格单变量：A0→A1 只开向量，A1→A2 只加 BM25，A2→A3 只加重排。
每一步只动一个组件，差值才可归因。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.eval import metrics as M  # noqa: E402
from app.eval.cases import cases_for  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig  # noqa: E402
from app.store.vector_store import get_vector_store  # noqa: E402

logger = logging.getLogger("eval")

PRESETS: dict[str, RetrievalConfig] = {
    "A0": RetrievalConfig(use_bm25=True, use_vector=False, use_rerank=False),    # 仅 BM25
    "A1": RetrievalConfig(use_bm25=False, use_vector=True, use_rerank=False),    # 仅向量
    "A2": RetrievalConfig(use_bm25=True, use_vector=True, use_rerank=False),     # RRF 融合
    "A3": RetrievalConfig(use_bm25=True, use_vector=True, use_rerank=True),      # 再加重排
}


@dataclass
class CaseOutcome:
    case_id: str
    label: str
    recall: float
    hit: float
    mrr: float
    ndcg: float
    latency_ms: int
    top_titles: list[str]


async def run_one(
    label: str, game: str, top_k: int, variant: str,
    embed_model: str | None, rerank_candidates: int = 0,
) -> dict:
    s = get_settings()
    llm = get_llm()
    # 清掉 embedding / rerank 缓存：A0 只用 BM25 但 A1/A2/A3 共用同一条 query 的
    # embedding，若不清理，后跑的 label 会白捡前一个 label 的缓存，
    # 延迟列就不是同一条件下的可比数字了（指标不受影响，缓存是确定性的）。
    llm.clear_caches()
    retriever = HybridRetriever(settings=s, llm=llm, store=get_vector_store(s))
    cfg = RetrievalConfig(
        game=game, embed_model=embed_model, variant=variant, final_top_k=top_k,
        # 0 = 全量送重排（优化前的行为，也是本实验的对照组）
        rerank_candidates=rerank_candidates,
        **{k: v for k, v in vars(PRESETS[label]).items() if k.startswith("use_")},
    )
    cfg.min_revision_ts = None

    rows: list[dict] = []
    for case in cases_for(game):
        # 没有 gold_pages 的 case（纯算术 / 幻觉测试 / 跨游戏辨析）无法判定检索召回，
        # 计入均值会人为拉低所有配置，所以只在生成侧评测里用。
        if not case.gold_pages:
            continue
        t0 = time.perf_counter()
        try:
            hits = await retriever.retrieve(case.question, cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] %s 检索失败: %s", label, case.case_id, exc)
            continue
        dt = int((time.perf_counter() - t0) * 1000)
        ids = [h.chunk_id for h in hits]
        page_of = {h.chunk_id: h.chunk.page_title for h in hits}
        # gold 标的是页面，所以指标必须在页面级排名上算，
        # 否则永远是 0（拿 chunk_id 去比页面标题）
        page_rank = M.page_ranking(ids, page_of)
        m = M.evaluate_pages(page_rank, case.gold_pages, k=top_k)
        rows.append(
            {
                **m.as_dict(),
                "case_id": case.case_id,
                "page_hit": M.page_level_hit(ids, case.gold_pages, page_of),
                "latency_ms": dt,
                "top_titles": [h.chunk.page_title for h in hits[:3]],
            }
        )

    agg = M.aggregate(
        [{k: r[k] for k in r if k.startswith(("recall", "hit_rate", "mrr", "ndcg"))} for r in rows]
    )
    agg["page_hit_rate"] = round(
        sum(1 for r in rows if r["page_hit"]) / len(rows), 4
    ) if rows else 0.0
    agg["avg_latency_ms"] = int(sum(r["latency_ms"] for r in rows) / len(rows)) if rows else 0
    return {"label": label, "config": cfg.as_dict(), "aggregate": agg, "rows": rows}


async def persist_results(args, results: list[dict]) -> None:
    """评测集 + 每条 case 的得分一起落 MySQL。"""
    from app.store.db import init_db
    from app.store.persist import save_eval_cases, save_eval_run

    if not await init_db():
        print("MySQL 不可用，跳过落库（结果仍已写入 JSON）")
        return

    try:
        n_cases = await save_eval_cases(args.game)
    except Exception as exc:  # noqa: BLE001
        print(f"评测集落库失败（不影响结果）：{type(exc).__name__}: {exc}")
        return

    total = 0
    for r in results:
        rows = [dict(x, k=args.top_k) for x in r["rows"]]
        total += await save_eval_run(exp_label=f"A{r['label']}", config=r["config"], rows=rows)
    print(f"MySQL：评测集 {n_cases} 条 + 评测结果 {total} 行已落库")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="A", choices=["A"])
    ap.add_argument("--game", default="sr")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--labels", default="A0,A1,A2,A3")
    ap.add_argument("--rerank-candidates", type=int, default=0,
                    help="送入重排的候选上限；0=全量（对照组）。与 --no-persist 配合可只比指标不落库")
    ap.add_argument("--no-persist", action="store_true", help="只写 JSON 不落 MySQL")
    ap.add_argument("--out", default=None, help="结果 JSON 落盘路径")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx / openai / httpcore 默认也是 INFO，会把每一次 embedding/rerank 请求
    # 都打出来 —— 29 道题 × 4 配置就是几百行噪音，真正要看的结论被埋掉了。
    for noisy in ("httpx", "httpcore", "openai", "lance", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    s = get_settings()
    store = get_vector_store(s)
    n = store.count(args.game, args.embed_model, args.variant)
    if n == 0:
        raise SystemExit(f"collection 为空：{store.collection_name(args.game, args.embed_model, args.variant)}，先跑入库")

    results = []
    for label in args.labels.split(","):
        label = label.strip()
        logger.info("=== 运行 %s ===", label)
        r = await run_one(label, args.game, args.top_k, args.variant, args.embed_model,
                          args.rerank_candidates)
        results.append(r)
        a = r["aggregate"]
        logger.info(
            "%s: page_hit=%.3f recall@%d=%.3f mrr=%.3f ndcg=%.3f 平均延迟=%dms",
            label, a["page_hit_rate"], args.top_k, a.get(f"recall@{args.top_k}", 0),
            a["mrr"], a["ndcg"], a["avg_latency_ms"],
        )

    table = [
        {
            "label": r["label"],
            "page_hit": r["aggregate"]["page_hit_rate"],
            f"recall@{args.top_k}": r["aggregate"].get(f"recall@{args.top_k}"),
            "mrr": r["aggregate"]["mrr"],
            "ndcg": r["aggregate"]["ndcg"],
            "latency_ms": r["aggregate"]["avg_latency_ms"],
            "n": r["aggregate"]["n"],
        }
        for r in results
    ]
    print("\n=== 实验 A：检索策略阶梯消融（单变量）===")
    print(f"{'配置':<6}{'page_hit':>10}{f'recall@{args.top_k}':>14}{'mrr':>8}{'ndcg':>8}{'延迟ms':>9}{'n':>4}")
    for row in table:
        print(
            f"{row['label']:<6}{row['page_hit']:>10.3f}{row[f'recall@{args.top_k}']:>14.3f}"
            f"{row['mrr']:>8.3f}{row['ndcg']:>8.3f}{row['latency_ms']:>9}{row['n']:>4}"
        )

    # ---- 配对显著性检验 ----
    # 只报两两相邻的两级（A0→A1→A2→A3）。均值差本身不能说明差异是真的：
    # 消融实验跑的是同一批题，case 难度方差极大，配对检验先消掉这个方差源。
    if len(results) >= 2:
        print("\n=== 配对 t 检验（metric=mrr，与上一级比）===")
        print(f"{'对比':<16}{'Δmrr(后-前)':>13}{'t':>8}{'p':>8}{'Cohen d':>9}{'显著':>7}")
        sig_rows = []
        for i in range(1, len(results)):
            prev, cur = results[i - 1], results[i]
            # 传 (当前, 上一档)：Δ 与 d 都是「后减前」，正数 = 这次改进是正收益。
            t = M.paired_test(cur["rows"], prev["rows"], "mrr")
            sig_rows.append({"comparison": f"{prev['label']}→{cur['label']}", **t})
            if t.get("n", 0) >= 3:
                print(f"{prev['label'] + '→' + cur['label']:<16}{t['mean_diff']:>9.4f}"
                      f"{t['t']:>8.2f}{t['p']:>8.4f}{t['cohens_d']:>9.3f}"
                      f"{'是' if t['significant'] else '否':>7}")
            else:
                print(f"{prev['label'] + '→' + cur['label']:<16}{'样本不足，跳过':>40}")
        results[0]["significance"] = sig_rows

    out = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / "data" / "eval" / "exp_A.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")

    # 同时落 MySQL：eval_cases / eval_results 这两张表只有接上写入才不是摆设，
    # 落库之后才能做"改了分块参数再跑一遍，直接对比两次差多少"的回归。
    if not args.no_persist:
        await persist_results(args, results)


if __name__ == "__main__":
    asyncio.run(main())