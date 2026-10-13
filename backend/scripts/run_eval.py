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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import get_settings  # noqa: E402
from app.eval import metrics as M  # noqa: E402
from app.eval.cases import cases_for  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig  # noqa: E402
from app.store.vector_store import get_vector_store  # noqa: E402
from exp_common import build_meta, is_kept, parse_games  # noqa: E402

logger = logging.getLogger("eval")

PRESETS: dict[str, RetrievalConfig] = {
    "A0": RetrievalConfig(use_bm25=True, use_vector=False, use_rerank=False),    # 仅 BM25
    "A1": RetrievalConfig(use_bm25=False, use_vector=True, use_rerank=False),    # 仅向量
    "A2": RetrievalConfig(use_bm25=True, use_vector=True, use_rerank=False),     # RRF 融合
    "A3": RetrievalConfig(use_bm25=True, use_vector=True, use_rerank=True),      # 再加重排
}

#: 有 gold_pages 的题才进检索统计。剩下的（纯算术 / 幻觉测试 / 跨库辨析）
#: 检索层无法判定对错，硬算 recall 只会得到 0，把所有配置一起拉低。
#: 但它们**不是不存在** —— 数量单列出来，避免"题好像变少了"这种无声消失。
NO_GOLD_EXEMPT = "no_gold_pages"


def build_config(label: str, game: str, top_k: int, variant: str,
                 embed_model: str | None, rerank_candidates: int) -> RetrievalConfig:
    cfg = RetrievalConfig(
        game=game, embed_model=embed_model, variant=variant, final_top_k=top_k,
        # 0 = 全量送重排（优化前的行为，也是本实验的对照组）
        rerank_candidates=rerank_candidates,
        **{k: v for k, v in vars(PRESETS[label]).items() if k.startswith("use_")},
    )
    cfg.min_revision_ts = None
    return cfg


def summarize(rows: list[dict], top_k: int) -> dict:
    """把一批 per-case 结果聚合成一行指标 —— 整体与分库**共用这一个实现**。

    为什么必须共用：早先均值与配对检验各有一份过滤逻辑，
    同一份产物因此打出两个均值（4.77 / 4.13）。分库汇总如果再写一份，
    就是同一个坑的第三次复现。
    """
    # 均值只用成功的 case —— 失败行是链路故障，不是检索能力差。
    # 与 exp_c/exp_e 共用同一个 is_kept 谓词，避免同一个项目两套口径。
    ok_rows = [r for r in rows if is_kept(r)]
    agg = M.aggregate(
        [{k: r[k] for k in r if k.startswith(("recall", "hit_rate", "mrr", "ndcg"))}
         for r in ok_rows]
    )
    # 全档检索失败时 aggregate 收到的是空列表，返回的 dict **没有指标键**，
    # 下游读 agg["mrr"] 会直接 KeyError —— 「全军覆没」不该表现得像「格式变了」。
    # 这里补齐键名，值为 0，配合下面的 n_ok=0 / errors=n 表示"一条都没成功"。
    for key in (f"recall@{top_k}", "hit_rate", "mrr", "ndcg"):
        agg.setdefault(key, 0.0)
    agg["page_hit_rate"] = round(
        sum(1 for r in ok_rows if r["page_hit"]) / len(ok_rows), 4
    ) if ok_rows else 0.0
    agg["avg_latency_ms"] = int(sum(r["latency_ms"] for r in ok_rows) / len(ok_rows)) if ok_rows else 0
    # n / n_ok / errors 一起给出：只看均值会以为失败的那些 case 从未存在
    agg["errors"] = len(rows) - len(ok_rows)
    agg["n"] = len(rows)
    agg["n_ok"] = len(ok_rows)
    return agg


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
    retriever=None,
) -> dict:
    """跑一个 label 的全部 case。

    `game` 可以是 `sr`，也可以是 `sr,ys,zzz` / `all`（多库）。

    `retriever` 是注入点：默认 None 时按生产路径自建 HybridRetriever，
    测试则传入可控替身，用来验证「检索失败时会发生什么」。
    故障注入没法靠 mock.patch 私有调用链完成，而这条路径恰恰最需要守卫 ——
    早先就是「失败即消失」把各档样本集弄得不一致。
    """
    s = get_settings()
    llm = get_llm()
    # 清掉 embedding / rerank 缓存：A0 只用 BM25 但 A1/A2/A3 共用同一条 query 的
    # embedding，若不清理，后跑的 label 会白捡前一个 label 的缓存，
    # 延迟列就不是同一条件下的可比数字了（指标不受影响，缓存是确定性的）。
    llm.clear_caches()
    if retriever is None:
        retriever = HybridRetriever(settings=s, llm=llm, store=get_vector_store(s))

    games = parse_games(game)
    rows: list[dict] = []
    no_gold = 0
    configs: dict[str, dict] = {}
    for g in games:
        cfg = build_config(label, g, top_k, variant, embed_model, rerank_candidates)
        configs[g] = cfg.as_dict()
        for case in cases_for(g):
            # 没有 gold_pages 的 case（纯算术 / 幻觉测试 / 跨游戏辨析）无法判定检索召回，
            # 计入均值会人为拉低所有配置，所以只在生成侧评测里用。
            if not case.gold_pages:
                no_gold += 1
                continue
            t0 = time.perf_counter()
            try:
                hits = await retriever.retrieve(case.question, cfg)
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] %s 检索失败: %s", label, case.case_id, exc)
                # 记一条失败，而不是让这条 case 直接消失。
                #
                # 早先这里是 `continue` 且不落任何行，问题有两层：
                # 1. 若 A0 的某条 case 检索失败，A0 的 rows 就少一条，A1/A2/A3 却是完整的
                #    —— **各档样本集不同**，均值不再是同一批 case 上的对比，配对检验也失真；
                # 2. 失败只在日志里 error，聚合结果里看不出少了 case，等于静默丢数据。
                #
                # 指标键名（recall@k 等）由 RetrievalMetrics.as_dict() 统一生成，
                # 这里直接复用它，不手写字段名 —— 手写就会和 k 的动态命名脱钩。
                rows.append(
                    {
                        **M.RetrievalMetrics(0.0, 0.0, 0.0, 0.0, top_k).as_dict(),
                        "case_id": case.case_id,
                        "game": g,
                        "category": case.category,
                        "page_hit": False,
                        "latency_ms": int((time.perf_counter() - t0) * 1000),
                        "top_titles": [],
                        "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                    }
                )
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
                    "game": g,
                    "category": case.category,
                    "page_hit": M.page_level_hit(ids, case.gold_pages, page_of),
                    "latency_ms": dt,
                    "top_titles": [h.chunk.page_title for h in hits[:3]],
                }
            )

    # 分库聚合。三库的语料规模差 2.7 倍（sr 706 页 / ys 435 / zzz 260），
    # 合成一个均值会把「语料多所以好召回」和「策略真的更好」搅在一起 ——
    # 所以 by_game 是必看项，整体均值只当总览。
    by_game = {}
    for g in games:
        grows = [r for r in rows if r.get("game") == g]
        if grows:
            by_game[g] = summarize(grows, top_k)

    agg = summarize(rows, top_k)
    agg[NO_GOLD_EXEMPT] = no_gold
    # 单库时 config 就是那一个库的，多库时另给 configs，
    # 但 config 仍保留 —— 落库和 exp_b 的 `is_stale()` 都读它。
    return {
        "label": label,
        "games": games,
        "config": configs[games[0]] if games else {},
        "configs": configs,
        "aggregate": agg,
        "by_game": by_game,
        "rows": rows,
    }


async def persist_results(args, results: list[dict]) -> None:
    """评测集 + 每条 case 的得分一起落 MySQL。"""
    from app.store.db import init_db
    from app.store.persist import save_eval_cases, save_eval_run

    if not await init_db():
        print("MySQL 不可用，跳过落库（结果仍已写入 JSON）")
        return

    games = parse_games(args.game)
    n_cases = 0
    try:
        for g in games:
            n_cases += await save_eval_cases(g)
    except Exception as exc:  # noqa: BLE001
        print(f"评测集落库失败（不影响结果）：{type(exc).__name__}: {exc}")
        return

    total = 0
    for r in results:
        # config 按库分组落：eval_results.config 是 JSON 列，
        # 三库跑出来的 config 里 game 字段不同，混成一份就分不出是哪库的分数。
        # 单库时 suffix 为空，exp_label 与改造前完全一致，历史产物仍可比。
        configs = r.get("configs") or {"": r["config"]}
        multi = len(configs) > 1
        for g, cfg in configs.items():
            rows = [dict(x, k=args.top_k) for x in r["rows"] if x.get("game") == g]
            if not rows:
                continue
            total += await save_eval_run(
                exp_label=f"A{r['label']}_{g}" if multi else f"A{r['label']}",
                config=cfg, rows=rows,
            )
    print(f"MySQL：评测集 {n_cases} 条 + 评测结果 {total} 行已落库")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="A", choices=["A"])
    ap.add_argument("--game", default="sr",
                    help="要评测的库：sr / ys / zzz，可逗号分隔多库，或 all")
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
    games = parse_games(args.game)
    if not games:
        raise SystemExit("--game 没有解析出任何库")
    # 每个库都要单独查 —— 只查第一个库的话，ys/zzz 的 collection 缺失
    # 要一直跑到那一步才炸，而 rerank / embed 请求已经发出去了。
    for g in games:
        n = store.count(g, args.embed_model, args.variant)
        if n == 0:
            raise SystemExit(
                f"collection 为空：{store.collection_name(g, args.embed_model, args.variant)}，"
                f"先跑入库（--game {g}）"
            )
        logger.info("库 %s：索引 %d chunk", g, n)

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

    def line(label: str, a: dict) -> str:
        return (
            f"{label:<6}{a['page_hit_rate']:>10.3f}"
            f"{a.get(f'recall@{args.top_k}', 0.0):>14.3f}"
            f"{a['mrr']:>8.3f}{a['ndcg']:>8.3f}{a['avg_latency_ms']:>9}{a['n']:>4}"
        )

    header = (f"{'配置':<6}{'page_hit':>10}{f'recall@{args.top_k}':>14}"
              f"{'mrr':>8}{'ndcg':>8}{'延迟ms':>9}{'n':>4}")

    print("\n=== 实验 A：检索策略阶梯消融（单变量）===")
    if len(games) > 1:
        # 分库小节优先。三库语料规模差 2.7 倍（sr 706 / ys 435 / zzz 260 页），
        # 混在一张表里算平均，等于把「语料多所以好召回」和「策略真的更好」
        # 搅进同一个数字 —— 这张总表只能当总览，不能用来下结论。
        for g in games:
            print(f"\n--- [{g}] ---")
            print(header)
            for r in results:
                a = r["by_game"].get(g)
                if a:
                    print(line(r["label"], a))
        print(f"\n--- 合计（{'/'.join(games)}，口径混合，仅作总览）---")
        print(header)
        for r in results:
            print(line(r["label"], r["aggregate"]))
        exempt = results[0]["aggregate"].get(NO_GOLD_EXEMPT, 0)
        if exempt:
            print(f"\n注：有 gold_pages 的题才进上表；另有 {exempt} 条无 gold 题"
                  f"（幻觉测试 / 纯算术 / 跨库辨析）不参与检索统计，只在生成侧评测里用。")
    else:
        print(header)
        for r in results:
            print(line(r["label"], r["aggregate"]))
        exempt = results[0]["aggregate"].get(NO_GOLD_EXEMPT, 0)
        if exempt:
            print(f"\n注：另有 {exempt} 条无 gold 题（幻觉测试 / 纯算术 / 跨库辨析）"
                  f"不参与检索统计。")

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
        if len(games) > 1:
            # 分库再检一次。合并跑的配对检验里，case 间方差同时包含了
            # 「题目难度差异」和「库本身的召回难度差异」（sr 706 页 vs zzz 260 页），
            # 两者混在一个 sd_diff 里，检验的功效被稀释 ——
            # 合并不显著不代表每库都不显著，反之亦然。
            print("\n=== 分库配对 t 检验（同上，仅本库的配对样本）===")
            by_game_sig: dict[str, list[dict]] = {g: [] for g in games}
            for g in games:
                print(f"\n--- [{g}] ---")
                print(f"{'对比':<16}{'Δmrr(后-前)':>13}{'t':>8}{'p':>8}{'Cohen d':>9}{'显著':>7}")
                for i in range(1, len(results)):
                    prev_g, cur_g = results[i - 1], results[i]
                    # 命名带 _g 是为了和上面合并跑的 prev/cur 区分开：
                    # 两个循环里的 prev/cur 含义一样，但筛出的行不同，
                    # 混用名字会让人以为传错了档。
                    prows = [x for x in prev_g["rows"] if x.get("game") == g]
                    crows = [x for x in cur_g["rows"] if x.get("game") == g]
                    t = M.paired_test(crows, prows, "mrr")
                    by_game_sig[g].append(
                        {"comparison": f"{prev_g['label']}→{cur_g['label']}", **t}
                    )
                    if t.get("n", 0) >= 3:
                        print(f"{prev_g['label'] + '→' + cur_g['label']:<16}{t['mean_diff']:>9.4f}"
                              f"{t['t']:>8.2f}{t['p']:>8.4f}{t['cohens_d']:>9.3f}"
                              f"{'是' if t['significant'] else '否':>7}")
                    else:
                        # n 小到连检验都不成立时必须直说，不能让上面那张表里的
                        # 「显著=否」被读成「检验做了，没显著」。
                        reason = f"样本不足（n={t.get('n', 0)}），无法检验"
                        print(f"{prev_g['label'] + '→' + cur_g['label']:<16}{reason:>40}")
            results[0]["significance_by_game"] = by_game_sig

    default_name = "exp_A.json" if len(games) == 1 else f"exp_A_{'_'.join(games)}.json"
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / "data" / "eval" / default_name
    out.parent.mkdir(parents=True, exist_ok=True)

    # 产物必须自己记录**当时那个索引的规模**。
    #
    # 为什么：2026-10-13 又踩了一次静默混入两个时期数据的坑 ——
    # 补抓角色页后 sr 索引从 2306 涨到 3932 chunk，跑 `exp_b.py --skip-rebuild`
    # 时 B0 的产物文件还是补抓前写的，汇总表里 B0 显示 0.966 而 B1/B2/B3 是新数字，
    # **一行旧数据混在一张新表里，没有任何报错**。
    #
    # 靠文件 mtime 判不出来（碰一下就变），所以规模直接写进产物里，
    # 汇总方拿它和 collection 的实际行数一比就知道这份产物过没过期。
    meta = build_meta(args)
    results[0]["_meta"] = meta
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if meta.get("index_by_game"):
        scale = "，".join(
            f"{g} {v['index_chunks']} chunk / {v['index_pages']} 页"
            for g, v in meta["index_by_game"].items()
        )
        print(f"\n结果已写入 {out}（{scale} @ {meta['generated_at']}）")
    else:
        print(f"\n结果已写入 {out}（索引 {meta['index_chunks']} chunk / "
              f"{meta['index_pages']} 页 @ {meta['generated_at']}）")

    # 同时落 MySQL：eval_cases / eval_results 这两张表只有接上写入才不是摆设，
    # 落库之后才能做"改了分块参数再跑一遍，直接对比两次差多少"的回归。
    if not args.no_persist:
        await persist_results(args, results)


if __name__ == "__main__":
    asyncio.run(main())