"""检索指标。

只算检索层指标，不涉及生成 —— 这样实验 A/B（检索策略）和实验 C/D/E（生成侧）
可以分开归因，不会把"检索没召回"误判成"模型答得差"。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    hit_rate: float
    mrr: float
    ndcg: float
    k: int

    def as_dict(self) -> dict:
        return {
            f"recall@{self.k}": round(self.recall_at_k, 4),
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
        }


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """gold 命中比例。多 gold 时用比例而非 0/1，避免"全中才算对"的惩罚过重。"""
    if not gold_ids:
        return 0.0
    got = set(retrieved_ids[:k])
    return sum(1 for g in gold_ids if g in got) / len(gold_ids)


def hit_rate(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """是否有任意一条命中（二值，衡量"能不能答"的门槛）。"""
    if not gold_ids:
        return 0.0
    return 1.0 if any(g in set(retrieved_ids[:k]) for g in gold_ids) else 0.0


def mrr(retrieved_ids: list[str], gold_ids: list[str], k: int = 10) -> float:
    """首个命中排名的倒数，越靠前越好。"""
    if not gold_ids:
        return 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg(retrieved_ids: list[str], gold_ids: list[str], k: int = 10) -> float:
    """二值增益版 NDCG。"""
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved_ids[:k], start=1)
        if rid in gold
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def normalize(s: str) -> str:
    return s.strip().replace(" ", "").replace("_", "")


def page_ranking(chunk_ids: Sequence[str], page_of: dict[str, str]) -> list[str]:
    """把 chunk 级命中折算成页面级排名（同一页面只保留首次出现的位置）。

    评测集标的是页面而不是 chunk：
    chunk 边界会随 chunk_size 变化，绑死 chunk_id 等于让"重建索引"等价于"评测集失效"。
    折算成页面级排名后，实验 B 改分块参数时指标仍然可比。
    """
    seen: set[str] = set()
    out: list[str] = []
    for cid in chunk_ids:
        page = page_of.get(cid, "")
        key = normalize(page)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(page)
    return out


def evaluate_pages(retrieved_pages: Sequence[str], gold_pages: Sequence[str], k: int = 5) -> RetrievalMetrics:
    """页面级检索指标。

    两侧都必须归一化：wiki 标题里有「凝滞虚影 • 巽风之形」这类带空格和间隔号的形式，
    只归一化 gold 会出现"明明召回了却判 0"的假阴性。
    """
    got = [normalize(p) for p in retrieved_pages]
    gold = [normalize(g) for g in gold_pages]
    return RetrievalMetrics(
        recall_at_k=recall_at_k(got, gold, k),
        hit_rate=hit_rate(got, gold, k),
        mrr=list_mrr(got, gold),
        ndcg=ndcg(got, gold),
        k=k,
    )


def list_mrr(retrieved: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    gold_set = {normalize(g) for g in gold}
    if not gold_set:
        return 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if normalize(item) in gold_set:
            return 1.0 / rank
    return 0.0


def evaluate(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> RetrievalMetrics:
    return RetrievalMetrics(
        recall_at_k=recall_at_k(retrieved_ids, gold_ids, k),
        hit_rate=hit_rate(retrieved_ids, gold_ids, k),
        mrr=mrr(retrieved_ids, gold_ids),
        ndcg=ndcg(retrieved_ids, gold_ids),
        k=k,
    )


def page_level_hit(retrieved_ids: list[str], gold_pages: list[str], page_of: dict[str, str]) -> bool:
    """页面级判定：chunk 变了但落在正确页面上也算召回。

    wiki 页面结构会变，绑定 chunk_id 会让评测集随索引重建而失效，
    所以评测集标注 gold_pages，判定时再映射到 chunk。
    """
    gold_norm = {normalize(p) for p in gold_pages}
    return any(normalize(page_of.get(rid, "")) in gold_norm for rid in retrieved_ids)


def aggregate(list_of: list[dict]) -> dict:
    """对一批 case 的指标取均值，顺带给标准差方便判断差异是否显著。"""
    if not list_of:
        return {}
    keys = list_of[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(d[key]) for d in list_of if key in d and d[key] is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        out[key] = round(mean, 4)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            out[f"{key}_std"] = round(var**0.5, 4)
    out["n"] = len(list_of)
    return out


def paired_test(a_rows: list[dict], b_rows: list[dict], metric: str = "mrr") -> dict:
    """配对 t 检验：两个配置在同一批 case 上的差异是否显著。

    **方向约定：`mean_diff` 和 `cohens_d` 都是「a 减 b」。**
    调用方一律传 `(改进档, 基线档)`，所以 **正数 = 改进**。
    早先这里传的是 `(基线, 改进档)`，结果 A2→A3 打出 Δ=−0.092，
    而 A3 的 mrr 明明高 9.2 个点 —— 结论方向是对的，符号读起来是反的，
    照抄进简历就会写成「A3 比 A2 下降」。方向必须和直觉一致，否则等于埋雷。

    **为什么必须做配对而不是比较两个均值**：
    消融实验里所有配置跑的是同一批题，case 之间的难度差异极大
    （结构化字段题几乎必中，语义题经常不中）。
    非配对比较会把"这批题恰好简单"误判成"这个配置更好"。
    配对检验先消掉 case 这个方差源，只看同一题上两个配置的差。

    本机没有 scipy，所以手写 t 分布的近似 p 值：
    用 Welch-Satterthwaite 自由度 + 正态近似（n≥20 时 t 分布已经很接近正态）。
    n 小时这个近似不可靠，结果里会标注。

    返回里同时给效应量 Cohen's d —— 统计显著不等于实际有意义，
    d 太小的话结论也不该写进简历。
    """
    common = sorted(set(r.get("case_id", "") for r in a_rows) & set(r.get("case_id", "") for r in b_rows))
    common = [c for c in common if c]
    if len(common) < 3:
        return {"n": 0, "note": "配对样本不足 3 条，无法检验"}

    amap = {r["case_id"]: r for r in a_rows}
    bmap = {r["case_id"]: r for r in b_rows}
    diffs = [float(amap[c][metric]) - float(bmap[c][metric]) for c in common]

    n = len(diffs)
    mean_d = sum(diffs) / n
    sd = (sum((d - mean_d) ** 2 for d in diffs) / (n - 1)) ** 0.5
    if sd == 0:
        # 差值全为 0：两个配置在这批题上完全等价
        return {"n": n, "mean_diff": 0.0, "t": 0.0, "p": 1.0, "cohens_d": 0.0,
                "significant": False, "note": "所有配对差值均为 0，两配置在这批题上完全等价"}

    se = sd / (n**0.5)
    t_stat = mean_d / se
    # Welch-Satterthwaite 近似自由度
    df = n - 1

    try:
        from math import erf, sqrt

        # 正态近似 CDF：n≥20 时 t 分布与正态的偏离 < 3%
        p = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
        approx = "正态近似"
    except Exception:  # noqa: BLE001
        p, approx = float("nan"), "无法计算"

    sd_a = (sum((float(amap[c][metric]) - sum(amap[c][metric] for c in common) / n) ** 2
                 for c in common) / (n - 1)) ** 0.5
    sd_b = (sum((float(bmap[c][metric]) - sum(bmap[c][metric] for c in common) / n) ** 2
                 for c in common) / (n - 1)) ** 0.5
    pooled = ((sd_a**2 + sd_b**2) / 2) ** 0.5
    d = mean_d / pooled if pooled else 0.0

    return {
        "n": n,
        "metric": metric,
        "mean_a": round(sum(float(amap[c][metric]) for c in common) / n, 4),
        "mean_b": round(sum(float(bmap[c][metric]) for c in common) / n, 4),
        "mean_diff": round(mean_d, 4),
        "sd_diff": round(sd, 4),
        "std_err": round(se, 4),
        "t": round(t_stat, 3),
        "df": df,
        "p": round(p, 4),
        "cohens_d": round(d, 3),
        "significant": bool(p < 0.05),
        "approx": approx,
        "note": ("n<20，t 分布与正态偏离较大，p 值仅供参考"
                 if n < 20 else ""),
    }