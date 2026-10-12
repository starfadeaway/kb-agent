"""延迟优化前后 A/B（单变量：只改 rerank_candidates，其余全部冻结）。

对照组 rerank_candidates=0 = 优化前行为（全量送重排）
实验组 rerank_candidates=20 = 优化后（只重排融合后的前 20）

同时验证缓存命中率 —— 缓存只影响延迟，不影响排名，
所以两组（缓存开/关）应当给出逐字相同的 top-5 页面序列。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_io import save_probe

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000/api"

QUERIES = [
    "太空封印站位面饰品的效果是什么",
    "布洛妮娅的遗器词条怎么配",
    "怎么配队才能提高输出",
    "佩拉是几星角色",
    "行迹的存护有哪些",
]


def post(path: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, (time.perf_counter() - t0) * 1000


def run(label: str, candidates: int, repeat: int = 3,
        collect: bool = True) -> tuple[list[float], list[list[str]]]:
    """collect=False 时只把缓存打热，不计入统计。

    必须有这一轮：原来只预热了 QUERIES[0]，后面 14 条查询的冷缓存全部落在
    「对照组先跑」的那一侧 —— 对照组均值 228ms、实验组 120ms，看起来像裁剪
    省了一半时间，其实那 100ms 是缓存从冷变热。实测两次方向都能反过来
    （一次中位数 +18.4%、一次 -31.2%），所以这不是小偏差，是结论不可复现。
    """
    lat: list[float] = []
    ranks: list[list[str]] = []
    for q in QUERIES:
        for _ in range(repeat):
            data, ms = post(
                "/retrieve",
                {"query": q, "top_k": 5, "rerank_candidates": candidates},
            )
            if not collect:
                continue
            lat.append(ms)
            ranks.append([h["page_title"] for h in data["hits"]])
    if not collect:
        return [], []
    print(f"  {label:<28} 中位 {statistics.median(lat):>7.1f} ms   "
          f"均值 {statistics.mean(lat):>7.1f}   最小 {min(lat):>7.1f}")
    return lat, ranks


def main() -> None:
    print("\n" + "=" * 64)
    print("  延迟优化 A/B —— 唯一变量：rerank_candidates")
    print("=" * 64)

    # 预热：两种配置各把全部查询跑一遍打热缓存，都不计入统计。
    # 只预热一条是不够的 —— 剩下 14 条的冷启动会全落在先跑的那一组，
    # 于是"谁先跑"就变成了第二个变量。
    run("预热 candidates=0", 0, collect=False)
    run("预热 candidates=20", 20, collect=False)
    print("  （上面两行是预热轮，不计入统计）")

    before, r_before = run("对照组 candidates=0 (优化前)", 0)
    after, r_after = run("实验组 candidates=20 (优化后)", 20)

    # 排名一致性 —— 裁剪不能改变 top-5 页面
    same = sum(1 for a, b in zip(r_before, r_after, strict=False) if a == b)
    print(f"\n  排名一致性：{same}/{len(r_before)} 组 top-5 页面序列完全相同")
    if same != len(r_before):
        for i, (a, b) in enumerate(zip(r_before, r_after, strict=False)):
            if a != b:
                print(f"    第 {i} 组不一致：")
                print(f"      前: {a}")
                print(f"      后: {b}")

    # 方向必须由数据决定，不能把措辞写死。
    # 原来这里固定打印"下降"，于是中位数真的涨了（30.5 -> 36.1ms，cut 为负）
    # 时屏幕上会显示"下降 -18.2%" —— 与括号里两个数字直接矛盾，
    # 而落盘字段名 median_drop_pct 又把这个方向锁死，下游没法发现。
    def _pct(new: float, old: float) -> float:
        return (new / old - 1) * 100

    med_pct = _pct(statistics.median(after), statistics.median(before))
    mean_pct = _pct(statistics.mean(after), statistics.mean(before))
    print(f"\n  中位数 {med_pct:+.1f}%  "
          f"({statistics.median(before):.1f}ms -> {statistics.median(after):.1f}ms, "
          f"{'变慢' if med_pct > 0 else '变快' if med_pct < 0 else '持平'})")
    print(f"  均值   {mean_pct:+.1f}%  "
          f"({statistics.mean(before):.1f}ms -> {statistics.mean(after):.1f}ms, "
          f"{'变慢' if mean_pct > 0 else '变快' if mean_pct < 0 else '持平'})")
    print(f"  最慢   {max(before):.1f} -> {max(after):.1f}ms（单请求最大值）")
    if med_pct * mean_pct < 0:
        print("  ⚠ 中位数与均值方向相反：裁剪削掉的是长尾慢请求，不是整体水平。"
              "\n    只看中位数会得出与均值相反的结论，两项必须一起报。")

    health = json.loads(urllib.request.urlopen(BASE + "/health", timeout=10).read().decode("utf-8"))
    print(f"  缓存命中率：embed {health['cache']['embed']['hit_rate']:.0%}  "
          f"rerank {health['cache']['rerank']['hit_rate']:.0%}")
    print("=" * 64)

    # 落盘：这 15 组对照是"裁剪有损"这个结论的全部证据，
    # 原先只打屏上，改了 rerank_candidates 之后无从判断结论还成不成立。
    save_probe(
        "bench_latency_rerank_cut",
        {
            "queries": QUERIES,
            "control_candidates0": {"latency_ms": before, "top5_titles": r_before},
            "treat_candidates20": {"latency_ms": after, "top5_titles": r_after},
            "ranking_identical_groups": same,
            "total_groups": len(r_before),
            # 字段名不带 drop/decline —— 方向由数值符号表达（正=变慢）。
            # 原来的 median_drop_pct 在中位数上升时会存进一个负的"下降"百分比，
            # 名字与内容互相矛盾，下游只能靠猜。
            "median_before_ms": round(statistics.median(before), 2),
            "median_after_ms": round(statistics.median(after), 2),
            "median_change_pct": round(med_pct, 1),
            "mean_before_ms": round(statistics.mean(before), 2),
            "mean_after_ms": round(statistics.mean(after), 2),
            "mean_change_pct": round(mean_pct, 1),
            "max_before_ms": round(max(before), 2),
            "max_after_ms": round(max(after), 2),
            "mean_median_disagree": med_pct * mean_pct < 0,
            "cache_hit_rate": health["cache"],
        },
        context={"variable": "rerank_candidates", "control": 0, "treat": 20,
                 "n_queries": len(QUERIES), "repeat_per_query": 3,
                 "warm_cache": True,
                 "note": "两组均已预热；冷缓存下对照/实验顺序会让缓存升温被误记为裁剪收益"},
    )


if __name__ == "__main__":
    main()