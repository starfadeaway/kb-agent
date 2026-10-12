"""延迟优化前后 A/B（单变量：只改 rerank_candidates，其余全部冻结）。

对照组 rerank_candidates=0 = 优化前行为（全量送重排）
实验组 rerank_candidates=20 = 优化后（只重排融合后的前 20）

同时验证缓存命中率 —— 缓存只影响延迟，不影响排名，
所以两组（缓存开/关）应当给出逐字相同的 top-5 页面序列。
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.request
from pathlib import Path

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


def run(label: str, candidates: int, repeat: int = 3) -> tuple[list[float], list[list[str]]]:
    lat: list[float] = []
    ranks: list[list[str]] = []
    for q in QUERIES:
        for _ in range(repeat):
            data, ms = post(
                "/retrieve",
                {"query": q, "top_k": 5, "rerank_candidates": candidates},
            )
            lat.append(ms)
            ranks.append([h["page_title"] for h in data["hits"]])
    print(f"  {label:<28} 中位 {statistics.median(lat):>7.1f} ms   "
          f"均值 {statistics.mean(lat):>7.1f}   最小 {min(lat):>7.1f}")
    return lat, ranks


def main() -> None:
    print("\n" + "=" * 64)
    print("  延迟优化 A/B —— 唯一变量：rerank_candidates")
    print("=" * 64)

    # 第一遍预热缓存（两种模式都会命中同一条 embed）
    post("/retrieve", {"query": QUERIES[0], "top_k": 5, "rerank_candidates": 20})

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

    cut = (1 - statistics.median(after) / statistics.median(before)) * 100
    print(f"\n  延迟中位数下降 {cut:.1f}%  "
          f"({statistics.median(before):.0f}ms -> {statistics.median(after):.0f}ms)")

    health = json.loads(urllib.request.urlopen(BASE + "/health", timeout=10).read().decode("utf-8"))
    print(f"  缓存命中率：embed {health['cache']['embed']['hit_rate']:.0%}  "
          f"rerank {health['cache']['rerank']['hit_rate']:.0%}")
    print("=" * 64)


if __name__ == "__main__":
    main()