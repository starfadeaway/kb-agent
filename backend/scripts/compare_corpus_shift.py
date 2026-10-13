"""新语料 vs 旧语料：实验 A 逐 case 对比，定位是谁在退。

为什么必须做这一步
------------------
新语料下 A0(BM25) 从 0.897 掉到 0.793、A3 从 0.966 掉到 0.897，
而 A1(向量) 一点没动。**"BM25 掉、向量不掉"本身不是结论** ——
它可以是「新增的长文本攻略拖累了词面检索」，也可以是
「恰好某几道题的金标准页变了」。这两种解释对应完全不同的修法，
所以必须逐题看。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OLD = Path("data/eval/exp_A_k3_v2.json")
NEW = Path("data/eval/exp_A_role_pages.json")


def load(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for block in data:
        label = block["label"]
        out[label] = {
            r["case_id"]: (bool(r.get("page_hit")), (r.get("top_titles") or [])[:3])
            for r in block["rows"]
        }
    return out


old, new = load(OLD), load(NEW)

print("=" * 78)
print("  各档 page_hit 对比（旧语料 2306 → 新语料 3932）")
print("=" * 78)
print(f"{'label':<6}{'旧':>7}{'新':>7}{'Δ':>8}   变好 / 变差")
for label in ("A0", "A1", "A2", "A3"):
    o, n = old[label], new[label]
    ids = sorted(set(o) & set(n))
    o_rate = sum(o[i][0] for i in ids) / len(ids)
    n_rate = sum(n[i][0] for i in ids) / len(ids)
    better = [i for i in ids if n[i][0] and not o[i][0]]
    worse = [i for i in ids if o[i][0] and not n[i][0]]
    print(f"{label:<6}{o_rate:>7.3f}{n_rate:>7.3f}{n_rate - o_rate:>+8.3f}"
          f"   {len(better)} / {len(worse)}")

print("\n" + "=" * 78)
print("  A3（最终配置）退化的题")
print("=" * 78)
o3, n3 = old["A3"], new["A3"]
for cid in sorted(o3):
    if o3[cid][0] and not n3[cid][0]:
        print(f"\n  ✗ {cid}")
        print(f"      旧命中: {o3[cid][1]}")
        print(f"      新 top3: {n3[cid][1]}")

print("\n" + "=" * 78)
print("  A0（纯 BM25）退化的题")
print("=" * 78)
o0, n0 = old["A0"], new["A0"]
for cid in sorted(o0):
    if o0[cid][0] and not n0[cid][0]:
        print(f"\n  ✗ {cid}")
        print(f"      旧命中: {o0[cid][1]}")
        print(f"      新 top3: {n0[cid][1]}")