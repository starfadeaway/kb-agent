"""实验 D 的跨模型配对检验：bge-m3 vs Qwen3-Embedding-0.6B。

为什么必须单独写这个实验
-----------------------
原实验 D 只在 **Qwen 内部**做相邻档检验（A0 vs A1 vs A2），
于是"bge 更强"这个核心命题**从来没有被直接检验过** ——
它是靠拿 Qwen 的 A2 绝对值去和历史 bge 的 A2 绝对值对比得出的，
两条数据来自不同批次，样本集是否一致也没验证过。
两个独立批次的均值相减，不能当成一个配对检验的结论。

这里改成**同一批 29 条 case 上的配对比较**：
同一道题、同一个 gold，只是检索用的 embedding 模型不同。

冻结项：语料、chunk 参数（500/50）、BM25、RRF、rerank 模型、
top_k=3、评测集与 gold 全部一致；唯一变量 = embedding 模型。

A0 是天然的**阴性对照**：纯 BM25 完全不用向量，
两组数据应当逐位相同。若 A0 也算出差异，说明还有别的变量没冻住。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.eval.metrics import paired_test  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"

# (文件, 该文件代表哪个 embedding 模型)
BGE_FILE = EVAL_DIR / "exp_A.json"
QWEN_GLOB = "exp_D_Qwen_Qwen3-Embedding-0.6B_sr_k3.json"

METRICS = [
    ("mrr", "MRR"),
    ("ndcg", "NDCG"),
    ("recall@3", "recall@3"),
    ("page_hit", "page_hit"),
    ("hit_rate", "hit_rate"),
    ("latency_ms", "延迟ms"),
]
# higher_is_better=False 的指标，差值方向相反才叫"更好"
LOWER_IS_BETTER = {"latency_ms"}


def _load(p: Path) -> dict[str, list[dict]]:
    """文件是 [{label, rows}, ...]，转成 {label: rows}。"""
    data = json.loads(p.read_text(encoding="utf-8"))
    return {r["label"]: r["rows"] for r in data}


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bge", default=str(BGE_FILE))
    ap.add_argument("--other", default=str(EVAL_DIR / QWEN_GLOB))
    ap.add_argument("--labels", default="A0,A1,A2",
                    help="A3 只有 bge 有（重排后 embedding 影响被掩盖），默认不参与")
    ap.add_argument("--out", default="model_compare.json")
    args = ap.parse_args()

    bge = _load(Path(args.bge))
    other = _load(Path(args.other))

    labels = [x.strip().upper() for x in args.labels.split(",") if x.strip()]
    report: dict = {"_meta": {
        "script": "analyze_model_compare.py",
        "note": "跨模型配对检验：同一批 case，只变 embedding 模型",
        "bge_file": Path(args.bge).name,
        "other_file": Path(args.other).name,
        "labels": labels,
    }}

    print("=" * 100)
    print("  跨模型配对检验：bge-m3  vs  Qwen3-Embedding-0.6B")
    print("  唯一变量 = embedding 模型；语料/分块/BM25/RRF/rerank/top_k/评测集全部冻结")
    print("=" * 100)
    print(f"{'档位':<6}{'指标':<12}{'bge':>10}{'Qwen':>10}{'Δ(bge-Qwen)':>14}"
          f"{'t':>10}{'p':>12}{'d':>9}")

    for label in labels:
        if label not in bge or label not in other:
            print(f"  (跳过 {label}：两侧并非都有)")
            continue
        rb, ro = bge[label], other[label]
        mb = {r["case_id"]: r for r in rb}
        mo = {r["case_id"]: r for r in ro}
        common = sorted(set(mb) & set(mo))

        if not common:
            continue

        entry = {"n_common": len(common),
                 "n_bge": len(rb), "n_other": len(ro),
                 "metrics": {}}
        if len(common) != len(rb) or len(common) != len(ro):
            print(f"  ⚠ {label} 样本集不一致：bge={len(rb)} other={len(ro)} "
                  f"交集={len(common)}，以下只用交集")

        for key, name in METRICS:
            if key not in rb[0] or key not in ro[0]:
                continue
            a = [{**mb[c], key: mb[c][key]} for c in common]
            b = [{**mo[c], key: mo[c][key]} for c in common]
            t = paired_test(a, b, key)
            mbv, mov = _mean(a, key), _mean(b, key)
            p = t["p"]
            ptxt = "<1e-10" if p == 0 else f"{p:.4f}"
            dtxt = f"{t['cohens_d']:.3f}"
            print(f"{label:<6}{name:<12}{mbv:>10.4f}{mov:>10.4f}"
                  f"{t['mean_diff']:>+14.4f}{t['t']:>10.3f}{ptxt:>12}{dtxt:>9}")
            entry["metrics"][key] = {
                "bge": round(mbv, 4), "other": round(mov, 4),
                "mean_diff": round(t["mean_diff"], 4), "t": round(t["t"], 3),
                "p": p, "cohens_d": round(t["cohens_d"], 3),
                "n": t["n"],
                "significant": bool(t.get("significant")),
                "higher_is_better": key not in LOWER_IS_BETTER,
            }
        report[label] = entry
        print()

    # A0 阴性对照专项：理论上应当逐位相同
    if "A0" in labels and "A0" in bge and "A0" in other:
        same = all(
            {r["case_id"]: r["mrr"] for r in bge["A0"]}[c]
            == {r["case_id"]: r["mrr"] for r in other["A0"]}[c]
            for c in sorted({r["case_id"] for r in bge["A0"]}
                            & {r["case_id"] for r in other["A0"]})
        )
        print(f"阴性对照 A0（纯 BM25，不用向量）逐位相同: {same}")
        report["_meta"]["a0_identical"] = bool(same)
        if same:
            print("  → 其余档位的差异确实来自 embedding 模型，没有第二个隐藏变量。\n")

    out = EVAL_DIR / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()