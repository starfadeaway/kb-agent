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

METRICS = [
    ("mrr", "MRR"),
    ("ndcg", "NDCG"),
    ("recall@3", "recall@3"),
    ("page_hit", "page_hit"),
    ("hit_rate", "hit_rate"),
    ("latency_ms", "延迟ms"),
]
# 这些指标越小越好：mean_diff > 0 意味着「A 更慢」，
# 也就是 **B 更好**。不显式处理方向，就会把
# 「bge 慢 37.7ms，p<1e-10」写成「bge 更好」——正负号刚好反着。
LOWER_IS_BETTER = {"latency_ms"}


def _winner(mean_diff: float, higher_is_better: bool) -> str:
    """给定差值与方向，返回谁更好。

    为什么必须有这个函数：`significant=True` 只说明「差异不是噪声」，
    不说明「谁赢」。裸看 mean_diff=+37.7 / p<1e-10，
    对延迟这种越低越好的指标，含义恰恰是 **bge 更差**。
    输出里不给 winner，读者就得自己反推符号 ——
    而本实验唯二显著的结果**全是延迟**，正好落在这个陷阱里。
    这和 `judge.pairwise` 把解析失败静默算成 TIE 是同一类错误：
    把需要语义判断的信息编码成要读者自己解码的裸值。
    """
    if mean_diff == 0:
        return "tie"
    bge_is_better = (mean_diff > 0) if higher_is_better else (mean_diff < 0)
    return "bge" if bge_is_better else "other"


def _load(p: Path) -> dict[str, list[dict]]:
    """文件是 [{label, rows}, ...]，转成 {label: rows}。"""
    data = json.loads(p.read_text(encoding="utf-8"))
    return {r["label"]: r["rows"] for r in data}


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sr",
                    help="目标库。产物文件名里带库名，三库必须分别跑 —— "
                         "混着跑等于拿不同批次的均值相减")
    ap.add_argument("--bge", default=None)
    ap.add_argument("--other", default=None)
    ap.add_argument("--labels", default="A0,A1,A2",
                    help="A3 只有 bge 有（重排后 embedding 影响被掩盖），默认不参与")
    ap.add_argument("--out", default=None,
                    help="相对 data/eval/；默认按库命名，不覆盖别的库的结果")
    args = ap.parse_args()

    game = args.game
    bge_path = Path(args.bge) if args.bge else EVAL_DIR / f"exp_D_BAAI_bge-m3_{game}_k3.json"
    other_path = Path(args.other) if args.other else (
        EVAL_DIR / f"exp_D_Qwen_Qwen3-Embedding-0.6B_{game}_k3.json")

    bge = _load(bge_path)
    other = _load(other_path)

    labels = [x.strip().upper() for x in args.labels.split(",") if x.strip()]
    out_name = args.out or f"model_compare_{game}.json"
    report: dict = {"_meta": {
        "script": "analyze_model_compare.py",
        "note": "跨模型配对检验：同一批 case，只变 embedding 模型",
        "game": game,
        "bge_file": bge_path.name,
        "other_file": other_path.name,
        "labels": labels,
    }}

    print("=" * 100)
    print(f"  跨模型配对检验（{game}）：bge-m3  vs  Qwen3-Embedding-0.6B")
    print("  唯一变量 = embedding 模型；语料/分块/BM25/RRF/rerank/top_k/评测集全部冻结")
    print("=" * 100)
    print(f"{'档位':<6}{'指标':<12}{'bge':>10}{'Qwen':>10}{'Δ(bge-Qwen)':>14}"
          f"{'t':>10}{'p':>12}{'d':>9}   结论")
    print("  （Δ 为正 = bge 数值更大；对延迟这类越低越好的指标，Δ 为正意味着 **bge 更慢**）")

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
            underflow = p == 0.0
            ptxt = "<1e-10" if underflow else f"{p:.4f}"
            hi = key not in LOWER_IS_BETTER
            win = _winner(t["mean_diff"], hi)
            if win == "tie":
                verdict = "-"
            elif t.get("significant"):
                verdict = f"{'bge' if win == 'bge' else 'Qwen'} 显著更优"
            else:
                verdict = f"{'bge' if win == 'bge' else 'Qwen'} 略优(未显著)"
            print(f"{label:<6}{name:<12}{mbv:>10.4f}{mov:>10.4f}"
                  f"{t['mean_diff']:>+14.4f}{t['t']:>10.3f}{ptxt:>12}"
                  f"{t['cohens_d']:>9.3f}   {verdict}")
            entry["metrics"][key] = {
                "bge": round(mbv, 4), "other": round(mov, 4),
                "mean_diff": round(t["mean_diff"], 4), "t": round(t["t"], 3),
                "p": p, "p_is_underflow": underflow,
                "cohens_d": round(t["cohens_d"], 3),
                "n": t["n"],
                "significant": bool(t.get("significant")),
                "higher_is_better": hi,
                # 谁更好，显式给出，避免读者自己反推符号
                "winner": win,
                "verdict": verdict,
            }
        report[label] = entry
        print()

    # A0 阴性对照专项：理论上应当逐位相同
    a0_ok: bool | None = None
    if "A0" in labels and "A0" in bge and "A0" in other:
        a0b = {r["case_id"]: r["mrr"] for r in bge["A0"]}
        a0o = {r["case_id"]: r["mrr"] for r in other["A0"]}
        shared = sorted(set(a0b) & set(a0o))
        diff = [c for c in shared if a0b[c] != a0o[c]]
        a0_ok = not diff
        report["_meta"]["a0_identical"] = a0_ok
        report["_meta"]["a0_diff_case_ids"] = diff
        if a0_ok:
            print(f"阴性对照 A0（纯 BM25，不用向量）逐位相同: True"
                  f"（{len(shared)} 条全同）")
            print("  → 其余档位的差异确实来自 embedding 模型，没有第二个隐藏变量。\n")
        else:
            # 这里必须显式失败，不能只打一行。
            #
            # A0 是纯 BM25，两侧**根本没用向量**，逐位相同是「除了 embedding
            # 以外没有第二个变量在动」的唯一自证。一旦它不相等，说明语料版本、
            # BM25 参数、top_k、评测集样本这些里有东西没冻住 ——
            # 此时 A1/A2 那些 p 值全都不能当作「embedding 模型造成的差异」来读，
            # 但表格照样会把它们打出来，看起来和正常结果一模一样。
            #
            # 2026-10-13 实测踩到：三库版跑完后默认拿 sr 的老产物去比，
            # A0 对不上却仍打出了「bge 显著更优 p=0.0142」。
            print(f"❌ 阴性对照 A0 不成立：{len(diff)}/{len(shared)} 条逐位不同 → {diff[:10]}")
            print("   纯 BM25 两侧不该有任何差异，这说明**还有第二个隐藏变量没冻住**")
            print("   （常见原因：两侧产物不是同一次语料版本 / 不是同一个 top_k / 评测集样本集不同）。")
            print("   在查清之前，上表 A1/A2 的 p 值**不能**读作 embedding 模型造成的差异。\n")

    out = EVAL_DIR / out_name
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()