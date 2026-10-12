"""对已完成实验的原始逐条分数做配对检验。

为什么单独一个脚本：实验脚本的汇总表只给均值，
而"多给工具到底有没有变差"这种结论必须看显著性 ——
n=15 时 4.60 vs 4.47 的差距很可能只是噪声，
不检验就下结论等于自己骗自己。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_common import EVAL_DIR, aggregate_rows, paired_scores, print_category_matrix


def load(name: str) -> list[dict]:
    # 曾经这里写死本机绝对路径（C:\Users\Liu\...），换台机器就找不到文件 ——
    # 而这个脚本的产物是要写进简历的量化数字，不能只在作者本机跑得通。
    # 复用 exp_common 的 EVAL_DIR，和 save 落盘的那份永远是同一个目录。
    p = EVAL_DIR / name
    return json.loads(p.read_text(encoding="utf-8"))


def is_infra_failure(row: dict) -> bool:
    """这一条是**基础设施失败**（429 / 断网 / 额度用尽），不是模型答错。

    为什么必须单独剔除：Agent 把模型异常捕获后会把错误文案写进 answer，
    `error` 字段仍是空的 —— 所以聚合口径里的"只统计未报错 case"抓不到它，
    这条会以 1.0 分（最低分）混进均值，把被测配置**系统性拉低**。

    判据用 `prompt_tokens == 0`：模型根本没被调用成功过，
    不可能有真实 token 消耗。这是能被客观验证的，不靠看答案文本猜。
    """
    return (row.get("prompt_tokens") or 0) == 0 and (row.get("completion_tokens") or 0) == 0


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "exp_C.json"
    pairs = [tuple(p.split(":")) for p in sys.argv[2].split(",")] if len(sys.argv) > 2 else None

    data = load(name)
    by_label = {d["label"]: d for d in data}
    if pairs is None:
        labels = [d["label"] for d in data]
        pairs = [(labels[i], labels[i + 1]) for i in range(len(labels) - 1)]

    # ---- 先报基础设施失败，避免把基础设施问题当成模型能力差 ----
    infra = {lab: [r["case_id"] for r in by_label[lab]["rows"] if is_infra_failure(r)]
             for lab in by_label}
    if any(infra.values()):
        print("=" * 78)
        print("  ⚠️ 基础设施失败（已从下方均值与检验中剔除）")
        print("=" * 78)
        for lab, ids in infra.items():
            if ids:
                print(f"  {lab}: {', '.join(ids)}  —— 模型调用未成功，与答得好不好无关")

    clean = {
        lab: {**d, "rows": [r for r in d["rows"] if not is_infra_failure(r)]}
        for lab, d in by_label.items()
    }
    clean_by_label = {lab: d for lab, d in clean.items()}

    print("\n" + "=" * 78)
    print(f"  配对检验 · {name}（剔除基础设施失败后）")
    print("=" * 78)
    for a, b in pairs:
        t = paired_scores([clean_by_label[a], clean_by_label[b]])
        if not t:
            print(f"  {a} → {b}: 数据不足")
            continue
        verdict = "显著" if t["p"] < 0.05 else "**未达显著**"
        print(
            f"  {a} → {b}   Δ={t['mean_diff']:+.4f}  t={t['t']:.3f}  p={t['p']:.4f}  "
            f"Cohen's d={t['cohens_d']:+.3f}  n={t['n']}  →  {verdict}"
        )

    # 对照：如果不剔除，结论会不会反过来 —— 数字不同就说明剔除是必要的
    if any(infra.values()):
        print("\n  对照：若不剔除这些失败样本")
        for a, b in pairs:
            t = paired_scores([by_label[a], by_label[b]])
            if t:
                print(f"    {a} → {b}   Δ={t['mean_diff']:+.4f}  p={t['p']:.4f}  n={t['n']}")

    print("\n" + "=" * 78)
    print("  各档均值（已剔除基础设施失败）")
    print("=" * 78)
    # 均值改走 exp_common.aggregate_rows —— 以前这里是手写的 4 行 sum/len，
    # 和 aggregate_rows 的"只统计未报错 case"口径是两份独立实现。
    # 两份实现一旦漂移，同一份 exp_C.json 会算出两个 judge 均值，
    # 而且没人会发现 —— 这个数字是要写进简历的。
    for lab, d in clean_by_label.items():
        rows = d["rows"]
        if not rows:
            continue
        a = aggregate_rows(rows, exclude=lambda r: not is_infra_failure(r))
        print(f"  {lab:<4} judge={a['judge_score']:.2f}  n={a['n_ok']}  步数={a['avg_steps']:.1f}  "
              f"prompt tok={a['avg_prompt_tokens']}  延迟={a['avg_latency_ms']}ms")

    print("\n" + "=" * 78)
    print("  按题型")
    print("=" * 78)
    # 按题型矩阵同样改调 exp_common 里的实现（列宽 10 是这里的历史版式）
    print_category_matrix(
        list(clean_by_label.values()), exclude=lambda r: not is_infra_failure(r), width=10
    )


if __name__ == "__main__":
    main()
