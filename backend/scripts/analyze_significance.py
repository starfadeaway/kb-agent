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

from exp_common import (
    EVAL_DIR,
    aggregate_rows,
    is_kept,
    judge_failed,
    paired_scores,
    print_category_matrix,
)


def load(name: str) -> list[dict]:
    # 曾经这里写死本机绝对路径（C:\Users\Liu\...），换台机器就找不到文件 ——
    # 而这个脚本的产物是要写进简历的量化数字，不能只在作者本机跑得通。
    # 复用 exp_common 的 EVAL_DIR，和 save 落盘的那份永远是同一个目录。
    p = EVAL_DIR / name
    data = json.loads(p.read_text(encoding="utf-8"))
    # 产物尾部可能跟一个 `_meta` 出处块（没有 label）——
    # exp_b 走 build_meta 会带，exp_E 经 merge_exp_e.py 合并后也会带。
    # 不滤掉的话 `{d["label"]: d}` 直接 KeyError，而且报错信息
    # （KeyError: 'label'）完全指不到真正的原因。
    blocks = [d for d in data if isinstance(d, dict) and d.get("label")]
    dropped = len(data) - len(blocks)
    if dropped:
        print(f"  （{name} 里有 {dropped} 个非档位块，已跳过："
              f"多半是 _meta 出处块）")
    return blocks


def is_infra_failure(row: dict) -> bool:
    """这一条是**基础设施失败**（429 / 断网 / 额度用尽），不是模型答错。

    为什么必须单独剔除：Agent 把模型异常捕获后会把错误文案写进 answer，
    `error` 字段仍是空的 —— 所以聚合口径里的"只统计未报错 case"抓不到它，
    这条会以 1.0 分（最低分）混进均值，把被测配置**系统性拉低**。

    判据用 `prompt_tokens == 0`：模型根本没被调用成功过，
    不可能有真实 token 消耗。这是能被客观验证的，不靠看答案文本猜。
    """
    # ⚠️ 本函数只管模型侧。Judge 侧失败由 exp_common.is_kept 统一处理，
    # 这里不重复实现 —— 早先两件事混在同一个谓词里，导致同一个脚本里
    # 配对检验剔了 Judge 失败、均值却没剔，同一份 exp_C.json 打出两个均值。
    return (row.get("prompt_tokens") or 0) == 0 and (row.get("completion_tokens") or 0) == 0


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "exp_C.json"
    pairs = [tuple(p.split(":")) for p in sys.argv[2].split(",")] if len(sys.argv) > 2 else None

    data = load(name)
    by_label = {d["label"]: d for d in data}
    if pairs is None:
        labels = [d["label"] for d in data]
        pairs = [(labels[i], labels[i + 1]) for i in range(len(labels) - 1)]

    # ---- 先报两类失败，避免把链路问题当成模型能力差 ----
    infra = {lab: [r["case_id"] for r in by_label[lab]["rows"] if is_infra_failure(r)]
             for lab in by_label}
    judge_broken = {lab: [r["case_id"] for r in by_label[lab]["rows"] if judge_failed(r)]
                    for lab in by_label}
    if any(infra.values()) or any(judge_broken.values()):
        print("=" * 78)
        print("  ⚠️ 链路失败（已从下方均值与检验中剔除）")
        print("=" * 78)
        for lab, ids in infra.items():
            if ids:
                print(f"  {lab} [模型侧]: {', '.join(ids)}  —— 模型调用未成功，与答得好不好无关")
        for lab, ids in judge_broken.items():
            if ids:
                print(f"  {lab} [Judge 侧]: {', '.join(ids)}  —— 评分链路失败，不代表答错")

    # 统一走 exp_common.is_kept（Agent 侧 + Judge 侧）。
    # 早先这里只用 is_infra_failure 过滤，于是 paired_scores 内部又按 is_kept
    # 剔了一遍（配对 n 正确），但均值与题型矩阵拿的是未剔干净的 rows ——
    # 同一份 exp_C.json 因此打出两个 judge 均值：4.77（这里）与 4.13。
    clean = {
        lab: {**d, "rows": [r for r in d["rows"] if is_kept(r)]}
        for lab, d in by_label.items()
    }
    clean_by_label = {lab: d for lab, d in clean.items()}

    print("\n" + "=" * 78)
    print(f"  配对检验 · {name}（剔除链路失败后）")
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
    print("  各档均值（已剔除链路失败）")
    print("=" * 78)
    # 均值走 exp_common.aggregate_rows，且**不再传 exclude** ——
    # rows 在上面已按 is_kept 过滤过，再传一个只认模型侧的谓词等于又加了一层
    # 与配对检验不一致的过滤（这正是 4.77 vs 4.13 的来源）。
    for lab, d in clean_by_label.items():
        rows = d["rows"]
        if not rows:
            continue
        a = aggregate_rows(rows)
        print(f"  {lab:<4} judge={a['judge_score']:.2f}  n={a['n_ok']}  步数={a['avg_steps']:.1f}  "
              f"prompt tok={a['avg_prompt_tokens']}  延迟={a['avg_latency_ms']}ms")

    print("\n" + "=" * 78)
    print("  按题型")
    print("=" * 78)
    # 按题型矩阵同样走 exp_common 里的实现，且同样不传 exclude（口径与均值一致）
    print_category_matrix(list(clean_by_label.values()), width=10)


if __name__ == "__main__":
    main()
