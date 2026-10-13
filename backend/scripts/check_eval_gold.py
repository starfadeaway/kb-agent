"""评测集健康检查：每条题的 gold 页面是否真的在本地语料里。

为什么要有这个脚本：扩充评测集（把 ys / zzz 的题也纳入检索实验）之前，
必须先回答一个问题：**这些题的 gold 页面是不是真的存在？**

不验证就扩，会得到一个更隐蔽的失败模式：
题量从 29 涨到 53，看起来"覆盖三库了"，但其中一半的 gold 页面压根不在语料里，
recall 天生算不满，而输出里不会有一处提示 —— 汇总表上只是一个偏低的数字，
被读成"这个库检索效果差"，实际是标注时指向了不存在的页面。

判据是**按标题**在 data/raw/{game}/ 下找正文文件（文件名是 `{page_id}_{标题}.txt`），
而不是按 page_id —— 评测集标注的就是标题。

跑法：python backend/scripts/check_eval_gold.py
退出码非 0 表示有 gold 缺失（可作为 CI/守卫用）。
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.eval.cases import ALL_CASES, SR_CASES, YS_CASES, ZZZ_CASES  # noqa: E402

RAW = ROOT / "data" / "raw"


def local_titles(game: str) -> set[str]:
    """该库本地已抓页面的标题集合。"""
    d = RAW / game
    if not d.is_dir():
        return set()
    out: set[str] = set()
    for p in d.glob("*.txt"):
        stem = p.stem
        # 文件名形如 `1494_三月七`，page_id 与标题之间用第一个下划线分隔
        m = re.match(r"^\d+_(.*)$", stem)
        if m:
            out.add(m.group(1))
    return out


def norm(s: str) -> str:
    return s.strip().replace(" ", "").replace("_", "")


def main() -> int:
    groups = [("sr", SR_CASES), ("ys", YS_CASES), ("zzz", ZZZ_CASES)]
    cache: dict[str, set[str]] = {}
    report: dict = {"groups": {}, "missing": []}
    bad = 0

    for gname, case_list in groups:
        titles = cache.setdefault(gname, local_titles(gname))
        normed = {norm(t) for t in titles}
        cats = Counter()
        n_gold = 0
        missing: list[tuple[str, str]] = []
        for c in case_list:
            cats[c.category] += 1
            gold = list(getattr(c, "gold_pages", []) or [])
            if not gold:
                continue
            n_gold += 1
            for g in gold:
                if norm(g) not in normed:
                    missing.append((c.case_id, g))
        pct = len(case_list) and round(len(case_list) and (n_gold - len({m[0] for m in missing})) / n_gold * 100, 1)
        print(f"== {gname}：{len(case_list)} 条，本地语料 {len(titles)} 页")
        print(f"   题型 {dict(cats)}")
        print(f"   有 gold 的题 {n_gold} 条，gold 页面全部命中：{len(case_list) and n_gold - len({m[0] for m in missing})}"
              f"（{pct}%）")
        if missing:
            bad += len(missing)
            for cid, g in missing[:12]:
                print(f"   ✗ {cid}: gold 页面不在语料里 -> {g}")
            if len(missing) > 12:
                print(f"   ... 还有 {len(missing) - 12} 条")
        report["groups"][gname] = {
            "cases": len(case_list), "local_pages": len(titles),
            "cases_with_gold": n_gold, "missing": len(missing),
            "categories": dict(cats),
        }
        report["missing"].extend({"game": gname, "case_id": c, "gold": g} for c, g in missing)

    print("\n== 汇总 ==")
    total = sum(len(v) for _, v in groups)
    with_gold = sum(report["groups"][g]["cases_with_gold"] for g, _ in groups)
    print(f"  三库合计 {total} 条，有 gold 的 {with_gold} 条，gold 缺失 {bad} 处")
    if ALL_CASES:
        print(f"  另有 {len(ALL_CASES)} 条跨库路由题（ALL_CASES，无 gold 页面要求）")

    out = ROOT / "data" / "eval" / "check_eval_gold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  产物已写 {out.relative_to(ROOT)}")
    print("\n-- 判定 --")
    print(f"  {'✅ 所有 gold 页面都能在本地语料里找到' if not bad else f'❌ {bad} 处 gold 缺失，先修标注再扩评测集'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())