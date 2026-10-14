"""把实验 C 的三库混跑产物按库拆成 `exp_C_<game>.json`。

## 为什么必须拆

`exp_c.py --game all` 会把三库 56 条题**堆成一个均值**。但三库的题型构成差得很远
（`exp_common.load_cases` 的 docstring 自己承认「分层只是意图，行为上并没有做」）：
实测混跑产物里 single_hop 占 28/56（50%），而三库分别看差异明显。

堆在一起的均值里混着题型构成，读者会把它读成「Agent 能力」，
实际有相当一部分来自「哪库的题更简单」。项目纪律一直是
**三库不合并成一个均值**，所以这里把它拆回去。

## 拆的方式，以及它的代价

`exp_c.py` 的 row **不带 `game` 字段**，只能按 `case_id` 前缀（`sr-` / `ys-` / `zzz-`）
拆。拆出来的行**与单独跑 `--game sr` 不是同一次运行**，差别来自：

- 同一个进程里三库共享 WAF 配额、HTTP 连接池、BM25 索引缓存；
- 56 条题的调度顺序与单库跑不同。

所以本脚本产出的数只能用于**库内**的 C0/C1/C2 对比，
不能声称「等于按库单独跑的结果」。这个限制写进 `_meta.caveat`。

要完全消除这个差异只能三库各跑一次（3 × 3 档 × LLM+Judge 调用），
当前没有必要 —— 库内三档的对比不受这个差异影响，
因为三档在同一次运行里共享完全相同的环境。

## C0 为什么是单独一次跑

旧 `exp_C*.json` 的 C0 档把未格式化的 `SYSTEM_PROMPT` 当系统消息发了出去，
游戏名和注入防护那段都没进去（见 commit 6a22f20）。所以 C0 必须重跑，
由 `exp_C0_fixed.json` 提供；C1/C2 沿用混跑产物里对应的行。

用法：
    python scripts\\merge_exp_c.py
    python scripts\\merge_exp_c.py --games sr
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_common import EVAL_DIR, aggregate_rows, fmt_p, paired_scores  # noqa: E402

#: C0 与 C1/C2 的来源不同文件。C0 必须来自重跑产物 ——
#: 旧产物的 C0 提示词是坏的，混用会让「给不给工具」这条差值被污染。
C0_SOURCE = "exp_C0_fixed.json"
C12_SOURCE = "exp_C_sr_ys_zzz.json"

_PFX = ("sr-", "ys-", "zzz-")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _load_blocks(p: Path) -> dict[str, list[dict]]:
    """{label: rows}。只认带 label 的块（出处块以 `_meta` 为键）。"""
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        d["label"]: d["rows"]
        for d in data
        if isinstance(d, dict) and d.get("label") and isinstance(d.get("rows"), list)
    }


def _split_by_game(rows: list[dict]) -> dict[str, list[dict]]:
    """按 case_id 前缀拆库。识别不出的 id 直接丢弃，**不**猜。

    猜前缀的话，一条 sr 的题被误判成 ys 就会静默进入错误的库均值，
    而配对检验的 n 看起来完全正常。
    """
    out: dict[str, list[dict]] = {g: [] for g in ("sr", "ys", "zzz")}
    unknown: list[str] = []
    for r in rows:
        cid = str(r.get("case_id") or "")
        # 命中后要**去掉前缀里的连字符**再当键用 —— case_id 是 `sr-001`，
        # 而 out 的键是 `sr`。这里曾经写成直接 `out[hit]`，
        # 于是一跑就 KeyError: 'sr-'。
        hit = next((g for g in _PFX if cid.startswith(g)), None)
        if hit is None:
            unknown.append(cid)
        else:
            out[hit[:-1]].append(r)
    if unknown:
        print(f"  !! 有 {len(unknown)} 条 case_id 识别不出库，已丢弃：{unknown[:5]}")
    return out


def _paired_pair(a_rows, b_rows):
    """对同一组 case 的两档做配对检验。复用 exp_common.paired_scores 的方向约定。"""
    blocks = [{"label": "C0", "rows": a_rows}, {"label": "C2", "rows": b_rows}]
    return paired_scores(blocks)


def _paired_adjacent(a_rows, b_rows):
    """C1 -> C2（相邻两档）。paired_scores 只比首尾，所以这里手工换标签再调。"""
    blocks = [{"label": "C1", "rows": a_rows}, {"label": "C2", "rows": b_rows}]
    return paired_scores(blocks)


def merge_game(game: str, *, verbose: bool = True) -> tuple[bool, dict]:
    c0_path = EVAL_DIR / C0_SOURCE
    c12_path = EVAL_DIR / C12_SOURCE
    for p in (c0_path, c12_path):
        if not p.is_file():
            print(f"  !! 缺源文件：{p.name}")
            return False, {}

    c0_all = _split_by_game(_load_blocks(c0_path).get("C0", []))
    c12 = _load_blocks(c12_path)

    parts: list[dict] = []
    for label, rows in (("C0", c0_all[game]), ("C1", c12.get("C1", [])), ("C2", c12.get("C2", []))):
        sub = [r for r in _split_by_game(rows)[game]]
        if not sub:
            print(f"  !! {game} 的 {label} 档拆出来 0 行")
            return False, {}
        # 聚合必须**按拆出来的行重算**，不能沿用混跑文件里的 aggregate ——
        # 那个数是 56 条题的均值，与这里的分母不是一回事。
        parts.append({
            "label": label,
            "tools": {"C0": [], "C1": ["search_kb"],
                      "C2": ["search_kb", "get_page", "calculator", "list_games"]}[label],
            "aggregate": aggregate_rows(sub),
            "rows": sub,
        })

    case_sets = {b["label"]: sorted(r["case_id"] for r in b["rows"]) for b in parts}
    union = set().union(*case_sets.values())
    for label, ids in case_sets.items():
        missing = sorted(union - set(ids))
        if verbose and missing:
            print(f"    ⚠ {label} 缺题 {missing}（配对时会按交集收窄）")

    out = list(parts)
    out.append({"_meta": {
        "script": Path(__file__).name,
        "experiment": "C",
        "game": game,
        "merged_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": [
            {"file": C0_SOURCE, "sha256_16": _sha256(c0_path), "labels": ["C0"],
             "why": "旧 exp_C*.json 的 C0 提示词未格式化（游戏名与注入防护都没进去），必须重跑"},
            {"file": C12_SOURCE, "sha256_16": _sha256(c12_path), "labels": ["C1", "C2"],
             "why": "沿用三库混跑产物里对应库的行"},
        ],
        "case_set_per_label": {k: len(v) for k, v in case_sets.items()},
        "paired_C0_C2": _paired_pair(parts[0]["rows"], parts[2]["rows"]),
        "paired_C1_C2": _paired_adjacent(parts[1]["rows"], parts[2]["rows"]),
        "caveat": (
            "按 case_id 前缀从三库混跑产物拆出，与「--game 单库跑一次」不是同一份数据："
            "进程内三库共享 WAF 配额 / 连接池 / BM25 缓存，调度顺序也不同。"
            "库内 C0/C1/C2 的对比不受影响（三档共享同一环境），"
            "但不要声称这些数等于单库独立运行的结果。"
        ),
    }})

    out_name = f"exp_C_{game}.json"
    (EVAL_DIR / out_name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"  -> {out_name}")
    return True, {"game": game, "labels": {b["label"]: b["aggregate"] for b in parts}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="sr,ys,zzz")
    args = ap.parse_args()

    summaries = []
    for g in [x.strip() for x in args.games.split(",") if x.strip()]:
        print(f"=== 拆分 {g} ===")
        ok, s = merge_game(g)
        if not ok:
            return 1
        summaries.append(s)

    print("\n" + "=" * 78)
    print("实验 C：Agent 能力消融（逐库）")
    print("=" * 78)
    print(f"{'库':<5}{'档':<4}{'judge':>8}{'correct':>9}{'faith':>8}"
          f"{'tok':>8}{'步数':>7}{'引用':>7}{'失败':>8}{'n_ok':>7}")
    for s in summaries:
        for label, a in s["labels"].items():
            print(f"{s['game']:<5}{label:<4}{a['judge_score']:>8.2f}{a['correctness']:>9.2f}"
                  f"{a['faithfulness']:>8.2f}{a['avg_prompt_tokens']:>8}"
                  f"{a['avg_steps']:>7.1f}{a['ref_coverage']:>7.2f}"
                  f"{a['tool_failure_rate']:>8.3f}{str(a['n_ok']) + '/' + str(a['n']):>7}")

    print()
    for g in [x.strip() for x in args.games.split(",") if x.strip()]:
        meta = json.loads((EVAL_DIR / f"exp_C_{g}.json").read_text(encoding="utf-8"))[-1]["_meta"]
        for key, label in (("paired_C0_C2", "C0→C2"), ("paired_C1_C2", "C1→C2")):
            p = meta.get(key)
            if p:
                print(f"  [{g}] {label}  Δ={p['mean_diff']:+.4f} t={p['t']} "
                      f"p={fmt_p(p['p'])} d={p['cohens_d']} n={p['n']}")
            else:
                print(f"  [{g}] {label}  样本不足，无法配对（两档交集 < 3）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())