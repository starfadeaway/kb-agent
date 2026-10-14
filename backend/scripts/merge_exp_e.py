"""把分档 + 分库跑出来的实验 E 产物，按库合并成 `exp_E_<game>.json`。

## 为什么需要它

两重拆分，都是被硬约束逼出来的：

1. **分档**：实验 E 的两档串行跑会超过单任务 3600s 上限（E2 还会撞
   MiniMax 额度），必须 `--labels E1` / `--labels E2` 分两次跑。
   但 `exp_e.py` 每次都把 results 整个写进 `--out`，第二次跑会覆盖第一次
   （E1 的结果就这么丢过一次）。
2. **分库**：三库一起跑同样会超上限，所以又按 `--game` 拆成 sr/ys/zzz 三次。

于是产物散成 6 个文件：`exp_E1_sr.json` … `exp_E2_zzz.json`。

## 为什么不合并成一份

**三库的 chunk 数、页数、题型构成都不一样**（sr 15 题里 multi_hop 2，
ys 只有 1；zzz 以 single_hop 为主）。合并成一个均值，被题型构成主导，
而不是被模型能力主导 —— 结论会变成「sr 比 zzz 难」，而那不是本实验要回答的
问题。所以**每个库各自出一份 E1/E2 配对**，跨库只在终端打印对照表，不落盘均值。

## 合并产物必须自证来源

每份合并文件都写进 `_meta.sources`（源文件名 + sha256 + 每档行数）、
`_meta.case_set_per_label`（各档 case 集合大小）以及 `_meta.paired`
（配对检验结果）。否则合并后的数字没人能说清是哪几次跑拼的。

`script` 字段这里写的是**本脚本**的真实名字 —— `exp_common.build_meta` 里的
`"script": Path(__file__).name` 恒为 `exp_common.py`（那是它自己），
所以靠它判来源会得到「所有实验都是 exp_common」。合并文件不受影响，
因为这里没用 build_meta。

用法：
    python scripts\\merge_exp_e.py                    # 默认 sr,ys,zzz
    python scripts\\merge_exp_e.py --games sr,ys
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

from exp_common import EVAL_DIR, fmt_p, paired_scores  # noqa: E402

# 配对必须和 exp_e.py 用**同一个**谓词：`is_counted` 定义在 exp_e 里，
# 因为它含 E 特有的 `prompt_tokens > 0` 判断。从这里 import 而不是复制一份，
# 复制的话哪天改了一处，两边的样本集就会悄悄分叉。
from exp_e import is_counted  # noqa: E402


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _load(p: Path) -> list[dict]:
    """只取带 label 的档位块（出处块以 `_meta` 为键，没有 label）。"""
    data = json.loads(p.read_text(encoding="utf-8"))
    return [d for d in data if isinstance(d, dict) and d.get("label")]


def merge_game(game: str, *, verbose: bool = True) -> tuple[bool, dict]:
    """合并单个库的 E1/E2 两档。返回 (是否成功, 摘要)。"""
    names = [f"exp_E1_{game}.json", f"exp_E2_{game}.json"]
    parts: list[dict] = []
    sources: list[dict] = []

    for name in names:
        p = EVAL_DIR / name
        if not p.is_file():
            print(f"  !! 缺源文件：{name}")
            return False, {}
        blocks = _load(p)
        if not blocks:
            print(f"  !! {name} 里没有带 label 的块，疑似格式不对")
            return False, {}
        for b in blocks:
            label = b.get("label")
            if any(q.get("label") == label for q in parts):
                # 同一库内档位重复是静默出错的高发点，宁可停下
                print(f"  !! 档位重复：{label} 在 {name} 与前一个源文件里都出现")
                return False, {}
            parts.append(b)
        sources.append({
            "file": name,
            "sha256_16": _sha256(p),
            "labels": [b.get("label") for b in blocks],
            "rows_per_label": {b.get("label"): len(b.get("rows", [])) for b in blocks},
        })

    # 各档 case 集合：E1/E2 必须跑同一批评测题，否则配对检验无意义。
    case_sets = {b["label"]: sorted(r["case_id"] for r in b.get("rows", [])) for b in parts}
    union = set().union(*case_sets.values())
    for label, ids in case_sets.items():
        missing = union - set(ids)
        flag = "⚠️ 缺题 " + ",".join(sorted(missing)) if missing else "与并集一致"
        if verbose:
            print(f"    {label}: {len(ids)} 条 case，{flag}")

    # 配对检验在这里跑 —— exp_e.py 只在两档同文件时才有数据算它，
    # 分档跑完 merge 之前，那个 t/p/d 一直没被算出来过。
    paired = paired_scores(parts, exclude=is_counted)

    out = list(parts)
    out.append({"_meta": {
        "script": Path(__file__).name,
        "experiment": "E",
        "game": game,
        "merged_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": sources,
        "case_set_per_label": {k: len(v) for k, v in case_sets.items()},
        "n_ok_per_label": {b["label"]: b.get("aggregate", {}).get("n_ok") for b in parts},
        "paired": paired,
        "note": (
            "per-label 聚合值沿用各自源文件，未重算。"
            "配对检验在交集样本上做（两档都被 judge 判为可用的 case 才入配对），"
            "n 会小于各档的 n_ok —— 这是被剔除样本导致的，不是计算错误。"
        ),
    }})

    out_name = f"exp_E_{game}.json"
    (EVAL_DIR / out_name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"  -> {out_name}")
    return True, {
        "game": game,
        "labels": {b["label"]: b.get("aggregate", {}) for b in parts},
        "paired": paired,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="sr,ys,zzz",
                    help="逗号分隔的库名；每个库各自出一份合并产物")
    args = ap.parse_args()
    games = [g.strip() for g in args.games.split(",") if g.strip()]

    summaries: list[dict] = []
    for g in games:
        print(f"=== 合并 {g} ===")
        ok, summary = merge_game(g)
        if not ok:
            print(f"!! {g} 合并失败")
            return 1
        summaries.append(summary)

    # 跨库只打印，不落盘均值 —— 三库题型构成不同，合并均值会被构成主导。
    print("\n" + "=" * 78)
    print("实验 E：Chat 基座对比（逐库，配对检验在交集样本上）")
    print("=" * 78)
    for s in summaries:
        for label, agg in s["labels"].items():
            print(f"[{s['game']}] {label}: judge={agg.get('judge_score')} "
                  f"correct={agg.get('correctness')} faith={agg.get('faithfulness')} "
                  f"n_ok={agg.get('n_ok')}/{agg.get('n')} "
                  f"tok={agg.get('avg_prompt_tokens')} 引用={agg.get('ref_coverage')} "
                  f"延迟={agg.get('avg_latency_ms')}ms "
                  f"judge_err={agg.get('judge_errors')}")
        p = s["paired"]
        if p:
            print(f"  配对 Δ={p['mean_diff']:+.4f} t={p['t']} p={fmt_p(p['p'])} "
                  f"Cohen's d={p['cohens_d']} n={p['n']}")
        print()

    print("注意：三库题型构成不同（见各自产物的 _meta.subset_composition），"
          "上面的数只应在同一库内做 E1→E2 对比，跨库不做均值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
