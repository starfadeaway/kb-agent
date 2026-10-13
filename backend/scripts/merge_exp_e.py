"""把分档跑出来的实验 E 产物合并成一份 `exp_E.json`。

为什么需要它：实验 E 的两档串行跑会超过单任务 3600s 上限（E2 还会撞
MiniMax 额度），所以必须 `--labels E1` / `--labels E2` 分两次跑。
但 `exp_e.py` 每次都把自己的 results 写进 `--out` 指定的文件 ——
第二次跑会把第一次的结果**整个覆盖掉**（E1 的结果就这么丢过一次）。

所以流程是：
    exp_e.py --labels E1 --out exp_E1.json
    exp_e.py --labels E2 --out exp_E2.json
    merge_exp_e.py                     # -> exp_E.json

合并产物必须自证来源：两档的 case 集合、每档的行数、以及每个源文件的
sha256 都写进 `_meta.sources`。否则合并后的数字没人能说清是哪两次跑拼的 ——
这正是 `build_meta` 想解决的那类问题，这里不能只打个时间戳了事。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_common import EVAL_DIR  # noqa: E402

SOURCES = ["exp_E1.json", "exp_E2.json"]
OUT = "exp_E.json"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _load(p: Path) -> list[dict]:
    """只取带 label 的档位块（出处块以 `_meta` 为键，没有 label）。"""
    data = json.loads(p.read_text(encoding="utf-8"))
    return [d for d in data if isinstance(d, dict) and d.get("label")]


def main() -> int:
    parts: list[dict] = []
    sources: list[dict] = []
    seen_labels: set[str] = set()

    for name in SOURCES:
        p = EVAL_DIR / name
        if not p.is_file():
            print(f"!! 缺源文件：{name}（请先跑对应的那一档）")
            return 1
        blocks = _load(p)
        for b in blocks:
            label = b.get("label")
            if not label:
                print(f"!! {name} 里有缺 label 的块，疑似格式不对")
                return 1
            if label in seen_labels:
                # 合并重复档是静默出错的高发点，宁可停下
                print(f"!! 档位重复：{label} 同时出现在多个源文件里，拒绝合并")
                return 1
            seen_labels.add(label)
            parts.append(b)
        sources.append({
            "file": name,
            "sha256_16": _sha256(p),
            "labels": [b.get("label") for b in blocks],
            "rows_per_label": {b.get("label"): len(b.get("rows", [])) for b in blocks},
        })
        print(f"  读入 {name}: {[b.get('label') for b in blocks]}")

    if not parts:
        print("!! 没有可合并的档位")
        return 1

    # 各档 case 集合写进出处块：E1/E2 必须跑同一批评测题，否则配对检验无意义
    case_sets = {b["label"]: sorted(r["case_id"] for r in b.get("rows", []))
                 for b in parts}
    union = set().union(*case_sets.values()) if case_sets else set()
    for label, ids in case_sets.items():
        missing = union - set(ids)
        flag = "⚠️ 缺题" if missing else "与并集一致"
        print(f"  {label}: {len(ids)} 条 case，{flag}"
              + (f"（缺 {sorted(missing)}）" if missing else ""))

    out = list(parts)
    out.append({"_meta": {
        "script": Path(__file__).name,
        "merged_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": sources,
        "case_set_per_label": {k: len(v) for k, v in case_sets.items()},
        "note": "本文件由分档产物合并而来，per-label 聚合值沿用各自源文件",
    }})

    dst = EVAL_DIR / OUT
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已合并 {len(parts)} 档 -> {OUT}")
    print(f"  各档 case 数：{ {k: len(v) for k, v in case_sets.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
