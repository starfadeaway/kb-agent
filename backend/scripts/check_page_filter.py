"""用真实语料验证页面过滤器：必须只剔掉模板定义页，不误伤任何内容页。

直接拿 data/raw 下三个游戏的全部页面跑一遍，输出被剔掉的清单。
这是"宁可漏杀不可错杀"这类过滤器的唯一可信验收方式 ——
单元测试用构造样本只能证明函数逻辑，证明不了阈值选得对不对。

刻意遍历 `.txt` 而不是 `.meta.json`：语料里有 3 个页面只有正文没有侧车元数据
（抓取中途被打断），只遍历 meta 会把它们整个漏掉 ——
其中 `1273_模板_遗器套装.txt` 恰好就是本次要过滤的模板页，漏了它这个检查就是假绿。

用法：
    python scripts/check_page_filter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ingest.page_filter import should_index  # noqa: E402
from probe_io import save_probe  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"

# 落盘时文件名里的非法字符（含全角冒号）会被替换成 _
_SANITIZED = {"：": ":", "／": "/", "？": "?", "！": "!", "＂": '"', "＃": "#"}


def title_from_meta(game_dir: Path, page_id: str) -> str | None:
    meta_p = game_dir / f"{page_id}.meta.json"
    if not meta_p.is_file():
        return None
    try:
        return json.loads(meta_p.read_text(encoding="utf-8")).get("title") or None
    except ValueError:
        return None


def title_from_filename(name: str) -> str:
    """`1273_模板_遗器套装.txt` → `模板:遗器套装`（把被替换掉的冒号还原）。"""
    stem = name[:-4] if name.endswith(".txt") else name
    _, _, title = stem.partition("_")
    for bad, good in _SANITIZED.items():
        title = title.replace("_" + bad, good)
    return title.replace("_", ":", 1) if title.startswith("_") else title


def main() -> int:
    if not DATA_ROOT.is_dir():
        print(f"找不到语料目录 {DATA_ROOT}，请先跑 crawl_all.py")
        return 1

    total = kept = 0
    dropped: list[tuple[str, str, str]] = []
    no_meta: list[str] = []

    for game_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        for txt_p in sorted(game_dir.glob("*.txt")):
            total += 1
            page_id = txt_p.name.split("_", 1)[0]
            title = title_from_meta(game_dir, page_id)
            if title is None:
                no_meta.append(f"{game_dir.name}/{txt_p.name}")
                title = title_from_filename(txt_p.name)
            ok, why = should_index(title)
            if ok:
                kept += 1
            else:
                dropped.append((game_dir.name, title, why))

    print(f"扫描 {total} 页，保留 {kept}，过滤 {len(dropped)}")
    print("-" * 66)
    for game, title, why in dropped:
        print(f"  [{game}] {title}   ← {why}")

    if no_meta:
        print("-" * 66)
        print(f"⚠ {len(no_meta)} 个页面缺 .meta.json（离线重建会丢页，跑 backfill_meta.py 补）：")
        for f in no_meta:
            print(f"    {f}")

    # 过滤比例是安全阀：真实语料里模板页应该是"个位数"，
    # 一旦占比超过 5%，多半是阈值太狠把内容页也砍了
    ratio = len(dropped) / max(total, 1)
    print("-" * 66)
    print(f"过滤比例 {ratio:.2%}")

    # 落盘。"内容启发式误杀 60%"是 README 里第 16 条教训的全部证据，
    # 而"现在只按标题过滤、比例 0.39%"是它被否掉的依据 —— 两者都只有 stdout。
    save_probe(
        "check_page_filter",
        {
            "total": total, "kept": kept, "dropped_count": len(dropped),
            "drop_ratio": round(ratio, 6),
            "dropped": [{"game": g, "title": t, "reason": w} for g, t, w in dropped],
            "missing_meta": no_meta,
            "threshold": "drop_ratio > 0.05 判定为阈值过狠",
        },
        context={"corpus_root": str(DATA_ROOT), "games": sorted(
            p.name for p in DATA_ROOT.iterdir() if p.is_dir())},
    )

    if ratio > 0.05:
        print("✗ 过滤比例过高，阈值可能误杀内容页")
        return 1
    if not dropped:
        print("⚠ 一页都没过滤掉，过滤器可能没生效")
        return 1
    print("✓ 比例正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())