"""从本地语料里提取真实字段值，用于撰写评测集的 gold。

**为什么不直接凭记忆写 gold**：EVALUATION.md 记了教训 —— 早期一条算术题
的前提本身就不成立（星铁暴击率副词条并不按每级 5% 成长），
模型答不出来反而是对的，一个没核对过的 gold 会把正确行为判成错。
所以这里的做法是：先从 data/raw 抽出真实值，再据此写题。

用法：
    python scripts/dump_fields.py --game zzz --title 艾莲
    python scripts/dump_fields.py --game ys --grep 圣遗物 --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.template_parser import clean_wikitext  # noqa: E402

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def all_pages(game: str) -> list[tuple[int, str, Path]]:
    d = RAW / game
    out = []
    for meta_p in sorted(d.glob("*.meta.json")):
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        hits = list(d.glob(f"{meta['page_id']}_*.txt"))
        if hits:
            out.append((meta["page_id"], meta["title"], hits[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--grep", default="")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--chars", type=int, default=700)
    args = ap.parse_args()

    if not (RAW / args.game).is_dir():
        print(f"没有 {args.game} 语料")
        return 1

    pages = all_pages(args.game)
    if args.title:
        pages = [p for p in pages if p[1] == args.title]
    if args.grep:
        pages = [p for p in pages if args.grep in p[1]]

    if not pages:
        print("没匹配到页面")
        return 1

    for pid, title, txt_p in pages[: args.limit]:
        raw = txt_p.read_text(encoding="utf-8", errors="ignore")
        text = clean_wikitext(raw)
        print("=" * 70)
        print(f"[{args.game}] page_id={pid}  {title}")
        print("=" * 70)
        print(text[: args.chars])
        print()

    print(f"（共匹配 {len(pages)} 页）")
    return 0


if __name__ == "__main__":
    sys.exit(main())