"""补齐本地缓存的侧车元数据（page_id / 标题 / 修订时间 / 修订号）。

只请求修订信息不请求正文，比重新下载全文轻得多。
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.ingest.pipeline import raw_meta_path  # noqa: E402
from app.ingest.wiki_client import WikiClient  # noqa: E402


def cached_pages(game: str) -> list[tuple[int, str]]:
    d = Path(__file__).resolve().parents[2] / "data" / "raw" / game
    if not d.is_dir():
        return []
    out = []
    for f in d.glob("*.txt"):
        m = re.match(r"^(\d+)_(.+)\.txt$", f.name)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


async def main() -> None:
    game = sys.argv[1] if len(sys.argv) > 1 else "sr"
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    s = get_settings()
    pages = cached_pages(game)
    todo = [(pid, title) for pid, title in pages if not raw_meta_path(pid, game).is_file()]
    print(f"缓存 {len(pages)} 页，待补元数据 {len(todo)} 页")
    if not todo:
        return
    # 按 page_id 查，不按 title —— 文件名里的 `:` 已被替换成 `_`，
    # 拿反推出来的标题去查会一个都查不到，还不报错。
    print("待补 page_id: " + ", ".join(str(pid) for pid, _ in todo))

    written = 0
    async with WikiClient(
        game, base_url=s.wiki_base, min_interval=interval, max_retry=s.wiki_max_retry
    ) as wc:
        for i in range(0, len(todo), 20):
            ids = [pid for pid, _ in todo[i : i + 20]]
            try:
                metas = await wc.fetch_meta_by_ids(ids)
            except Exception as exc:  # noqa: BLE001
                print(f"批次 {i} 失败: {type(exc).__name__}: {exc}")
                continue
            for m in metas:
                raw_meta_path(m.page_id, game).write_text(
                    json.dumps(
                        {
                            "page_id": m.page_id,
                            "title": m.title,
                            "last_revision": m.last_revision or "",
                            "last_revision_ts": m.last_revision or "",
                            "revision_id": m.revision_id,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                written += 1
            print(f"进度 {min(i + 20, len(todo))}/{len(todo)}，已写 {written}")
    print(f"完成，写入 {written} 份侧车元数据")


if __name__ == "__main__":
    asyncio.run(main())