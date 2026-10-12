"""LanceDB 可行性验证：写入 → 检索 → **换进程读回**。

ChromaDB 就是在最后一步挂的（索引不落盘，Error loading hnsw index），
所以这里必须真的起第二个进程验证，不能只看同一个进程里能不能查。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
LANCE_DIR = ROOT / "data" / "lance_probe"


def write() -> dict:
    import lancedb
    from app.core.config import get_settings
    from app.store.vector_store import get_vector_store

    s = get_settings()
    vs = get_vector_store(s)
    col = vs.get_collection("sr")
    n = col.count()
    ids, docs, metas = col.ids, col.documents, col.metadatas
    emb = col._emb  # noqa: SLF001

    rows = [
        {
            "id": ids[i],
            "vector": emb[i].tolist(),
            "text": docs[i],
            "page_title": metas[i].get("page_title", ""),
            "section": metas[i].get("section", ""),
            "last_revision_ts": metas[i].get("last_revision_ts", ""),
            "game": metas[i].get("game", "sr"),
        }
        for i in range(n)
    ]

    if LANCE_DIR.exists():
        import shutil

        shutil.rmtree(LANCE_DIR)
    LANCE_DIR.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    db = lancedb.connect(str(LANCE_DIR))
    tbl = db.create_table("kb_sr_probe", data=rows, mode="overwrite")
    tbl.create_index(metric="cosine")
    write_s = time.perf_counter() - t0

    # 同进程内立刻查一次，确认索引真的建起来了
    probe = rows[0]["vector"]
    tbl.search(probe).limit(3).to_list()

    size_mb = sum(f.stat().st_size for f in LANCE_DIR.rglob("*") if f.is_file()) / 1024 / 1024
    return {
        "rows": n,
        "write_s": round(write_s, 2),
        "disk_mb": round(size_mb, 2),
        "dim": len(rows[0]["vector"]),
        "files": sorted(p.name for p in LANCE_DIR.iterdir()),
    }


def read() -> dict:
    """第二个进程：只靠磁盘上的文件把索引读回来并检索。"""
    import asyncio

    import lancedb

    db = lancedb.connect(str(LANCE_DIR))
    tbl = db.open_table("kb_sr_probe")
    n = tbl.count_rows()

    from app.core.config import get_settings
    from app.llm.client import get_llm

    s = get_settings()
    qv = asyncio.run(get_llm(s).embed(["太空封印站的两件套效果是什么"], model=s.embed_model_main, normalize=True))

    t0 = time.perf_counter()
    res = tbl.search(qv[0]).limit(5).to_list()
    dt = int((time.perf_counter() - t0) * 1000)

    # 带过滤条件的检索，验证 metadata 过滤在跨进程后依然可用
    filtered = tbl.search(qv[0]).where("page_title = '太空封印站'").limit(5).to_list()

    return {
        "rows_read_back": n,
        "search_ms": dt,
        "top1": res[0]["page_title"],
        "top1_score": round(1 - float(res[0]["_distance"]), 4),
        "filtered_hits": len(filtered),
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "write"
    out = write() if mode == "write" else read()
    print(json.dumps(out, ensure_ascii=False, indent=2))