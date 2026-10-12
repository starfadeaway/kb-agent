"""验收：`/api/retrieve` 是否真的听 `.env` 的话。

背景：`RETRIEVE_TOP_K` / `RETRIEVE_CANDIDATE` / `RRF_K` 三个 env var
曾经是**死的** —— `config.py` 里定义了、`.env.example` 里列了，
但 `main.py` 构造 `RetrievalConfig(...)` 时压根没传，
于是改 `.env` 完全不生效、检索照常跑、不报任何错。

只验证「默认值恰好等于 .env 里的默认值」是**假绿** —— 那样即使接线断了
也照样通过。所以这个脚本的断言全部**从 Settings 动态读取**，
配了非默认值时一样成立。

跑法（后端起着时）：
    python backend/scripts/check_config_wiring.py
想验证非默认值，先改 .env 再重启后端。
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings  # noqa: E402

BASE = "http://127.0.0.1:8000/api"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    s = get_settings()
    print(f"当前 .env 生效值：TOP_K={s.retrieve_top_k} "
          f"CANDIDATE={s.retrieve_candidate} RRF_K={s.rrf_k}\n")

    q = {"query": "太空封印站的两件套效果是什么"}

    # 1) 请求里什么都不给 —— 必须全部由 .env 兜底
    d = post("/retrieve", dict(q))
    c = d["config"]
    print("== 请求不带任何检索参数（应由 .env 兜底）==")
    print(f"  rrf_k             = {c['rrf_k']}")
    print(f"  final_top_k       = {c['final_top_k']}   命中 {len(d['hits'])} 条")
    print(f"  bm25_top          = {c['bm25_top']}")
    print(f"  vector_top        = {c['vector_top']}")
    print(f"  rerank_candidates = {c['rerank_candidates']}")
    print(f"  embed_model       = {c['embed_model']}")

    # 2) 请求显式给 top_k —— 必须压过 .env
    d2 = post("/retrieve", {**q, "top_k": 2})
    print(f"\n== 请求显式 top_k=2（必须压过 .env={s.retrieve_top_k}）==")
    print(f"  final_top_k = {d2['config']['final_top_k']}  命中 {len(d2['hits'])} 条")

    checks = {
        "RRF_K 由 .env 决定": c["rrf_k"] == s.rrf_k,
        "RETRIEVE_TOP_K 由 .env 决定": c["final_top_k"] == s.retrieve_top_k,
        "实际命中条数 = 配置的 top_k": len(d["hits"]) == s.retrieve_top_k,
        "RETRIEVE_CANDIDATE 三个候选字段一致且等于 .env": (
            c["bm25_top"] == c["vector_top"] == c["rerank_candidates"] == s.retrieve_candidate
        ),
        "EMBED_MODEL_MAIN 生效": c["embed_model"] == s.embed_model_main,
        "请求里的 top_k 压过 .env": (
            d2["config"]["final_top_k"] == 2 and len(d2["hits"]) == 2
        ),
    }
    print("\n-- 判定 --")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\n  注：这套断言是从 Settings 动态取值，因此把 .env 改成非默认值也能验。")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())