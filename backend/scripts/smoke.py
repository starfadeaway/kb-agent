"""端到端冒烟：一条命令验证整条链路是否还活着。

覆盖：配置加载 → 向量索引可读 → 混合检索 → 工具 → ReAct → FastAPI 接口。
任何一步失败都会明确指出是哪一层，不让问题被吞掉。

用法：
    .venv\\Scripts\\python.exe backend\\scripts\\smoke.py
    .venv\\Scripts\\python.exe backend\\scripts\\smoke.py --skip-agent   # 跳过消耗 token 的 Agent 环节
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402

PASS = "\u2713"
FAIL = "\u2717"


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{PASS if ok else FAIL} {name}{('  ' + detail) if detail else ''}")
    return ok


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sr")
    ap.add_argument("--skip-agent", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    failures = 0

    # 1. 配置与密钥
    failures += not check("配置加载", True, f"embed={s.embed_model_main} rerank={s.rerank_model}")
    failures += not check("DeepSeek 密钥", bool(s.deepseek_api_key))
    failures += not check("SiliconFlow 密钥", bool(s.siliconflow_api_key))

    # 2. 向量索引（必须能从磁盘读回来，这是曾经崩掉的地方）
    from app.store.vector_store import get_vector_store

    store = get_vector_store(s)
    n = store.count(args.game)
    failures += not check("向量索引可读", n > 0, f"{store.collection_name(args.game)}: {n} chunk")

    # 3. 混合检索
    from app.llm.client import get_llm
    from app.retrieval.hybrid import HybridRetriever, RetrievalConfig

    llm = get_llm(s)
    r = HybridRetriever(settings=s, llm=llm, store=store)
    t0 = time.perf_counter()
    hits = await r.retrieve("位面饰品太空封印站的两件套效果是什么", RetrievalConfig(game=args.game, final_top_k=3))
    dt = int((time.perf_counter() - t0) * 1000)
    top = hits[0].chunk.page_title if hits else ""
    failures += not check("混合检索 top-1 命中", top == "太空封印站", f"{top!r} ({dt}ms, {len(hits)} hits)")

    # 4. 工具
    from app.agent.tools import ToolContext, call_tool

    ctx = ToolContext.build(args.game)
    text, ok = await call_tool("calculator", {"expression": "(0.5+0.25)*200"}, ctx, timeout=10)
    failures += not check("calculator", ok and "150" in text, text.strip())
    bad, ok2 = await call_tool("search_kb", {"query": "测试", "game": "xyz"}, ctx, timeout=15)
    failures += not check("工具参数校验", (not ok2) and bad.startswith("[错误]"), bad[:46].replace("\n", " "))
    inj, ok3 = await call_tool("calculator", {"expression": "__import__('os')"}, ctx, timeout=10)
    failures += not check("注入防护", (not ok3) and "非法字符" in inj, inj[:40])

    # 5. Agent（消耗 token，默认开启）
    if not args.skip_agent:
        from app.agent.react import ReActAgent, route_game

        failures += not check(
            "游戏路由", route_game("绝区零的音擎怎么配", "sr", "sr", "zzz") == "zzz"
        )
        agent = ReActAgent(settings=s, llm=llm)
        st = await agent.run("位面饰品太空封印站的两件套效果是什么？", ctx=ctx)
        good = st.stop_reason == "answer" and "12" in st.answer
        failures += not check(
            "ReAct 端到端", good,
            f"stop={st.stop_reason} steps={len(st.steps)} {st.elapsed_ms}ms tok={st.prompt_tokens}/{st.completion_tokens}",
        )
        print(f"    答案节选：{st.answer[:90].replace(chr(10), ' ')}")

    # 6. 清洗/分块（离线，直接读缓存）
    from app.ingest.template_parser import clean_wikitext

    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / args.game
    cached = sorted(raw_dir.glob("*.txt")) if raw_dir.is_dir() else []
    if cached:
        wt = cached[0].read_text(encoding="utf-8")
        clean = clean_wikitext(wt)
        ratio = len(clean) / max(len(wt), 1)
        failures += not check("模板展平未丢内容", ratio > 0.35, f"保留率 {ratio:.0%}（{len(clean)}/{len(wt)} 字符）")
    else:
        failures += not check("原始缓存存在", False, f"未找到 {raw_dir}")

    print("\n" + ("全部通过" if failures == 0 else f"{failures} 项未通过"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
