"""感知延迟实测：用户等多久才看到第一个字。

背景：用户反馈"延迟还有点高"。但"延迟"对用户来说不是总耗时，
而是**从点发送，到屏幕上出现第一个字**的时间（TTFT）。
总耗时一样的情况下，流式能让感知延迟降低一个数量级。

这个脚本测三件事：
  A. 非流式：多久拿到完整答案
  B. 流式：多久看到第一个字（TTFT），多久全部出完
  C. 流式首字 vs 检索耗时：端到端时间到底花在哪

用法：
    .venv\\Scripts\\python.exe scripts\\probe_stream_latency.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.llm.client import get_llm

API = "http://127.0.0.1:8000/api"
Q = "太空封印站的两件套效果是什么？"


def post(path: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8")), (time.perf_counter() - t0) * 1000


async def measure_stream() -> dict:
    """SSE 流式：记录首字时间 / 全部完成时间 / 收到的事件类型。"""
    req = urllib.request.Request(
        API + "/chat/stream", data=json.dumps({"question": Q}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    first_ms = None
    events: list[str] = []
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            now = (time.perf_counter() - t0) * 1000
            ev = json.loads(line[5:].strip())
            events.append(ev.get("event", "?") if "event" in ev else "data")
            if first_ms is None and ev:
                first_ms = now
    return {"ttft_ms": first_ms or 0, "total_ms": (time.perf_counter() - t0) * 1000,
            "n_events": len(events)}


async def main() -> None:
    s = get_settings()
    llm = get_llm(s)
    print(f"模型：{s.deepseek_chat_model}\n")

    print("=" * 74)
    print("  A. 直接调 LLM（非流式）—— 后端 /api/chat 走的就是这条路")
    print("=" * 74)
    msgs = [{"role": "system", "content": "你是星穹铁道知识库助手，用中文简短回答。"},
            {"role": "user", "content": Q}]
    llm.clear_caches()
    t0 = time.perf_counter()
    out = await llm.chat(msgs, max_tokens=400)
    t_llm = (time.perf_counter() - t0) * 1000
    print(f"  纯 LLM 调用耗时        : {t_llm:.0f} ms")
    print(f"  completion tokens      : {out['usage']['completion_tokens']}")

    print("\n" + "=" * 74)
    print("  B. 检索环节（/api/retrieve，A3 完整链路）")
    print("=" * 74)
    for label, kw in (("A3 完整", {}), ("A2 无重排", {"use_rerank": False}),
                      ("A0 纯本地", {"use_rerank": False, "use_vector": False})):
        _, ms = post("/retrieve", {"query": Q, "top_k": 5, **kw})
        print(f"  {label:<22}: {ms:>6.0f} ms")

    print("\n" + "=" * 74)
    print("  C. 端到端：/api/chat（非流式）vs /api/chat/stream（流式）")
    print("=" * 74)
    _, t_chat = post("/chat", {"question": Q})
    print(f"  非流式总耗时           : {t_chat:.0f} ms   ← 用户全程干等")

    try:
        st = await measure_stream()
        print(f"  流式首字 TTFT          : {st['ttft_ms']:.0f} ms   ← 用户第一次看到反馈")
        print(f"  流式全部完成           : {st['total_ms']:.0f} ms   （{st['n_events']} 个事件）")
        gain = t_chat - st["ttft_ms"]
        print(f"\n  感知延迟改善           : {gain:.0f} ms（{t_chat:.0f} → {st['ttft_ms']:.0f}，"
              f"降 {gain / t_chat * 100:.0f}%）")
        print(f"  总耗时差异             : {abs(t_chat - st['total_ms']):.0f} ms（基本一致）")
    except Exception as exc:
        print(f"  [流式不可用] {type(exc).__name__}: {str(exc)[:120]}")
        print("  （后端没起？在项目根跑 .venv\\Scripts\\python.exe run_server.py）")


if __name__ == "__main__":
    asyncio.run(main())