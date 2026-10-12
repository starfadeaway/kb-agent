"""判断 KV cache 能不能省延迟 —— 先测，再决定上不上。

背景：用户反馈延迟还高。直觉是"上下文长，上 KV cache（DeepSeek
上下文缓存）能省钱省时间"。但这个直觉可能是错的：

DeepSeek 命中缓存的**收益是按 token 计费的，不是按墙钟时间**。
而且缓存按「前缀」命中 —— 本项目每一步都会追加 tool 消息，
历史几乎不变，理论上应该能命中。

所以要分清三件事：
  1. 缓存命中率能不能上去（靠前缀稳定性）
  2. 命中之后**墙钟时间**到底降不降（这才是用户关心的"延迟"）
  3. token 费用降多少（这才是 KV cache 的真实卖点）

用法：
    .venv\\Scripts\\python.exe scripts\\probe_kv_cache.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import get_settings
from app.llm.client import get_llm
from probe_io import save_probe


def _cache_field(resp) -> tuple[int, int]:
    """DeepSeek 在 usage 里返回 prompt_cache_hit_tokens / prompt_cache_miss_tokens。"""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0),
        int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0),
    )


async def probe_prefix_stability() -> None:
    """模拟 Agent 的消息增长模式：每步追加 tool 结果，历史不变。

    这正是 KV cache 能命中的理想场景。测 5 轮看命中率是否上去、
    每轮墙钟时间是否随轮次下降。
    """
    s = get_settings()
    llm = get_llm(s)

    # 造一段足够长的静态前缀（模拟系统提示 + 多轮历史 + 检索结果）
    filler = "以下是知识库检索到的游戏资料片段，请据此回答用户问题。\n"
    body = filler * 60
    messages: list[dict] = [
        {"role": "system", "content": body},
        {"role": "user", "content": "太空封印站的两件套效果是什么？"},
    ]

    print("=" * 78)
    print("  场景 1：前缀稳定，每轮追加一条 tool 结果（= Agent 的真实行为）")
    print("=" * 78)
    print(f"{'轮':>3}{'prompt tokens':>16}{'cache hit':>12}{'命中率':>9}{'墙钟ms':>10}")

    turns: list[dict] = []
    for turn in range(1, 6):
        t0 = time.perf_counter()
        client = llm.deepseek
        resp = await client.chat.completions.create(
            model=s.deepseek_chat_model,
            messages=messages,
            max_tokens=40,
            temperature=0.0,
        )
        ms = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        pt = int(usage.prompt_tokens or 0)
        hit, miss = _cache_field(resp)
        rate = (hit / pt * 100) if pt else 0.0
        turns.append({"turn": turn, "prompt_tokens": pt, "cache_hit_tokens": hit,
                      "cache_miss_tokens": miss, "hit_rate_pct": round(rate, 1),
                      "wall_ms": round(ms)})
        print(f"{turn:>3}{pt:>16}{hit:>12}{rate:>8.1f}%{ms:>10.0f}")

        # 追加 assistant + tool 消息，模拟 Agent 继续往下走
        messages.append({"role": "assistant", "content": resp.choices[0].message.content or ""})
        messages.append(
            {"role": "user", "content": f"补充资料（第{turn}轮）：位面饰品两件套效果为攻击力提升。"}
        )

    print("\n  → 若命中率随轮次上升，说明前缀稳定，DeepSeek 侧缓存能生效。")
    return turns


async def probe_latency_vs_length() -> None:
    """延迟到底由什么决定：上下文长度，还是首 token 生成？

    如果是首 token 生成占大头，加 KV cache 对延迟帮助有限。
    """
    s = get_settings()
    llm = get_llm(s)

    print("\n" + "=" * 78)
    print("  场景 2：延迟到底由什么决定？（同一模型，只变上下文长度）")
    print("=" * 78)
    print(f"{'上下文':>12}{'prompt tokens':>16}{'首token ms':>14}{'总耗时 ms':>12}")

    rows: list[dict] = []
    for mult in (5, 30, 60, 120):
        body = ("知识库检索到的游戏资料片段，请据此回答。\n" * mult)
        messages = [
            {"role": "system", "content": body},
            {"role": "user", "content": "太空封印站的两件套效果是什么？用一句话。"},
        ]
        t0 = time.perf_counter()
        stream = await llm.deepseek.chat.completions.create(
            model=s.deepseek_chat_model, messages=messages,
            stream=True, max_tokens=128, temperature=0.0,
        )
        first_ms = None
        n = 0
        async for _chunk in stream:
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000
            n += 1
        total_ms = (time.perf_counter() - t0) * 1000
        pt = len(body) // 2 + 30     # 粗略估计，中文约 1 字 0.5-0.7 token
        rows.append({"filler_lines": mult, "prompt_tokens_est": pt,
                     "first_token_ms": round(first_ms or 0),
                     "total_ms": round(total_ms), "n_chunks": n})
        print(f"{mult:>10} 行{pt:>16}{first_ms or 0:>14.0f}{total_ms:>12.0f}")

    print("\n  → 首 token 延迟随上下文增长缓慢 = 预处理已并行；")
    print("    总耗时的大头是生成。加 KV cache 主要省【费用】，省【延迟】有限。")
    return rows


async def main() -> None:
    s = get_settings()
    print(f"模型：{s.deepseek_chat_model}")
    print(f"端点：{s.deepseek_base_url}\n")
    turns = await probe_prefix_stability()
    by_len = await probe_latency_vs_length()

    # 落盘。ARCHITECTURE 里"KV cache 自动生效、命中率 80%+"这句结论
    # 全部依赖这张五轮表 —— 没有原始文件就只剩一个没法核对的百分比。
    save_probe(
        "probe_kv_cache",
        {"prefix_stability": turns, "latency_vs_length": by_len},
        context={"model": s.deepseek_chat_model, "endpoint": s.deepseek_base_url,
                 "note": "prompt_tokens 在场景 2 是按字符数粗估的，不是 usage 实测值"},
    )


if __name__ == "__main__":
    asyncio.run(main())