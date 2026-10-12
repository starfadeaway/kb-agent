"""MiniMax 429 的归因排查。

现象：充值后仍报 429 `已达到 Token Plan 用量上限：请升级 Token Plan 套餐或购买积分补充用量`

关键点：报错说的是 **Token Plan（套餐额度）**，不是「账户余额不足」。
这两者在 MiniMax 是两套东西 —— 充值可能加到了余额，但套餐的月度/日度
token 额度是独立的硬上限。

所以要分清三种可能，不能只凭「我充了钱」就断定已恢复：
  A. 端点不对 —— 国际站 / 国内站的账号和额度是分开的
  B. 模型名不对 —— 套餐可能不覆盖 M2.5
  C. 确实是套餐额度用尽，需要在控制台升级 Token Plan

用法：
    .venv\\Scripts\\python.exe scripts\\probe_minimax_endpoints.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings

# 这个探针存在的意义就是排查"端点/模型名配错导致 429"，
# 所以它必须读**和线上走的那份完全一样的配置**。
# 原来这里自己 load_dotenv + os.getenv 绕过了配置层：一旦 .env 里同时存在
# 国际站/国内站两个键、或键名和 config.py 不一致，这个探针给出的结论就是错的 ——
# 而它恰恰是排查 429 时最该被信任的那个脚本。
_s = get_settings()
KEY = _s.minimax_api_key

# 国际站 vs 国内站是两套账号体系，套餐额度不互通
ENDPOINTS = [
    ("国际站", "https://api.minimax.chat/v1"),
    ("国内站", "https://api.minimaxi.com/v1"),
]
MODELS = ["MiniMax-M2.5", "MiniMax-M2", "MiniMax-M1", "abab6.5s-chat", "MiniMax-Text-01"]


async def try_one(url: str, model: str) -> tuple[bool, str, int]:
    from openai import AsyncOpenAI

    try:
        c = AsyncOpenAI(api_key=KEY, base_url=url, timeout=25.0)
        t0 = asyncio.get_event_loop().time()
        r = await c.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "回复'好'"}],
            max_tokens=16, temperature=0.0,
        )
        dt = int((asyncio.get_event_loop().time() - t0) * 1000)
        return True, (r.choices[0].message.content or "")[:40], dt
    except Exception as exc:  # noqa: BLE001
        m = str(exc)
        if "Token Plan" in m or "2056" in m:
            why = "套餐额度用尽"
        elif "does not exist" in m or "20012" in m:
            why = "模型不存在"
        elif "401" in m or "invalid" in m.lower():
            why = "密钥无效"
        elif "429" in m:
            why = "限流(非套餐)"
        else:
            why = type(exc).__name__
        return False, why, 0


async def main() -> None:
    if not KEY:
        print("MINIMAX_API_KEY 未配置")
        return
    print(f"密钥前缀：{KEY[:8]}…（不显示完整）\n")
    print(f"{'端点':<10}{'模型':<22}{'状态':<16}{'耗时':>8}")
    print("-" * 60)

    ok_any = False
    for label, url in ENDPOINTS:
        for model in MODELS:
            ok, msg, dt = await try_one(url, model)
            if ok:
                ok_any = True
            print(f"{label:<10}{model:<22}{'✅ OK' if ok else '❌ ' + msg:>16}{dt if ok else '-':>8}")
        print()

    print("=" * 60)
    if ok_any:
        print("至少有一个组合可用 —— 把对应的 base_url 和模型名写进 .env 即可跑实验 E。")
    else:
        print("全部组合都失败。三种可能，按排查优先级：")
        print("  1. **Token Plan 额度确实用尽** —— 报错原文是「升级 Token Plan 套餐」，")
        print("     充值【余额】不等于提升【套餐额度】，需要在控制台单独升级套餐。")
        print("  2. **充值的是另一个站点** —— 国际站 api.minimax.chat 和国内站 api.minimaxi.com")
        print("     账号体系与额度完全独立，充错站的话这里永远 429。")
        print("  3. **套餐不覆盖 M2.5** —— 老套餐可能只含 M1/Text-01。")
        print()
        print("最快的确认方式：登录 MiniMax 控制台看【账户余额】和【Token Plan 额度】两项，")
        print("看哪一项没动，或者套餐里列的模型名是什么。")


# 同 probe_ctx_reuse：模块顶层 asyncio.run 会让 `import` 触发
# 2 端点 × 5 模型 = 10 次真实 API 请求，排查 429 时最不该被误触发的脚本。
if __name__ == "__main__":
    asyncio.run(main())