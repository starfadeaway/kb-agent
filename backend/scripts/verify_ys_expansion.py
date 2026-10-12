"""原神语料扩展后的端到端验收。

设计要点（单变量）：
- 三个 query 的答案分别落在**本次新增的三个分类**里（七国徽印 / 神瞳 / 冒险道具），
  扩展前的 236 chunk 覆盖不到，扩展后必须命中。
- 同时跑两个跨游戏隔离 query：一个 ys 库内、一个明确未收录的游戏。
- 先跑 /retrieve（确定性、零 LLM）判检索本身，再跑 /chat 判 Agent 全链路。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000/api"
TIMEOUT = 180

# (query, game, 期望命中的页面关键词)
#
# 用例只选**具体条目页**，不选「什么是光锥」这类定义题：
# 知识库按页面标题建的，存的是「苍雷紫电的记忆」这种具体光锥，
# 没有一篇讲"光锥是什么"的概览页 —— 定义题结构上就无解，
# 拿它当断言只会测出用例设计有问题，而不是检索有没有变好。
RETRIEVE_CASES = [
    ("原神里岩之印怎么获得？", "ys", "岩之印"),
    ("散失的草神瞳在哪里可以找到？", "ys", "草神瞳"),
    ("冒险家这个角色的突破材料是什么？", "ys", "冒险家"),
    ("绝区零的音擎「电磁暴」-壹式有什么效果？", "zzz", "电磁暴"),
    ("星穹铁道的光锥等价交换有什么效果？", "sr", "等价交换"),
]

CHAT_CASES = [
    # 库内问题：扩展前 Agent 会因为系统提示词写死"只覆盖两款"而误拒
    ("原神里岩之印怎么获得？", "in_scope"),
    # 未收录表里的游戏：必须走规则层拦截（0 步 out_of_scope），不是模型自觉
    ("明日方舟的罗德岛是什么？", "refuse"),
]


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    fails: list[str] = []

    print("=" * 72)
    print("[1] /retrieve —— 检索本身（确定性，无 LLM）")
    print("=" * 72)
    for q, game, expect in RETRIEVE_CASES:
        t0 = time.perf_counter()
        try:
            res = post("/retrieve", {"query": q, "game": game, "top_k": 5})
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  ✗ {q}\n      请求失败: {e}")
            fails.append(f"retrieve 请求失败: {q}")
            continue
        dt = (time.perf_counter() - t0) * 1000

        results = res.get("hits") or []
        titles = [h.get("page_title", "") for h in results]
        blob = " ".join(titles)
        ok = expect in blob
        print(f"  {'✓' if ok else '✗'} [{game}] {q}  ({dt:.0f}ms)")
        print(f"      top: {titles[:3]}")
        if not ok:
            fails.append(f"retrieve 未命中 {expect}: {q}")

    print()
    print("=" * 72)
    print("[2] /chat —— Agent 全链路（含 LLM）")
    print("=" * 72)
    for q, kind in CHAT_CASES:
        t0 = time.perf_counter()
        try:
            res = post("/chat", {"question": q, "game": "ys"})
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  ✗ {q}\n      请求失败: {e}")
            fails.append(f"chat 请求失败: {q}")
            continue
        dt = (time.perf_counter() - t0) * 1000
        answer = res.get("answer", "")
        steps = res.get("steps")
        stop = res.get("stop_reason")
        unc = res.get("uncovered_game")

        print(f"  · [{kind}] {q}")
        print(f"      {dt:.0f}ms  steps={steps}  stop_reason={stop}"
              + (f"  uncovered={unc}" if unc else ""))
        print(f"      {answer[:220].replace(chr(10), ' ')}")
        print()

        if kind == "refuse":
            if stop != "out_of_scope":
                fails.append(f"未收录游戏未被规则层拦截: {q} (stop={stop})")
            if steps not in (0, None):
                fails.append(f"拒答路径仍在调工具({steps} 步): {q}")
            if "知识库" not in answer and "未收录" not in answer:
                fails.append(f"拒答文案未说明未收录: {answer[:80]}")
        else:
            if stop == "out_of_scope":
                fails.append(f"库内问题被误拒: {q}")
            if "岩之印" not in answer:
                fails.append(f"库内答案未提及岩之印: {answer[:80]}")

    print("=" * 72)
    if fails:
        print(f"失败 {len(fails)} 项：")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())