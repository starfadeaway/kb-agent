"""验证版本感知过滤**在 HTTP 入口上真的生效**（第 4 项收尾用）。

为什么不能用单元测试代替：单测只证明 `build_where` 会拼出正确的 where 子句，
证明不了"前端传进来的值一路活到向量库查询"。中间隔着 BM25 分支、
融合、重排，任何一段把 cfg 丢掉都不会报错，只是结果不变。

跑法：后端起着的时候执行 `python backend/scripts/probe_version_entry.py`
"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000/api/retrieve"


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        BASE, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    base_payload = {"query": "三月七 命途", "game": "sr", "top_k": 5}

    plain = post(base_payload)
    print("== 无过滤 ==")
    print(f"  config.min_revision_ts = {plain['config'].get('min_revision_ts')!r}")
    print(f"  config.page_titles     = {plain['config'].get('page_titles')!r}")
    for h in plain["hits"]:
        ts = h.get("last_revision_ts")
        print(f"  {h['rank']}. {h['page_title'][:24]}  rev_ts={ts}")

    stamps = [h.get("last_revision_ts") for h in plain["hits"] if h.get("last_revision_ts")]
    if not stamps:
        print("!! 响应里没有 last_revision_ts，无法继续验证")
        return 1

    # 核心不变式：返回的每一条都必须满足 rev_ts >= cutoff。
    # 判空用**远超语料上限**的阈值，不用"基线最晚时间戳" ——
    # 过滤语义是 >=（含等于），等于阈值那条本来就该留下，用它判空会误报。
    far_future = "2099-01-01T00:00:00Z"
    print(f"\n== min_revision_ts={far_future}（远超语料上限）==")
    none_left = post({**base_payload, "min_revision_ts": far_future})
    print(f"  命中 {len(none_left['hits'])} 条（期望 0 条）")
    ok_empty = len(none_left["hits"]) == 0

    # 反向对照：阈值早于全部语料时，结果必须和无过滤时**逐条一致**。
    # 少了这条就分不清"过滤生效"和"过滤根本没接上但恰好排序没变"。
    print("\n== 阈值早于全部语料（对照：应与无过滤逐条一致）==")
    far_past = "2020-01-01T00:00:00Z"
    ctrl = post({**base_payload, "min_revision_ts": far_past})
    same_ids = ([h["chunk_id"] for h in ctrl["hits"]]
                == [h["chunk_id"] for h in plain["hits"]])
    print(f"  命中 {len(ctrl['hits'])} 条，与基线逐条一致={same_ids}")
    ok_control = same_ids and len(ctrl["hits"]) == len(plain["hits"])

    # 取基线里**最晚**的时间戳当阈值：结果必须整体换一批，
    # 且每条都达标 —— 这才是"过滤真的改变了召回集"的正面证据。
    cutoff = max(stamps)
    print(f"\n== min_revision_ts={cutoff}（=基线最晚时间戳）==")
    filt = post({**base_payload, "min_revision_ts": cutoff})
    print(f"  命中 {len(filt['hits'])} 条")
    for h in filt["hits"]:
        print(f"  {h['rank']}. {h['page_title'][:24]}  rev_ts={h.get('last_revision_ts')}")
    ok_all_ge = all(h.get("last_revision_ts") >= cutoff for h in filt["hits"])
    changed = ([h["chunk_id"] for h in filt["hits"]]
               != [h["chunk_id"] for h in plain["hits"]])
    ok_shifted = ok_all_ge and changed

    # 纯日期（不带时区）入口 —— 前端日期控件送的就是这种格式
    early = min(stamps)
    date_only = post({**base_payload, "min_revision_ts": early[:10]})
    ok_date = date_only["config"].get("min_revision_ts") == early[:10] + "T00:00:00Z"
    date_equiv = post({**base_payload, "min_revision_ts": early})
    same = ([h["chunk_id"] for h in date_only["hits"]]
            == [h["chunk_id"] for h in date_equiv["hits"]])
    print("\n== 纯日期归一化 ==")
    print(f"  送 {early[:10]!r} -> config 记 {date_only['config'].get('min_revision_ts')!r}")
    print(f"  命中 {len(date_only['hits'])} 条，与带时区版本逐条一致={same}")

    # page_titles 入口
    title = plain["hits"][0]["page_title"]
    print(f"\n== page_titles=[{title}] ==")
    t = post({**base_payload, "page_titles": [title]})
    print(f"  命中 {len(t['hits'])} 条")
    for h in t["hits"][:6]:
        print(f"  {h['rank']}. {h['page_title'][:24]}")
    ok_title = all(h["page_title"] == title for h in t["hits"])

    checks = {
        "阈值远超语料上限时结果为空": ok_empty,
        "阈值早于全部语料时与无过滤逐条一致（对照）": ok_control,
        "阈值收紧后召回集整体换批且每条达标": ok_shifted,
        "纯日期被归一化成当天零点": ok_date,
        "纯日期与带时区版本召回逐条一致": same,
        "page_titles 只返回该标题": ok_title,
        "config 回读得到 min_revision_ts": "min_revision_ts" in plain["config"],
        "config 回读得到 page_titles": "page_titles" in plain["config"],
    }
    print("\n-- 判定 --")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())