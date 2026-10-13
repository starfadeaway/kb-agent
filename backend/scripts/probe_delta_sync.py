"""实测：知识库的增量同步到底值不值得做。

背景：版本迭代的直觉做法是「定期全量重抓」，但到底该全量还是增量，
不能靠感觉 —— 取决于两个数：

1. **接口可用性**：MediaWiki 的 `list=recentchanges` 能不能穿过 EdgeOne？
   早期结论是「只能靠拉长间隔硬扛」，后来发现批量取页才是有效手段，
   那 `recentchanges` 是否也在这条豁免路径上，必须实测。

2. **增量口径**（最容易算错的一处）：全站变更页数 **不是** 增量成本。
   全站含活动公告、版本专题、玩家投稿闲聊页，而本项目只收录 12 个分类。
   真正的成本是 |变更页 ∩ 已收录页|。实测两者差了一倍 ——
   全站 7 天变更 > 1000 页，落到本地收录范围只有 ~150 页。

   ⚠️ 把全站数当增量成本，会得出「增量没意义」的错误结论。

三个阶段各自可单跑（WAF 预算有限）：
    --stage api      只测接口是否可用
    --stage scope    变更页与本地收录的交集（需要 data/checkpoints 存在）

产物：data/eval/probe_delta_sync.json

**读数纪律**：窗口翻不完（被 WAF 截断）时，去重页数是**下界**，
交集数也是下界。脚本会把 exhausted=False 明确打出来，别把下界当精确值。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
OUT = ROOT / "data" / "eval" / "probe_delta_sync.json"

# 退避基数：实测 45s/90s 两级仍然会被继续拦，说明惩罚窗口更长；
# 这里用 60/120/180 三级，且每次重试前重新判断是否值得继续。
BACKOFF = (60, 120, 180)


def api_of(slug: str) -> str:
    return f"https://wiki.biligame.com/{slug}/api.php"


def get(api: str, params: dict, *, verbose: bool = True) -> tuple[int, str, str]:
    """带分级退避的请求。567 是 EdgeOne 挑战页，不是数据。"""
    for attempt in range(len(BACKOFF) + 1):
        q = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{api}?{q}", headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            st, ct = e.code, e.headers.get("Content-Type", "")
            body = e.read().decode("utf-8", "replace")
            if st == 567 and attempt < len(BACKOFF):
                wait = BACKOFF[attempt]
                if verbose:
                    print(f"    WAF 567，等 {wait}s 后重试…")
                time.sleep(wait)
                continue
            return st, ct, body
        except Exception as e:  # noqa: BLE001
            return -1, "", f"{type(e).__name__}: {e}"
    return -1, "", "retries exhausted"


def classify(status: int, ctype: str, body: str) -> str:
    if status == -1:
        return "网络异常"
    if status == 567:
        return "WAF 限流挑战页"
    if "json" not in ctype and body.lstrip()[:1] == "<":
        return "HTML 挑战页"
    if status == 200:
        return "OK"
    return f"HTTP {status}"


def local_titles(game: str) -> set[str]:
    """已抓过的全部标题（各分类 checkpoint 的并集），即本项目的收录范围。"""
    out: set[str] = set()
    d = ROOT / "data" / "checkpoints"
    if not d.is_dir():
        return out
    for p in sorted(d.glob(f"{game}_*.json")):
        try:
            out.update(json.loads(p.read_text(encoding="utf-8-sig")).get("done_titles", []))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"    !! {p.name} 解析失败：{exc}")
    return out


def stage_api(slug: str) -> dict:
    """阶段一：增量所需的三个接口是否都能穿过 EdgeOne。"""
    api = api_of(slug)
    probes = {
        "recentchanges": {
            "action": "query", "list": "recentchanges", "rcnamespace": "0",
            "rclimit": "20", "rcprop": "title|timestamp", "format": "json",
            "formatversion": "2",
        },
        "recentchanges_window": {
            "action": "query", "list": "recentchanges", "rcnamespace": "0",
            "rclimit": "50", "rcprop": "title|timestamp", "format": "json",
            "formatversion": "2",
        },
        "allpages_rv": {
            "action": "query", "generator": "allpages", "gapnamespace": "0",
            "gaplimit": "20", "prop": "revisions", "rvprop": "ids|timestamp",
            "format": "json", "formatversion": "2",
        },
    }
    res: dict = {}
    for name, params in probes.items():
        st, ct, body = get(api, params, verbose=False)
        v = classify(st, ct, body)
        entry = {"status": st, "content_type": ct.split(";")[0], "verdict": v}
        if v == "OK":
            try:
                data = json.loads(body)
                if name == "allpages_rv":
                    entry["count"] = len(data["query"]["pages"])
                else:
                    rows = data["query"]["recentchanges"]
                    entry["count"] = len(rows)
                    if rows:
                        entry["sample"] = [r["title"] for r in rows[:3]]
            except (json.JSONDecodeError, KeyError) as exc:
                entry["verdict"] = f"OK 但解析失败({exc})"
        res[name] = entry
        print(f"  [{'可用' if entry['verdict'] == 'OK' else '不可用'}] {name}: {entry['verdict']}")
        time.sleep(8)
    return res


def stage_scope(slug: str, game: str, days: int, interval: float, cap: int) -> dict:
    """阶段二：变更页与本地收录的交集 —— 这才是增量的真实成本。"""
    api = api_of(slug)
    local = local_titles(game)
    print(f"  本地已收录（{game} checkpoint 并集）：{len(local)} 页")

    since = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
    cursor: str | None = None
    titles: set[str] = set()
    changes = 0
    requests = 0
    waf_hits = 0

    for _ in range(cap):
        params = {
            "action": "query", "list": "recentchanges", "rcnamespace": "0",
            "rcstart": since.strftime("%Y-%m-%dT%H:%M:%SZ"), "rcdir": "older",
            "rclimit": "500", "rcprop": "title|timestamp", "format": "json",
            "formatversion": "2",
        }
        if cursor:
            params["rccontinue"] = cursor
        st, ct, body = get(api, params)
        requests += 1
        if st != 200 or "json" not in ct:
            waf_hits += 1
            print(f"    第 {requests} 次请求失败：HTTP {st} {classify(st, ct, body)}")
            break
        data = json.loads(body)
        rows = data.get("query", {}).get("recentchanges", [])
        if not rows:
            cursor = None
            break
        changes += len(rows)
        titles.update(r["title"] for r in rows)
        cursor = data.get("continue", {}).get("rccontinue")
        print(f"    第 {requests} 次：{len(rows)} 条 → 去重 {len(titles)} 页")
        if not cursor:
            break
        time.sleep(interval)

    exhausted = cursor is None
    inter = titles & local
    pct = round(len(inter) / len(local) * 100, 1) if local else 0.0

    if not exhausted:
        print("  ⚠️ 窗口未翻完（被 WAF 截断）：下面的去重页数与交集都是**下界**，不是精确值")
    if waf_hits and not titles:
        print("  ⚠️ 本次窗口一个页面都没取到，交集数不可用 —— 别把它当成「零变更」")

    print(f"  → 全站变更 {len(titles)} 页，交集 {len(inter)} 页（本地 {pct}%）")
    return {
        "window_days": days, "local_pages": len(local),
        "changed_pages": len(titles), "intersect": len(inter), "local_pct": pct,
        "changes": changes, "requests": requests, "waf_hits": waf_hits,
        "exhausted": exhausted, "lower_bound": not exhausted,
        "sample_intersect": sorted(inter)[:8],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="sr")
    ap.add_argument("--game", default="sr")
    ap.add_argument("--stage", default="all", choices=["all", "api", "scope"])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--interval", type=float, default=15.0,
                    help="翻页间隔；实测 6s 会在第 3 页吃 567，15s 可行")
    ap.add_argument("--cap", type=int, default=8, help="最多翻几页")
    args = ap.parse_args()

    print(f"探针目标：{args.slug} / 阶段 {args.stage} / 窗口 {args.days} 天\n")
    res: dict = {
        "probed_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slug": args.slug, "game": args.game,
    }

    if args.stage in ("all", "api"):
        print("== 阶段一：接口可用性 ==")
        res["api"] = stage_api(args.slug)

    if args.stage in ("all", "scope"):
        print(f"\n== 阶段二：近 {args.days} 天交集 ==")
        res["scope"] = stage_scope(args.slug, args.game, args.days,
                                   args.interval, args.cap)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    sc = res.get("scope")
    if sc and sc["local_pages"]:
        print(f"\n-- 判定 --\n  本地 {sc['local_pages']} 页，近 {sc['window_days']} 天"
              f"交集 {sc['intersect']} 页 = {sc['local_pct']}%"
              f"{'（下界）' if sc['lower_bound'] else ''}")
    print(f"\n产物已写 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
