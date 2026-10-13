"""实验脚本的公共底座。

`exp_c.py`（Agent 能力消融）和 `exp_e.py`（Chat 基座对比）本来各写了一份
评测子集、结果聚合、表格打印。三个实验再各写一遍的话，改一个统计口径要
改三个文件，而且一定会漏。

抽到这里之后，**统计口径只有一份** —— 这是比省代码更重要的理由：
三个实验的数字必须可比，口径一旦分叉，A/B/C/D/E 的对比就失去意义。

用法：
    from exp_common import SUBSET_IDS, aggregate_rows, print_summary
"""
from __future__ import annotations

import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

#: 实验结果的落盘目录。读写两端共用这一个常量 ——
#: 之前 save() 里内联拼 ROOT/"data"/"eval"，而 analyze_significance.py
#: 读的时候写死了本机绝对路径，两边不是同一个推导方式就可能对不上。
EVAL_DIR = ROOT / "data" / "eval"


def load_cases(limit: int = 0):
    """取实验用的 15 条代表性子集，覆盖全部 6 个题型。

    为什么不跑全部 32 条：每条要一次 Judge 调用（8B 模型约 15s），
    3 档 × 32 条 = 96 次 ≈ 24 分钟。15 条 × 3 档 ≈ 11 分钟，
    换来的是能反复迭代的实验节奏。检索侧实验（A/B/D）跑的是全部 29 条，
    因为那边没有 Judge 开销。
    """
    from app.eval.cases import SR_CASES

    by_id = {c.case_id: c for c in SR_CASES}
    cases = [by_id[i] for i in SUBSET_IDS if i in by_id]
    return cases[:limit] if limit else cases


# 15 条子集：结构化字段 / 纯语义 / 多跳 / 数值 / 幻觉 / 假前提 各覆盖
SUBSET_IDS = [
    "sr-001", "sr-002", "sr-004", "sr-006", "sr-008", "sr-023",
    "sr-007", "sr-011", "sr-013", "sr-025", "sr-028",
    "sr-009", "sr-031", "sr-032", "sr-030",
]


JUDGE_ERROR_PREFIX = "judge_error"


def judge_failed(row: dict) -> bool:
    """这条 case 是不是败在 Judge 侧（超时 / 限流 / 输出解析失败）。

    为什么必须单独认出来：`is_kept` 早先只看 `error` 字段，而那是 **Agent
    侧**的异常。Judge 自己失败时 `error` 是空的，但 `judge_score` 已经被
    `score()` 兜成了 0 分，于是这个 0 会被当成「模型答得一塌糊涂」算进均值。

    实测踩到的例子（实验 E 的 E1 档 sr-008）：
        judge_score = 0.0
        judge_reason = "judge_error:APITimeoutError"
        error        = None
    DeepSeek 那条其实答得不错，是 Judge 的一次 APITimeout 把它判成 0 分。
    混进均值后 E1 从 4.60 掉到 4.30 —— 看起来像 DeepSeek 能力不行，
    实际是评测链路抖了一下。**这种分不能进能力指标。**

    `score()` 里空回答走的是「空回答」、Judge 调不通走的是
    `judge_error:<异常名>`，两者前缀不同，所以按前缀判定不会误伤空回答。
    """
    return str(row.get("judge_reason") or "").startswith(JUDGE_ERROR_PREFIX)


def is_kept(row: dict) -> bool:
    """默认的「这条结果算不算数」：Agent 与 Judge 两侧都没失败才算。

    早先只判 `error`（Agent 侧），漏掉了 Judge 侧失败 —— 见 judge_failed
    的说明。只算 Agent 侧，会让链路故障伪装成能力差距。
    """
    return not row.get("error") and not judge_failed(row)


def aggregate_rows(rows: Sequence[dict], *, exclude=None) -> dict:
    """把per-case 结果聚合成一行指标。

    **均值只统计没报错的 case** —— 报错的 case judge 会给 0 分，
    但那是链路失败不是能力差，混进去会让均值被悄悄拉低，
    看起来像「这个配置更差」，实际是「它根本没跑起来」。
    `errors` / `n_ok` 两个字段就是给这个场景留的证据。

    `exclude` 是额外的排除谓词（默认 `is_kept`）。
    为什么要有这个参数：实验 E 里 MiniMax 有一次撞 429，
    Agent 把模型异常写进 `answer` 而 `error` 字段是空的 ——
    靠读答案文本猜不可靠，于是用 `prompt_tokens == 0` 作为基础设施失败的判据。
    这种"额外的排除规则"以前只有 analyze_significance 自己一份实现，
    现在统一走这里，避免**同一份数据被两个脚本算出两个均值**。
    """
    keep = exclude if exclude is not None else is_kept
    kept = [r for r in rows if keep(r)]
    # 全部被排除时退回全量，避免除零；但必须留下 all_failed 标记 ——
    # 否则「评测链路全挂」会被读成「模型全答错」，方向完全相反。
    ok = kept or list(rows)
    n_ok = len(ok) or 1
    all_failed = not kept and bool(rows)

    def avg(key: str, cast=float, default=0):
        vals = [cast(r[key]) for r in ok if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else default

    steps_total = sum(r.get("steps", 0) for r in ok)
    return {
        "judge_score": avg("judge_score"),
        "correctness": avg("correctness"),
        "faithfulness": avg("faithfulness"),
        "avg_prompt_tokens": int(avg("prompt_tokens")),
        "avg_completion_tokens": int(avg("completion_tokens")),
        "avg_steps": round(avg("steps"), 2),
        # 分母用总步数而不是 case 数：某个 case 调了 3 次工具、另一个 1 次，
        # 按 case 平均会让「工具用得多」看起来像「工具更容易失败」。
        "tool_failure_rate": round(sum(r.get("tool_failures", 0) for r in ok) / max(1, steps_total), 4),
        "ref_coverage": round(sum(1 for r in ok if r.get("has_refs")) / n_ok, 4),
        "avg_latency_ms": int(avg("latency_ms")),
        "errors": sum(1 for r in rows if r.get("error")),
        # Judge 侧失败单列：它是评测链路的健康度，不是被测系统的能力，
        # 但确实会拉低均值，必须能一眼看出某个 0 分是「答错」还是「没评成」
        "judge_errors": sum(1 for r in rows if judge_failed(r)),
        "all_failed": all_failed,
        "n": len(rows),
        "n_ok": len(ok),
    }


def print_summary(results: Sequence[dict], title: str) -> None:
    """打印总表 + 分类别拆解。

    分类别是重点：整体均值会把「幻觉题崩掉、语义题还行」这种结构
    藏在平均数里 —— 而那恰恰是最值得讲的部分。
    """
    if not results:
        print("（无结果）")
        return

    print("\n" + "=" * 96)
    print(f"  {title}")
    print("=" * 96)
    print(f"{'配置':<5}{'judge':>8}{'correct':>9}{'faith':>8}"
          f"{'prompt tok':>12}{'步数':>8}{'引用':>7}{'工具失败':>10}{'延迟ms':>9}{'错误':>6}")
    for r in results:
        a = r["aggregate"]
        print(f"{r['label']:<5}{a['judge_score']:>8.2f}{a['correctness']:>9.2f}"
              f"{a['faithfulness']:>8.2f}{a['avg_prompt_tokens']:>12}{a['avg_steps']:>8.1f}"
              f"{a['ref_coverage']:>7.2f}{a['tool_failure_rate']:>10.3f}"
              f"{a['avg_latency_ms']:>9}{a['errors']:>6}")

    cats = sorted({r["category"] for r in results[0]["rows"] if r.get("category")})
    if cats:
        print("\n--- 按题型拆开（幻觉 / 假前提题是「无 RAG 时会怎样」的照妖镜）---")
        print_category_matrix(results)


def category_means(results: Sequence[dict], *, metric: str = "judge_score",
                   exclude=None) -> dict[str, list[float | None]]:
    """题型 × 各档label 的均值矩阵。抽出来是为了让 print_summary 和
    analyze_significance 用**同一份实现** —— 以前两边各写一遍，
    现在改一处两边同时生效。
    """
    keep = exclude if exclude is not None else is_kept
    cats = sorted({r["category"] for r in results[0]["rows"] if r.get("category")})
    matrix: dict[str, list[float | None]] = {}
    for cat in cats:
        col: list[float | None] = []
        for r in results:
            vals = [x[metric] for x in r["rows"]
                    if x.get("category") == cat and keep(x) and x.get(metric) is not None]
            col.append(round(statistics.mean(vals), 4) if vals else None)
        matrix[cat] = col
    return matrix


def print_category_matrix(results: Sequence[dict], *, metric: str = "judge_score",
                          exclude=None, width: int = 12) -> None:
    matrix = category_means(results, metric=metric, exclude=exclude)
    if not matrix:
        return
    print(f"{'题型':<14}" + "".join(f"{r['label']:>{width}}" for r in results))
    for cat, col in matrix.items():
        cells = [(f"{v:>{width}.2f}" if v is not None else f"{'-':>{width}}") for v in col]
        print(f"{cat:<14}" + "".join(cells))


def paired_scores(results: Sequence[dict], metric: str = "judge_score",
                  *, exclude=None) -> dict | None:
    """相邻两档之间的配对比较（首尾对比）。

    用「首尾」而不是「相邻」：实验 C 的问题就是「有 RAG vs 没 RAG」，
    中间那档只是过渡。首尾对比最能说明项目存在的理由。

    **样本集必须和 aggregate_rows 用同一个谓词。**
    早先这里硬编码 `not r.get("error")`，只认 Agent 侧异常，于是
    aggregate_rows 剔除的 Judge 侧失败（judge_error:*）在这里又被算了回来：
    均值按 n=14 算、配对检验按 n=15 算，两份数字来自不同样本集，
    Δ 和 p 却会被当成同一份数据的结论写进文档 —— 这类不自洽很难事后发现，
    因为两个数字单独看都「正常」。
    """
    from app.eval.metrics import paired_test

    if len(results) < 2:
        return None
    keep = exclude if exclude is not None else is_kept
    first, last = results[0], results[-1]
    common = [r for r in first["rows"] if keep(r)]
    common_ids = {r["case_id"] for r in common}
    last_rows = [r for r in last["rows"] if r["case_id"] in common_ids and keep(r)]
    if len(last_rows) < 3:
        return None
    return paired_test(
        # 传 (末档, 首档)：Δ 与 d 都是「末档减首档」，正数 = 加了 RAG 之后变好。
        # 反过来传的话，C0→C1 会打出负的 Δ，而 "C1 更好" 这个结论要靠读者
        # 自己把符号翻过来才成立 —— 与直觉相反的方向约定迟早会被抄错。
        [{**r, metric: r[metric]} for r in last_rows],
        [{**r, metric: r[metric]} for r in common],
        metric,
    )


def save(results: Sequence[dict], name: str) -> Path:
    import json

    out_dir = EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    out.write_text(json.dumps(list(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    return out


def build_meta(args) -> dict:
    """产物的出处块：脚本、时间、commit，以及**当时那个索引的规模**。

    索引规模这一项不是可有可无的装饰。2026-10-13 踩过一次坑：
    补抓角色页后 sr 索引 2306 -> 3932 chunk，跑 `exp_b.py --skip-rebuild`
    时 B0 的产物文件还是补抓前写的，汇总表里 B0 显示 0.966 而 B1/B2/B3 是新数字 ——
    **一行旧数据混在一张新表里，没有任何报错**。
    靠文件 mtime 判不出来（改一下就变），所以规模必须写进产物内容里。

    判据是能证明不一致才排除，所以取不到 collection 时降级为 None 而不是抛错：
    老格式产物没有这个字段时，汇总方应当放行而不是把所有步骤都踢掉。
    """
    import datetime as _dt
    import subprocess as _sp

    meta = {
        "script": Path(__file__).name,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "experiment": getattr(args, "experiment", None),
        "game": getattr(args, "game", None),
        "variant": getattr(args, "variant", None),
        "top_k": getattr(args, "top_k", None),
        "labels": getattr(args, "labels", None),
        "index_chunks": None,
        "index_pages": None,
    }
    try:
        commit = _sp.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
        if commit.returncode == 0:
            meta["git_commit"] = commit.stdout.strip()
    except Exception:
        pass
    try:
        from app.core.config import get_settings
        from app.store.vector_store import get_vector_store

        s = get_settings()
        model = getattr(args, "embed_model", None) or s.embed_model_main
        variant = meta["variant"] or "base"
        games = parse_games(meta["game"])
        if len(games) > 1:
            # 多库评测时每库的规模差得很远，而且**页数与 chunk 数不是同一个比例**：
            # sr 706 页 / 3932 chunk，ys 435 页 / 5413 chunk，zzz 260 页 / 1498 chunk。
            # ys 的 chunk 数反而最多（页面更长、分得更碎），所以只报一个总数
            # 反而看不出「指标差异里有多少来自语料规模」—— 逐库记，汇总方才好核对。
            meta["index_by_game"] = {}
            for g in games:
                try:
                    data = get_vector_store(s).get_collection(g, model, variant).get(
                        include=["metadatas"]
                    )
                    metas = data.get("metadatas") or []
                    meta["index_by_game"][g] = {
                        "index_chunks": len(data.get("ids") or []),
                        "index_pages": len({(m or {}).get("page_title", "") for m in metas} - {""}),
                    }
                except Exception:
                    meta["index_by_game"][g] = {"index_chunks": None, "index_pages": None}
        else:
            game = games[0] if games else s.game_primary
            col = get_vector_store(s).get_collection(game, model, variant)
            data = col.get(include=["metadatas"])
            metas = data.get("metadatas") or []
            meta["index_chunks"] = len(data.get("ids") or [])
            meta["index_pages"] = len({(m or {}).get("page_title", "") for m in metas} - {""})
    except Exception:
        pass
    return meta


def parse_games(game: str | None) -> list[str]:
    """`--game` 参数归一成库列表。

    支持三种写法：`sr` / `sr,ys` / `all`。
    库顺序**固定按 sr, ys, zzz**（与 `cases.ALL_CASES` 的声明顺序一致），
    不按命令行里写的顺序 —— 否则 `--game zzz,sr` 和 `--game sr,zzz`
    会产出顺序不同的 rows，配对检验按 case_id 求交集还好，
    但逐条产物 diff 出来全是噪声，两次运行也就没法互相核对了。
    """
    from app.eval.cases import ALL_CASES

    order = ["sr", "ys", "zzz"]
    raw = (game or "").strip()
    if not raw:
        return []
    if raw == "all":
        return [g for g in order if g in ALL_CASES]
    picked = {x.strip() for x in raw.split(",") if x.strip()}
    unknown = picked - set(ALL_CASES)
    if unknown:
        raise SystemExit(f"未知的库：{', '.join(sorted(unknown))}；可用：{', '.join(ALL_CASES)} 或 all")
    return [g for g in order if g in picked]
