"""实验 E：Chat 基座对比（DeepSeek vs MiniMax）。

这是唯一一个「换大脑」的实验。检索、分块、评测集、Judge 全部冻结，
唯一变量是 Chat 基座 —— 所以差异只能归因到模型本身。

**冻结清单**（不冻结就不是单变量对比）：
  temperature = 0
  tools schema = 同一套（get_schemas()）
  系统提示 = SYSTEM_PROMPT，不改一个字
  评测集 = 实验 C 的同一批 15 条
  Judge = 同一个独立基座（SiliconFlow Qwen3-8B）

**注意 MiniMax 的两个已知风险**（先探测再跑，scripts/probe_minimax.py）：
  1. 会输出 `<think>` 标签 —— 混进 assistant content 会污染后续轮次的消息链
  2. 工具调用协议可能与 OpenAI 不完全一致

所以这里做了两处适配：
  - 剥除 `<think>...</think>` 再放进消息
  - 把 content 里的 JSON 工具调用解析回标准 tool_calls（兜底）

用法：
    .venv\\Scripts\\python.exe scripts\\exp_e.py
    .venv\\Scripts\\python.exe scripts\\exp_e.py --limit 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.agent.react import ReActAgent  # noqa: E402
from app.agent.tools import ToolContext, get_schemas  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.judge import Judge, contexts_from_hits  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from app.retrieval.hybrid import RetrievalConfig  # noqa: E402
from exp_common import (  # noqa: E402
    SUBSET_IDS as _SUBSET_IDS,
)
from exp_common import (
    aggregate_rows,
    fmt_p,
    is_kept,
    load_cases,
    paired_scores,
    parse_games,
    print_summary,
    save,
    subset_composition,
)

MODELS = {
    "E1": None,                       # None = .env 的 deepseek_chat_model
    "E2": "MiniMax-M2.5",
}

# 评测子集（15 条）与实验 C 完全一致 —— 两个实验要能横向对比，
# 子集就必须来自同一个来源，所以复用 exp_common.SUBSET_IDS，不在这里抄一份。
# 抄一份的后果是：哪天给 C 换子集，E 还停在旧子集上，对比结论悄悄失效。
SUBSET_IDS = _SUBSET_IDS


def is_counted(row) -> bool:
    """这条 case 的分数能不能进 E 的能力指标。

    `is_kept` 已经挡住 Agent 报错与 Judge 侧失败（judge_error:*）两种情况。
    这里再补一条 E 特有的：模型侧限流（429）时 Agent 会把模型异常写进
    `answer` 文本，而 `error` 字段是空的 —— 靠读答案文本猜不可靠，
    于是用 `prompt_tokens == 0` 作为「压根没调通模型」的判据。
    这种 0 分是额度问题，不是模型能力问题。

    aggregate_rows 与 paired_scores 必须共用这一个谓词，否则两者会算出
    不同样本集上的均值与 Δ（见 paired_scores 的 docstring）。
    """
    return is_kept(row) and (row.get("prompt_tokens") or 0) > 0

_THINK = re.compile(r"<think>.*?</think>", re.S)
_JSON_CALL = re.compile(r"\{[\s\S]*\}")

from openai import AsyncOpenAI  # noqa: E402


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def repair_tool_calls(content: str) -> list[dict]:
    """兜底：模型把工具调用写成了 content 里的 JSON 而不是标准 tool_calls。

    没有这层，ReAct 的消息链会断 —— assistant 没有 tool_calls，
    后面就挂不上 role='tool' 的消息。
    """
    if not content:
        return []
    m = _JSON_CALL.search(content)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    # 兼容几种常见写法
    for name in ("tool", "tool_name", "function", "name"):
        if isinstance(obj, dict) and name in obj:
            fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
            arg = fn.get("arguments") or fn.get("args") or fn.get("parameters") or {}
            if isinstance(arg, str):
                try:
                    arg = json.loads(arg)
                except json.JSONDecodeError:
                    arg = {"raw": arg}
            return [{"id": "call_repaired_0", "name": fn.get(name) or fn.get("name"),
                     "args": arg if isinstance(arg, dict) else {}}]
    return []


@asynccontextmanager
async def _client_for(label: str):
    """E2 的 client 整个 label 只建一次，且异常路径也一定关闭。

    之前每次 chat_with 都新建一个 AsyncOpenAI，而 close() 写在 return 之前、
    并不在 finally 里 —— 于是任何一次异常（429、超时、网络抖动）都会泄漏
    一个 httpx 连接池和 TLS 会话。ReAct 每步都调一次，泄漏累积到 socket
    耗尽后新连接再也建不出去，协程就永久挂在 await 上。

    实测症状（2026-10-13 那次跑挂了 65 分钟就是这个）：
      - CPU 增量 0s、读入增量 0KB、内存不变，连续 90s 纹丝不动
      - py-spy dump 出来主线程停在 select，13 个 asyncio_* 线程全部 idle
      - 同一时刻全进程只有 1 条 Established，看不出在忙什么

    E1 走全局单例 get_llm()，本来就不归本脚本关，所以 yield None。
    """
    if label == "E1":
        yield None
        return
    s = get_settings()
    if not s.minimax_api_key:
        raise RuntimeError("MINIMAX_API_KEY 未配置")
    c = AsyncOpenAI(api_key=s.minimax_api_key, base_url=s.minimax_base_url)
    try:
        yield c
    finally:
        await c.close()


async def chat_with(label: str, model_name: str | None, messages: list[dict],
                    tools: list[dict], client=None,
                    timeout: float = 120.0) -> dict:
    """单次调用，带 MiniMax 适配与工具调用修复。

    timeout 是真的生效的（asyncio.wait_for），不是摆设 —— 声明了却不接上，
    等于单点故障可以无限期挂住整个实验，而实验通常跑在无人值守的后台里。
    """
    llm = get_llm()

    if label == "E1":
        out = await asyncio.wait_for(
            llm.chat(messages, model=model_name, tools=tools,
                     temperature=0.0, max_tokens=1024),
            timeout=timeout,
        )
        return {"content": out["content"], "tool_calls": out["tool_calls"], "usage": out["usage"]}

    # E2：MiniMax 走独立 client（不同 base_url / key），由 _client_for 统一持有
    if client is None:
        raise RuntimeError("E2 必须在 _client_for 里传入共享 client，不能临时新建")
    kwargs = {"model": model_name, "messages": messages, "temperature": 0.0, "max_tokens": 1024}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    r = await asyncio.wait_for(
        client.chat.completions.create(**kwargs), timeout=timeout)
    msg = r.choices[0].message
    content = strip_think(msg.content or "")

    calls = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"__raw__": tc.function.arguments}
        calls.append({"id": tc.id, "name": tc.function.name, "args": args})
    if not calls:
        calls = repair_tool_calls(content)

    usage = {
        "prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
        "completion_tokens": r.usage.completion_tokens if r.usage else 0,
    }
    return {"content": content, "tool_calls": calls, "usage": usage}


async def run_case(case, *, label: str, model_name: str, judge: Judge, tools,
                   ctx, sem: asyncio.Semaphore, client=None) -> dict:
    """跑单条 case。每条自己建/关 client，不共享可变状态，并发安全。"""
    s = get_settings()
    # 证据固定：两个模型看到完全一样的检索结果，
    # 否则「谁答得好」可能只是因为「谁检索到的证据更好」
    try:
        ev = await ctx.retriever.retrieve(case.question, RetrievalConfig(game=case.game, final_top_k=5))
        contexts = contexts_from_hits(ev)
    except Exception:  # noqa: BLE001
        contexts = ""

    # 只注入 chat 函数，ReAct 循环、工具执行、安全阀、引用收集
    # 全部走生产代码本身 —— 不在脚本里抄一份副本。
    # 抄一份的话，测出来的就不是线上真正在跑的东西。
    async def chat_fn(messages, **kw):
        return await chat_with(label, model_name, messages, kw.get("tools") or tools,
                               client=client)

    agent = ReActAgent(settings=s, llm=get_llm(), tools=tools, chat_fn=chat_fn)

    async with sem:
        t0 = time.perf_counter()
        err = None
        try:
            st = await agent.run(case.question, ctx=ctx)
            answer = st.answer
            steps, failures, refs = len(st.steps), st.tool_failures, st.refs
            usage = {"prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens}
        except Exception as exc:  # noqa: BLE001
            answer, usage, steps, failures, refs = "", {}, 0, 0, []
            err = f"{type(exc).__name__}: {str(exc)[:160]}"
        dt = int((time.perf_counter() - t0) * 1000)

        js = await judge.score(question=case.question, gold=case.gold_answer,
                               answer=answer, contexts=contexts)
    d = js.as_dict()
    # 报错的 case 不能和"答得差"混为一谈 —— judge 对空回答给 0 分，
    # 但那是链路失败不是能力差，不加标记的话均值会被悄悄拉低，
    # 看起来像"这个模型更差"，实际是它根本没跑起来。
    if err:
        d["errored"] = True
    return {
        "case_id": case.case_id, "category": case.category,
        "correctness": d["correctness"], "faithfulness": d["faithfulness"],
        "completeness": d["completeness"], "judge_score": d["judge_score"],
        "judge_reason": d["reason"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "steps": steps, "tool_failures": failures, "has_refs": bool(refs),
        "latency_ms": dt, "error": err, "answer": answer[:500],
    }


async def run_one(label: str, case_list, judge: Judge, *, concurrency: int = 2) -> dict:
    s = get_settings()
    model_name = MODELS[label]
    if label == "E1" and not model_name:
        model_name = s.deepseek_chat_model
    tools = get_schemas()

    # 先串行建好 ToolContext（首次建 BM25 ~1.8s，之后命中进程缓存 ~17ms），
    # 再并发跑题 —— 与 exp_c 同一套并发策略。
    for case in case_list:
        ToolContext.build(case.game)

    sem = asyncio.Semaphore(max(1, concurrency))
    # 整个 label 共用一个 client，finally 保证异常路径也关闭
    async with _client_for(label) as client:
        rows = list(await asyncio.gather(*[
            run_case(case, label=label, model_name=model_name, judge=judge, tools=tools,
                     ctx=ToolContext.build(case.game), sem=sem, client=client)
            for case in case_list
        ]))

    for r in rows:
        print(f"  [{label}] {r['case_id']:<8} score={r['judge_score']:<6} "
              f"steps={r['steps']} tok={r['prompt_tokens']:>6} {r['latency_ms']:>6}ms"
              + (f"  ⚠ ERR {str(r['error'])[:70]}" if r["error"] else ""), flush=True)

    # 聚合口径统一在 exp_common.aggregate_rows —— 与实验 C 共用同一个函数
    return {"label": label, "model": model_name, "concurrency": concurrency,
            "aggregate": aggregate_rows(rows, exclude=is_counted), "rows": rows}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="E1,E2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--game", default="sr",
                    help="目标库：sr / ys / zzz，可逗号分隔或 all")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="同档内并发跑几条题。1=串行。注意 latency_ms 含排队时间，跨并发度不可比")
    ap.add_argument("--out", default=None, help="结果文件名；不给则按样本量自动命名")
    args = ap.parse_args()


    case_list = load_cases(args.limit, game=args.game)
    print(f"评测子集：{len(case_list)} 条，标签 {args.labels}")
    print(f"题型构成：{subset_composition(case_list)}")
    if len(parse_games(args.game)) > 1:
        print("  ⚠ 多库题混跑：各库题型构成不同（见上），**跨库均值不可直接比较**，"
              "只能看同一库内的模型对比。\n")

    # MODELS[E1] 是 None（= 用 .env 默认值），直接打印会显示 "None"，
    # 看起来像脚本坏了；这里显示实际会用的模型名。
    def _label_of(label: str) -> str:
        return MODELS.get(label) or get_settings().deepseek_chat_model

    results = []
    judge = Judge(get_llm())
    games = parse_games(args.game)
    # 单库沿用历史命名；多库带库名。用 `len(case_list)==15` 判断单库会在多库下误判，
    # 恰好 15 条的多库子集会被写成 exp_E.json 覆盖单库历史产物。
    if args.out:
        out_name = args.out
    elif len(games) > 1:
        out_name = f"exp_E_{'_'.join(games)}.json"
    else:
        out_name = f"exp_E{'' if len(case_list) == 15 else f'_{len(case_list)}'}.json"
    # 中间产物另存为 .partial.json，绝不占用正式文件名 ——
    # 否则半截结果会被后续读取当成完整实验，静默污染结论。
    stem = out_name[:-5] if out_name.endswith(".json") else out_name
    partial_name = f"{stem}.partial.json"

    # 题型构成写进产物：三库题型分布差很多，跨库比均值必须能看出构成差异。
    def _stamp_meta() -> None:
        """把出处块写进第一档。

        必须在 for 循环**之后**首次调用 —— results 那时才有元素，
        在循环前写 `results[0]` 会直接 IndexError。
        每档之后都补一次，partial 落盘也要带上它。
        """
        if not results:
            return
        results[0]["_meta"] = {
            "script": "exp_e.py",
            "experiment": "E",
            "game": args.game,
            "games": games,
            "n": len(case_list),
            "subset_composition": subset_composition(case_list),
            "concurrency": args.concurrency,
            "note": "concurrency>1 时 latency_ms 含排队时间，跨并发度不可比",
        }

    for label in args.labels.split(","):
        label = label.strip().upper()
        print(f"=== 运行 {label}（{_label_of(label)}）===", flush=True)
        results.append(await run_one(label, case_list, judge, concurrency=args.concurrency))
        _stamp_meta()
        # 每档一落盘：E1 跑完就写，不必等 E2。上次 E2 挂死把 E1 的有效结果
        # 一起带走了，正是因为 save() 只在两档全跑完之后才调用 ——
        # 65 分钟的 API 花费，磁盘上一个字节都没留下。
        save(results, partial_name)

    _stamp_meta()
    save(results, out_name)
    print_summary(results, f"实验 E：Chat 基座对比（冻结 temperature/tools/prompt/评测集，n={len(case_list)}）")

    print("--- 各档 Chat 基座 ---")
    for r in results:
        print(f"  {r['label']:<5}{r.get('model') or ''}")

    t = paired_scores(results, exclude=is_counted)
    if t:
        print(f"\n--- 配对检验 E1 → E2（metric=judge_score，n={t['n']}）---")
        print(f"  Δ={t['mean_diff']:+.4f}  t={t['t']}  p={fmt_p(t['p'])}  Cohen's d={t['cohens_d']}")


if __name__ == "__main__":
    asyncio.run(main())

