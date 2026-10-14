"""LLM-as-a-Judge。

三条不可妥协的规则（每条都对应一个真实的评测陷阱）：

1. **Judge 必须与被测模型不同源**。
   用 DeepSeek 答、用 DeepSeek 评，会系统性偏好自己的表达风格（self-preference bias）。
   这里 Judge 固定走 SiliconFlow 上的 Qwen。

2. **成对比较必须交换位置跑两遍**。
   LLM Judge 对"先出现的答案"有稳定偏好，单次 A/B 的胜负有一半是位置带来的。
   `Judge.pairwise` 会正反各跑一次，只有两次都指向同一方才算赢。

3. **给出上下文再问忠实度**。
   不给证据就问"这段话有幻觉吗"，Judge 只能凭常识猜，测的是它自己的知识而不是系统的事实性。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Judge 单次实测（SiliconFlow Qwen3-8B，真实 6000 字符 payload）：37–46s，
# 短输入可低至 2s。120s 相对 46s 约 2.6 倍余量 —— 原注释写的「28.5s、四倍余量」
# 已不成立，实测尾部到过 51s。
#
# 保持 120s 不改：改超时会让新旧产物落在不同样本集上，等于给实验换一个变量。
# 真要解决超时，先查 `LLMClient.judge_client` 的 SDK 默认 `max_retries=2`
# （静默重试会撑爆外层 wait_for），而不是放宽上限。
#
# 这个预算是**单次尝试**的：judge_client 已关掉 SDK 重试，所以不存在
# 「一次调用其实偷偷重试了 3 次」的情况，wait_for 到点砍断就是真的砍断。
# 不含的部分：并发场景下 Semaphore 的排队等待 —— 那是实验脚本自己的闸门。
DEFAULT_JUDGE_TIMEOUT = 120.0

_JSON_RE = re.compile(r"\{[\s\S]*\}")

SCORE_PROMPT = """你是游戏知识库问答系统的评测员。请根据【参考答案】和【检索到的证据】评估【待评回答】。

评分维度（各 1-5 分，5 为最好）：
- correctness：与参考答案的事实一致性。数值、名称、条件完全对才算 5；表述不同但事实相同算 4；部分正确算 3；关键事实错误算 1-2。
- faithfulness：回答中的每个事实性陈述是否都能在【检索到的证据】里找到依据。出现证据里没有的具体数值或名称，faithfulness 直接降到 2 以下。
- completeness：是否覆盖了问题的全部要点。

只输出 JSON，不要任何解释：
{{"correctness": <1-5>, "faithfulness": <1-5>, "completeness": <1-5>, "reason": "<30字内理由>"}}

【问题】
{question}

【参考答案】
{gold}

【检索到的证据】
{contexts}

【待评回答】
{answer}"""

PAIRWISE_PROMPT = """你是游戏知识库问答系统的评测员。请比较两个回答，选出更好的一个。

评判优先级：正确性 > 忠实于证据 > 覆盖完整 > 表达简洁。
只要其中一个答案有证据里查不到的具体数值或名称，就判它输。

【问题】
{question}

【参考答案】
{gold}

【检索到的证据】
{contexts}

【回答 A】
{a}

【回答 B】
{b}

只输出 JSON：{{"winner": "A" 或 "B" 或 "TIE", "reason": "<30字内理由>"}}"""


@dataclass
class JudgeScore:
    correctness: float
    faithfulness: float
    completeness: float
    reason: str
    raw: str

    def as_dict(self) -> dict:
        """落进实验产物的投影（exp_c / exp_e 把这一份直接摊进 row）。

        `judge_score` = (correctness + faithfulness) / 2，**不含 completeness** ——
        所以它衡量的是「答对 + 不编」，而「答得不全」在总分里被刻意不计。
        所有实验的总均值都是这个口径，换算法等于换指标。
        `raw` 不进这个 dict：它是模型原文（含证据片段），随产物落盘等于把
        检索内容抄一遍，体积翻倍；排查解析失败时再从 JudgeScore.raw 取。
        """
        return {
            "correctness": self.correctness,
            "faithfulness": self.faithfulness,
            "completeness": self.completeness,
            "judge_score": round((self.correctness + self.faithfulness) / 2, 3),
            "reason": self.reason,
        }


def _parse_json(text: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _clamp(v) -> float:
    """把 Judge 给的分夹进 [1, 5]；非数、缺字段一律 0.0。

    注意 0.0 越过了下界：它表示「这一维没拿到分」，不是「打了 1 分」，
    但两者在产物 JSON 里长得一模一样。所以任何一维异常都必须同时看 reason，
    否则均值里会混进来源不明的 0，而且没有任何报错提示。
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(1.0, min(5.0, f))


class Judge:
    """给回答打分。**每一次对 Judge 的调用都有超时上限。**

    为什么必须有：早先 `score()` / `pairwise()` 直接
    `await self.llm.judge(...)`，超时完全依赖 OpenAI SDK 的默认值（600 秒）。
    实测已经踩过两次：
      - 实验 E 的 sr-008 拿到 `judge_error:APITimeoutError`；
      - 并发一致性探针的第二轮出现连续 60 秒零读入。

    评测跑在无人值守的后台里，一次 Judge 挂住就足以让整轮实验停在那里，
    而 **SDK 的 600 秒兜底远长于任何一次正常的打分（28.5s）**，
    等下去既拿不到结果，也看不出是哪一档出的问题。
    超时后走既有 except 分支，标记为 `judge_error:*`，
    由聚合层剔除（见 exp_common.is_kept）——**失败可识别，而不是静默挂起**。
    """

    def __init__(
        self,
        llm: LLMClient,
        model: str | None = None,
        timeout: float = DEFAULT_JUDGE_TIMEOUT,
    ) -> None:
        self.llm = llm
        self.model = model
        self.timeout = timeout

    async def score(
        self, *, question: str, gold: str, answer: str, contexts: str, max_context_chars: int = 6000
    ) -> JudgeScore:
        """单条打分。**失败不抛异常，一律折成 0 分 + 可读的 reason。**

        调用方（exp_c / exp_e）不区分「异常」与「低分」，只读 as_dict()，
        所以想让失败可见，就必须让 reason 带上一眼能认出来的前缀。
        """
        # 空回答在这里短路：不发请求，也就不会被超时误伤。
        # 但 reason 是「空回答」而非 `judge_error:*`，exp_common.judge_failed
        # 认不出来 —— 只有 Agent 侧同时抛了异常（row["error"] 非空）才会被剔除。
        # 也就是说：Agent 正常返回但答案是空串时，这个 0 分是真的进能力均值的。
        if not answer.strip():
            return JudgeScore(0, 0, 0, "空回答", "")
        # 证据截到 max_context_chars（6000 字符，37-51s 就是这个量级下的实测）。
        # **faithfulness 是对这份截断视图打的分**：答案引用了被截掉的后半段正文，
        # Judge 会判成幻觉。所以 faithfulness 天然偏低，读它时要知道这是
        # 「视图上界」而不是「模型爱编」。
        prompt = SCORE_PROMPT.format(
            question=question.strip(),
            gold=gold.strip(),
            contexts=contexts[:max_context_chars] or "（无检索证据）",
            answer=answer.strip(),
        )
        try:
            t0 = time.perf_counter()
            raw = await asyncio.wait_for(
                self.llm.judge(prompt, model=self.model),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            # 必须打类型 + 耗时。原实现只打 `%s`（异常消息），而 asyncio.TimeoutError
            # 的 str() 是空串 —— 日志里只剩一行「Judge 调用失败: 」，看不出是超时、
            # 429 还是连接断开。这个空日志已经让实验 E 白跑了两轮（2026-10-14）。
            logger.warning(
                "Judge 调用失败: %s: %s (耗时 %.1fs, 上限 %.0fs)",
                type(exc).__name__, exc or "<无消息>",
                time.perf_counter() - t0, self.timeout,
            )
            return JudgeScore(0, 0, 0, f"judge_error:{type(exc).__name__}", "")

        data = _parse_json(raw)
        if not data:
            return JudgeScore(0, 0, 0, "judge 输出无法解析", raw[:300])
        # 返回的 reason 会原样进 row["judge_reason"]，聚合层靠 `judge_error`
        # 前缀识别「这次没评成」。**这个前缀是跨模块契约**：改写它不会报错，
        # 只会静默改掉剔除口径（0 分于是又以能力差的形式回到均值里）。
        return JudgeScore(
            correctness=_clamp(data.get("correctness")),
            faithfulness=_clamp(data.get("faithfulness")),
            completeness=_clamp(data.get("completeness")),
            reason=str(data.get("reason", ""))[:80],
            raw=raw[:500],
        )

    async def pairwise(
        self,
        *,
        question: str,
        gold: str,
        answer_a: str,
        answer_b: str,
        contexts: str,
    ) -> tuple[str, str]:
        """正反各跑一次，抵消位置偏好。返回 (结论, 理由)。

        结论取 A / B / TIE。只有两次都选同一个才判该方获胜，
        不一致说明差异在噪声内，直接判 TIE —— 不硬凑出"我们更优"。

        **"拿不到结论"不等于"打平"。**
        Judge 输出无法解析时（模型没按 JSON 输出、响应被截断），
        早先的实现写的是 `_parse_json(raw) or {}`，于是 `winner` 取默认值
        "TIE" —— 一次基础设施失败被伪装成"A 和 B 一样好"，而且连日志都没有。

        后果不只是噪声：正反两次里只要有一次解析失败，r1 就是 TIE，
        后面 `if r1 == "TIE"` 直接返回平局，**把另一次可能明确的胜负一起盖掉**。
        于是系统性故障反而会制造出"两档没差别"的假象 —— 正好是评测最该避免的结论。

        所以这里用 None 表示"本次没拿到有效结论"，与真实的 TIE 分开：
        只要有一次无效，整体判 TIE 但原因里标死 `judge_parse_error`，
        下游能统计、能告警，不会把故障读成结论。
        同文件的 score() 早就是 `if not data: return ... "judge 输出无法解析"`，
        两处口径必须一致，否则同一套 Judge 会按两套标准计分。

        证据长度这里**写死 6000**，而 score() 走的是 max_context_chars 参数：
        两条路径必须保持同一个证据长度，否则同一份产物里 faithfulness 与
        成对胜负不是同一把尺子量的。

        注意：目前没有任何实验脚本调用 pairwise（只有测试用），
        它是为模型对比准备的组件。接入时要一并确认 reason 串的落盘位置 ——
        丢弃它可以（统计口径不依赖），丢弃结论则会把基础设施失败读成打平。
        """
        results: list[str | None] = []
        reasons: list[str] = []
        # 两次尝试的区别只有 a/b 顺序；任何一次失败都不是"这一次不好"，
        # 而是整条比较作废 —— 所以 except 分支直接返回，不再跑第二次。
        for first, second in ((answer_a, answer_b), (answer_b, answer_a)):
            prompt = PAIRWISE_PROMPT.format(
                question=question.strip(),
                gold=gold.strip(),
                contexts=contexts[:6000] or "（无检索证据）",
                a=first.strip(),
                b=second.strip(),
            )
            try:
                t0 = time.perf_counter()
                raw = await asyncio.wait_for(
                    self.llm.judge(prompt, model=self.model),
                    timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Judge 成对比较失败: %s: %s (耗时 %.1fs, 上限 %.0fs)",
                    type(exc).__name__, exc or "<无消息>",
                    time.perf_counter() - t0, self.timeout,
                )
                return "TIE", f"judge_error:{type(exc).__name__}（未得到有效结论，非真实平局）"

            data = _parse_json(raw)
            if not data or "winner" not in data:
                logger.warning("Judge 成对比较输出无法解析，原始输出: %s", (raw or "")[:200])
                reasons.append("parse_error")
                results.append(None)
                continue

            winner = str(data["winner"]).upper()
            reasons.append(str(data.get("reason", ""))[:80])
            results.append("A" if winner == "A" else "B" if winner == "B" else "TIE")

        if any(r is None for r in results):
            return "TIE", "judge_parse_error（未得到有效结论，非真实平局）：" + " / ".join(reasons)

        r1, r2 = results
        # 分支触发条件全在这里，按顺序读：
        #   任一次真 TIE → 整体 TIE（模型自己都说分不出来）；
        #   两次都指向同一个字母 → 那一方赢（位置偏好已被正反抵消）；
        #   两次指向不同答案 → 差异落在噪声内，判 TIE 而不是硬挑一个。
        if r1 == "TIE" or r2 == "TIE":
            return "TIE", " / ".join(reasons)
        if r1 == r2:
            return r1, " / ".join(reasons)
        # 两次顺序下指向不同答案 → 差异落在噪声内
        return "TIE", f"位置敏感，结论不一致({r1}/{r2})：" + " / ".join(reasons)


def contexts_from_hits(hits) -> str:
    """把检索结果拼成 Judge 唯一的证据视图。

    三条对下游成立的口径约束：

    - `[i]` 编号就是 hits 的下标顺序。它是**另取的一份检索**（exp_c / exp_e
      为 Judge 单独跑一次 retriever），所以编号不必和 Agent 引用里的编号对齐 ——
      两边各看各的顺序，不要拿答案里的 [n] 去对这里的 [n]。
    - 只有被拼进来的内容才算「证据」。top_k 调小、被 rerank 挤掉的候选
      对 Judge 是不可见的，于是 top_k 一变 faithfulness 就会跟着变 ——
      那是测量设置的差异，不是能力变化。
    - 检索抛异常时调用方兜成空串，Judge 收到的是「（无检索证据）」：
      此时任何具体数值都会被判不忠实，这条 0 分来自链路而不是模型。
    """
    return "\n\n".join(
        f"[{i}] {h.chunk.page_title} > {h.chunk.metadata.get('section', '')}\n{h.chunk.text}"
        for i, h in enumerate(hits, 1)
    )