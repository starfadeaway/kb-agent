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

import json
import logging
import re
from dataclasses import dataclass

from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

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
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(1.0, min(5.0, f))


class Judge:
    def __init__(self, llm: LLMClient, model: str | None = None) -> None:
        self.llm = llm
        self.model = model

    async def score(
        self, *, question: str, gold: str, answer: str, contexts: str, max_context_chars: int = 6000
    ) -> JudgeScore:
        if not answer.strip():
            return JudgeScore(0, 0, 0, "空回答", "")
        prompt = SCORE_PROMPT.format(
            question=question.strip(),
            gold=gold.strip(),
            contexts=contexts[:max_context_chars] or "（无检索证据）",
            answer=answer.strip(),
        )
        try:
            raw = await self.llm.judge(prompt, model=self.model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Judge 调用失败: %s", exc)
            return JudgeScore(0, 0, 0, f"judge_error:{type(exc).__name__}", "")

        data = _parse_json(raw)
        if not data:
            return JudgeScore(0, 0, 0, "judge 输出无法解析", raw[:300])
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
        """
        results: list[str | None] = []
        reasons: list[str] = []
        for first, second in ((answer_a, answer_b), (answer_b, answer_a)):
            prompt = PAIRWISE_PROMPT.format(
                question=question.strip(),
                gold=gold.strip(),
                contexts=contexts[:6000] or "（无检索证据）",
                a=first.strip(),
                b=second.strip(),
            )
            try:
                raw = await self.llm.judge(prompt, model=self.model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Judge 成对比较失败: %s", exc)
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
        if r1 == "TIE" or r2 == "TIE":
            return "TIE", " / ".join(reasons)
        if r1 == r2:
            return r1, " / ".join(reasons)
        # 两次顺序下指向不同答案 → 差异落在噪声内
        return "TIE", f"位置敏感，结论不一致({r1}/{r2})：" + " / ".join(reasons)


def contexts_from_hits(hits) -> str:
    return "\n\n".join(
        f"[{i}] {h.chunk.page_title} > {h.chunk.metadata.get('section', '')}\n{h.chunk.text}"
        for i, h in enumerate(hits, 1)
    )