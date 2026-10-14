"""LLM 故障分类：哪些错误值得换个基座再试，哪些不值得。

为什么需要单独一个模块
----------------------
"捕获异常然后重试"是最容易写出静默错误的地方。两个具体的坑：

1. **对确定性失败重试**。400 参数错、401 鉴权错、内容过滤 ——
   同样的请求再发一次，结果必然一样。降级链在这里重试，
   只会把一次 200ms 的失败拉成 N 次 N 倍延迟，还多烧一份 token，
   而最后的错误信息反而被重试噪声淹掉。

2. **降级不可见**。如果 `chat()` 悄悄换了基座，调用方拿到的答案
   其实是另一个模型生成的。这在演示里无所谓，在**跨模型对比实验**里
   是致命的隐藏变量 —— 实验 D 刚刚因为"检验对象用错"推翻了结论，
   同一个错误不能再犯一次。

所以本模块只负责一件事：把异常归类成「可降级」与「不可降级」，
并要求调用方把降级事实**显式带回**给上层。

分类依据
--------
只按「同样的请求再发一次，结果会不会变」来判断，不看错误消息长什么样：

- **可降级**：429（限流）、5xx（服务端故障）、超时、连接中断
  —— 这些是环境/配额问题，换个供应商可能就绕过去了。
- **不可降级**：4xx 里除 429 外全部、参数校验失败、内容过滤、
  本地构造错误（未配置密钥、未知基座名）
  —— 这些是请求本身或本地配置的问题，换谁都一样。
"""
from __future__ import annotations

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

#: 明确不可降级的异常类型。这里必须**枚举**，不能写"其余都算可重试" ——
#: 新增一个 openai SDK 的异常类型时，默认必须落在安全的一侧。
_FATAL: tuple[type[Exception], ...] = (
    BadRequestError,      # 400 参数/上下文/消息格式错
    AuthenticationError,  # 401 密钥无效
    PermissionDeniedError,  # 403
    NotFoundError,        # 404 模型名不存在
    ValueError,           # 本地构造错：未配置密钥、未知基座名、参数非法
    TypeError,
    KeyError,
)

#: 可降级的异常类型。
_RETRYABLE: tuple[type[Exception], ...] = (
    RateLimitError,       # 429
    InternalServerError,  # 5xx
    APIConnectionError,   # 连不上
    APITimeoutError,      # 超时
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


def classify(exc: BaseException) -> str:
    """返回 "retryable" / "fatal"，供降级链决策。

    顺序有讲究：先判不可降级。白名单式（只认可重试类型）其实更安全，
    但 openai SDK 对 5xx 抛的是 `InternalServerError`、对连接失败抛的是
    `APIConnectionError`，而 `APIStatusError` 的其余子类散在 SDK 内部，
    枚举不全就会把 500 当成 fatal 直接放弃降级 —— 所以这里两侧都列，
    并且**未知异常默认 fatal**（`else` 分支），宁可漏降级也不要把
    ValueError 之类的本地 bug 拿去重试三遍。
    """
    if isinstance(exc, _FATAL):
        return "fatal"
    if isinstance(exc, _RETRYABLE):
        return "retryable"
    # openai SDK 的 5xx 家族还包含 APIStatusError 的其它实例（521/522/524 等），
    # 用状态码兜住，否则 Cloudflare 层错误会被当成 fatal。
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return "retryable" if status == 429 or status >= 500 else "fatal"
    return "fatal"


def is_retryable(exc: BaseException) -> bool:
    """`classify(...) == "retryable"` 的布尔投影，供降级链一处判断。

    **读这个 bool 之前要清楚它管的是什么、不管什么。**
    唯一消费点是 `client.chat` 的降级链：不可重试就当场把异常抛给上层，
    可重试才 `continue` 换下一个基座。所以这里的"可重试"实际含义是
    「**换个基座重发一次**可能有用」，而不是「同一条请求再发一次」——
    本项目**没有**针对同一个基座的重试，`_chat_once` 失败即换人。
    把它当成「可以原地重试」的信号会高估系统的容错。

    判断错的后果是不对称的，这也是宁可漏降级的原因：
    - 误判 fatal → 一次限流直接冒泡成 500，用户看到的是硬失败；
    - 误判 retryable → 真实故障被逐个基座重试一遍，错误信息被重试噪声淹没，
      而且 `/api/health` 的 fallback_stats 会显示「降级过」，
      跨模型对比实验里这些样本其实已经不是原模型产出的了（隐藏变量）。

    判定逻辑本身全在 `classify`，本函数不引入任何额外口径 ——
    加新判断必须改 `classify`（及其两侧元组），只改这里会造成两套口径。
    """
    return classify(exc) == "retryable"


def describe(exc: BaseException) -> str:
    """给人看的一句话说明，不含任何密钥/请求体内容。"""
    kind = classify(exc)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return f"{kind}/{type(exc).__name__}/status={status}"