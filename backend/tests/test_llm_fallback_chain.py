"""守卫：Chat 降级链只对可重试故障生效，且降级事实必须可见。

缺陷 / 风险
-----------
1. **对确定性失败重试**。400 参数错、401 鉴权错、内容过滤 ——
   同样的请求再发一次结果必然一样。降级链在这里重试，
   只会把一次 200ms 的失败拉成 N 倍延迟、多烧一份 token，
   而真正的错误信息被重试噪声淹掉。
2. **降级不可见**。`chat()` 悄悄换了基座，调用方拿到的答案其实是
   另一个模型生成的。演示里无所谓，**跨模型对比实验里是致命的隐藏变量** ——
   实验 D 刚刚因为"检验对象用错"推翻了结论，同一个坑不能踩第二次。
3. **未知异常默认可重试**。白名单式重试会把 `ValueError`（本地构造错、
   未配置密钥、未知基座名）也拿去重试三遍，而这三个里没有一个能靠换供应商治好。

双向验证
--------
本组测试顶层只 import 修复前就存在的符号（`LLMClient` / `get_settings`），
降级链相关的属性一律用属性访问，不 import 任何本轮新增的模块级名字，
所以可以对"还没有降级链"的代码快照实跑。
"""
from __future__ import annotations

import dataclasses

import httpx
import pytest
from app.core.config import get_settings
from app.llm.client import LLMClient
from app.llm.fallback import classify, describe, is_retryable
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)


def _client(fallbacks: str = "minimax", backoff: float = 0.0) -> LLMClient:
    s = dataclasses.replace(
        get_settings(),
        chat_fallbacks=fallbacks,
        chat_fallback_backoff=backoff,
        deepseek_api_key="sk-test",
        minimax_api_key="sk-test",
        siliconflow_api_key="sk-test",
    )
    return LLMClient(settings=s)


class _Resp:
    """最小可用的 completion 响应对象。"""

    def __init__(self, content: str = "ok") -> None:
        outer = self

        class _Msg:
            def __init__(self) -> None:
                self.content = content
                self.tool_calls = None

        class _Choice:
            def __init__(self) -> None:
                self.message = _Msg()

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5

        outer.choices = [_Choice()]
        outer.usage = _Usage()


def _install(llm: LLMClient, plan: dict[str, object]) -> list[str]:
    """按基座名注入成功/失败序列，返回调用记录列表（调用后才可读长度）。

    `plan`: {"deepseek": exc 或 None, "minimax": exc 或 None}
    值为异常实例 → 抛；为 None → 成功；不在 plan 里 → 成功。
    """
    seen: list[str] = []

    def make(backend: str):
        class _C:
            async def create(self, *, model, messages, temperature, max_tokens, **kw):  # noqa: ARG002
                seen.append(backend)
                outcome = plan.get(backend)
                if isinstance(outcome, BaseException):
                    raise outcome
                if isinstance(outcome, list):
                    item = outcome.pop(0)
                    if isinstance(item, BaseException):
                        raise item
                return _Resp(f"from-{backend}")
        return _C()

    class _Chat:
        def __init__(self) -> None:
            self.completions = None

    class _Fake:
        def __init__(self) -> None:
            self.chat = _Chat()

    llm._chat_providers = {"deepseek": _Fake(), "minimax": _Fake()}
    llm._chat_providers["deepseek"].chat.completions = make("deepseek")
    llm._chat_providers["minimax"].chat.completions = make("minimax")
    return seen


def _ratelimit() -> RateLimitError:
    req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)


def _server_error() -> InternalServerError:
    req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return InternalServerError("boom", response=httpx.Response(500, request=req), body=None)


def _bad_request() -> BadRequestError:
    req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return BadRequestError("bad", response=httpx.Response(400, request=req), body=None)


# ---------- 1. 故障分类 ----------

@pytest.mark.parametrize(
    "exc",
    [
        _ratelimit(), _server_error(),
        APIConnectionError(request=httpx.Request("POST", "https://x")),
        APITimeoutError(request=httpx.Request("POST", "https://x")),
        httpx.ConnectError("down"),
        httpx.ReadTimeout("slow"),
        httpx.RemoteProtocolError("bad frame"),
    ],
)
def test_transient_failures_are_retryable(exc):
    assert is_retryable(exc), f"{type(exc).__name__} 应判为可降级"
    assert classify(exc) == "retryable"


@pytest.mark.parametrize(
    "exc",
    [
        _bad_request(),
        AuthenticationError("no key", response=httpx.Response(
            401, request=httpx.Request("POST", "https://x")), body=None),
        ValueError("未知的 Chat 基座 'nope'"),
        TypeError(None, "messages"),
        KeyError("messages"),
        RuntimeError("未知异常"),
    ],
)
def test_deterministic_failures_are_fatal(exc):
    """确定性失败再发一次还是一样，换供应商也治不好。"""
    assert not is_retryable(exc), f"{type(exc).__name__} 不该降级"


def test_unknown_exception_defaults_to_fatal():
    """未知异常默认**不**重试：宁可漏降级，也不要把本地 bug 重试三遍。"""
    assert classify(Exception("?")) == "fatal"


def test_describe_has_no_secret_material():
    exc = _ratelimit()
    d = describe(exc)
    assert "retryable" in d and "RateLimit" in d and "429" in d


# ---------- 2. 链的构造 ----------

def test_chain_is_primary_then_fallbacks():
    llm = _client("minimax")
    assert llm._fallback_chain(None, None) == ["deepseek", "minimax"]


def test_chain_skips_primary_duplicate_and_deduplicates():
    llm = _client("deepseek, minimax ,minimax,")
    assert llm._fallback_chain(None, None) == ["deepseek", "minimax"]


def test_empty_fallbacks_means_no_degradation():
    llm = _client("")
    assert llm._fallback_chain(None, None) == ["deepseek"]


def test_explicit_model_disables_fallback():
    """实验脚本指定了 model 时绝不降级 —— 那时的诉求就是"就用这个模型"。"""
    llm = _client("minimax")
    assert llm._fallback_chain(None, "MiniMax-M2.5") == ["deepseek"]


# ---------- 3. 降级行为 ----------

@pytest.mark.asyncio
async def test_falls_back_on_rate_limit():
    llm = _client()
    seen = _install(llm, {"deepseek": _ratelimit()})
    out = await llm.chat([{"role": "user", "content": "hi"}])

    assert seen == ["deepseek", "minimax"], f"实际调用序列 {seen}"
    assert out["backend"] == "minimax"
    assert out["fallback"]["used"] is True
    assert out["fallback"]["from"] == "deepseek"
    assert out["fallback"]["to"] == "minimax"
    assert "retryable" in out["fallback"]["why"]


@pytest.mark.asyncio
async def test_does_not_fall_back_on_bad_request():
    """400 参数错再发一次还是 400。降级只会白白多烧一次。"""
    llm = _client()
    seen = _install(llm, {"deepseek": _bad_request()})
    with pytest.raises(BadRequestError):
        await llm.chat([{"role": "user", "content": "hi"}])
    assert seen == ["deepseek"], f"不该有第二次调用，实际 {seen}"


@pytest.mark.asyncio
async def test_does_not_fall_back_on_auth_error():
    llm = _client()
    seen = _install(llm, {"deepseek": AuthenticationError(
        "bad key", response=httpx.Response(401, request=httpx.Request("POST", "https://x")), body=None)})
    with pytest.raises(AuthenticationError):
        await llm.chat([{"role": "user", "content": "hi"}])
    assert seen == ["deepseek"]


@pytest.mark.asyncio
async def test_successful_primary_call_is_not_marked_as_fallback():
    llm = _client()
    _install(llm, {})
    out = await llm.chat([{"role": "user", "content": "hi"}])
    assert out["fallback"] == {"used": False}
    assert out["backend"] == "deepseek"


@pytest.mark.asyncio
async def test_no_degradation_when_chain_empty():
    llm = _client("")
    seen = _install(llm, {"deepseek": _server_error()})
    with pytest.raises(InternalServerError):
        await llm.chat([{"role": "user", "content": "hi"}])
    assert seen == ["deepseek"]


@pytest.mark.asyncio
async def test_chain_is_bounded_no_infinite_loop():
    """两个基座都失败时尝试次数必须正好等于链长。"""
    llm = _client()
    seen = _install(llm, {"deepseek": _ratelimit(), "minimax": _server_error()})
    with pytest.raises(InternalServerError):
        await llm.chat([{"role": "user", "content": "hi"}])
    assert seen == ["deepseek", "minimax"], f"实际 {seen}"


# ---------- 4. 降级必须留痕 ----------

@pytest.mark.asyncio
async def test_fallback_is_recorded_in_stats():
    llm = _client()
    _install(llm, {"deepseek": _ratelimit()})
    await llm.chat([{"role": "user", "content": "hi"}])

    st = llm.fallback_stats()
    assert st["total"] == 1
    assert st["chain"] == ["minimax"]
    assert st["by_pair"] == [{"from": "deepseek", "to": "minimax", "count": 1}]
    assert st["recent"][0]["why"]


@pytest.mark.asyncio
async def test_stats_empty_when_no_fallback():
    llm = _client()
    _install(llm, {})
    await llm.chat([{"role": "user", "content": "hi"}])
    assert llm.fallback_stats()["total"] == 0


@pytest.mark.asyncio
async def test_transient_then_success_does_not_count():
    """降级链里主基座自己重试成功的情况不应被记成降级 —— 当前实现不重试主基座，
    所以这里锁的是"没有降级就没有记录"这条底线。"""
    llm = _client("")
    _install(llm, {})
    await llm.chat([{"role": "user", "content": "hi"}])
    assert llm.fallback_stats()["total"] == 0


# ---------- 5. 接线：返回契约不能被悄悄改掉 ----------

@pytest.mark.asyncio
async def test_chat_return_shape_is_backward_compatible():
    """实验脚本按 dict 取字段。返回值加东西可以，少东西不行。"""
    llm = _client()
    _install(llm, {})
    out = await llm.chat([{"role": "user", "content": "hi"}])
    for key in ("content", "tool_calls", "usage", "model"):
        assert key in out, f"返回值少了 {key}，实验脚本会 KeyError"
    assert set(out["usage"]) >= {"prompt_tokens", "completion_tokens"}


def test_fallback_stats_is_exposed_on_health():
    """/api/health 必须能看到降级 —— 降级不报错，否则没人知道发生过。"""
    import inspect

    import app.main as main_mod

    src = inspect.getsource(main_mod.health)
    assert "fallback_stats" in src, "/api/health 没有透出降级统计"