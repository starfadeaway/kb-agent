"""守卫：`aclose()` 必须关掉全部网络客户端，且关完还能继续用。

缺陷
----
原实现只有一行：

    async def aclose(self) -> None:
        await self._rerank_http.aclose()

两个问题叠在一起：

1. **只关了一半。** 本实例实际持有 5 个网络客户端 ——
   `_rerank_http` 加上四个 `AsyncOpenAI`（deepseek / minimax /
   silicon / judge）。四个 `AsyncOpenAI` 各自建了独立的 httpx 连接池，
   **从不释放**。进程退出时残留连接就是那条
   `RuntimeError: Event loop is closed` 噪音的来源。

2. **关的是全局单例，而且关完就废了。** `aclose()` 的调用方是
   `main.py` 的 shutdown、`crawl.py` / `bulk_ingest.py` 的脚本收尾、
   `pipeline.py` 的自检入口 —— 它们拿到的都是 `get_llm()` 单例。
   只关不复位的话，单例里那几个槽位仍指向**已关闭**的 client，
   而它们是懒加载的（`is None` 才新建），所以永远不会被重建。
   同进程内任何 close 之后发生的调用都会拿到一个死 client，
   报 `Event has been closed`。也就是说旧实现只能"进程最后调一次"，
   一旦被调两次、或在 close 之后还有请求，整个 LLM 层就废了。

为什么单测没抓到
----------------
没有任何测试碰过 `aclose()`。测试替身（`test_agent_loop.py` 里的 FakeLLM）
自带一个空实现的 `aclose`，天然不可能失败。

双向验证
--------
本组测试顶层只 import 修复前就存在的符号（`LLMClient` / `get_settings`），
`_chat_providers` 等实例属性也用属性访问而非 import，
所以可以对修复前的代码快照实跑，确认能抓到真缺陷。
"""
from __future__ import annotations

import dataclasses

import pytest
from app.core.config import get_settings
from app.llm.client import LLMClient


def _client() -> LLMClient:
    """用假密钥构造一个真客户端。

    只创建对象不发请求，所以假密钥是安全的 —— 本组测试一个网络包都不打，
    也不读取真实 `.env` 里的任何密钥。
    """
    s = dataclasses.replace(
        get_settings(),
        deepseek_api_key="sk-test-not-a-real-key",
        minimax_api_key="sk-test-not-a-real-key",
        siliconflow_api_key="sk-test-not-a-real-key",
    )
    return LLMClient(settings=s)


def _all_clients(llm: LLMClient) -> list:
    """触发全部懒加载槽位，返回 [rerank_http, deepseek, minimax, silicon, judge]。"""
    return [llm._rerank_http, llm.deepseek, llm.minimax, llm.silicon, llm.judge_client]


def _closed(c) -> bool:
    """`is_closed` 在两类客户端上形态不同：httpx 是 property，openai SDK 是方法。

    统一探测。写死 `c.is_closed` 拿到的是 bound method，它恒为真 →
    `not c.is_closed` 恒为 False → 守卫静默失效（假绿）。
    """
    attr = getattr(c, "is_closed", None)
    return bool(attr() if callable(attr) else attr)


# ---------- 1. 必须全部关闭，而不只是 rerank_http ----------

@pytest.mark.asyncio
async def test_aclose_closes_every_openai_client():
    llm = _client()
    clients = _all_clients(llm)
    assert not any(_closed(c) for c in clients), "构造出来就该全是 open 的"

    await llm.aclose()

    still_open = [type(c).__name__ for c in clients if not _closed(c)]
    assert not still_open, f"这些客户端没被关闭：{still_open}"


@pytest.mark.asyncio
async def test_aclose_is_not_a_noop_for_openai_clients():
    """四个 AsyncOpenAI 都真的被构造出来了，且状态是 open。"""
    llm = _client()
    for c in (llm.deepseek, llm.minimax, llm.silicon, llm.judge_client):
        assert not _closed(c), "构造出来就该是 open 的，不然这个守卫测不出东西"


# ---------- 2. 关的是全局单例 → 关完必须能复位继续用 ----------

@pytest.mark.asyncio
async def test_slots_are_reset_after_aclose():
    """单例被 close 后，懒加载槽位必须归零，否则永远拿不到可用 client。

    注意必须**先创建再关**：如果从头到尾没碰过某个槽位，它本来就是 None，
    断言 `is None` 在修复前也能过 —— 那是假绿，抓不到真缺陷。
    """
    llm = _client()
    _all_clients(llm)
    assert llm._chat_providers["deepseek"] is not None, "前置条件没成立"
    await llm.aclose()

    assert llm._chat_providers["deepseek"] is None, "chat provider 槽位没复位"
    assert llm._chat_providers["minimax"] is None
    assert llm._silicon is None, "silicon 槽位没复位"
    assert llm._judge is None, "judge 槽位没复位"


@pytest.mark.asyncio
async def test_provider_is_recreated_and_usable_after_aclose():
    """close 之后再次访问必须拿到一个**新的、open 的** client。"""
    llm = _client()
    first = llm.deepseek
    await llm.aclose()

    second = llm.deepseek
    assert not _closed(second), "复位后拿到的仍是一个已关闭的 client"
    assert _closed(first), "旧 client 应已被关闭"
    # 顺带证明两个 client 的连接池确实是独立的，而不是被复用了
    assert second is not first


@pytest.mark.asyncio
async def test_rerank_http_is_usable_after_aclose():
    """`_rerank_http` 被关掉后必须重建，否则 rerank 这条路永久不可用。"""
    llm = _client()
    old = llm._rerank_http
    await llm.aclose()
    assert _closed(old), "旧的 rerank_http 应当被关闭"

    new = llm._rerank_http
    assert new is not old, "rerank_http 没有重建"
    assert not _closed(new), "重建出来的 rerank_http 却是关闭的"
    await new.aclose()


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    """连调两次不能抛 —— shutdown + 脚本收尾可能都调。"""
    llm = _client()
    _all_clients(llm)
    await llm.aclose()
    await llm.aclose()  # 不应抛
    assert not _closed(llm._rerank_http), "第二次 aclose 把新建的也关了"


@pytest.mark.asyncio
async def test_aclose_before_any_use_is_safe():
    """一个 provider 都没用过就 close —— 不能因为 None 而炸。"""
    llm = _client()
    await llm.aclose()
    assert llm._chat_providers["deepseek"] is None


@pytest.mark.asyncio
async def test_one_failing_client_does_not_block_the_others():
    """单个 client 关失败不能让其余的漏掉 —— 连接泄漏正是要修的东西。"""
    llm = _client()
    healthy = llm.deepseek

    class _Boom:
        is_closed = False

        async def close(self):
            raise RuntimeError("模拟关闭失败")

    llm._silicon = _Boom()
    await llm.aclose()  # 不应抛

    assert _closed(healthy), "前一个 client 没能关掉（说明异常把循环打断了）"
    assert llm._silicon is None, "失败的 client 槽位也要复位，否则永远卡在坏对象上"


# ---------- 3. 动态注册的基座不能被 aclose 清掉 ----------

@pytest.mark.asyncio
async def test_aclose_keeps_registered_backend_keys():
    """`CHAT_BACKENDS` 是可扩展的，aclose 不能把新注册的 key 弄丢。"""
    llm = _client()
    llm._chat_providers["future_backend"] = None
    await llm.aclose()
    assert "future_backend" in llm._chat_providers, "aclose 把已注册的基座 key 清掉了"


@pytest.mark.asyncio
async def test_unknown_backend_still_rejected_after_aclose():
    llm = _client()
    await llm.aclose()
    with pytest.raises(ValueError, match="未知的 Chat 基座"):
        llm._provider("not-a-backend")