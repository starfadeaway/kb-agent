"""LLM / Embedding / Rerank 客户端。

三个 provider 都走 OpenAI 兼容协议，但基座、鉴权头、向量维度都不同，
所以这里统一封装一层，避免上层代码出现 provider 分支。
"""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings


@dataclass(frozen=True)
class RerankHit:
    index: int
    score: float


# 超过这个条数就走批量接口而不是逐条查缓存（入库路径）
_CACHE_BATCH_THRESHOLD = 8

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_think(text: str) -> str:
    """剥掉 <think>…</think> 推理块。

    MiniMax-M2.5 把心智链写在 content 里，而 DeepSeek 收到这段会当成普通文本，
    自己的推理也会被带偏。统一在客户端剥除，上层永远看不到 think 标签。

    未闭合的 <think>（被 max_tokens 截断时很常见）按「从 <think> 到结尾全丢」处理 ——
    宁可少一段文本，也不能把半截推理当答案输出给用户。
    """
    out = text
    while True:
        a = out.find(_THINK_OPEN)
        if a == -1:
            return out.strip()
        b = out.find(_THINK_CLOSE, a)
        out = out[:a] + out[b + len(_THINK_CLOSE) :] if b != -1 else out[:a]


def _balanced_object(text: str, start: int) -> str | None:
    """取出 text[start]=='{' 起配平的那一段 JSON；字符串里的括号不计入深度。"""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalize_call(obj: Any) -> tuple[str, dict]:
    """把见过的几种内联写法归一成 (工具名, 参数 dict)。认不出就返回空。"""
    if not isinstance(obj, dict):
        return "", {}
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    raw = obj.get("arguments", obj.get("parameters", obj.get("args")))
    if isinstance(name, dict):  # {"function": {"name": ..., "arguments": ...}}
        name, raw = name.get("name"), name.get("arguments", raw)
    if not isinstance(name, str):
        return "", {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {"__raw__": raw}
    return name, raw if isinstance(raw, dict) else {}


def parse_inline_tool_calls(content: str, allowed: set[str]) -> list[dict]:
    """兜底解析「模型写在正文里」的工具调用。

    没有这层，ReAct 的消息链会断：assistant 消息不带 tool_calls，
    后面的 role='tool' 消息就没有锚点，下一轮 API 直接 400。

    `allowed` 由调用方传入 —— 也就是本次请求**声明过的**工具名集合。
    这样兜底解析只可能认出"本来就允许调"的工具，不会凭空造出一个
    模型/schema 里都不存在的函数名。
    """
    calls: list[dict] = []
    if not content or "{" not in content:
        return calls
    for i, ch in enumerate(content):
        if ch != "{":
            continue
        blob = _balanced_object(content, i)
        if blob is None:
            continue
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # 正文里的花括号（示例代码、数学写法）远多于工具调用，
            # 解析失败是常态，直接跳过，不能因此中断整段扫描。
            continue
        name, args = _normalize_call(obj)
        if name in allowed:
            calls.append({"id": f"inline-{len(calls)}", "name": name, "args": args})
    return calls


def l2_normalize(vec: list[float]) -> list[float]:
    """bge-m3 实测输出范数不为 1，不归一化会让不同长度的向量抢相似度。

    这是实验 D（换 embedding 模型）的隐藏变量之一：必须双侧归一化，
    否则对比里混进了"范数分布"这个不该有的变量。
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class _LRU(dict):
    """进程内 LRU。命中/未命中只影响延迟，不影响结果 —— 最多多重算一次。"""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def put(self, key, value) -> None:
        if len(self) >= self.maxsize:
            # 丢最早插入的那个（Python 3.7+ dict 保序）
            self.pop(next(iter(self)), None)
        self[key] = value

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


class LLMClient:
    #: 可选的 Chat 基座。key 是对外名（API / 前端下拉框用的就是它），
    #: value 指向 Settings 里的模型名字段。
    #: 加新 provider 只要在这里加一行 —— 上层（Agent / API / 前端）都不需要改，
    #: 这就是"把 provider 选择做成数据而不是分支"的价值。
    CHAT_BACKENDS: ClassVar[dict[str, str]] = {
        "deepseek": "deepseek_chat_model",
        "minimax": "minimax_chat_model",
    }

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        # key = Settings 里的 api_key 字段名，value = 已构造的 OpenAI 兼容客户端
        self._chat_providers: dict[str, AsyncOpenAI | None] = {
            "deepseek": None, "minimax": None,
        }
        self._silicon: AsyncOpenAI | None = None
        self._judge: AsyncOpenAI | None = None
        self._rerank_http = httpx.AsyncClient(timeout=30.0)
        # embedding / rerank 都是确定性函数，同输入必然同输出，缓存不影响正确性。
        # 主要收益场景：评测重跑、Agent 连续几步问同一个 query、用户重复提问。
        self._embed_cache = _LRU(maxsize=512)
        self._rerank_cache = _LRU(maxsize=256)

    # ---------------- provider 懒加载 ----------------
    def _provider(self, name: str) -> AsyncOpenAI:
        key_field = f"{name}_api_key"
        url_field = f"{name}_base_url"
        if name not in self._chat_providers:
            raise ValueError(f"未知的 Chat 基座 {name!r}，可选：{', '.join(self.CHAT_BACKENDS)}")
        if self._chat_providers[name] is None:
            api_key = getattr(self.s, key_field, "")
            if not api_key:
                raise RuntimeError(f"{key_field.upper()} 未配置")
            self._chat_providers[name] = AsyncOpenAI(
                api_key=api_key, base_url=getattr(self.s, url_field, "")
            )
        return self._chat_providers[name]

    @property
    def deepseek(self) -> AsyncOpenAI:
        return self._provider("deepseek")

    @property
    def minimax(self) -> AsyncOpenAI:
        return self._provider("minimax")

    @property
    def silicon(self) -> AsyncOpenAI:
        if self._silicon is None:
            if not self.s.siliconflow_api_key:
                raise RuntimeError("SILICONFLOW_API_KEY 未配置")
            self._silicon = AsyncOpenAI(
                api_key=self.s.siliconflow_api_key, base_url=self.s.siliconflow_base_url
            )
        return self._silicon

    @property
    def judge_client(self) -> AsyncOpenAI:
        """Judge 专用客户端 —— 走 `JUDGE_BASE_URL`，而不是 SiliconFlow。

        之前 `judge()` 直接用 `self.silicon`，于是 `JUDGE_BASE_URL` 这个
        env var 定义了却**永远读不到**：想把 Judge 挪到独立基座（避免和被测
        模型共用一个供应商、或换推理服务）时，改 .env 没有任何效果，
        而且不报任何错。

        默认值仍然回落到 SiliconFlow，所以不配 `JUDGE_BASE_URL` 的行为不变。
        """
        if self._judge is None:
            if not self.s.siliconflow_api_key:
                raise RuntimeError("Judge 需要 SILICONFLOW_API_KEY")
            self._judge = AsyncOpenAI(
                api_key=self.s.siliconflow_api_key,
                base_url=self.s.judge_base_url or self.s.siliconflow_base_url,
            )
        return self._judge

    async def aclose(self) -> None:
        await self._rerank_http.aclose()

    # ---------------- Chat ----------------
    def chat_backend_name(self, backend: str | None) -> str:
        """把对外名解析成实际模型名，顺便校验这个 provider 配没配。"""
        key = (backend or "deepseek").lower()
        if key not in self.CHAT_BACKENDS:
            raise ValueError(f"未知的 Chat 基座 {backend!r}，可选：{', '.join(self.CHAT_BACKENDS)}")
        model_name = getattr(self.s, self.CHAT_BACKENDS[key], "")
        if not model_name:
            raise RuntimeError(f"{key} 对应的模型名未配置（检查 .env）")
        return model_name

    def available_backends(self) -> list[dict]:
        """已配好密钥的 Chat 基座，/api/health 透给前端渲染下拉框。

        只暴露 key / model / 是否可用，不返回任何密钥内容。
        """
        out = []
        for key, field in self.CHAT_BACKENDS.items():
            model = getattr(self.s, field, "")
            if not model:
                continue
            out.append(
                {"id": key, "model": model, "configured": bool(getattr(self.s, f"{key}_api_key", ""))}
            )
        return out

    @staticmethod
    def assistant_tool_message(content: str, calls: list[dict]) -> dict:
        """构造带 tool_calls 的 assistant 消息。

        必须原样回传 tool_calls：后一条 role='tool' 消息靠 tool_call_id 挂到它上面，
        只发一句纯文本 assistant，下一轮就会报
        "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"。
        """
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)},
                }
                for c in calls
            ],
        }

    def _chat_provider(self, backend: str | None, model: str | None) -> tuple[AsyncOpenAI, str]:
        """把 (backend, model) 解析成 (客户端, 模型名)。

        `model` 直接给实际模型名时优先于 backend 的配置值 ——
        这样实验脚本能指定模型而不必改 .env，配置与实验互不干扰。
        """
        key = (backend or "deepseek").lower()
        if key not in self.CHAT_BACKENDS:
            raise ValueError(f"未知的 Chat 基座 {backend!r}，可选：{', '.join(self.CHAT_BACKENDS)}")
        model_name = model or self.chat_backend_name(key)
        return self._provider(key), model_name

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        backend: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> dict:
        """返回 {'content': str, 'tool_calls': [...], 'usage': {...}, 'model': str}。

        `backend` 是对外的 provider 名（deepseek / minimax），
        上层不需要知道 key 和 base_url 在哪 —— 那是这里的事。
        """
        provider, model_name = self._chat_provider(backend, model)

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = await provider.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        msg = resp.choices[0].message
        content = strip_think(msg.content or "")
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__raw__": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "args": args})
        # MiniMax 偶尔把工具调用写成 content 里的 JSON 而不是标准 tool_calls。
        # 兜底只在「本次确实声明了工具」时启用，且只认声明过的名字。
        if not calls and tools:
            allowed = {t.get("function", {}).get("name", "") for t in tools}
            calls = parse_inline_tool_calls(content, allowed)
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }
        return {"content": content, "tool_calls": calls, "usage": usage, "model": model_name}

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        backend: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ):
        """SSE 流式，供前端打字机效果。

        逐 token 剥 <think>：推理块整段丢弃，且不会因为半截标签
        （"thi" + "nk>" 跨两个 chunk）而漏出前缀。
        """
        provider, model_name = self._chat_provider(backend, model)
        kwargs.setdefault("temperature", 0.0)
        stream = await provider.chat.completions.create(
            model=model_name, messages=messages, stream=True, **kwargs
        )
        in_think = False
        buf = ""
        async for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if not piece:
                continue
            buf += piece
            # 只在缓冲区里已出现闭合标签或明显不是推理时才吐字
            while True:
                if in_think:
                    end = buf.find(_THINK_CLOSE)
                    if end == -1:
                        buf = buf[-len(_THINK_CLOSE) :]  # 留住可能跨 chunk 的标签尾巴
                        break
                    buf, in_think = buf[end + len(_THINK_CLOSE) :], False
                    continue
                start = buf.find(_THINK_OPEN)
                if start == -1:
                    keep = len(_THINK_OPEN) - 1  # 留住可能跨 chunk 的标签开头
                    if len(buf) > keep:
                        yield buf[:-keep] if keep else buf
                        buf = buf[-keep:] if keep else ""
                    break
                if start > 0:
                    yield buf[:start]
                buf, in_think = buf[start + len(_THINK_OPEN) :], True
        if buf and not in_think:
            yield buf

    # ---------------- Embedding ----------------
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        normalize: bool = True,
        batch_size: int = 16,
    ) -> list[list[float]]:
        """Embedding 入口。

        两条路径，因为两种场景的最优策略相反：
          - **少量文本**（检索时的 1 条 query）：逐条查缓存，命中就完全不发 HTTP。
            实测 bge-m3 单条 147ms，Agent 连续几步或评测重跑时省得很实在。
          - **大量文本**（入库时的几千条）：必须走批量接口，逐条串行会退化成
            上千次 HTTP，比原来的 16 条/批慢一个数量级。

        两条路径都会回填缓存，所以入库过的语料在检索时也能命中。
        """
        model_name = model or self.s.embed_model_main
        if len(texts) > _CACHE_BATCH_THRESHOLD:
            return await self.embed_batch(
                texts, model=model_name, normalize=normalize, batch_size=batch_size
            )

        def key_of(t: str) -> tuple:
            return (model_name, normalize, t)

        # 命中/未命中都要计数：这两个数是 /api/health 展示缓存效果的唯一来源，
        # 少记一次就意味着侧栏永远显示 0%，可观测性直接失效。
        for t in dict.fromkeys(texts):
            if key_of(t) in self._embed_cache:
                self._embed_cache.hits += 1
            else:
                self._embed_cache.misses += 1
        missing = [t for t in dict.fromkeys(texts) if key_of(t) not in self._embed_cache]
        if missing:
            # 未命中的攒成一批，一次 HTTP 搞定（而不是每条一次）
            resp = await self.silicon.embeddings.create(
                model=model_name, input=[t if t.strip() else " " for t in missing]
            )
            for text, d in zip(missing, resp.data, strict=False):
                vec = d.embedding
                self._embed_cache.put(key_of(text), l2_normalize(vec) if normalize else vec)
        return [self._embed_cache[key_of(t)] for t in texts]

    async def embed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        """真正的大批量 embedding（入库用）。串行分批 + 间隔，限速友好。"""
        model_name = kwargs.pop("model", None) or self.s.embed_model_main
        normalize = kwargs.pop("normalize", True)
        batch_size = kwargs.pop("batch_size", 16)
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            resp = await self.silicon.embeddings.create(model=model_name, input=batch)
            for text, d in zip(batch, resp.data, strict=False):
                vec = d.embedding
                vec = l2_normalize(vec) if normalize else vec
                out.append(vec)
                self._embed_cache.put((model_name, normalize, text), vec)
            if len(texts) > batch_size:
                await asyncio.sleep(0.15)
        return out

    async def embed_dim(self, model: str | None = None) -> int:
        vecs = await self.embed(["维度探测"], model=model)
        return len(vecs[0])

    # ---------------- Rerank ----------------
    async def rerank(
        self, query: str, documents: list[str], *, model: str | None = None, top_n: int = 20
    ) -> list[RerankHit]:
        """硅基流动 rerank 接口；返回按分数降序的原始下标。

        注意：rerank 模型只对 bge 系向量模型有效，
        换 embedding 做对比实验时必须关闭，否则引入隐藏变量。

        缓存键包含全部文档内容 —— 只有「同样的 query + 同样的候选集」才复用，
        候选集变了就必须重算，否则会拿到旧排序。
        """
        if not documents:
            return []
        model_name = model or self.s.rerank_model
        key = (model_name, query, hash(tuple(documents)))
        if key in self._rerank_cache:
            self._rerank_cache.hits += 1
            return self._rerank_cache[key]
        self._rerank_cache.misses += 1

        resp = await self._rerank_http.post(
            f"{self.s.siliconflow_base_url.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {self.s.siliconflow_api_key}"},
            json={
                "model": model_name,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        out = [RerankHit(index=r["index"], score=float(r["relevance_score"])) for r in data["results"]]
        self._rerank_cache.put(key, out)
        return out

    def cache_stats(self) -> dict:
        """给 /api/health 展示用：延迟优化的效果要可观测才算数。"""
        return {
            "embed": self._embed_cache.stats(),
            "rerank": self._rerank_cache.stats(),
        }

    def clear_caches(self) -> None:
        """清空结果缓存。

        评测脚本在每个 label 前调用：否则后一个配置会命中前一个配置的缓存，
        延迟列就不是同条件下的可比数字了。
        """
        self._embed_cache.clear()
        self._embed_cache.hits = self._embed_cache.misses = 0
        self._rerank_cache.clear()
        self._rerank_cache.hits = self._rerank_cache.misses = 0

    # ---------------- Judge ----------------
    async def judge(self, prompt: str, *, model: str | None = None, max_tokens: int = 512) -> str:
        """Judge 走独立基座（Qwen），严禁用被测模型自评。"""
        model_name = model or self.s.judge_model
        resp = await self.judge_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


_default: LLMClient | None = None


def get_llm(settings: Settings | None = None) -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient(settings)
    return _default