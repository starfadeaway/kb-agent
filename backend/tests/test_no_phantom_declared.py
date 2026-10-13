"""死代码防回流 + 「名单里的名字必须真的存在」

## 为什么要有这个文件

本项目 2026-10 做了一次全仓审计，找到了三类"看起来能用、实际是死的或假的"东西，
其中最严重的一条是**我自己造的**：`TOOL_TRANSPORT` 被注释和文档描述成
「设成 mcp 就走 MCP 协议」，但 `app/` 与 `scripts/` 里没有任何一处代码分支读它。
当时的守卫 `test_mcp_is_not_default_transport` 断言的是
`getattr(settings, "tool_transport", "builtin") != "mcp"` ——
只读了一个**没人用的配置值**，什么都没测。

根因值得单独记住：**断言「配置的值」不等于断言「行为真的变了」**。
下面第 10 条就是为此设计的通用网。

## 本文件的边界

本项目**不做**「扫描全部未使用符号」那种大网：装饰器注册（`@tool`、`@api_router.*`）、
注册表（`REGISTRY[name]`）、`__init__` 协议方法都会误报，噪声一大就没人看。
所以这里用显式清单 —— 每条写明「删的是什么、为什么删、谁可能想把它加回来」。

真正的通用网只有一条：`_SECRET_FIELDS` 名单校验。
它能泛化，因为它验的是**一个结构化名单 vs 真实字段**，而不是猜调用关系。
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest


# --------------------------------------------------------------------------- #
# 1. LLMClient：删掉的两个方法不得回来
# --------------------------------------------------------------------------- #
def test_no_chat_stream_method():
    """`LLMClient.chat_stream` 已删：48 行上游流式，全仓从未被调用一次。

    `/api/chat/stream` 走的是「Agent 拿完整答案 → `_split_for_stream()` 切片推送」，
    不是 token 级流式。曾有个 `probe_stream_latency.py` 疑似依赖它，
    核实后发现那里的 `"chat_stream"` 只是**产物 JSON 的 key 名**，不是方法调用。
    """
    from app.llm.client import LLMClient

    assert not hasattr(LLMClient, "chat_stream"), (
        "chat_stream 又加回来了。如果现在要真做上游流式，"
        "必须同时：改 _run_stream 去消费它、加一条断言首个 token 事件早于"
        "生成完成的守卫，并更新 README 延迟章节的口径说明。"
    )


def test_no_embed_dim_method():
    """`embed_dim()` 已删：无调用点，且 .env.example 曾拿它当论据（也已改）。"""
    from app.llm.client import LLMClient

    assert not hasattr(LLMClient, "embed_dim")


# --------------------------------------------------------------------------- #
# 2. 两个 store 的 RetrievedChunk.cite 已删
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod_name", ["app.store.vector_store", "app.store.lance_store"])
def test_retrieved_chunk_has_no_cite(mod_name: str):
    """`cite()` 已删：两个后端都定义了一份，全仓没有任何调用点。

    引用出处实际走的是 tools.py 的「出处：」行，由 `react.py::_collect_refs`
    从工具结果文本里扫出来 —— 不是这个方法。
    """
    import importlib

    cls = importlib.import_module(mod_name).RetrievedChunk
    assert not hasattr(cls, "cite"), (
        "cite() 加回来了。如果要用它生成引用，"
        "必须同步改 _collect_refs —— 否则会出现两套并存的引用来源，"
        "而忠实度评测只会认其中一套。"
    )


# --------------------------------------------------------------------------- #
# 3. fallback.NotRetryable 已删
# --------------------------------------------------------------------------- #
def test_no_notretryable_class():
    """`NotRetryable` 异常类已删：定义了但全仓没有 raise、也没有 except。

    降级链实际用的是 `_FATAL` / `_RETRYABLE` 元组 + `is_retryable()`。
    """
    from app.llm import fallback

    assert not hasattr(fallback, "NotRetryable"), (
        "NotRetryable 又加回来了。要用它就得让 _FATAL 元组或 is_retryable "
        "真正认它，否则又是一个看着能控制降级、其实控制不了的开关。"
    )


# --------------------------------------------------------------------------- #
# 4. /api/sessions 端点已删
# --------------------------------------------------------------------------- #
def test_no_sessions_endpoint():
    """会话列表端点已删：前端零调用（frontend/src 里没有 session 字样），
    README 的端点清单也没列它。留着就是一个没人用的对外接口。"""
    src = inspect.getsource(__import__("app.main", fromlist=["api_router"]))
    assert '"/sessions"' not in src, (
        "/api/sessions 又加回来了。如果要做会话列表 UI，"
        "请连带把 frontend 的调用一起加上 —— 后端单方面加接口只会让它再次变成死代码。"
    )


def test_no_list_sessions_function():
    """`persist.list_sessions` 随端点一起删：它只被那个端点调用。"""
    from app.store import persist

    assert not hasattr(persist, "list_sessions")


# --------------------------------------------------------------------------- #
# 5. 死参数与恒空值
# --------------------------------------------------------------------------- #
def test_delta_report_has_no_new_pages():
    """`DeltaReport.new_pages` 已删：全仓没有任何 append，恒为空。

    于是 `new_pages_count` 恒为 0 —— 一个恒零的统计项被写进了
    `data/eval/delta_sync_*.json` 这类要写进简历的证据文件，
    读的人要花时间确认「0 是不是真的没有新增页」。
    """
    from app.ingest.delta_sync import DeltaReport

    fields = {f.name for f in dataclasses.fields(DeltaReport)}
    assert "new_pages" not in fields
    assert "new_pages_count" not in DeltaReport(game="sr", since="2026-01-01").as_dict(), (
        "产物里又出现 new_pages_count。如果这次真的会统计新增页，"
        "请确保它有 append 路径并有测试覆盖 —— 别放一个恒 0 的占位数。"
    )


def test_detect_changes_has_no_dry_run_param():
    """`detect_changes(dry_run=)` 已删：函数体从未读取，11 处调用也全不传。

    注意别和 `crawl.py` 的 `--dry-run` 混淆 —— 那个是真在用的
    （`crawl.py` 函数体里有 `if dry_run:` 分支），别一起删掉。
    """
    from app.ingest.delta_sync import detect_changes

    params = inspect.signature(detect_changes).parameters
    assert "dry_run" not in params, (
        "dry_run 加回来了。函数本来就不写任何状态，"
        "一个控制不了写入的开关比没有开关更糟。"
    )


def test_dry_run_still_exists_in_crawl():
    """反向锁定：`crawl.py` 的 `dry_run` 是活的，别被当成同类问题误删。"""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "crawl.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "crawl"
    )
    assert any(
        isinstance(node, ast.Name) and node.id == "dry_run" and isinstance(node.ctx, ast.Load)
        for node in ast.walk(fn)
    ), "crawl() 的 dry_run 真的在读了，不该被一起删掉"


def test_crawl_category_has_no_on_progress():
    """`on_progress` 已删：函数体里实现了两处回调，但没有任何调用方传入。

    留着一个永远不触发的回调 = 留了一条没人验证的分支。
    """
    from app.ingest.pipeline import Ingestor

    params = inspect.signature(Ingestor.crawl_category).parameters
    assert "on_progress" not in params


def test_ingest_request_has_no_interval():
    """`IngestRequest.interval` 已删：收下后从未使用。

    WikiClient 的 min_interval 来自 `settings.wiki_min_interval`。
    CLI 的 `crawl.py --interval` 是另一个东西，那个仍在用。
    """
    from app.main import IngestRequest

    assert "interval" not in IngestRequest.model_fields


# --------------------------------------------------------------------------- #
# 6. 通用网：名单里的名字必须真的存在
# --------------------------------------------------------------------------- #
def test_secret_fields_all_exist():
    """`_SECRET_FIELDS` 的每个名字都必须对应一个真实的 Settings 字段。

    这条是本文件唯一的**通用**断言，也是本轮审计里最值得留下的一条。
    `judge_api_key` 曾经在打码名单里，但 Settings 上从来没有这个字段
    （Judge 复用 `siliconflow_api_key`）。它无害，但会让人误以为
    「Judge 有独立 Key 可以单独轮换」—— 一个会误导人的空条目。

    同类风险会一直复发：任何「名单型」结构（白名单、豁免表、配置映射）
    都可能残留已删除的条目。所以对这类名单要有一条「名字必须存在」的断言，
    而不是等到有人读到它才发现。
    """
    from app.core.config import Settings

    real = {f.name for f in dataclasses.fields(Settings)}
    missing = [n for n in Settings._SECRET_FIELDS if n not in real]
    assert not missing, (
        f"_SECRET_FIELDS 里有不存在的字段：{missing}\n"
        f"真实字段：{sorted(real)}\n"
        f"这类空条目会误导人以为存在一个并不存在的独立配置。"
    )


def test_transitive_allowlist_entries_are_real_packages():
    """`_TRANSIITIVE` 白名单里的包名必须是**真实存在**的依赖。

    防的是同一种病：白名单里留着一个早已卸掉的包名，
    于是「这个 import 靠 X 带入」的理由永远不会被发现已经不成立。
    """
    from importlib.metadata import PackageNotFoundError, version
    from importlib.metadata import PackageNotFoundError as _PNF

    from tests.test_requirements_honest import _IMPORT_TO_PIP, _TRANSIITIVE

    absent: list[str] = []
    for mod in _TRANSIITIVE:
        pip = _IMPORT_TO_PIP.get(mod, mod).lower().replace("_", "-")
        try:
            version(pip)
        except (PackageNotFoundError, _PNF):
            absent.append(pip)
    assert not absent, (
        f"_TRANSIITIVE 里的这些包在当前环境里并不存在：{absent}\n"
        f"它们应该由 requirements.txt 里的直接依赖带入，"
        f"装不上就说明那条理由已经不成立了。"
    )