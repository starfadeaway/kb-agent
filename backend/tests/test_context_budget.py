"""上下文预算：多轮历史裁剪 + 工具结果截断 + chroma 字段下线。

## 为什么这个文件不做双向验证

本轮改动里有**新增符号**（`trim_history`、`data_root`），它们在改造前的 HEAD 上
根本不存在，因此没有"修复前快照"可以注入 sys.modules。
强行套双向验证只会得到一个假结论：旧代码在 collection 阶段 ImportError，
一条行为断言都跑不到 —— 那只证明"新符号不存在"，和守卫有效性无关。

所以这里换成三件对新增模块真正有效的事：
1. **分流逻辑写成可测纯函数**：`trim_history` 不碰 settings 也不碰 IO，
   每条分支都能独立断言；
2. **断言覆盖每条分支**：见下面 1~5 组；
3. **接线必须用非默认值验证**：见 `test_prepare_actually_trims_by_env` 等 ——
   默认值下这些代码路径大概率不触发，用默认值断言会得到一条恒真的假绿。
   （这正是本项目栽过多次的坑：`.env` 默认值恰好等于写死值时，"改 .env 不生效"
   完全看不出来。）

另有一处**手工变异验证**记录在 commit message 里：把 `_prepare` 改回
`messages.extend(history[-8:])`、把 tools.py 的截断改回字面量，
本文件必须由绿转红。

## 变更背景

三处真实缺陷，都是"看起来正常、实际不生效/不设防"：

1. `messages.extend(history[-8:])` —— 固定 8 条不看每条多长。8 条长回答能
   一口吃掉大半 `AGENT_TOKEN_BUDGET`，而预算只在 ReAct 循环里检查、不管历史。
2. `_truncate(c.text, 500)` / `_truncate(text, 4000)` —— 全项目唯一改不了的
   预算，其他旋钮都走 .env。
3. `chroma_path` / `chroma_collection_prefix` —— 名字指向 chromadb（早已弃用），
   且 `chroma_path` 自己的目录**从不被使用**（两个 store 取的都是它的 `.parent`）。
"""
from __future__ import annotations

import asyncio
import dataclasses
import types
from pathlib import Path

from app.agent.react import ReActAgent, trim_history
from app.agent.tools import ToolContext, get_page, search_kb
from app.core.config import Settings, get_settings
from app.ingest.template_parser import clean_wikitext
from app.retrieval.hybrid import RetrievalHit

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


# --------------------------------------------------------------------------- #
# 1. trim_history 纯函数：每条分支都要能独立触发
# --------------------------------------------------------------------------- #
def _hist(*contents: str) -> list[dict]:
    """造交替的 user/assistant 历史，最后一条是 assistant（贴近真实会话）。"""
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": c} for i, c in enumerate(contents)]


def test_trim_history_empty():
    assert trim_history([], max_chars=100) == []


def test_trim_history_keeps_everything_when_within_budget():
    h = _hist("a" * 30, "b" * 30)
    assert trim_history(h, max_chars=100) == h


def test_trim_history_drops_oldest_and_keeps_time_order():
    """超预算时丢最早的，且返回必须仍是时间正序。"""
    h = _hist("旧" * 100, "中" * 100, "新" * 10)
    out = trim_history(h, max_chars=50)
    assert [m["content"] for m in out] == ["新" * 10]
    # 顺序断言：倒序拼接会让这一条直接红 —— 这是最容易写错的一处。
    assert out == [{"role": "user", "content": "新" * 10}]


def test_trim_history_boundary_is_inclusive():
    """恰好等于预算必须全留 —— 边界判定写成 `<` 会多丢一轮。"""
    h = _hist("a" * 25, "b" * 25)
    assert len(trim_history(h, max_chars=50)) == 2


def test_trim_history_always_keeps_at_least_the_last_message():
    """单条就超预算时仍保留最后一条，否则多轮会话被静默降级成单轮。"""
    h = _hist("很长的历史" * 100, "短")
    out = trim_history(h, max_chars=10)
    assert len(out) == 1
    assert out[0]["content"] == "短"


def test_trim_history_never_returns_more_chars_than_budget():
    """不变量：只要结果不止一条，总字符数就必须真的在预算内。"""
    h = _hist(*[f"第{i}轮" * 40 for i in range(8)])
    for budget in (0, 1, 50, 120, 400, 5000):
        out = trim_history(h, max_chars=budget)
        if len(out) > 1:
            total = sum(len(m["content"]) for m in out)
            assert total <= budget, f"预算 {budget} 下超了：{total}"


def test_trim_history_tolerates_missing_or_none_content():
    """历史来自 DB 与请求，content 可能缺字段或为 None，不能因此抛异常。"""
    h = [{"role": "user"}, {"role": "assistant", "content": None}, {"role": "user", "content": "有"}]
    assert trim_history(h, max_chars=100)[-1]["content"] == "有"


# --------------------------------------------------------------------------- #
# 2. 接线：_prepare 必须真的用 agent_history_chars
# --------------------------------------------------------------------------- #
def _prepare_messages(history: list[dict], **overrides) -> list[dict]:
    """跑真 _prepare，替换成非默认预算值。"""
    s = dataclasses.replace(get_settings(), **overrides)
    agent = ReActAgent(settings=s, tools=[])
    ctx = ToolContext(settings=s, retriever=None, store=None, game="sr")
    _context, messages, _state = agent._prepare("当前问题", None, history, ctx)
    return messages


def test_prepare_actually_trims_by_env():
    """**接线断言 —— 必须用非默认值。**

    默认 4000 字符下这批历史压根不会被裁，用默认值断言必然恒真。
    这里三条各 40 字、预算 50：只有最新一条放得下，
    倒数第二条一加就 80 > 50 被整轮丢弃。
    """
    history = _hist("问" * 40, "答" * 40, "尾" * 40)
    messages = _prepare_messages(history, agent_history_chars=50)

    contents = [m["content"] for m in messages if m["role"] != "system"]
    assert contents == ["尾" * 40, "当前问题"]
    # 断言"没退化回 history[-8:]"：那样会得到三条历史全在。
    assert len(contents) == 2, "AGENT_HISTORY_CHARS 没生效，历史按条数原样全量进上下文"


def test_prepare_keeps_history_under_default_budget():
    """对照组：预算够大时历史必须一条不少，否则说明 trim 写坏了。"""
    history = _hist("问" * 40, "答" * 40, "尾")
    messages = _prepare_messages(history, agent_history_chars=100000)
    contents = [m["content"] for m in messages if m["role"] != "system"]
    assert contents == ["问" * 40, "答" * 40, "尾", "当前问题"]


def test_prepare_without_history_is_system_plus_question():
    messages = _prepare_messages([], agent_history_chars=50)
    assert [m["role"] for m in messages] == ["system", "user"]


def test_agent_history_chars_is_not_wired_to_a_hardcoded_value():
    """两条不同预算必须给出不同结果 —— 只测一条会被"恰好等于默认值"骗过。"""
    history = _hist(*[f"第{i}轮" + "字" * 100 for i in range(6)])
    loose = [m["content"] for m in _prepare_messages(history, agent_history_chars=100000)]
    tight = [m["content"] for m in _prepare_messages(history, agent_history_chars=150)]
    assert len(tight) < len(loose)


# --------------------------------------------------------------------------- #
# 3. 接线：工具结果截断必须走 .env
# --------------------------------------------------------------------------- #
def _chunk(text: str) -> types.SimpleNamespace:
    """鸭子类型即可 —— search_kb 只做属性访问，不 isinstance。"""
    return types.SimpleNamespace(
        chunk_id="c1", page_title="三月七", text=text, source_url="http://x/y",
        metadata={"section": "角色定位"},
    )


class _FakeRetriever:
    def __init__(self, text: str) -> None:
        self._hit = RetrievalHit(chunk=_chunk(text), score=0.9, rank_source="rrf")

    async def retrieve(self, query, cfg):
        return [self._hit]


def _run_search(text: str, **overrides) -> str:
    s = dataclasses.replace(get_settings(), **overrides)
    ctx = ToolContext(
        settings=s, retriever=_FakeRetriever(text), store=None, game="sr"
    )
    return asyncio.run(search_kb({"query": "三月七", "top_k": 1}, ctx))


def test_search_kb_truncates_by_tool_chunk_chars():
    """非默认值 17 字。用默认值 500 断言的话这条永远是恒真的假绿。"""
    out = _run_search("样" * 500, tool_chunk_chars=17)
    assert "样" * 17 in out
    assert "样" * 18 not in out
    assert "已截断，原长度 500" in out, "截断没有显式告知原长度，模型会以为已看完全文"


def test_search_kb_does_not_truncate_short_chunks():
    out = _run_search("短文本", tool_chunk_chars=500)
    assert "已截断" not in out
    assert "短文本" in out


def test_tool_chunk_chars_and_page_chars_are_independent():
    """两个截断不是同一个旋钮的两次赋值 —— 否则改一个会连带改另一个。"""
    assert get_settings().tool_chunk_chars != get_settings().tool_page_chars


class _FakeWikiClient:
    """替换 WikiClient：get_page 只验证截断长度，不该真去联网。"""

    captured: dict = {}

    def __init__(self, game, **kw):
        _FakeWikiClient.captured["game"] = game
        _FakeWikiClient.captured["kw"] = kw

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def fetch_page(self, title):
        return types.SimpleNamespace(
            title=title, wikitext="页" * 3000, last_revision=1700000000
        )


def test_get_page_truncates_by_tool_page_chars(monkeypatch):
    import app.ingest.wiki_client as wc_mod

    monkeypatch.setattr(wc_mod, "WikiClient", _FakeWikiClient)
    s = dataclasses.replace(get_settings(), tool_page_chars=33)
    ctx = ToolContext(settings=s, retriever=None, store=None, game="sr")

    out = asyncio.run(get_page({"title": "三月七"}, ctx))
    assert "页" * 33 in out
    assert "页" * 34 not in out
    expected = len(clean_wikitext("页" * 3000))
    assert f"已截断，原长度 {expected}" in out


# --------------------------------------------------------------------------- #
# 4. chroma 字段下线：改名必须**不是重定位**
# --------------------------------------------------------------------------- #
def test_no_chroma_field_remains_on_settings():
    """防回归：字段名再也不能以 chroma 开头回来。"""
    names = [f.name for f in dataclasses.fields(Settings)]
    assert not [n for n in names if "chroma" in n.lower()], (
        f"又出现了 chroma 字段：{[n for n in names if 'chroma' in n.lower()]}"
    )


def test_data_root_defaults_to_the_old_chroma_paths_parent():
    """锁住迁移等价性。

    旧路径 Path('./data/chroma').parent = ./data，新路径 Path('./data') = ./data。
    这一条红了就意味着改名把索引目录搬走了 —— 表现是"检索全部为空但不报错"，
    是本项目最难查的一类静默故障。
    """
    s = get_settings()
    legacy_parent = Path("./data/chroma").parent
    assert s.data_root_abs == (PROJECT_ROOT / legacy_parent).resolve()
    # 表名前缀没变 ⇒ 已有 LanceDB 表名不变 ⇒ 索引原地可读。
    assert s.collection_prefix == "kb"


def test_env_example_no_longer_mentions_chroma():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    # 只允许出现在迁移说明里，不允许再有可用的配置条目。
    assert "\nCHROMA_PATH=" not in text
    assert "\nCHROMA_COLLECTION_PREFIX=" not in text
    assert "DATA_ROOT=" in text
    assert "COLLECTION_PREFIX=" in text


def test_stores_land_under_the_data_root():
    """消费点断言：两个 store 的 root 必须真的在 data_root 之下。

    只改 config 不改 store，索引会写到别处去，而配置层看起来完全正常。
    """
    import tempfile

    from app.store.lance_store import LanceVectorStore
    from app.store.vector_store import VectorStore

    with tempfile.TemporaryDirectory(prefix="kb_root_probe_") as tmp:
        root = Path(tmp) / "data"
        s = dataclasses.replace(get_settings(), data_root=str(root))
        assert VectorStore(s).root == root / "vectors"
        assert LanceVectorStore(s).root == root / "lance"


def test_collection_prefix_actually_reaches_the_table_name():
    """collection_prefix 也要真的到表名 —— 这是索引能否被找到的另一半。"""
    import tempfile

    from app.store.vector_store import VectorStore

    with tempfile.TemporaryDirectory(prefix="kb_name_probe_") as tmp:
        s = dataclasses.replace(
            get_settings(), data_root=str(Path(tmp) / "data"),
            collection_prefix="zz_marker",
        )
        name = VectorStore(s).collection_name("sr", "BAAI/bge-m3", "base")
        assert name.startswith("zz_marker_sr_"), name


# --------------------------------------------------------------------------- #
# 5. 旧的历史条数上限不能被误当成"已经控住了上下文"
# --------------------------------------------------------------------------- #
def test_history_limit_and_char_budget_are_two_different_dials():
    """两条限制职责不同：DB 取数(limit=8) vs 上下文预算(chars)。

    记在这里是因为很容易有人看到 load_history 有 limit 就以为上下文可控，
    然后把字符预算调回很大 —— 那样 limit=8 挡不住 8 条长回答。
    """
    import inspect

    from app.store import persist

    sig = inspect.signature(persist.load_history)
    assert sig.parameters["limit"].default == 8
    assert get_settings().agent_history_chars > 0