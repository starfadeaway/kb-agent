"""守卫：全量语料扫描 / BM25 建索引不得发生在事件循环线程上。

缺陷
----
`HybridRetriever.retrieve()` 原先在协程里**直接同步调用**
`self._load_corpus(cfg)`（实测 sr 全量 ~800ms）和 `self._get_bm25(cfg)`
（实测建索引 ~1.9s）。这两步内部没有任何 `await`，所以在 asyncio 下
**整个事件循环会被卡住**：这期间所有并发的 `/api/chat`、`/api/retrieve`
连 `/health` 都答不了。缓存没命中的第一个请求就是一次全站停顿。

为什么"同步调用所以不会交错"是错的
----------------------------------
修复前的论证是「同步且内部无 await，所以无法交错执行」。
这个论证成立的前提是**只有一个线程会进来**。一旦把这两步改成
`asyncio.to_thread`，多线程立刻变成常态：

1. 两个请求同时命中冷缓存 → **重复拉全量 + 重复建索引**（1.9s × N）
2. `_get_bm25` 的慢路径会在持锁状态下调用 `_load_corpus`，
   而 `_load_corpus` 自己也要取同一把锁 → 必须 `RLock`，
   换普通 `Lock` 会直接死锁

所以锁和 to_thread 是一套的：只做一半就是新缺陷。
"""
from __future__ import annotations

import ast
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from app.retrieval import hybrid as hybrid_mod
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig

HYBRID_SRC = Path(__file__).resolve().parents[1] / "app" / "retrieval" / "hybrid.py"


class _FakeCol:
    """最小 chroma collection 替身；get 故意慢，用来放大竞态。"""

    def __init__(self, n: int = 120, delay: float = 0.15) -> None:
        self.calls = 0
        self._n = n
        self._delay = delay

    def get(self, where=None, include=None):  # noqa: ARG002
        self.calls += 1
        time.sleep(self._delay)  # 同步阻塞，正是要防的那种调用
        return {
            "ids": [f"c{i}" for i in range(self._n)],
            "documents": [f"命途 文档 {i} 星魂 养成" for i in range(self._n)],
            "metadatas": [{"page_title": f"P{i}"} for i in range(self._n)],
        }


class _FakeStore:
    def __init__(self, col: _FakeCol) -> None:
        self.col = col
        self.where_calls = 0

    def get_collection(self, game, embed_model, variant):  # noqa: ARG002
        return self.col

    def build_where(self, **kwargs):  # noqa: ARG002
        self.where_calls += 1
        return {"game": "sr"}

    def query(self, **kwargs):  # pragma: no cover - 本组测试都关掉向量路
        raise AssertionError("本组测试 use_vector=False，不应走到向量检索")


def _retriever(col: _FakeCol | None = None) -> tuple[HybridRetriever, _FakeStore]:
    col = col or _FakeCol()
    store = _FakeStore(col)
    r = HybridRetriever(store=store)  # 传 store 就不会去建真的向量库
    return r, store


# ---------- 1. 冷缓存并发不得重复拉全量 ----------

def test_concurrent_load_corpus_fetches_once():
    r, store = _retriever()
    cfg = RetrievalConfig(game="sr", use_bm25=False, use_vector=False)
    barrier = threading.Barrier(8)

    def work():
        barrier.wait()
        return r._load_corpus(cfg)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = [f.result(timeout=30) for f in [ex.submit(work) for _ in range(8)]]

    assert store.col.calls == 1, f"全量扫描被重复执行 {store.col.calls} 次"
    assert all(r0 == results[0] for r0 in results)


def test_concurrent_get_bm25_builds_once(monkeypatch):
    """8 个线程同时要同一个索引 → BM25Index.build 只该跑一次。"""
    r, store = _retriever()
    cfg = RetrievalConfig(game="sr", use_bm25=True, use_vector=False)

    orig_build = hybrid_mod.BM25Index.build
    n_build = 0
    build_lock = threading.Lock()

    def counting_build(ids, docs, tok):
        nonlocal n_build
        with build_lock:
            n_build += 1
        time.sleep(0.15)
        return orig_build(ids, docs, tok)

    monkeypatch.setattr(hybrid_mod.BM25Index, "build", staticmethod(counting_build))

    barrier = threading.Barrier(8)

    def work():
        barrier.wait()
        return r._get_bm25(cfg)

    with ThreadPoolExecutor(max_workers=8) as ex:
        got = [f.result(timeout=30) for f in [ex.submit(work) for _ in range(8)]]

    assert n_build == 1, f"BM25 索引被重复构建 {n_build} 次"
    assert store.col.calls == 1
    assert all(g is got[0] for g in got)


# ---------- 2. RLock：重入不死锁 ----------

def test_get_bm25_reentrant_lock_does_not_deadlock():
    """_get_bm25 → _load_corpus 是同锁重入。用非重入锁会在这里挂死。"""
    r, _ = _retriever(_FakeCol(n=40, delay=0.01))
    cfg = RetrievalConfig(game="sr", use_bm25=True, use_vector=False)
    barrier = threading.Barrier(4)
    done = threading.Event()

    def work():
        barrier.wait()
        r._get_bm25(cfg)
        done.set()

    threads = [threading.Thread(target=work, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert done.is_set(), "疑似死锁：4 个线程拿不到索引（检查是不是把 RLock 换成了 Lock）"
    assert not any(t.is_alive() for t in threads)


def test_cache_lock_is_reentrant():
    r, _ = _retriever(_FakeCol(n=10, delay=0))
    lock = r._cache_lock
    assert isinstance(lock, type(threading.RLock())), "缓存锁必须是 RLock"
    # 真重入一次：RLock 允许同线程重复 acquire
    assert lock.acquire(blocking=False)
    try:
        assert lock.acquire(blocking=False), "同线程二次 acquire 失败 → 这把锁不可重入"
    finally:
        lock.release()
        lock.release()


# ---------- 3. retrieve 不得阻塞事件循环 ----------

@pytest.mark.asyncio
async def test_retrieve_yields_to_other_coroutines():
    """retrieve 执行期间，另一个协程必须还能被调度。"""
    r, _ = _retriever(_FakeCol(n=200, delay=0.3))
    cfg = RetrievalConfig(game="sr", use_bm25=True, use_vector=False, final_top_k=3)

    ticks = 0
    stop = False

    async def ticker():
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.005)

    t = asyncio.create_task(ticker())
    await asyncio.sleep(0)  # 让 ticker 起步
    hits = await r.retrieve("命途 怎么配队", cfg)
    stop = True
    t.cancel()

    assert len(hits) == 3
    assert ticks > 5, f"事件循环在 retrieve 期间几乎没被调度（ticks={ticks}）→ 同步调用仍在协程里跑"


@pytest.mark.asyncio
async def test_retrieve_empty_query_never_touches_store():
    r, store = _retriever()
    assert await r.retrieve("   ", RetrievalConfig(game="sr")) == []
    assert store.col.calls == 0


# ---------- 4. 语义回归：缓存 / 淘汰 / 失效行为不变 ----------

def test_invalidate_clears_both_caches():
    r, _ = _retriever(_FakeCol(n=20, delay=0))
    cfg = RetrievalConfig(game="sr")
    r._load_corpus(cfg)
    r._get_bm25(cfg)
    assert r.stats() == {"bm25_cached": 1, "corpus_cached": 1}
    r.invalidate()
    assert r.stats() == {"bm25_cached": 0, "corpus_cached": 0}


def test_eviction_cap_still_applies():
    r, _ = _retriever(_FakeCol(n=20, delay=0))
    r._max_cached_indexes = 1
    r._load_corpus(RetrievalConfig(game="sr"))
    r._load_corpus(RetrievalConfig(game="zzz"))
    st = r.stats()
    assert st["corpus_cached"] <= 1, st


def test_corpus_roundtrip_preserves_text_and_meta():
    r, _ = _retriever(_FakeCol(n=5, delay=0))
    ids, docs, mapping = r._load_corpus(RetrievalConfig(game="sr"))
    assert len(ids) == len(docs) == len(mapping) == 5
    assert mapping["c0"].text.startswith("命途")
    assert mapping["c0"].metadata["page_title"] == "P0"


# ---------- 5. 结构守卫：源码级锁死"必须经 to_thread" ----------

def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    out: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def test_retrieve_only_calls_blocking_helpers_via_to_thread():
    """未来有人把这两行挪回协程里，直接从源码层抓住，不等线上卡顿。"""
    tree = ast.parse(HYBRID_SRC.read_text(encoding="utf-8"))
    parents = _parents(tree)

    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "retrieve"
    )
    offenders: list[str] = []
    checked = 0
    for node in ast.walk(fn):
        # 注意捕获的是**属性引用**而不是 Call：`asyncio.to_thread(self._load_corpus, cfg)`
        # 里 `self._load_corpus` 是实参，不是 func 位。
        if not (isinstance(node, ast.Attribute) and node.attr in {"_load_corpus", "_get_bm25"}):
            continue
        if isinstance(node.ctx, ast.Store):  # pragma: no cover
            continue
        checked += 1
        parent = parents.get(node)
        wrapped = (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Attribute)
            and parent.func.attr == "to_thread"
        )
        if not wrapped:
            offenders.append(ast.unparse(node))
    assert checked == 2, f"retrieve 里应恰好有 2 处阻塞调用，实际 {checked}"
    assert not offenders, f"以下调用未包 asyncio.to_thread，会阻塞事件循环：{offenders}"