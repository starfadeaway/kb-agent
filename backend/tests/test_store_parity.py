"""两个 store 后端的**语义一致性**测试。

为什么这组测试比看起来重要：项目全部量化结论都建立在"单变量消融"上 ——
换检索策略、换 embedding、换分块参数，每次只动一个变量。
但 `VECTOR_BACKEND` 是可切换的（numpy / lancedb），如果两个实现在
**同一个调用下表现不同**，那么"A3 比 A0 好"这种结论换个后端就可能不成立，
而且**不会报任何错**。

这不是理论风险。已确认的分叉：
- `upsert`：NumPy 端对已存在 id 只覆盖 embedding，正文/元数据走 `fresh` 分支
  永不更新；Lance 端先 delete 再 add，整体刷新。
- `drop`：NumPy 端集合不存在返回 False；Lance 端返回 True。
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import numpy as np
from app.core import config
from app.store.vector_store import get_vector_store


def _write(store, ids, texts, metas, dim: int = 4):
    """写入向量，**必须先 L2 归一化**。

    Lance 端用 `cos = 1 - d/2` 从平方 L2 距离还原余弦，这个换算**只在单位向量上成立**。
    生产链路 `llm.embed(..., normalize=True)` 保证归一化；测试若随手塞未归一化的随机向量，
    两端的排序会天然不一致 —— 那是测试造错了，不是产品有 bug。
    """
    rng = np.random.default_rng(0)
    vectors = rng.random((len(ids), dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store.upsert(game="sr", model=None, variant="base", ids=ids,
                 embeddings=vectors.tolist(), documents=texts, metadatas=metas)


def test_both_backends_expose_the_same_methods(stores_probe, chunk_meta):
    """接口对齐：Lance 侧多出的方法要么提到基类，要么在类型上承认它是默认后端。"""
    np_store, lance = stores_probe
    required = {"get_collection", "upsert", "count", "drop", "list_collections",
                "build_where", "query", "get_by_ids"}
    missing = {name for name in required
               if not (hasattr(np_store, name) and hasattr(lance, name))}
    assert not missing, f"NumPy/Lance 接口不一致，缺：{sorted(missing)}"


def test_upsert_updates_text_and_metadata_on_both_backends(stores_probe, chunk_meta):
    """**已存在 id 再次 upsert 时，正文与元数据必须一起刷新。**

    这是最关键的一条：改分块参数 / 改 embed_mode 后重跑入库，
    同一批 chunk_id 的正文会变。如果 NumPy 端保留旧正文，
    那"重跑入库"在 numpy 后端下等于只换了向量、没换内容 ——
    实验 B 那种"换 chunk_size 对比"会直接失效，而且不报错。
    """
    np_store, lance = stores_probe
    _write(np_store, ["c1"], ["旧正文"], [chunk_meta("旧页")])
    _write(lance, ["c1"], ["旧正文"], [chunk_meta("旧页")])

    _write(np_store, ["c1"], ["新正文"], [chunk_meta("新页")])
    _write(lance, ["c1"], ["新正文"], [chunk_meta("新页")])

    for name, store in (("numpy", np_store), ("lance", lance)):
        got = store.get_by_ids(game="sr", model=None, variant="base", ids=["c1"])
        assert got["c1"].text == "新正文", (
            f"{name} 后端：已存在 chunk 再入库后正文仍是旧值 {got['c1'].text!r}。"
            f"重跑入库只换了向量没换内容，会让分块/embed_mode 实验失效"
        )
        assert got["c1"].metadata.get("page_title") == "新页", (
            f"{name} 后端：已存在 chunk 再入库后元数据没刷新 "
            f"-> {got['c1'].metadata.get('page_title')!r}"
        )


def test_count_is_consistent(stores_probe, chunk_meta):
    np_store, lance = stores_probe
    _write(np_store, ["a", "b"], ["x", "y"], [chunk_meta("A"), chunk_meta("B")])
    _write(lance, ["a", "b"], ["x", "y"], [chunk_meta("A"), chunk_meta("B")])
    assert np_store.count("sr", None, "base") == 2
    assert lance.count("sr", None, "base") == 2


def test_upsert_appends_new_ids_without_losing_old(stores_probe, chunk_meta):
    np_store, lance = stores_probe
    _write(np_store, ["a"], ["xa"], [chunk_meta("A")])
    _write(np_store, ["b"], ["xb"], [chunk_meta("B")])
    _write(lance, ["a"], ["xa"], [chunk_meta("A")])
    _write(lance, ["b"], ["xb"], [chunk_meta("B")])
    for name, store in (("numpy", np_store), ("lance", lance)):
        assert store.count("sr", None, "base") == 2, f"{name} 追加后条数不对"
        got = store.get_by_ids(game="sr", model=None, variant="base", ids=["a", "b"])
        assert got["a"].text == "xa" and got["b"].text == "xb", f"{name} 追加时覆盖了旧记录"


def test_list_collections_exposes_same_metadata_keys(stores_probe, chunk_meta):
    """`list_collections` 的元数据是前端算"生产语料量"的唯一依据 ——
    之前就是这里读了不回来，前端退回正则解析，把实验变体也算进去，虚高 4.4 倍。"""
    np_store, lance = stores_probe
    for store in (np_store, lance):
        _write(store, ["a"], ["xa"], [chunk_meta("A")])
    for name, store in (("numpy", np_store), ("lance", lance)):
        cols = store.list_collections()
        assert cols, f"{name} 没有列出 collection"
        for c in cols:
            assert c.get("game") and c.get("embed_model") and c.get("variant"), (
                f"{name} 的 list_collections 缺元数据字段：{c}"
            )


def test_build_where_produces_equivalent_filters(stores_probe, chunk_meta):
    """版本感知检索：两端必须能理解同一套过滤字典。"""
    np_store, lance = stores_probe
    a = np_store.build_where(game="sr", page_titles=["X"], min_revision_ts="2025-01-01T00:00:00Z")
    b = lance.build_where(game="sr", page_titles=["X"], min_revision_ts="2025-01-01T00:00:00Z")
    assert a == b, f"两端 build_where 结果不同：\n  numpy={a}\n  lance={b}"


def test_collection_name_slugs_match(stores_probe, chunk_meta):
    """模型名 → collection 名的 slug 规则必须一致，否则同名模型在两个后端
    落到不同 collection，看起来像"索引丢了"。"""
    np_store, lance = stores_probe
    np_name = np_store.collection_name("sr", "BAAI/bge-m3", "base")
    lance_name = lance.collection_name("sr", "BAAI/bge-m3", "base")
    assert np_name == lance_name, f"collection 名不一致：{np_name} vs {lance_name}"


def test_search_returns_same_ranking_for_same_query(stores_probe, chunk_meta):
    """同一条 query 在两个后端上必须给出相同的 top-k chunk_id。
    这是"换后端不影响指标"的前提。"""
    np_store, lance = stores_probe
    texts = {f"c{i}": f"内容{i}" for i in range(6)}
    _write(np_store, list(texts), list(texts.values()),
           [chunk_meta(t) for t in texts.values()])
    _write(lance, list(texts), list(texts.values()),
           [chunk_meta(t) for t in texts.values()])

    rng = np.random.default_rng(7)
    q = rng.random(4, dtype=np.float32)
    q /= np.linalg.norm(q)
    q = q.tolist()
    n_ids = [h.chunk_id for h in np_store.query(query_embedding=q, game="sr", top_k=3)]
    l_ids = [h.chunk_id for h in lance.query(query_embedding=q, game="sr", top_k=3)]
    assert n_ids == l_ids, f"两个后端 top-3 排序不同：\n  numpy={n_ids}\n  lance={l_ids}"




# --------------------------------------------------------------------------
# 注释不得宣称不存在的实现
# --------------------------------------------------------------------------
# 这条不是风格洁癖。`get_vector_store` 的 docstring 与 config.py 的注释
# 都曾写着「sqlite（最小演示后端）」，把后端写成三个 ——
# 而 `sqlite3` / `SQLiteStore` 在 `app/` 下**从来没有任何实现**。
#
# 一句注释里的虚构实现比不写更糟：面试官追问「第三个后端怎么实现的」
# 会当场露底，而且它恰好出现在要写进简历的这个项目里。
#
# 双向验证：对修复前的代码快照实跑本文件，这组断言必须失败。

# 后端名 -> 该后端在源码中必须能找到的实现证据
_BACKEND_EVIDENCE = {
    "lancedb": ("app.store.lance_store", "LanceVectorStore"),
    "numpy": ("app.store.vector_store", "VectorStore"),
}
# 归一化与存储无关，不算后端声明
_NOT_BACKENDS = {"vector", "store", "backend", "ann", "numpy", "default",
                 "env", "field", "factory", "dict", "str", "int", "bool"}

# 后端声明有两种写法都要认：
#   - lancedb（默认）：真实向量库…
#   - numpy：零依赖兜底…
# 半角 '(' 不能当说明括号 —— 否则 `field(default_factory=…)` 会被误认成后端。
_BACKEND_DECL = re.compile(r"-\s*([a-z][a-z0-9_]*)\s*[（:：]")
_BACKEND_DECL_PIPE = re.compile(r"(?:^|[|｜])\s*([a-z][a-z0-9_]*)\s*[（:：]")


def _declared_backends(doc: str) -> set[str]:
    """从 docstring 里抠出后端声明。"""
    names = {m.group(1) for m in _BACKEND_DECL.finditer(doc)}
    names |= {m.group(1) for m in _BACKEND_DECL_PIPE.finditer(doc)}
    return names


def test_no_phantom_backend_declared():
    """docstring 里提到的每个后端都必须真的有可 import 的实现。"""
    doc = inspect.getdoc(get_vector_store) or ""
    for name in _declared_backends(doc):
        if name in _NOT_BACKENDS:
            continue
        assert name in _BACKEND_EVIDENCE, (
            f"get_vector_store 的 docstring 宣称了 {name!r} 后端，"
            f"但 _BACKEND_EVIDENCE 里没有它 —— 虚构的实现比不宣传更糟"
        )
    for name, (mod_name, attr) in _BACKEND_EVIDENCE.items():
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, attr), \
            f"{name} 后端声称的实现 {mod_name}.{attr} 不存在"


def test_config_comment_matches_real_backends():
    """config.py 里 vector_backend 上方的注释不能提未实现的后端。"""
    cfg_src = Path(config.__file__).read_text(encoding="utf-8")
    lines = cfg_src.splitlines()
    idx = next(i for i, ln in enumerate(lines)
               if "vector_backend" in ln and "field(" in ln)
    comment = "\n".join(lines[max(0, idx - 8):idx])
    for name in _declared_backends(comment):
        if name in _NOT_BACKENDS:
            continue
        assert name in _BACKEND_EVIDENCE, \
            f"config.py 注释提到后端 {name!r}，但它没有实现"


def test_phantom_backend_is_named_in_doc_as_history():
    """这段纠正说明必须留着：否则下一个人会把 sqlite 再当成真后端加回来。"""
    doc = (inspect.getdoc(get_vector_store) or "").lower()
    assert "sqlite" in doc, "删掉了 sqlite 的纠正说明 —— 它会被当成真后端再加回来"
