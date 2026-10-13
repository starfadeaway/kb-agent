"""变体一致性核对：新语料下，实验 B / D 的单变量前提还成立吗。

为什么必须查这个
----------------
`EVALUATION.md` 里记过一次教训：重建 base 时忘了同步变体，
b3 停在 2293 而 base 已到 2306，**13 个 chunk 的差**就足以让
「只改 embed_mode」的对比变成双变量。

这次补抓角色页后 sr 从 2306 涨到 3932，四个变体全部重建过，
但「重建过」不等于「口径一致」—— 本脚本逐条验证。

三组要验的
----------
1. **b3 vs base**：chunk_size 与 chunk_overlap 相同、embed_mode 不同。
   chunk_id 集合必须**完全一致**；唯一差别是送进 embedding 的文本开头。
2. **b1 / b2 vs base**：chunk_size 不同，chunk_id 集合必须**不同**
   （否则说明分块参数没生效）。
3. **四个变体页面覆盖**：706 页必须一个不少 —— 语料少一页同样是变量。
"""
import sys

sys.path.insert(0, "backend")
sys.stdout.reconfigure(encoding="utf-8")

from app.core.config import get_settings
from app.store.vector_store import get_vector_store

s = get_settings()
store = get_vector_store(s)

VARIANTS = [
    ("base", s.embed_model_main, "chunk=500 / embed_mode=title（生产基线）"),
    ("b3", s.embed_model_main, "chunk=500 / embed_mode=text"),
    ("b1", s.embed_model_main, "chunk=800 / embed_mode=title"),
    ("b2", s.embed_model_main, "chunk=1200 / embed_mode=title"),
    ("d_Qwen_Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B",
     "Qwen embedding / chunk=500 / embed_mode=title"),
]

data = {}
print("=" * 78)
print("  sr 各变体 chunk 规模与页面覆盖")
print("=" * 78)
print(f"{'variant':<34}{'chunk':>8}{'页面':>7}   参数")
for name, model, desc in VARIANTS:
    col = store.get_collection("sr", model, name)
    d = col.get(include=["metadatas"])
    metas = d.get("metadatas") or []
    ids = set(d.get("ids") or [])
    pages = {(m or {}).get("page_title", "") for m in metas} - {""}
    data[name] = (ids, pages)
    print(f"{name:<34}{len(ids):>8}{len(pages):>7}   {desc}")

print("\n" + "=" * 78)
print("  单变量前提校验")
print("=" * 78)
base_ids, base_pages = data["base"]
ok = True

# 1. b3 vs base：chunk_id 必须完全一致
b3_ids, b3_pages = data["b3"]
same_ids = b3_ids == base_ids
same_pages = b3_pages == base_pages
print(f"1. b3 vs base  chunk_id 集合完全一致 : {same_ids} "
      f"(base {len(base_ids)} / b3 {len(b3_ids)})")
print(f"   b3 vs base  页面集合完全一致      : {same_pages}")
if not (same_ids and same_pages):
    ok = False
    print(f"   ✗ 仅 base 有 {len(base_ids - b3_ids)} 个 chunk，"
          f"仅 b3 有 {len(b3_ids - base_ids)} 个 → embed_mode 差异被 chunk 划分差异污染")

# 2. b1 / b2 vs base：chunk_id 必须不同（否则分块参数没生效）
for v in ("b1", "b2"):
    ids, pages = data[v]
    differs = ids != base_ids
    print(f"2. {v} vs base chunk_id 集合不同      : {differs} "
          f"({v} {len(ids)} vs base {len(base_ids)})")
    if not differs:
        ok = False
    if pages != base_pages:
        ok = False
        print(f"   ✗ {v} 页面集合与 base 不一致 → 语料覆盖不同，chunk_size 对比是双变量")

# 3. 页面覆盖必须一致
print("\n3. 页面覆盖（语料少一页同样是变量）")
for v, (_, pages) in data.items():
    same = pages == base_pages
    mark = "✔" if same else "✘"
    print(f"   {mark} {v:<34} {len(pages)} 页"
          + ("" if same else f"，与 base 差 {len(pages ^ base_pages)} 页"))
    if not same:
        ok = False

print("\n" + "=" * 78)
print(f"判定：{'全部一致，单变量前提成立' if ok else '存在不一致，相关实验需重跑'}")
print("=" * 78)
raise SystemExit(0 if ok else 1)