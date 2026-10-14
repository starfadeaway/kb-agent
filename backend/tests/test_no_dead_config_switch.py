"""守卫：`VECTOR_ANN` 这个假开关不能回来。

## 为什么要有这条

它曾经是一个**看起来能开、实际什么都不会发生**的配置项：
被读进 `Settings` → 传进 `LanceCollection` → 写进侧车 meta → 再读回来，
但**查询路径从不读它，全模块也没有任何 `create_index` 调用**。
配成 `true` 既不会变快也不会建近似索引。

它是本项目第二个同类问题（第一个是已删除的 `TOOL_TRANSPORT`），
所以守卫断言的不是「配置值等于某个状态」，而是**这个配置项根本不存在**——
因为不存在才是这里唯一诚实的状态：后端确实只有精确扫描一条路径。

## 双向验证说明

本模块是**新增守卫**，且被测行为是「某物不存在」，没有「修复前的代码快照」
可注入，所以**双向验证不适用**。替代做法：

1. 断言本身对「重新引入」的形态必然失败（下方三个用例覆盖了
   配置字段、构造签名、侧车内容三条可能的回归路径）；
2. 真实环境实测：`LanceCollection` 的实际查询路径不含任何 ANN 分支，
   由 `app/store/lance_store.py` 模块 docstring 里的 IVF-PQ 实测记录
   给出依据（同一句查询，暴力扫描与 IVF-PQ 的 top-3 第三名不同）。
"""
from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

from app.core.config import Settings
from app.store.lance_store import LanceCollection

BACKEND = Path(__file__).resolve().parent.parent
# `.env.example` 在**仓库根**，不在 backend/ 下。
# 早先这里写成 BACKEND / ".env.example"，文件找不到 → pytest.skip →
# 那条断言压根没跑，是典型的「跳过即通过」假绿。路径写对之后不许再 skip。
ENV_EXAMPLE = BACKEND.parent / ".env.example"


def test_env_example_exists():
    """先把路径钉死：上一条用例一旦 skip，这条会先失败并指明真正原因。"""
    assert ENV_EXAMPLE.is_file(), f"找不到 .env.example（找的是 {ENV_EXAMPLE}）"


def test_env_example_does_not_advertise_vector_ann():
    """`.env.example` 是新人照着抄的模板，模板里不能再出现这个键。

    用示例文件而不是 `.env`：`.env` 不入库，本机状态推不出仓库约束；
    而 `.env.example` 进库，是「这个键对本项目有效」的唯一公开声明。
    """
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "VECTOR_ANN" not in text, ".env.example 里仍在宣传 VECTOR_ANN"


def test_settings_has_no_vector_ann_field():
    """配置项必须从 Settings 上彻底消失。

    只删 .env 不够 —— 字段留在 dataclass 上，`getattr(settings, "vector_ann", ...)`
    这种带默认值的读法会永远读到兜底值，看起来一切正常。
    """
    names = {f.name for f in dataclasses.fields(Settings)}
    assert "vector_ann" not in names, "vector_ann 字段回到了 Settings 上"


def test_lance_collection_signature_has_no_ann_param():
    """构造签名里不能有 `ann`。

    这是第二条回归路径：即使 Settings 删了，只要 `LanceCollection(ann=...)`
    还在，调用方仍会传一个被忽略的参数 —— 那正是原来那个死参数的形态。
    """
    params = inspect.signature(LanceCollection.__init__).parameters
    assert "ann" not in params, f"LanceCollection.__init__ 仍有 ann 参数：{list(params)}"


def test_sidecar_meta_has_no_ann_key(tmp_path):
    """侧车 meta 里只能有 `meta`，不能回写 `ann`。

    这条是**行为断言**而不是配置断言：真的建一张表、真的读侧车文件，
    检查落盘内容。老 collection 的侧车里可能残留 `ann` 键，
    但 `open()` 只取 `blob.get("meta")`，残留键不参与任何行为，因此无需迁移。
    """
    coll = LanceCollection(tmp_path, "probe", {"game": "sr"})
    coll._save_meta()
    blob = json.loads((tmp_path / "probe.meta.json").read_text(encoding="utf-8"))
    assert "ann" not in blob, f"侧车里仍在写 ann：{sorted(blob)}"
    assert blob["meta"] == {"game": "sr"}
