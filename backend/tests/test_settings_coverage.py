"""**每个配置项都必须真的被读到** —— 跨层断言。

背景（同一类问题，本项目已经栽了三次）：
1. `LanceCollection` 的 sidecar 元数据写了不读回 —— 侧栏语料量虚高 4.4 倍。
2. `RetrievalConfig.as_dict()` 是手写字典，加了 `min_revision_ts` /
   `page_titles` 忘了同步 —— 带版本过滤的评测落库后看不出带过滤。
3. `RETRIEVE_TOP_K` / `RETRIEVE_CANDIDATE` / `RRF_K` 三个 env var
   在 `config.py` 里定义得整整齐齐，`.env.example` 也列了，
   但 `main.py` 构造 `RetrievalConfig` 时压根没传 ——
   **改 .env 完全不生效，且不报任何错**。

共同点：都是「声明了 ≠ 生效了」，而且**不会崩**。这类缺陷靠人读代码是查不完的，
因为一个 config 字段有没有读取点，取决于散落在十几个文件里的用法。

所以把它变成可执行断言：
- 每个 `Settings` 字段，在全仓 `.py` 里至少有一处**读取**（`x.<field>` 或
  `getattr(..., "<field>"`）。
- 每个 env var，在 `.env.example` 里必须有对应条目。

为什么用「文本扫描」而不是 `Settings` 的属性访问追踪：
真要追踪得给每个调用点插桩，成本远高于收益，而且会漏掉 `getattr` 这种动态读法。
这里要抓的是**更粗的一层**——「这个配置项全仓到底有没有人碰」。
能抓到上面第 3 条（那三个字段一次都没被读），已经值回票价。
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
from app.core.config import Settings

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

#: 全仓 Python 源码（含 scripts/ 与 tests/）。
#: **必须包含 tests/**：有些字段只被测试读取（比如 chroma_path 只在
#: conftest 里被 dataclasses.replace 构造过），排除掉会误报。
_PY_FILES = sorted(
    p for d in ("app", "scripts", "tests")
    for p in (BACKEND_DIR / d).rglob("*.py")
)

#: 显式豁免：这些字段只作为"写入目标"存在，或用法是纯声明。
#: 每个都要写清楚理由 —— 空豁免表才说明这张网真的有用。
_ALLOWLIST: dict[str, str] = {}


def _settings_fields() -> list[str]:
    return [f.name for f in dataclasses.fields(Settings)
            if f.name not in Settings._SECRET_FIELDS]


def _read_patterns(field: str) -> list[re.Pattern[str]]:
    """字段在全仓被"碰过"的几种写法。

    **这里刻意用宽松匹配，代价与边界写在下面 —— 别把它当精确证明用。**

    曾经试过限定接收者必须是 Settings 实例（`s.` / `settings.` / `self.s.`），
    想抓得更准，结果两头发火：
    - 假阴性：`RetrievalConfig.rrf_k` 也叫这名字，检索算法里天天写 `cfg.rrf_k`，
      于是 `Settings.rrf_k` 明明没人读，照样放行；
    - 假阳性：`(settings or get_settings()).vector_backend` 这种嵌套表达式匹配不上。

    靠文本猜「谁读的是 Settings 实例」本身就不稳，所以这条只当**粗网**：
    抓的是「这个配置项全仓一次都没被提到」这种最粗的漏配。

    真正要紧的那几个旋钮（RRF_K / RETRIEVE_TOP_K / RETRIEVE_CANDIDATE）
    由下面 `test_retrieval_config_from_settings_reflects_env` 做**语义化**验证 ——
    那是可证明的，这条只是网。
    """
    return [
        re.compile(rf"\.\s*{field}\b"),
        re.compile(rf'getattr\([^,]+,\s*["\']{field}["\']'),
        re.compile(rf"\b{field}\s*="),          # 构造/替换时的赋值目标
    ]


@pytest.mark.parametrize("field", _settings_fields())
def test_every_settings_field_has_a_read_site(field: str):
    """这个字段在全仓有没有被任何地方碰过。"""
    if field in _ALLOWLIST:
        pytest.skip(f"已豁免：{_ALLOWLIST[field]}")

    pats = _read_patterns(field)
    hits = []
    for p in _PY_FILES:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # 跳过 config.py 自身：字段就是在那里声明的，
        # 算进去等于「声明即算用过」，这道断言就废了。
        if p.name == "config.py" and p.parent.name == "core":
            continue
        if any(rx.search(text) for rx in pats):
            hits.append(p.relative_to(PROJECT_ROOT).as_posix())

    assert hits, (
        f"Settings.{field} 在 config.py 里定义了、.env.example 里也列了，"
        f"但全仓没有任何地方读到它 —— 改 .env 完全不生效，且不会报错。"
        f"\n修法二选一："
        f"\n  1. 在消费点真正使用它（优先：说明它本来就该生效）"
        f"\n  2. 确认它确实没用了就删掉字段、删 .env.example 条目，"
        f"并在本文件 _ALLOWLIST 里登记理由"
    )


def test_settings_fields_have_no_blank_allowlist_entries():
    """豁免表不许有空理由 —— 空理由等于给未来的坏味道开后门。"""
    empty = [k for k, v in _ALLOWLIST.items() if not v.strip()]
    assert not empty, f"这些豁免没有写理由：{empty}"


def test_retrieval_config_from_settings_reflects_env():
    """**语义化验证**：`from_settings()` 必须真的把 .env 的值带进检索配置。

    这是上面那张粗网真正要保护的东西。粗网只能证明"有人碰过这个名字"，
    这条能证明"改了 .env，检索层拿到的就是新值"。

    `RETRIEVE_TOP_K` / `RETRIEVE_CANDIDATE` / `RRF_K` 曾经三个都是死的：
    `config.py` 定义得整整齐齐，`.env.example` 也列了，
    但 `main.py` 构造 `RetrievalConfig(...)` 时压根没传这三个参数 ——
    **改 .env 完全不生效，检索照常跑、不报任何错。**
    做单变量消融的人会在这里白跑一整轮才发现参数没变。
    """
    import dataclasses as _dc

    from app.core.config import get_settings
    from app.retrieval.hybrid import RetrievalConfig

    tweaked = _dc.replace(
        get_settings(),
        rrf_k=137,
        retrieve_top_k=9,
        retrieve_candidate=33,
        embed_model_main="some/other-model",
    )
    cfg = RetrievalConfig.from_settings(settings=tweaked)
    assert cfg.rrf_k == 137, f"RRF_K 没生效：{cfg.rrf_k}"
    assert cfg.final_top_k == 9, f"RETRIEVE_TOP_K 没生效：{cfg.final_top_k}"
    assert cfg.embed_model == "some/other-model", f"EMBED_MODEL_MAIN 没生效：{cfg.embed_model}"
    for name in ("bm25_top", "vector_top", "rerank_candidates"):
        assert getattr(cfg, name) == 33, (
            f"RETRIEVE_CANDIDATE 没带进 {name}：{getattr(cfg, name)}"
        )


def test_retrieval_config_overrides_beat_env():
    """显式传入的实验参数必须压过 .env —— 否则调参调不动。"""
    import dataclasses as _dc

    from app.core.config import get_settings
    from app.retrieval.hybrid import RetrievalConfig

    tweaked = _dc.replace(get_settings(), rrf_k=137, retrieve_top_k=9)
    cfg = RetrievalConfig.from_settings(settings=tweaked, rrf_k=7, final_top_k=3)
    assert cfg.rrf_k == 7, f"显式 rrf_k 没压过 .env：{cfg.rrf_k}"
    assert cfg.final_top_k == 3, f"显式 final_top_k 没压过 .env：{cfg.final_top_k}"


@pytest.mark.parametrize("field", _settings_fields())
def test_env_example_covers_field(field: str):
    """`.env.example` 必须列出这个字段对应的 env var。

    这是既有脚本 `check_env_example.py` 的 pytest 版本，
    好处是 CI 不用单独跑那个脚本。
    """
    env_example = PROJECT_ROOT / ".env.example"
    if not env_example.is_file():
        pytest.skip("没有 .env.example")
    text = env_example.read_text(encoding="utf-8")
    key = field.upper()
    assert key in text, f".env.example 里没有 {key}（Settings.{field} 对应的 env var）"