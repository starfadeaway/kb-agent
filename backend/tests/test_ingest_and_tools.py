"""入库管线：页面过滤、清洗、分块、算术工具白名单。

页面过滤这组断言带一条**血泪教训**：
最初除了"按标题过滤模板页"，还加了一条"清洗后正文占比 < 5% 就跳过"的内容启发式。
拿 1035 页真实语料一跑，它会误杀 **60% 的页面** —— 因为这个 wiki 的内容页
本身就是模板驱动的，光锥页 95% 以上是 `{{光锥图鉴|…}}`。
按内容判等于把"模板展平"的价值又抹掉一次。所以只保留标题判据。
"""

from __future__ import annotations

import pytest
from app.agent.tools import REGISTRY, call_tool
from app.ingest.page_filter import is_template_page, should_index


@pytest.mark.parametrize(
    "title",
    [
        "模板:邦布图鉴",
        "Template:材料图鉴",
        "模板:遗器套装",
        "模块:Nav",
        "帮助:如何编辑",
        "分类:角色",
        "文件:立绘.png",
        "",
    ],
)
def test_template_pages_are_rejected(title):
    ok, why = should_index(title)
    assert not ok, f"{title!r} 应该被过滤"
    assert why, "跳过时必须给出原因，不做静默丢弃"


@pytest.mark.parametrize(
    "title",
    [
        "三月七",
        "苍雷紫电的记忆",
        "「电磁暴」-壹式",
        "太空封印站",
        "岩之印",
        "模板里的实际内容页",  # 标题以"模板"开头但不是"模板:"前缀
    ],
)
def test_content_pages_are_kept(title):
    ok, why = should_index(title)
    assert ok, f"{title!r} 不该被过滤（{why}）"


def test_template_prefix_needs_colon():
    """`模板里的实际内容页` 不是模板页 —— 判据是 `模板:` 前缀，不是"含模板"。"""
    assert not is_template_page("模板里的实际内容页")
    assert is_template_page("模板:邦布图鉴")


def test_wikitext_flattening_is_untouched_by_filter():
    """过滤只作用于"模板定义页"，不能误伤"内容页里的 infobox"。

    故意**不传 wikitext**：现在的判据只看标题。早期版本还会按内容占比过滤，
    拿 1035 页真实语料一跑误杀了 60% —— 因为内容页本身就是模板驱动的，
    光锥页 95% 以上是 `{{光锥图鉴|…}}`。这条测试锁住"只看标题"这个决定。
    """
    ok, _ = should_index("苍雷紫电的记忆")
    assert ok, "内容页即使 100% 是模板语法也必须保留 —— 它承载事实"


# --------------------------------------------------------------------------- #
# 算术工具：正则白名单 + AST 白名单
# --------------------------------------------------------------------------- #

# 传 None 上下文：calculator 不碰 ctx，这些用例不需要真的建检索器
NO_CTX = None


async def _calc(expression: str) -> tuple[str, bool]:
    return await call_tool("calculator", {"expression": expression}, NO_CTX, timeout=5)


@pytest.mark.parametrize(
    "bad",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "1 if True else 2",
        "x = 5",
    ],
)
async def test_calculator_rejects_code_execution(bad):
    """calculator 是白名单实现（正则 + AST），不是 eval。锁死绕过路径。"""
    out, ok = await _calc(bad)
    assert not ok, f"不该被执行：{bad} -> {out}"
    assert "[错误]" in out


async def test_calculator_computes_normal_arithmetic():
    out, ok = await _calc("(0.8 + 0.2) * 1000")
    assert ok, out
    assert "1000" in out


async def test_calculator_reports_division_by_zero():
    out, ok = await _calc("1 / 0")
    assert not ok
    assert "除零" in out or "[错误]" in out


async def test_calculator_rejects_empty_expression():
    out, ok = await _calc("")
    assert not ok
    assert "不能为空" in out


async def test_unknown_tool_returns_readable_error():
    out, ok = await call_tool("definitely_not_a_tool", {}, NO_CTX, timeout=5)
    assert not ok
    assert "[错误]" in out
    # 文案里要带上可用工具清单，否则模型没法自我修正
    assert "search_kb" in out


def test_all_registered_tools_have_valid_json_schema():
    """少一层 {"type": ...} 包装 schema 就废了，而废掉之后要到模型调用时才炸。"""
    from app.agent.tools import get_schemas

    schemas = get_schemas()
    assert schemas, "工具 schema 为空，@tool 装饰器没生效？"
    for s in schemas:
        assert s.get("type") == "function", f"{s.get('name')} 顶层 type 不对"
        fn = s.get("function", {})
        assert fn.get("name") and fn.get("description"), f"{s} 缺 name/description"
        params = fn.get("parameters", {})
        assert params.get("type") == "object", f"{fn.get('name')} parameters.type 不是 object"
        for pname, pschema in (params.get("properties") or {}).items():
            assert isinstance(pschema, dict) and "type" in pschema, (
                f"{fn.get('name')}.{pname} 缺 type：{pschema}"
            )


def test_search_kb_is_the_only_tool_enabled_by_default():
    """实验 C 结论：C1（只给 search_kb）比 C2（全工具）高 2.13 分且显著，
    但 token 多 22.6%。生产默认因此只挂 search_kb，
    别被"功能更全"说服改回去。其余工具仍在注册表里，需要时手动开。"""
    assert set(REGISTRY) == {"search_kb", "get_page", "calculator", "list_games"}


# --------------------------------------------------------------------------- #
# entity_type 清洗
# --------------------------------------------------------------------------- #
# 这组是原神 MySQL 回填崩溃的直接原因：ys 语料里有页面写成
#   {{多重系列任务<!-- 森林书等…可将模板名改为多重系列任务 -->|…}}
# 模板名后面拖着给人看的 HTML 注释。原样入库会：
#   1. 撑爆 facts.entity_type VARCHAR(32)，整条 INSERT 失败 → MySQL 1406，
#      表现为"原神整库回填崩掉"，而且崩在最后一步、前两个库已经写进去了；
#   2. 就算列够宽，存进去的也是没法当类型查的值，
#      而 facts 是给"精确匹配数值"用的，类型字段脏了等于表废了一半。

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # 真实语料里出现过的那个
        ("多重系列任务<!-- 森林书等嵌套系列任务可将模板名改为多重系列任务 -->", "多重系列任务"),
        ("角色<!--注释-->", "角色"),
        # 没管道参数的模板：正则只能成对吃掉 }}，会剩一个孤儿 }
        ("{{某某模板}}", "某某模板"),
        ("{{a|{{b}}}}", "ab"),
        ("[[链接|文字]]", "文字"),
        ("位面饰品", "位面饰品"),
        ("  光锥  ", "光锥"),
        ("", ""),
    ],
)
def test_clean_entity_type(raw, expected):
    from app.ingest.pipeline import _clean_entity_type

    assert _clean_entity_type(raw) == expected


def test_clean_entity_type_never_exceeds_column_width():
    """**任何输入都不能超出列宽** —— 超了就是整批 INSERT 失败，不是丢一条。

    列宽从 ORM 模型读，不在这里写死 32：改了模型这里会跟着变，
    抄一份的话两边迟早对不上。
    """
    from app.ingest.pipeline import _clean_entity_type, _fact_col_len

    limit = _fact_col_len("entity_type", 32)
    for raw in ("模板名<!-- " + "注释" * 200 + " -->", "角色" * 500, "{{" * 100 + "名}}"):
        assert len(_clean_entity_type(raw)) <= limit, f"超长：{raw[:40]!r}"


def test_fact_col_widths_come_from_the_model():
    """列宽必须真的取自 ORM，而不是藏在代码里的魔数。"""
    from app.ingest.pipeline import _fact_col_len

    assert _fact_col_len("entity_type", 999) == 32
    assert _fact_col_len("entity_name", 999) == 128
    assert _fact_col_len("attr", 999) == 64