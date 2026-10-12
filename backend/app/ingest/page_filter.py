"""判断一个 wiki 页面值不值得进索引。

**为什么要有这个模块**：抓分类列表时会把 MediaWiki 的**模板定义页**一起带回来
（`模板:邦布图鉴`、`Template:材料图鉴`…）。这些页面的 wikitext 几乎全是
`<includeonly>{{#set: |字段={{{字段|}}}}}…` 这种模板语法，不含任何游戏事实。
它们会占 embedding 成本、在"查邦布"这类 query 上挤召回位，指标上不会有任何收益。

**与"模板展平"是两回事，别混淆**（`template_parser.clean_wikitext`）：
- 展平处理的是**内容页内部**的 `{{光锥图鉴|名称=…}}` infobox —— 那是事实的载体，
  删了等于把整页洗没（实测 969 字 → 20 字），必须展平成 `字段：值`。
- 这里过滤的是**模板自身的定义页** —— 里面只有语法，没有事实，删掉零损失。

**只按标题判，不按内容判**（这是被真实语料打回来之后的设计）：
最初还加了一条"清洗后正文占比 < 5% 就跳过"的内容启发式，拿 1035 页真实语料一跑，
它会误杀 **60% 的页面** —— 因为这个 wiki 的内容页本身就是模板驱动的，
光锥页 95% 以上都是 `{{光锥图鉴|…}}`（README 里记着这条：展平后 9 chunk → 78 chunk）。
按内容判等于把展平的价值又抹掉一次。标题判据则是可证的、可预期的。
详见 `scripts/check_page_filter.py`（用真实语料回归这个取舍）。
"""
from __future__ import annotations

# 标题前缀。MediaWiki 本地化前缀会变，中英文都带上。
_TEMPLATE_TITLE_PREFIXES = (
    "模板:",
    "模板：",
    "Template:",
    "模板/",
    "模块:",
    "Module:",
    "帮助:",
    "Help:",
    "分类:",
    "Category:",
    "专题:",
    "Topic:",
    "文件:",
    "File:",
    "档案:",
    "Image:",
    "媒体:",
    "Media:",
)


def is_template_page(title: str) -> bool:
    """标题表明这是模板/模块/帮助/分类/图片等定义页，不是内容页。"""
    t = (title or "").strip()
    if not t:
        return True
    return t.startswith(_TEMPLATE_TITLE_PREFIXES)


def should_index(title: str) -> tuple[bool, str]:
    """返回 `(是否入库, 原因)`。

    调用方把被跳过的标题记进 IngestStats，让"为什么这批页没进去"在报告里可见 ——
    静默丢弃比丢弃本身更糟。
    """
    t = (title or "").strip()
    if not t:
        return False, "空标题"
    if is_template_page(t):
        return False, "模板/模块/帮助/分类/图片定义页"
    return True, ""