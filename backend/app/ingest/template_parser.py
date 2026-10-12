"""MediaWiki wikitext 解析。

biligame 的角色/光锥等页面把结构化数据放在 infobox 模板里,形如:

    {{角色
    |名称=刻晴
    |稀有度=5星
    |元素属性=雷
    |TAG=位移、能量回复…}}

这意味着**不需要依赖 Cargo 扩展**,直接解析模板的 key=value 即可拿到结构化字段,
比爬表格 HTML 稳定得多(字段定义写在模板里,页面改版也不影响)。

本模块负责两件事:
1. 从 wikitext 中切出模板并解析成 {name, named: dict, positional: list}
2. 把 wikitext 清洗成适合做向量化检索的纯文本(去模板、去引用、去文件链接)
"""

from __future__ import annotations

import re

__all__ = [
    "ParsedTemplate",
    "iter_templates",
    "find_infobox",
    "clean_wikitext",
]


class ParsedTemplate:
    """一个 MediaWiki 模板。"""

    __slots__ = ("name", "named", "positional")

    def __init__(self, name: str, named: dict[str, str], positional: list[str]) -> None:
        self.name = name
        self.named = named
        self.positional = positional

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"ParsedTemplate({self.name!r}, named={len(self.named)}, pos={len(self.positional)})"

    def get(self, key: str, default: str = "") -> str:
        return self.named.get(key, default).strip()


def _match_brace(text: str, start: int) -> int:
    """从 text[start] 处(应为 '{{')开始,返回配对 '{{...}}' 的结束下标(含)。

    需要正确处理嵌套:{{a|{{b}}}} 的右括号深度为 2。
    """
    depth = 0
    i = start
    n = len(text)
    while i < n - 1:
        two = text[i : i + 2]
        if two == "{{":
            depth += 1
            i += 2
            continue
        if two == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                return i
            continue
        i += 1
    return -1


def iter_templates(text: str):
    """产出 wikitext 中所有**最外层**模板的完整文本(含 {{ }})。"""
    i = 0
    n = len(text)
    while i < n - 1:
        if text[i : i + 2] == "{{":
            end = _match_brace(text, i)
            if end == -1:
                return
            yield text[i:end]
            i = end
            continue
        i += 1


def _split_top_level(body: str) -> list[str]:
    """按顶层管道符切分模板体,忽略嵌套 {{}} 与 [[]] 内部的管道符。"""
    parts: list[str] = []
    buf: list[str] = []
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(body)
    while i < n:
        two = body[i : i + 2]
        if two == "{{":
            depth_brace += 1
            buf.append(two)
            i += 2
            continue
        if two == "}}":
            depth_brace = max(0, depth_brace - 1)
            buf.append(two)
            i += 2
            continue
        if body[i : i + 2] == "[[":
            depth_brack += 1
            buf.append("[[")
            i += 2
            continue
        if body[i : i + 2] == "]]":
            depth_brack = max(0, depth_brack - 1)
            buf.append("]]")
            i += 2
            continue
        ch = body[i]
        if ch == "|" and depth_brace == 0 and depth_brack == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def parse_template(tpl: str) -> ParsedTemplate:
    """把 '{{名称|键=值|值}}' 解析成 ParsedTemplate。"""
    inner = tpl[2:-2] if tpl.startswith("{{") and tpl.endswith("}}") else tpl
    segments = _split_top_level(inner)
    if not segments:
        return ParsedTemplate("", {}, [])

    name = segments[0].strip()
    named: dict[str, str] = {}
    positional: list[str] = []

    for seg in segments[1:]:
        s = seg.strip()
        if not s:
            continue
        # 只有形如 k=v 且 v 里没有顶层管道符的才算具名参数
        m = re.match(r"^([^=|{}\[\]]+?)\s*=\s*(.*)$", s, re.DOTALL)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            if key:
                named[key] = val
        else:
            positional.append(s)

    return ParsedTemplate(name, named, positional)


# 这些是导航/面包屑/装饰性模板,不是数据 infobox
_SKIP_NAMES = {
    "面包屑",
    "角色导航",
    "导航模板",
    " infobox",
    "main",
    "注意",
    "黑体",
    "color",
    "gray",
    "kbd",
    "sic",
    "!",
}


def find_infobox(wikitext: str, skip: set[str] | None = None) -> ParsedTemplate | None:
    """找出页面的数据 infobox。

    策略:取第一个「有 ≥3 个具名参数且参数值非空」的模板。
    导航类模板通常参数少或值为空,因此能被这个规则自然排除。
    """
    skip = _SKIP_NAMES if skip is None else skip
    fallback: ParsedTemplate | None = None

    for raw in iter_templates(wikitext):
        tpl = parse_template(raw)
        if tpl.name.strip().lower() in {s.lower() for s in skip}:
            continue
        filled = [v for v in tpl.named.values() if v.strip() not in ("", "—", "-")]
        if len(filled) >= 3:
            return tpl
        if fallback is None and len(filled) >= 1:
            fallback = tpl

    return fallback


# --------------------------------------------------------------------------- #
# 清洗:wikitext -> 可检索纯文本
# --------------------------------------------------------------------------- #

_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_FILE_LINK = re.compile(r"\[\[\s*(?:File|Image|文件|图像)\s*:[^\]]*\]\]", re.IGNORECASE)
_CATEGORY = re.compile(r"\[\[\s*(?:Category|分类)\s*:[^\]]*\]\]", re.IGNORECASE)
_EXT_LINK = re.compile(r"\[https?://\S+\s+([^\]]*)\]")
_URL_BARE = re.compile(r"\[https?://\S+\]")
_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")

# 表格语法 → 空格
_TABLE_SPLIT = re.compile(r"\n\|")
_CELL = re.compile(r"\|\||\|\||!!")


# 模板里的图片类字段对问答没有价值，还会稀释向量
_NOISE_KEYS = {
    "图片", "图像", "图标", "头像", "缩略图", "配图", "立绘", "背景图",
    "image", "img", "file", "icon", "images", "图片尺寸", "文件",
}


# 纯格式包装类模板：{{颜色|描述2|12%}}、{{Color|描述2|12%}}
# 这类模板的第一个位置参数是"标签"、最后一个才是实际文本。
# 按普通模板展平会把标签一起吐出来，值变成"描述2；12%"，读起来像脏数据。
_WRAPPER_TEMPLATES = {
    "颜色", "color", "colored", "textcolor", "fontcolor",
    "字号", "font", "size", "高亮", "hl", "粗体", "bold", "斜体", "italic",
    "消歧义", "disambig", "lang", "langx", "注音", "ruby",
}


def _template_to_text(raw: str, depth: int = 0) -> str:
    """把一个模板展平成可读文本，而不是整条丢掉。

    这是踩过的最大的一个坑：星铁 wiki 的光锥页 / 角色页几乎全部内容都在
    `{{光锥图鉴}}`、`{{角色}}` 这类数据模板里。早期实现是"删掉所有模板"，
    结果 969 字的 wikitext 清洗后只剩 20 字 —— 页面看着抓到了，其实什么都没入库。
    改成展平成「字段：值」后，同一批 40 页从 9 chunk 变成 78 chunk。

    depth 是递归深度保护：模板可以互相嵌套（参数值里还有模板），
    但超过 6 层基本就是 wiki 模板写错了，继续递归会栈溢出。
    """
    if depth > 6:
        return " "
    tpl = parse_template(raw)
    name = tpl.name.strip()
    if name.lower() in {s.lower() for s in _SKIP_NAMES}:
        return " "

    # 格式包装模板只取最后一个位置参数（真正的文本），丢掉标签。
    # 不做这一步的话「两件套效果」的值会变成「描述2；12%」这种读起来像脏数据的东西，
    # 而且会污染 MySQL 里的 facts 表和向量文本。
    if name.lower() in _WRAPPER_TEMPLATES and tpl.positional:
        tail = _flatten_value(str(tpl.positional[-1]), depth)
        return tail or " "

    parts: list[str] = []
    for key, val in tpl.named.items():
        v = _flatten_value(str(val), depth)
        if not v or key.strip().lower() in _NOISE_KEYS:
            continue
        parts.append(f"{key.strip()}：{v}" if key.strip() else v)
    for val in tpl.positional:
        v = _flatten_value(str(val), depth)
        if v:
            parts.append(v)

    if not parts:
        return " "
    return "\n" + "；".join(parts) + "\n"


def _flatten_value(val: str, depth: int) -> str:
    """参数值里常还嵌着模板与链接，递归展平一次。

    内部链接 `[[目标|显示文字]]` 取**显示文字**而不是目标：
    用户问「太空封印站」时，链接文本通常是正常的中文名，
    而链接目标可能是带命名空间的 `遗器套装#太空封印站`，反而稀释了检索信号。
    """
    v = val
    if "{{" in v:
        v = _flatten_templates(v, depth + 1)
    v = re.sub(r"\[\[([^\[\]|]*)\|([^\[\]]*)\]\]", r"\2", v)
    v = re.sub(r"\[\[([^\[\]]*)\]\]", r"\1", v)
    v = re.sub(r"'''?", "", v)
    v = _HTML_TAG.sub(" ", v)
    return v.strip(" |\n\t")


def _flatten_templates(text: str, depth: int = 0) -> str:
    """扫描全文，把最外层模板逐个替换为展平后的文本。"""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        start = text.find("{{", i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        end = _match_brace(text, start)
        if end == -1:
            # 落单的 "{{"，原样保留并前进，避免死循环
            out.append(text[start : start + 2])
            i = start + 2
            continue
        out.append(_template_to_text(text[start:end], depth))
        i = end
    return "".join(out)


def clean_wikitext(wikitext: str) -> str:
    """把 wikitext 清洗成适合向量化的中文纯文本。

    处理顺序很关键:先摘掉嵌套最深的引用与注释,再逐层剥掉模板/链接,
    否则外层替换会破坏内层结构,产生残留噪声。
    """
    if not wikitext:
        return ""

    text = wikitext

    # 1. 注释与引用(可能嵌套模板,必须最先处理)
    prev = None
    while prev != text:
        prev = text
        text = _COMMENT.sub(" ", text)
        text = _REF.sub(" ", text)

    # 2. 文件/分类链接整条丢弃
    text = _FILE_LINK.sub(" ", text)
    text = _CATEGORY.sub(" ", text)

    # 3. 外链 → 保留锚文本
    text = _EXT_LINK.sub(r"\1", text)
    text = _URL_BARE.sub(" ", text)

    # 4. 内部链接 [[A|B]] → B,[[A]] → A
    text = re.sub(r"\[\[([^\[\]|]*)\|([^\[\]]*)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\[\]]*)\]\]", r"\1", text)

    # 5. 模板：展平成「字段：值」，不丢弃内容
    text = _flatten_templates(text)

    # 6. 表格行/单元格
    text = _TABLE_SPLIT.sub("\n", text)
    text = _CELL.sub(" ", text)
    text = re.sub(r"^\s*[-|!]+\s*$", "", text, flags=re.MULTILINE)

    # 7. 强调语法与 HTML
    text = re.sub(r"'''?", "", text)
    text = _HTML_TAG.sub(" ", text)

    # 8. 外部链接符号与残余括号
    text = re.sub(r"\[|\]", " ", text)

    # 9. 空白规整
    text = text.replace("&nbsp;", " ")
    text = _WS.sub(" ", text)
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    text = _NL.sub("\n\n", text)

    return text.strip()
