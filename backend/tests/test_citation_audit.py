"""C 类校验（生成后引用校验）的守卫测试。

这一批全部测**纯函数** —— 不碰网络、不碰 LLM、不碰索引。
理由和 `test_react_*` 那一批一样：能离线验的就不该依赖外部服务，
否则「本地全绿、CI 恒红」和反过来都会发生。

双向验证见本文件末尾的 `test_双向验证_*` 两条。
"""
from __future__ import annotations

import pytest
from app.agent.citation_audit import (
    CitationAudit,
    audit_answer,
    harvest_retrieved_pages,
    is_supported,
    looks_like_page_title,
    normalize_title,
    parse_citations,
)

# 一段**格式与 tools.py 完全一致**的 search_kb 输出。
#
# 「完全一致」曾经是句空话：夹具漏了命中头尾部的 ` | 页面修订 {rev}`
# （tools.py:258 在有修订号时一定会拼上），而当时没有任何测试把解析器
# 和真实工具输出对起来 —— 格式一旦漂移，harvest 会静默返回空 dict、
# checked 变成 False、校验器整体停摆，而这套测试**依然全绿**。
# 所以这里逐字照抄 tools.py:256-261 的拼接结果，含修订后缀。
SEARCH_KB_OUTPUT = """status=ok hits=3 top_score=0.9984
[1] 分数 0.9984 | 太空封印站 > 正文 | 页面修订 2025-04-13
太空封印站（正文）
名称：太空封印站；类别：位面饰品；两件套效果：使装备者的攻击力提高12%。
当装备者的速度大于等于120时，攻击力额外提高12%。
出处：https://wiki.biligame.com/sr/太空封印站 | 修订 2025-04-13
[2] 分数 0.9291 | 卡芙卡/攻略 > 配装推荐 > 深入详解 > 遗器 | 页面修订 2026-02-18
卡芙卡/攻略（配装推荐 > 深入详解 > 遗器）
位面饰品：沉欢醉饮的海隅；2件套；副C卡芙卡的最优选。提供12%攻击力和24%持续伤害增伤。
出处：https://wiki.biligame.com/sr/卡芙卡/攻略 | 修订 2026-02-18
[3] 分数 0.8756 | 那刻夏/攻略 > 配装推荐 > 简易推荐 | 页面修订 2025-12-10
那刻夏/攻略（配装推荐 > 简易推荐）
隧洞遗器1：晨昏交界的翔鹰；4件套。
出处：https://wiki.biligame.com/sr/那刻夏/攻略 | 修订 2025-12-10"""


# --------------------------------------------------------------------------- #
# 1. 从工具输出里收「本轮真正检索到的页面」
# --------------------------------------------------------------------------- #
def test_收页面与正文():
    got = harvest_retrieved_pages(SEARCH_KB_OUTPUT)
    assert set(got) == {"太空封印站", "卡芙卡/攻略", "那刻夏/攻略"}
    assert "攻击力提高12%" in got["太空封印站"]
    # 正文里不能把「出处：」那行吞进来 —— 它是 URL，不是证据
    assert "wiki.biligame.com" not in got["太空封印站"]


def test_同页多个_chunk的正文会拼起来():
    """L2 要在**整页**里找数值，只看一个 chunk 会误判「无出处」。"""
    two = SEARCH_KB_OUTPUT + """
[4] 分数 0.7 | 太空封印站 > 获取方式
获取方式：模拟宇宙·第三世界、差分宇宙。
出处：https://wiki.biligame.com/sr/太空封印站 | 修订 2025-04-13"""
    got = harvest_retrieved_pages(two)
    assert got["太空封印站"].count("出处") == 0
    assert "差分宇宙" in got["太空封印站"]
    assert "攻击力提高12%" in got["太空封印站"]


def test_非检索工具的输出不产出页面():
    """calculator / list_games 的输出没有命中块，不能被当成检索证据。"""
    assert harvest_retrieved_pages("[重复查询已拦截] 「abc」与本轮之前的检索等价") == {}
    assert harvest_retrieved_pages("") == {}


def test_格式被改动时优雅退化而不是抛异常():
    """tools.py 改了输出格式，这里必须返回空而不是让整轮运行崩掉。"""
    assert harvest_retrieved_pages("[1] 完全不同的格式 没有竖线没有分数") == {}


# --------------------------------------------------------------------------- #
# 2. 引用标记的抽取
# --------------------------------------------------------------------------- #
def test_抽出方括号引用():
    cites = parse_citations("推荐太空封印站[太空封印站]，备选[沉欢醉饮的海隅]。")
    assert [c[0] for c in cites] == ["太空封印站", "沉欢醉饮的海隅"]


def test_起止位置能定位到原文():
    """L1 要就地剥掉，位置错了就会削坏别的字。"""
    ans = "前面[甲页面]后面"
    (raw, s, e), = parse_citations(ans)
    assert ans[s:e] == "[甲页面]"
    assert raw == "甲页面"


def test_markdown链接不算编造引用():
    """`[文字](url)` 自带可点击链接，不属于「凭记忆编造出处」。"""
    assert parse_citations("见[太空封印站](https://x.com/a)") == []


def test_纯数字与过短的方括号不算引用():
    """编号、版本号、日期都不是页面引用。

    这条此前只覆盖长度 1 的 `[1]` / `[x]`，于是 `[12]`、`[2.7]` 因为
    「长度够、且含字母数字」一路过关，最后被 L1 当成编造引用从答案里
    **删掉**。断言加强到长度 ≥2 的纯数字形态，把这个洞锁死。
    """
    assert parse_citations("见表[1]与[x]") == []
    assert parse_citations("免责声明[12]，生效版本[2.7]") == []
    assert parse_citations("修订于[2025-04-13]") == []


def test_形状不像页面名的方括号不算引用():
    """纯 ASCII 的型号 / 等级不是页面名。

    代价是明确的：三个库 1405 个页面标题里只有 `Archer` / `Saber`
    不含中文，这两个页面的**编造**引用会漏剥。用这点漏剥换掉
    「`[Lv80]` 被当编造引用删掉」是划算的，而且被跳过的会进
    `CitationAudit.skipped`，可复核。
    """
    assert parse_citations("等级[Lv80]，面板[HP]") == []
    assert looks_like_page_title("太空封印站") is True
    assert looks_like_page_title("卡芙卡/攻略") is True
    assert looks_like_page_title("12") is False
    assert looks_like_page_title("Lv80") is False
    assert looks_like_page_title("2025-04-13") is False


def test_围栏代码块里的方括号不算引用():
    """代码下标 `arr[10]` 不是「凭记忆编造的出处」。"""
    ans = '示例如下：\n```python\nx = arr[10] + d["k"]\n```\n完毕。'
    assert parse_citations(ans) == []


def test_未闭合围栏也整段当代码():
    """模型输出撞 max_tokens 被截断时，围栏只有开头没有结尾。

    此时它后面全是代码 —— 不认未闭合围栏，就会在一条本来就残缺的
    回答上再删掉一段代码下标，把「截断」升级成「内容损坏」。
    """
    assert parse_citations('```python\nx = arr[10]\n') == []


def test_行内代码里的方括号不算引用():
    assert parse_citations("写成 `arr[10]` 即可。") == []


def test_代码块前后的真引用仍然正常抽取():
    """闸门只挡代码，不能把真引用一起挡掉。"""
    ans = "见[太空封印站]。\n```\nx = a[1]\n```\n另见[卡芙卡/攻略]。"
    assert [c[0] for c in parse_citations(ans)] == ["太空封印站", "卡芙卡/攻略"]


# --------------------------------------------------------------------------- #
# 3. 支持判定：归一化与双向包含
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "citation,titles,expected",
    [
        ("太空封印站", ["太空封印站"], True),
        # 模型写简称、检索到的是全称 —— 全等判定会误伤，这是必须支持的
        ("太空封印站", ["太空封印站（位面饰品）"], True),
        ("太空封印站（位面饰品）", ["太空封印站"], True),
        # 装饰性分隔符：wiki 标题里满是「•」「/」「·」
        ("凝滞虚影炎华之形", ["凝滞虚影 • 炎华之形"], True),
        ("卡芙卡", ["卡芙卡/攻略"], True),
        # 真·编造：本轮压根没检索到这个页面
        ("不老者的仙舟", ["太空封印站", "卡芙卡/攻略"], False),
        ("劫火莲灯", ["太空封印站"], False),
    ],
)
def test_支持判定(citation, titles, expected):
    assert is_supported(citation, titles) is expected


def test_归一化抹掉装饰字符():
    assert normalize_title("凝滞虚影 • 炎华之形") == normalize_title("凝滞虚影炎华之形")
    assert normalize_title("那刻夏/攻略") == normalize_title("那刻夏 攻略")


def test_单字标题不会被任意长引用命中():
    """否则标题「光」会被「光锥」命中，把编造判成合法。"""
    assert is_supported("光锥", ["光"]) is False


# --------------------------------------------------------------------------- #
# 4. L1 硬拦截：剥掉编造引用
# --------------------------------------------------------------------------- #
def test_剥掉未检索到的引用():
    ans = "推荐太空封印站[太空封印站]，也可以考虑[不老者的仙舟]。"
    out, audit = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert "[太空封印站]" in out, "真实引用必须留着"
    assert "[不老者的仙舟]" not in out, "编造引用必须被剥掉"
    assert audit.fabricated == ["不老者的仙舟"]
    assert audit.ok is False


def test_形状不像页面名的方括号一个都不删():
    """本模块最该守住的一条：校验器不许删正确内容。

    修复前 `[12]`、`[2.7]`、`[Lv80]` 会被判成编造引用从正文里删掉，
    于是「C 类已被拦截」的代价变成「正文被削」。它们现在进 `skipped`：
    不判、不删、但留痕。
    """
    ans = "免责声明[12]，生效版本[2.7]，等级[Lv80]。参考[太空封印站]。"
    out, audit = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert out == ans, "形状不像页面名的方括号一个都不能删"
    assert audit.fabricated == []
    assert audit.skipped == ["12", "2.7", "Lv80"]
    assert audit.ok is True, "跳过不是发现问题，不该把这一轮标成有问题"


def test_代码块不会被校验器改写():
    """代码被改写是数据损坏，不是防幻觉。"""
    ans = '```python\nx = arr[10] + d["k"]\n```'
    out, audit = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert out == ans
    assert audit.skipped == [], "代码不是「被跳过的引用」，不该进 skipped"


def test_剥掉后句子读起来仍然通顺():
    out, _ = audit_answer(
        "推荐[太空封印站]。另外[劫火莲灯]也不错。",
        harvest_retrieved_pages(SEARCH_KB_OUTPUT),
    )
    assert out == "推荐[太空封印站]。另外也不错。"


def test_多个编造引用去重但都剥掉():
    ans = "[不老者的仙舟]好，[不老者的仙舟]也好，[劫火莲灯]也行。"
    out, audit = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert "不老者的仙舟" not in out and "劫火莲灯" not in out
    assert audit.fabricated == ["不老者的仙舟", "劫火莲灯"], "只去重计数，不漏剥"


def test_多个编造引用的替换不互相顶偏():
    """从后往前替换 —— 正序替换会让第二处下标失效，削掉后面的正文。"""
    ans = "开头甲[伪造一]中间乙[伪造二]结尾丙"
    out, _ = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert out == "开头甲中间乙结尾丙"


def test_全部合法时原样返回():
    ans = "推荐[太空封印站]，或[卡芙卡/攻略]。"
    out, audit = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert out == ans
    assert audit.fabricated == []
    assert audit.ok is True


def test_没有任何引用时不改文本也不报错():
    assert audit_answer("普通回答，没有引用。", {})[0] == "普通回答，没有引用。"
    assert audit_answer("", {})[0] == ""


def test_判据缺失时一个引用都不剥():
    """本轮一个页面都没取到 → **不剥**，且 checked=False。

    这条锁的是本模块最容易造成大范围误伤的行为：
    `retrieved` 为空既可能是「真的没检索到」，也可能是「工具输出格式变了
    导致 harvest 解析不出来」。后者会让一次**完全正常**的检索回答被削掉
    全部引用 —— 比漏剥编造引用糟得多。

    判据与 SearchGuard.observe 的兜底同源：判据失效时宁可不动手。
    """
    ans = "根据知识库，答案是这样。[某页面]。"
    out, audit = audit_answer(ans, {})
    assert out == ans, "判据缺失时绝不能剥引用"
    assert audit.fabricated == []
    assert audit.checked is False
    assert audit.ok is False, "没校验过不能报成 ok"


def test_判据缺失时引用仍然被记录():
    """不剥 ≠ 不记。`cited` 照样要留下，否则评测看不出「这里本该有引用」。"""
    _, audit = audit_answer("引用了[某页面]。", {})
    assert audit.cited == ["某页面"]


def test_判据可用时checked为True():
    _, audit = audit_answer("答。", harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert audit.checked is True
    assert audit.ok is True


def test_没有引用但有证据时checked仍为True():
    """模型这次没标引用，不等于没取到页面 —— checked 说的是判据，不是引用数。"""
    _, audit = audit_answer("模型这次没标引用。", harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert audit.checked is True
    assert audit.cited == []


# --------------------------------------------------------------------------- #
# 5. L2 只记录、不改文本
# --------------------------------------------------------------------------- #
def test_数值有出处时不报():
    ans = "使装备者的攻击力提高 12%[太空封印站]。"
    out, audit = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert audit.unsupported_numbers == []
    assert out == ans


def test_数值查不到出处时记录但不动文本():
    """**不改文本**是这一档的全部意义：模型可能写出推导值，
    把推导值当幻觉削掉比留着更糟。"""
    ans = "攻击力提高 99%[太空封印站]。"
    out, audit = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert out == ans, "L2 绝不能改答案"
    assert audit.unsupported_numbers == [
        {"page": "太空封印站", "number": "99", "citation": "太空封印站"}
    ]


def test_L2不参与_ok_判定():
    """ok 只看 L1。否则一个正确的推导值会让整轮被标成有问题。"""
    _, audit = audit_answer("攻击力提高 99%[太空封印站]。", harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert audit.unsupported_numbers
    assert audit.ok is True


def test_导出的字段能直接落库():
    _, audit = audit_answer("[不老者的仙舟]", harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    d = audit.to_dict()
    assert set(d) == {
        "cited", "skipped", "fabricated", "unsupported_numbers",
        "retrieved_titles", "checked", "ok",
    }
    assert d["fabricated"] == ["不老者的仙舟"]
    assert d["skipped"] == []
    assert d["checked"] is True
    assert isinstance(d, dict)


def test_skipped与fabricated必须可区分():
    """「判定了，是编造」与「没判」是两种结论，绝不能合并。

    合并就等于退回那个把 `[12]`、`arr[10]` 从答案里删掉的版本。
    """
    ans = "生效版本[2.7]，另见[不老者的仙舟]。"
    _, audit = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert audit.skipped == ["2.7"]
    assert audit.fabricated == ["不老者的仙舟"]
    assert not set(audit.skipped) & set(audit.fabricated)


def test_默认构造的审计对象表示未校验():
    """默认 checked=False → ok=False：默认状态是「没校验过」而不是「校验通过」。

    这个默认值是有意的：让一个忘了跑校验的路径**显式暴露**成未校验，
    而不是安静地报成 ok —— 后者会让盲区在评测里显示成达标。
    """
    a = CitationAudit()
    assert a.checked is False
    assert a.ok is False
    assert a.to_dict()["fabricated"] == []


# --------------------------------------------------------------------------- #
# 6. 双向验证：这两条保证「校验器真的在干活」
# --------------------------------------------------------------------------- #
def test_双向验证_把判定短路成恒真则编造引用不再被剥():
    """模拟「校验逻辑坏掉」：is_supported 永远返回 True。

    期望：sr-007 那条 C 类失败（提及不存在的不老者的仙舟）
    在坏掉的校验器下**不会被剥掉** —— 证明剥除确实来自这段逻辑，
    而不是恰好文本里没有那个词。
    """
    import app.agent.citation_audit as mod

    ans = "推荐太空封印站[太空封印站]，也可考虑[不老者的仙舟]。"
    assert audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))[1].fabricated

    original = mod.is_supported
    try:
        mod.is_supported = lambda *a, **k: True  # 注入故障
        broken_out, broken = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    finally:
        mod.is_supported = original
    assert "[不老者的仙舟]" in broken_out, "校验器坏了就不该剥 —— 否则守卫本身是假绿"
    assert broken.fabricated == []


def test_双向验证_摘掉形状闸门则正文会被删():
    """证明「`[12]` 不被删」确实来自形状闸门，而不是它恰好没被匹配到。

    把闸门短路成「长度够就算页面名」（等价于修复前的行为），
    同一条答案立刻被削。没有这一条，闸门被谁顺手删掉都不会有人发现。
    """
    import app.agent.citation_audit as mod

    ans = "免责声明[12]，等级[Lv80]。"
    assert audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))[0] == ans

    original = mod.looks_like_page_title
    try:
        mod.looks_like_page_title = lambda raw: len(raw) >= mod.MIN_CITATION_LEN
        broken_out, broken = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    finally:
        mod.looks_like_page_title = original

    assert broken_out == "免责声明，等级。", "闸门失效了正文却没被删 —— 说明挡住它的不是闸门"
    assert sorted(broken.fabricated) == ["12", "Lv80"]


def test_双向验证_替换顺序错了会把正文削坏():
    """锁住「从后往前替换」这个约束，而且用**真实函数**验证。

    先前这个测试用一套自己写的假 harness 去模拟替换，等于在测测试自己写的
    替身，没碰到 `audit_answer` 的真实路径 —— 那种测试在实现被改坏时
    依然全绿。这里改为构造三个编造引用，直接比对真实输出。
    """
    # 三处编造引用 + 三段有辨识度的正文，谁被削错都会立刻看出来
    ans = "甲段[伪造一]乙段[伪造二]丙段[伪造三]丁段"
    out, audit = audit_answer(ans, harvest_retrieved_pages(SEARCH_KB_OUTPUT))
    assert out == "甲段乙段丙段丁段", "任何一段正文被削坏都是替换顺序错了"
    assert audit.fabricated == ["伪造一", "伪造二", "伪造三"]

    # 反证：如果按正序替换，第二处之后的下标全部失效，结果必然不是上面那样。
    # （手工模拟一次，把这个「顺序真的有后果」的事实写进测试而不是留在脑子里。）
    naive = ans
    for _raw, s, e in sorted(parse_citations(ans), key=lambda c: c[1]):
        naive = naive[:s] + naive[e:]
    assert naive != out, "正序替换会削坏正文 —— 所以倒序是必需的，不是风格选择"
