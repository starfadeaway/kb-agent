"""评测集定义。

放在代码里而不是 JSON 文件，是为了让 gold 标注和判定规则放在一起看：
每条 case 必须能回答"凭什么算召回成功"。

标注原则（直接影响实验结论能不能站住）：
1. `gold_pages` 标 wiki 页面标题，**不标 chunk_id** —— 索引一重建 chunk 就变了，
   绑死 chunk_id 会让"重建索引"直接等价于"评测集失效"；
2. gold_answer 必须是页面里真实存在的值，不允许写"以 wiki 为准"这种废话；
3. 幻觉测试单独成类：正确行为是"明确说不知道"，答不出来才算对；
4. 每个游戏一套标注，绝不混在一起评 —— 两款游戏的术语体系不同。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    case_id: str
    game: str
    category: str            # single_hop / multi_hop / numerical / routing / unanswerable
    question: str
    gold_answer: str
    gold_pages: list[str] = field(default_factory=list)
    difficulty: int = 1
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "game": self.game,
            "category": self.category,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "gold_pages": self.gold_pages,
            "difficulty": self.difficulty,
            "notes": self.notes,
        }


# 全部 gold 值都对照 data/raw/sr 下的原始页面核过
#
# 扩展记录：从 10 条扩到 32 条的原因不是"数量好看"，而是原版有致命缺陷 ——
# 8 条检索题里只有 1 条纯语义题，这正好是混合检索唯一能显示优势的场景。
# 样本不够时 A2（融合）的负结果无法区分"融合真的没用"和"题出得不好"。
# 扩展后的分布：
#   single_hop 14 / semantic 6 / multi_hop 4 / numerical 3 / unanswerable 3 / routing 2
SR_CASES: list[EvalCase] = [
    # ---------------- single_hop：结构化字段精确命中 ----------------
    EvalCase(
        case_id="sr-001",
        game="sr",
        category="single_hop",
        question="光锥「我」的诞生是几星？属于什么命途？",
        gold_answer="4星，命途为智识",
        gold_pages=["「我」的诞生"],
        notes="结构化字段类：模板展平后应能直接命中",
    ),
    EvalCase(
        case_id="sr-002",
        game="sr",
        category="single_hop",
        question="位面饰品太空封印站的两件套效果是什么？",
        gold_answer="使装备者攻击力提高12%，当速度大于等于120时攻击力额外提高12%",
        gold_pages=["太空封印站"],
    ),
    EvalCase(
        case_id="sr-003",
        game="sr",
        category="single_hop",
        question="凝滞虚影「巽风之形」在哪个地图，需要多少开拓力？",
        gold_answer="所在地图是铆钉镇，开拓力消耗30",
        gold_pages=["凝滞虚影 • 巽风之形"],
        notes="双跳问题：地图 + 消耗，都要答到",
    ),
    EvalCase(
        case_id="sr-004",
        game="sr",
        category="single_hop",
        question="NPC 阿然的称号是什么？",
        gold_answer="梦境鼓励师",
        gold_pages=["阿然"],
        notes="短文本，考察精确匹配",
    ),
    EvalCase(
        case_id="sr-005",
        game="sr",
        category="multi_hop",
        question="不老者的仙舟两件套对速度有什么要求？满足后加什么？",
        gold_answer="速度大于等于120时，我方全体攻击力提高8%（同时装备者生命上限提高12%）",
        gold_pages=["不老者的仙舟"],
        difficulty=2,
    ),
    EvalCase(
        case_id="sr-006",
        game="sr",
        category="single_hop",
        question="凝滞虚影「空海之形」的实现版本和可能获取的物品是什么？",
        gold_answer="实装版本1.0，可能获取虚幻铸铁",
        gold_pages=["凝滞虚影 • 空海之形"],
    ),
    EvalCase(
        case_id="sr-008",
        game="sr",
        category="numerical",
        question="太空封印站给攻击力 1000 的角色能加多少攻击力？",
        gold_answer="12%，即 120 点攻击力",
        gold_pages=["太空封印站"],
        notes=(
            "多工具题：必须先检索拿到 12% 这个系数，再用 calculator 算 1000×0.12。"
            "注意：早期版本我出过「5% 暴击率副词条多少级到 20%」——"
            "那个前提本身就是错的（星铁暴击率副词条不按每级 5% 成长），"
            "换成这题避免用一个假前提去考模型。"
        ),
    ),
    EvalCase(
        case_id="sr-009",
        game="sr",
        category="unanswerable",
        question="下一个版本会新增哪个角色？技能是什么？",
        gold_answer="知识库未收录未来版本内容，应明确说明无法确认，不要编造角色名",
        gold_pages=[],
        difficulty=2,
        notes="幻觉测试：模型很容易按训练数据编一个角色",
    ),
    EvalCase(
        case_id="sr-010",
        game="sr",
        category="routing",
        question="绝区零的音擎和星铁的光锥是一回事吗？",
        gold_answer="不是。音擎是绝区零的遗器装备，光锥是崩坏:星穹铁道的遗器装备，两套独立体系",
        gold_pages=[],
        difficulty=2,
        notes="跨游戏术语辨析，考察路由与术语隔离",
    ),

    # ---------------- semantic：问题里不含道具名，纯词面匹配召不回 ----------------
    # 这类是 A0/A1 分化的唯一试金石，扩评测集时优先补这一类。
    EvalCase(
        case_id="sr-007",
        game="sr",
        category="semantic",
        question="我攻击力不够还想更快，用哪个位面饰品比较合适？",
        gold_answer="太空封印站（速度≥120时攻击力额外提高12%）",
        gold_pages=["太空封印站"],
        difficulty=3,
        notes=(
            "语义类：问题里没有出现道具名，纯 BM25 应当召不回。"
            "注意 gold 标注本身有歧义 —— top-3 里返回的多个页面确实都是位面饰品，"
            "不止一个合理答案。保留它但不能单独解读其 top-1 结果。"
        ),
    ),
    EvalCase(
        case_id="sr-011",
        game="sr",
        category="semantic",
        question="我暴击率总是差一点到线，想让普攻和战技伤害更高，选哪个位面饰品？",
        gold_answer="繁星竞技场（暴击率≥70%时普攻和战技伤害提高20%）",
        gold_pages=["繁星竞技场"],
        difficulty=3,
        notes="纯语义：题干只描述目标状态，不含饰品名，BM25 无词面可匹配",
    ),
    EvalCase(
        case_id="sr-012",
        game="sr",
        category="semantic",
        question="队伍里效果抵抗不够高，想让全队的暴击伤害更高，有什么套装？",
        gold_answer="折断的龙骨（效果抵抗≥30%时我方全体暴击伤害提高10%）",
        gold_pages=["折断的龙骨"],
        difficulty=3,
        notes="纯语义：考查能否把「效果抵抗门槛 → 暴击伤害」这个因果链召回来",
    ),
    EvalCase(
        case_id="sr-013",
        game="sr",
        category="semantic",
        question="我想让开战斗就抢先手，有没有什么位面饰品能帮忙？",
        gold_answer="生命的翁瓦克（速度≥120时进入战斗立刻使行动提前40%）",
        gold_pages=["生命的翁瓦克"],
        difficulty=3,
        notes="纯语义：「抢先手」在 wiki 里写作「行动提前」，纯词面匹配必然召不回",
    ),
    EvalCase(
        case_id="sr-014",
        game="sr",
        category="semantic",
        question="带召唤物一起打本的时候，怎么进一步提升暴击伤害？",
        gold_answer="奇想蕉乐园（暴击伤害提高16%，存在召唤目标时额外提高32%）",
        gold_pages=["奇想蕉乐园"],
        difficulty=3,
        notes="纯语义：「召唤物」对应 wiki 里的「装备者召唤的目标」，术语不对齐",
    ),
    EvalCase(
        case_id="sr-015",
        game="sr",
        category="semantic",
        question="火属性的首领打不动，wiki 上说应该用什么属性配队比较好？",
        gold_answer="炎华之形建议物理、冰、量子（属性为火）",
        gold_pages=["凝滞虚影 • 炎华之形"],
        difficulty=2,
        notes="半语义：「打不动火」需要从属性克制反推建议属性",
    ),

    # ---------------- single_hop 扩充：数值阈值与版本号 ----------------
    EvalCase(
        case_id="sr-016",
        game="sr",
        category="single_hop",
        question="位面饰品繁星竞技场的暴击率门槛是多少？达到后提升什么伤害？",
        gold_answer="门槛是暴击率大于等于70%，达到后普攻和战技造成的伤害提高20%",
        gold_pages=["繁星竞技场"],
        notes="考察数值门槛抽取，答对门槛比答对效果更重要",
    ),
    EvalCase(
        case_id="sr-017",
        game="sr",
        category="single_hop",
        question="凝滞虚影「职司之形」是哪个版本实装的？在什么位置？",
        gold_answer="实装版本2.2，所在地图是克劳克影视乐园",
        gold_pages=["凝滞虚影 • 职司之形"],
    ),
    EvalCase(
        case_id="sr-018",
        game="sr",
        category="single_hop",
        question="凝滞虚影「凛月之形」的属性是什么？可能掉什么？",
        gold_answer="属性是量子，可能获取暗帷月华",
        gold_pages=["凝滞虚影 • 凛月之形"],
    ),
    EvalCase(
        case_id="sr-019",
        game="sr",
        category="single_hop",
        question="凝滞虚影「天人之形」在哪个地图？什么版本实装？",
        gold_answer="所在地图是丹鼎司，实装版本1.2",
        gold_pages=["凝滞虚影 • 天人之形"],
    ),
    EvalCase(
        case_id="sr-020",
        game="sr",
        category="single_hop",
        question="NPC 阿来的称号和职业分别是什么？",
        gold_answer="称号是相声爱好者，职业是码头工人",
        gold_pages=["阿来"],
        notes="同一页面两个字段，考察字段抽取的完整性",
    ),
    EvalCase(
        case_id="sr-021",
        game="sr",
        category="single_hop",
        question="位面饰品生命的翁瓦克两件套提升多少能量恢复效率？",
        gold_answer="提高5%",
        gold_pages=["生命的翁瓦克"],
    ),
    EvalCase(
        case_id="sr-022",
        game="sr",
        category="single_hop",
        question="位面饰品奇想蕉乐园是什么时候实装的？只能从哪里获取？",
        gold_answer="实装版本2.5，只能通过差分宇宙获取",
        gold_pages=["奇想蕉乐园"],
    ),
    EvalCase(
        case_id="sr-023",
        game="sr",
        category="single_hop",
        question="光锥「我」的诞生的基础生命值是多少？技能名称是什么？",
        gold_answer="生命值43，技能名称是画像少女",
        gold_pages=["「我」的诞生"],
        notes="数值字段 + 文本字段混合",
    ),
    EvalCase(
        case_id="sr-024",
        game="sr",
        category="single_hop",
        question="凝滞虚影「炎华之形」可能遭遇哪些敌人？",
        gold_answer="外宇宙之炎、火焚灾影",
        gold_pages=["凝滞虚影 • 炎华之形"],
    ),

    # ---------------- multi_hop：需要跨字段或跨页面组合 ----------------
    EvalCase(
        case_id="sr-025",
        game="sr",
        category="multi_hop",
        question="位面饰品折断的龙骨对效果抵抗的门槛是多少？达到后提升的是谁的暴击伤害？",
        gold_answer="门槛是效果抵抗大于等于30%，达到后我方全体暴击伤害提高10%",
        gold_pages=["折断的龙骨"],
        difficulty=2,
        notes="关键在「我方全体」还是「装备者」——这两者在数值上差一个数量级的作用对象",
    ),
    EvalCase(
        case_id="sr-026",
        game="sr",
        category="multi_hop",
        question="凝滞虚影「空海之形」的关卡属性标的是什么？它建议的配队属性有哪些？",
        gold_answer="属性是量子，建议属性为物理、风、虚数",
        gold_pages=["凝滞虚影 • 空海之形"],
        difficulty=2,
        notes="同一页面里「属性」和「建议属性」是两个不同字段，极易混淆",
    ),
    EvalCase(
        case_id="sr-027",
        game="sr",
        category="multi_hop",
        question="凝滞虚影「凛月之形」在什么位置？需要多少开拓力？",
        gold_answer="所在地图是「呓语密林」神悟树庭，开拓力消耗30",
        gold_pages=["凝滞虚影 • 凛月之形"],
        difficulty=2,
    ),
    EvalCase(
        case_id="sr-028",
        game="sr",
        category="multi_hop",
        question="位面饰品奇想蕉乐园的基础暴击伤害提升是多少？有召唤目标时总共提升多少？",
        gold_answer="基础提高16%，存在召唤目标时额外提高32%，合计48%",
        gold_pages=["奇想蕉乐园"],
        difficulty=2,
        notes="需要做加法：基础 16% + 额外 32% = 48%",
    ),

    # ---------------- numerical：多工具题（检索拿系数 + calculator 算数） ----------------
    EvalCase(
        case_id="sr-029",
        game="sr",
        category="numerical",
        question="位面饰品繁星竞技场给暴击率 50% 的角色能加多少暴击率？",
        gold_answer="8%，即加 8 个百分点，暴击率变为58%",
        gold_pages=["繁星竞技场"],
        notes="多工具题：先检索 8%，再 calculator 算 50×1.08 或直接报 8 个百分点",
    ),
    EvalCase(
        case_id="sr-030",
        game="sr",
        category="numerical",
        question="位面饰品折断的龙骨给效果抵抗 25% 的角色，暴击伤害能提高多少？",
        gold_answer="不提高。折断的龙骨要求效果抵抗≥30%才触发，25% 不满足条件",
        gold_pages=["折断的龙骨"],
        difficulty=2,
        notes=(
            "陷阱题：必须先检索门槛（30%）再判断 25% 不满足，"
            "答「提高10%」就是踩了没核对条件的坑。这条专门区分"
            "「检索到了数值」和「用对了数值」。"
        ),
    ),

    # ---------------- unanswerable：幻觉与越界 ----------------
    EvalCase(
        case_id="sr-031",
        game="sr",
        category="unanswerable",
        question="星铁 4.0 版本会实装哪个新遗器套装？两件套效果是什么？",
        gold_answer="知识库未收录该版本内容，应明确说明无法确认，不要编造套装名或效果",
        gold_pages=[],
        difficulty=2,
        notes="比 sr-009 更刁钻：问的是具体版本号，模型很容易自信地编一个套装名",
    ),
    EvalCase(
        case_id="sr-032",
        game="sr",
        category="unanswerable",
        question="位面饰品太空封印站的三件套效果是什么？",
        gold_answer="位面饰品只有两件套效果，没有三件套，应指出问题前提不成立",
        gold_pages=["太空封印站"],
        difficulty=2,
        notes=(
            "假前提题：页面里四件套效果字段本身是空的，"
            "位面饰品本身就只有两件套。答一个具体数值就是编造。"
            "与 sr-030 同类：都必须靠检索到的字段本身来否证问题。"
        ),
    ),
]

# 绝区零评测集（12 条）
#
# 之前只有 1 条路由题，README 里所有量化结论都只来自星铁一个库 ——
# "多库路由有效"只有 smoke 自检兜着，没被评测集覆盖。
#
# 全部 gold 值对照 data/raw/zzz 下的原始页面核过，用
# `scripts/dump_fields.py --game zzz --title <页面>` 可以复现核对过程。
ZZZ_CASES: list[EvalCase] = [
    EvalCase(
        case_id="zzz-001",
        game="zzz",
        category="routing",
        question="绝区零的音擎属于哪个部位？",
        gold_answer="音擎是绝区零的遗器装备，需要通过知识库 zzz 分片确认",
        gold_pages=[],
        difficulty=2,
        notes="仅用于验证知识库路由正确性（zzz 索引尚未建立时应明确报「知识库未收录」而不是编答案）",
    ),

    # ---------------- single_hop：结构化字段精确命中 ----------------
    EvalCase(
        case_id="zzz-002",
        game="zzz",
        category="single_hop",
        question="艾莲属于哪个阵营？是什么属性的代理人？",
        gold_answer="阵营是维多利亚家政，属性为冰",
        gold_pages=["艾莲"],
        notes="模板展平后「阵营/属性」两个字段应能直接命中",
    ),
    EvalCase(
        case_id="zzz-003",
        game="zzz",
        category="single_hop",
        question="艾莲是限定代理人吗？稀有度几星？",
        gold_answer="是限定角色，稀有度 S",
        gold_pages=["艾莲"],
    ),
    EvalCase(
        case_id="zzz-004",
        game="zzz",
        category="single_hop",
        question="安比是什么属性、什么特性？属于哪个阵营？",
        gold_answer="属性为电，特性是击破，阵营是狡兔屋",
        gold_pages=["安比"],
        notes="三个字段分属不同模板参数，测分块是否把它们聚到同一页可召回的范围",
    ),
    EvalCase(
        case_id="zzz-005",
        game="zzz",
        category="single_hop",
        question="爱丽丝属于哪个阵营？稀有度和限定情况如何？",
        gold_answer="阵营是怪啖屋，稀有度 S 且是限定角色",
        gold_pages=["爱丽丝"],
    ),
    EvalCase(
        case_id="zzz-006",
        game="zzz",
        category="single_hop",
        question="音擎「电磁暴」-壹式是几星音擎？攻击力词条是多少？",
        gold_answer="稀有度 B 阶，攻击力 32",
        gold_pages=["「电磁暴」-壹式"],
        notes="绝区零的音擎页字段命名与星铁光锥不同（攻击力/副词条数值），术语混用会答错",
    ),
    EvalCase(
        case_id="zzz-007",
        game="zzz",
        category="single_hop",
        question="「电磁暴」-壹式可以通过什么途径获得？",
        gold_answer="调频、玩具店",
        gold_pages=["「电磁暴」-壹式"],
    ),
    EvalCase(
        case_id="zzz-008",
        game="zzz",
        category="single_hop",
        question="安比的空洞鸣徽是什么？",
        gold_answer="【契合】充盈电鸣",
        gold_pages=["安比"],
    ),

    # ---------------- semantic：靠语义而非字面命中 ----------------
    EvalCase(
        case_id="zzz-009",
        game="zzz",
        category="semantic",
        question="艾莲的战斗定位是什么？怎么造成伤害？",
        gold_answer=(
            "灵活游走进行输出：进入「巡游」状态可快速冲刺提升机动性，"
            "在巡游状态下发动蓄力剪击可获得「急冻充能」，消耗急冻充能强化攻击"
        ),
        gold_pages=["艾莲"],
        notes="语义题：问题里没有「巡游」「急冻充能」任何字面词，纯关键词检索必然召不回",
    ),
    EvalCase(
        case_id="zzz-010",
        game="zzz",
        category="semantic",
        question="哪个绝区零代理人属于怪啖屋、是靠属性异常打输出的？",
        gold_answer="爱丽丝（怪啖屋，特性为异常）",
        gold_pages=["爱丽丝"],
        difficulty=2,
        notes="反向查询：不给出名字，要靠「阵营+特性」两个字段定位。干扰项是艾莲（也是 S 级限定）",
    ),

    # ---------------- numerical ----------------
    EvalCase(
        case_id="zzz-011",
        game="zzz",
        category="numerical",
        question="艾莲的生命值、攻击力、防御力分别是多少？",
        gold_answer="生命值 617，攻击力 135，防御力 49",
        gold_pages=["艾莲"],
        notes="三个字段连续出现，测分块是否把它们切在一起；也测别把 10 级/10+ 级的养成值当成基础值",
    ),

    # ---------------- unanswerable ----------------
    EvalCase(
        case_id="zzz-012",
        game="zzz",
        category="unanswerable",
        question="绝区零 3.0 版本会实装哪个新音擎？",
        gold_answer="知识库未收录该版本内容，应明确说明无法确认，不要编造音擎名或效果",
        gold_pages=[],
        difficulty=2,
        notes="与 sr-031 同类：问具体版本号时模型最容易自信地编一个名字出来",
    ),
]

# 原神评测集（12 条）
#
# 全部 gold 值对照 data/raw/ys 下的原始页面核过。
# 原神库 1293 chunk 里圣遗物/神瞳/徽印这类结构化页面最密集，所以单跳题占多数。
YS_CASES: list[EvalCase] = [
    # ---------------- single_hop ----------------
    EvalCase(
        case_id="ys-001",
        game="ys",
        category="single_hop",
        question="散失的草神瞳是什么类型的道具？用来做什么？",
        gold_answer="类型是冒险道具，用于提升须弥地区的七天神像等级",
        gold_pages=["散失的草神瞳"],
    ),
    EvalCase(
        case_id="ys-002",
        game="ys",
        category="single_hop",
        question="散失的草神瞳是怎么获得的？",
        gold_answer="来源是地图探索",
        gold_pages=["散失的草神瞳"],
    ),
    EvalCase(
        case_id="ys-003",
        game="ys",
        category="single_hop",
        question="岩之印是几星的什么东西？",
        gold_answer="稀有度 3，类型是七国徽印",
        gold_pages=["岩之印"],
    ),
    EvalCase(
        case_id="ys-004",
        game="ys",
        category="single_hop",
        question="岩之印能用来干什么？",
        gold_answer="在璃月港中的明星斋兑换道具",
        gold_pages=["岩之印"],
    ),
    EvalCase(
        case_id="ys-005",
        game="ys",
        category="single_hop",
        question="冒险家的稀有度范围是多少？",
        gold_answer="最低稀有度 1，最高稀有度 3",
        gold_pages=["冒险家"],
        notes="原神是「最低/最高稀有度」两个字段，不是星铁的单一「稀有度」，字段名不同",
    ),
    EvalCase(
        case_id="ys-006",
        game="ys",
        category="single_hop",
        question="冒险家的两件套效果是什么？",
        gold_answer="生命值上限提升 1000 点（绿字）",
        gold_pages=["冒险家"],
    ),
    EvalCase(
        case_id="ys-007",
        game="ys",
        category="single_hop",
        question="冒险家的四件套效果是什么？",
        gold_answer="开启各类宝箱后 5 秒内，持续恢复 30% 生命值",
        gold_pages=["冒险家"],
        difficulty=2,
        notes="与 ys-006 是同一页的不同字段：两件/四件套混淆是本库最常见的错答",
    ),

    # ---------------- multi_hop ----------------
    EvalCase(
        case_id="ys-008",
        game="ys",
        category="multi_hop",
        question="冒险家可以通过哪些途径获得？",
        gold_answer=(
            "宝箱（探索奖励、固定宝箱、精英怪物、BOSS）、打怪掉落、"
            "仲夏庭园：祝圣秘境：净化之炎，以及蒙德/须弥 NPC 处购买"
        ),
        gold_pages=["冒险家"],
        difficulty=2,
        notes="「获取途径」是一个多行长字段，测分块会不会把后半段切丢；"
              "只答「探索奖励」属于漏召回",
    ),

    # ---------------- semantic ----------------
    EvalCase(
        case_id="ys-009",
        game="ys",
        category="semantic",
        question="哪个道具是浓烈的草元素能量积聚成的？有什么用？",
        gold_answer="散失的草神瞳，用于提升须弥地区的七天神像等级",
        gold_pages=["散失的草神瞳"],
        notes="问题里没有「草神瞳」三个字，要靠「介绍」字段的描述反查",
    ),

    # ---------------- routing：跨库隔离 ----------------
    EvalCase(
        case_id="ys-010",
        game="ys",
        category="routing",
        question="原神的圣遗物和星穹铁道的遗器是一回事吗？",
        gold_answer="不是同一套体系：原神叫圣遗物（分花/羽/沙/杯/冠），星铁叫位面饰品，只有两件套",
        gold_pages=[],
        difficulty=2,
        notes=(
            "跨库隔离题：问的是两款游戏的同义不同名装备。"
            "必须分别检索 ys 和 sr 两个库才能答，只查一个库就会把两者的效果混着说"
        ),
    ),

    # ---------------- unanswerable ----------------
    EvalCase(
        case_id="ys-011",
        game="ys",
        category="unanswerable",
        question="原神 6.0 版本会实装哪个新圣遗物套装？",
        gold_answer="知识库未收录该版本内容，应明确说明无法确认，不要编造套装名或效果",
        gold_pages=[],
        difficulty=2,
        notes="与 sr-031 / zzz-012 同构：三库各一条，验证越界拒答在每个库上行为一致",
    ),
    EvalCase(
        case_id="ys-012",
        game="ys",
        category="unanswerable",
        question="派蒙的七元素弹弓精lv.9满级加多少攻击力？",
        gold_answer="知识库里没有派蒙这个武器页面，应说明未收录，不要给具体数值",
        gold_pages=[],
        difficulty=2,
        notes=(
            "假前提 + 未收录双层陷阱：派蒙是角色不是武器，"
            "且该条目不在库里。答一个看似合理的数值就是编造"
        ),
    ),
]

ALL_CASES: dict[str, list[EvalCase]] = {
    "sr": SR_CASES,
    "zzz": ZZZ_CASES,
    "ys": YS_CASES,
}


def cases_for(game: str) -> list[EvalCase]:
    return list(ALL_CASES.get(game, []))


def all_cases() -> list[EvalCase]:
    return [c for cases in ALL_CASES.values() for c in cases]