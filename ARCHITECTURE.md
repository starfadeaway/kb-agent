# 架构说明

这份文档回答三个问题：

1. **RAG 在这个项目里具体是哪几段代码？**（第三节，从入口到出口逐步展开）
2. **每个组件为什么放在那个位置？**（第四~八节）
3. **改这段代码会影响什么？**（最后一节的影响面对照表）

想改行为之前先看这里，能少踩很多"改了没生效"的坑。

---

## 一、30 秒总览

```
                       ┌─────────────── 入库（离线，一次性）───────────────┐
  biligame MediaWiki ──▶ 模板展平 ──▶ 分块 ──▶ Embedding ──▶ LanceDB
         ▲                                                    │
         │                                            MySQL ◀─┤ 页面元数据/infobox 数值
    get_page 工具                                                │
    （取全文）                                                   │
                                                               ▼
                    ┌─────────────── 查询（在线，每次提问）───────────────┐
  用户提问 ──▶ 关键词路由 ──▶ ReAct Agent ──▶ search_kb 工具 ───────────┤
                    ▲                     │                            │
                    │                     ▼                            │
                    │              ┌─ RAG 检索链 ─────────┐              │
                    │              │ ① BM25 稀疏召回      │              │
                    │              │ ② 向量稠密召回       │              │
                    │              │ ③ RRF 融合           │              │
                    │              │ ④ Cross-Encoder 重排 │              │
                    │              └──────────────────────┘              │
                    │                     │                            │
                    └──────────── Chat LLM 生成答案 ◀───────────────────┘
                                          │
                                          ▼
                            答案 + 引用出处 + 完整调用轨迹
```

**一句话分工**：数值走 SQL 精确查询，语义走向量。

---

## 二、目录地图

| 目录 | 职责 | 入口文件 |
|---|---|---|
| `backend/app/ingest/` | 抓取 → 清洗 → 分块 → 入库 | `pipeline.py`（编排） |
| `backend/app/retrieval/` | **RAG 检索链** | `hybrid.py`（唯一核心文件） |
| `backend/app/store/` | 向量库 + MySQL + 落库 | `lance_store.py` / `db.py` / `persist.py` |
| `backend/app/agent/` | 手搓 ReAct + 工具定义 | `react.py` / `tools.py` |
| `backend/app/llm/` | 四个基座的统一封装 | `client.py` |
| `backend/app/eval/` | 评测集 + 指标 + Judge | `cases.py` / `metrics.py` / `judge.py` |
| `backend/scripts/` | 抓取/重建/评测/剖析脚本 | `exp_b.py` / `exp_d.py` 等 |
| `frontend/src/` | React 界面 | `App.tsx` |

---

## 三、RAG 到底在哪：一次检索的完整路径

这是最常被问的问题，所以单独展开。**RAG = Retrieval（检索）+ Augmented（把检索结果喂给模型）+ Generation（生成）**。

### 3.1 入口：Agent 决定调 `search_kb`

```
ReActAgent._run_once()                    agent/react.py:912
  └─ await call_tool("search_kb", args)   agent/tools.py:481（分发入口）
                                          agent/tools.py:189（search_kb 本体）
```

Agent 不是每次都直接检索 —— 它先问 DeepSeek「要不要查知识库」，
这一步是 LLM 在做**工具决策**，不是 RAG 本身。

### 3.2 RAG 本体：检索链（`retrieval/hybrid.py`）

```
HybridRetriever.retrieve(query, cfg)      retrieval/hybrid.py:335（async）
   │
   ├─① BM25 稀疏召回
   │    tokenize(query)                     :178  jieba 切词 + 中文 2-gram
   │    BM25Index.search()                  :157  进程内 BM25Okapi
   │    → top20  chunk_id
   │
   ├─② 向量稠密召回
   │    llm.embed([query])                  llm/client.py:535   ← 远程 147ms
   │    store.query(query_embedding)        store/lance_store.py ← 本地 8ms
   │    → top20  chunk
   │
   ├─③ RRF 融合                             :395
   │    score = Σ 1/(60 + rank)             只用排名，不用分数
   │    → 约 31 条
   │
   └─④ Cross-Encoder 重排                  :464
        llm.rerank(query, 候选文档)         llm/client.py:611  ← 远程 449ms
        → top5
```

> 行号会漂：这份地图此前写的是一整组旧行号（`retrieve` 记成 200、
> `_run_once` 记成 303 …），逐条都对不上。**改文件时不要顺手改这些数字**，
> 它们只是导读；真值用 `grep -n "def 函数名"` 现取。

**四步各自解决什么，每一步都有实测数据支撑**（详见 `EVALUATION.md` 实验 A）：

| 步骤 | 解决什么问题 | 去掉它会怎样（实测） |
|---|---|---|
| ① BM25 | 低频专名（「位面饰品」「行迹」）词面匹配 | 纯向量 sr 恒定 0.828 且不随 top_k 增长 |
| ② 向量 | 问题里没有道具名的纯语义题 | 语义类题全部召不回 |
| ③ RRF 融合 | 两路排名合并 | 三库合计 MRR 0.865→0.854（−1.0pt） |
| ④ 重排 | 候选已对但排序不够靠前 | sr recall@3 0.897→0.828 |

> 这张表的数字取当前产物（`data/eval/exp_A_all_k3.json`）。此前写的是
> 2306 chunk 旧语料口径（③ 掉 2.3pt、④ 从 0.966 掉到 0.897），
> 与 README / EVALUATION 的当前值对不上。

**为什么用排名融合而不是分数加权**：BM25 分数和余弦相似度不同量纲，
直接加权相加会被其中一路的分数分布带偏。RRF 只看名次，无量纲问题。

### 3.3 Augmented：把检索结果喂回给模型

```
search_kb 的返回值                        agent/tools.py:189 起
  第 1 行：[search_kb] status=ok hits=3 top_score=0.912 top_page=… 675ms 来源=bm25+vector+rerank
  第 2 行起：[1] 分数 0.9312 | 太空封印站 > 两件套效果
             <文本片段，最多 500 字>
             出处：https://wiki.biligame.com/sr/太空封印站
```

**第一行是给 Agent 循环看的机器可读状态**，不是给人看的。
Agent 靠它判断「这次检索到底找到没有」，从而决定要不要继续搜。
没有这层，Agent 只能看到一段文本，会把「召回了但都不相关」
当成「再搜一次说不定有」—— 实测这会导致空转到顶满 8 步。

`出处：` 前缀的行同时被两处消费：
- Agent 拿它做引用标注
- `_collect_refs()` 抽出来存进 MySQL，供前端展示和忠实度评测

### 3.4 Generation：Chat LLM 组织答案

```
llm.chat(messages, tools=get_schemas())  llm/client.py:407   ← DeepSeek
```

模型在**已有的对话历史**上继续：系统提示 + 历史 + 用户问题 + 这一轮所有 tool 消息。
它看到检索到的文本片段，然后：
- 需要算数 → 调 `calculator`
- 片段不够全 → 调 `get_page`
- 信息够了 → 直接输出答案（不再有 tool_calls = 终止信号）

### 3.5 存回去：轨迹与评测

```
persist.save_run(state)                   store/persist.py
  → MySQL runs 表：步数 / 延迟 / token / 停止原因 / guard 统计
  → MySQL tool_calls 表：每次工具调用的参数、耗时、成败
```

这样每次提问都留下可复盘的轨迹，而不是一个孤立的答案。

### 3.6 RAG 之外的部分（容易和 RAG 混淆）

| 不属于 RAG | 属于 Agent / 工程层 |
|---|---|
| 检索链的四步 | 关键词路由（确定性，不用 LLM） |
| 检索结果拼进 prompt | 工具决策（LLM 决定调哪个工具） |
| | 安全阀 / 熔断 / 超时 |
| | 轨迹落库 |
| | 评测指标计算 |

**一句话区分**：RAG 解决「从知识库找到什么」，
Agent 解决「为了让回答正确，需要查几次、查什么、什么时候停」。

### 3.7 多游戏：为什么是「N 个」而不是「主库 + 副库」

最初 `Settings` 里只有 `game_primary` / `game_secondary` 两个字段。
问题是「当前覆盖了哪些游戏」这件事因此在**五处**各写了一遍：
配置校验、`search_kb` 的库内检查、`list_games` 工具、启动预热循环、前端下拉框。
加第三个库要改五个地方，漏一处就出现「库里有了但界面选不到」这类不一致。

现在收敛成一个属性：

```python
@property
def games(self) -> dict[str, str]:   # slug → 显示名
    table = {"sr": "崩坏：星穹铁道", "zzz": "绝区零", "ys": "原神"}
    configured = [self.game_primary, self.game_secondary, self.game_tertiary]
    return {g: table.get(g, g) for g in configured if g}
```

**加第 N+1 个库 = 加一个 `GAME_XXX` 字段 + 表里加一行**，其余代码自动跟上。
前端下拉框也改成读 `/api/health` 的 `games`，不再自己写死选项。

### 3.8 路由必须区分三种"没匹配上"

`route_game()` 曾经是「两个都不匹配就返回默认库」。这会产生一个**隐蔽的错误结论**：

> 问「原神甘雨的突破材料」→ 判定为星铁问题 → 搜星铁库 → 搜不到
> → 回答「知识库未收录」

问题在于：库里**压根没有原神这个库**，真实情况是"没有这个游戏"，
而不是"没有收录这一条"。这两句话用户分辨不出来，但我们必须让他分辨得出来。

所以现在返回三态：

| 输入 | 返回 | 后续 |
|---|---|---|
| 命中 `_GAME_HINTS[已收录]` | 该 slug | 正常进 ReAct 循环 |
| 命中 `UNCOVERED_GAMES`（崩坏3/明日方舟/…） | 该 slug | **不进循环**，`stop_reason=out_of_scope` |
| 都没命中 | `None` | 用 `ToolContext.game` 兜底（前端选的 / 上一轮的） |

越界拒答实测 **0 步 / 1.3s**，对比库内应答 4~5 步 / 6~8s ——
**拒答几乎零成本**，所以没有理由为了"看起来更热心"而去硬搜。

`UNCOVERED_GAMES` 只收**游戏名**和绝不歧义的专有词，刻意不收角色名：
漏判只是多搜一次库，误判会把一个本来能答的问题错误拒答掉，代价不对称。

---

## 一、MediaWiki API —— 语料从哪来

**调用点**：`backend/app/ingest/wiki_client.py`

走 biligame 的 MediaWiki API，三个接口：

| 接口 | 用途 |
|---|---|
| `list_category_pages` | 按「分类:」枚举页面（注意是中文命名空间，不是 `Category:`） |
| `fetch_pages` | 批量取 wikitext，**一次 10 页** |
| `fetch_page_meta` | 取修订号和时间戳 → 版本感知检索的基础 |

### 为什么这么调

- **批量 10 页**：EdgeOne 的 567 是限流挑战页，不是 404。对策优先级是
  「批量取页 > 拉长间隔 > 退避重试」。实测 518 页 0 次拦截。
- **错误在 `query.errors` 里而 HTTP 仍是 200**：只看状态码会把「分类不存在」误判成「这个分类没内容」。
- **567 有时伪装成 `application/json`**：只看 Content-Type 会漏判。

---

## 二、模板展平 —— 抓到了不等于入库了

**调用点**：`backend/app/ingest/template_parser.py`

星铁 wiki 的页面内容几乎全在模板里。光锥页 `{{光锥图鉴}}` 有 54 个具名参数。

**踩过的坑**：最初的清洗逻辑是"删掉所有模板"，把 969 字的 wikitext 洗成 20 字。
页面抓到了，索引里等于什么都没存。

**修法**：展平成 `字段：值` 的纯文本。另外专门处理了 `_WRAPPER_TEMPLATES`
（`{{颜色|描述2|12%}}` 这种），只取最后一个参数的实际文本，
否则 MySQL 里会存下 `{{颜色|描述2|12%}}` 这种脏值。

**效果**：同一批 40 页，从 9 chunk → 78 chunk。

---

## 三、分块 —— 检索粒度

**调用点**：`backend/app/ingest/chunker.py`

策略：先按标题切逻辑段 → 再按句边界贪心打包 → overlap 以整句回退。

**为什么 overlap 要"整句回退"**：按字符数硬切会把句子劈成两半，
半句话对 Embedding 是噪声，对 BM25 也是。

**实验 B 会动的就是这里**（`chunk_size` / `embed_mode`）。

---

## 四、检索层 —— RAG 的主体

**调用点**：`backend/app/retrieval/hybrid.py`

```
query ──┬─→ BM25 (jieba + 中文2-gram) ─→ top20 ─┐
        │                                          ├─→ RRF 融合 ─→ top31 ─→ Cross-Encoder ─→ top5
        └─→ embed(bge-m3) ─→ 向量暴力扫描 ─→ top20┘
```

### 每一步为什么存在（都有实测依据）

| 步骤 | 存在理由 | 实测数据 |
|---|---|---|
| BM25 | 低频专名（「位面饰品」「行迹」）词面匹配更稳 | A0 = 0.793（k=3，当前语料） |
| 向量 | 纯语义题需要 | 但单独用只有 0.828 且不随 k 增长 |
| RRF 融合 | 两路排名合并（用排名不用分数，因为量纲不同） | ⚠️ 当前语料下 **ΔMRR −0.0104、p=0.5664**（测不出增益，见下） |
| Rerank | 精排少量候选 | recall@3 0.828 → 0.897 |

> ⚠️ 上表的 BM25 / Rerank 两行此前写的是 2306 chunk 旧语料值
> （A0 = 0.897、recall@3 0.897→0.966），已按当前语料改正。
> **RRF 那一行更要注意**：旧值是「MRR +2.3pt」。2026-10-15 重跑确认
> （`data/eval/exp_A_rerun_sr_k3.json`）**ΔMRR = 0.0000、p = 1.0000**，
> sr / ys 两库完全等价、zzz 略负 —— **融合相对纯向量没有可测增益**。
> 别照抄「+2.3pt」。

### 用排名融合而不是分数加权

BM25 分数和余弦相似度不同量纲，直接加权相加会被其中一路的分数分布带偏。
RRF 只用排名：`score = Σ 1/(k + rank)`，k=60。

### 中文 2-gram 兜底

jieba 切词后，专名仍可能整词未命中。补一层字符 2-gram 能改善召回，
且**不引入新依赖**。副作用是索引变大，但实测可控。

### 版本感知

每个 chunk 带 `last_revision` / `last_revision_ts`，检索时可按时间戳过滤。
这是应对游戏版本更新的显式设计，不是把 wiki 当静态语料。

过滤条件走 `RetrievalConfig → store.build_where()`，两个 store 实现必须给出
**逐字相同**的 where 结构（`test_store_parity.py` 盯着），否则换 `VECTOR_BACKEND`
会静默改变召回结果 —— 而所有量化结论都建立在"换后端不影响指标"这个前提上。

入口有三处，缺一不可：`RetrieveRequest` 的字段、检索调试面板的「修订于」控件、
以及 `RetrievalConfig.as_dict()` —— 最后这个最容易漏。它同时喂给
`/api/retrieve` 的响应和 `run_eval.py` 落库的 config 快照，
手写白名单漏一个字段就意味着"这轮评测带没带版本过滤"在结果里看不出来。
所以它用 `dataclasses.asdict` 自动派生，并有断言强制与 dataclass 字段集相等。

---

## 五、存储层 —— 为什么是两个库

| 数据 | 存在哪 | 理由 |
|---|---|---|
| chunk 文本 + embedding | **LanceDB** | 向量检索要的就是这个 |
| 页面元数据 / infobox 数值 | **MySQL** | 结构化查询、聚合、追溯 |
| Agent 运行轨迹 | **MySQL** | `runs` / `tool_calls` |
| 评测结果 | **MySQL** | `eval_cases` / `eval_results`，能做回归对比 |

**数值走 SQL 精确查询，语义走向量。** 这是分工的判断依据，不是"两个库都存一份"。

### 为什么不用 MySQL 向量

社区版 8.4.5 没有 `VECTOR_DISTANCE()` 函数，Percona 虽然下放了但不是官方能力。
面试时能解释"为什么不用现成的向量列"比硬上更站得住。

### LanceDB 的坑

`table.create_index(metric="cosine")` 在 0.40 已废弃，它建的是 **IVF-PQ 近似索引**，
会把向量量化，导致同一句查询的 top-3 第三名变化。

**评测里悄悄改变排名不可接受** —— 指标差异就无法归因到检索策略本身。
所以默认 `VECTOR_ANN=false` 走暴力精确扫描（本地 8ms，不需要 ANN），
分数按 `cos = 1 - d/2` 还原。

### embedding 模型的批量上限（实测，影响所有入库）

SiliconFlow 对 embedding 请求有**单次请求总 token 数**的限制，
而且各模型上限不同。实测（用索引里最长的 64 条 chunk，平均 530 字符）：

| 模型 | batch=1 | batch=8 | batch=16 | batch=32 | 维度 |
|---|---|---|---|---|---|
| `BAAI/bge-m3` | OK | OK | OK | OK | 1024 |
| `BAAI/bge-large-zh-v1.5` | ✗ 超限 | ✗ 超限 | ✗ 超限 | ✗ 超限 | 1024 |
| `Pro/BAAI/bge-m3` | OK | OK | OK | OK | 1024 |
| `Qwen/Qwen3-Embedding-0.6B` | OK | OK | OK | OK | 1024 |

**踩坑过程**（值得记，因为它暴露了探测方法的问题）：
1. 第一次探测用两条**短**文本试，`bge-large-zh-v1.5` 返回 OK，判定"可用"
2. 实际重建索引（batch=16、真实 500 字 chunk）时炸 400 `The parameter is invalid`
3. 改用**真实长度的 chunk** 重测，它连 batch=1 都过不了

> **教训：探测必须用生产环境真实长度的输入。**
> 用短样本探容量，结论不可信 —— 这次差点让整个实验 D 得出错误结论。

`IngestConfig.embed_batch` 默认 16，对 `BAAI/bge-m3` 是安全的（实测 OK）。
换模型前先跑上面的批量探测，不要直接改模型名重建。

### 选型结论：为什么默认仍是 `bge-m3`（**依据是配套，不是质量**）

`scripts/analyze_model_compare.py` 在**同一批 29 条 case**（`sr`）上做了跨模型配对检验
（唯一变量 = embedding 模型，纯离线重算，不烧 API）。
下表取**当前语料**（3932 chunk）的三库逐库产物 `data/eval/model_compare_sr.json`：

| 档位 | MRR Δ(bge−Qwen) | p | Cohen's d | 延迟 Δ | p |
|---|---|---|---|---|---|
| A0（阴性对照） | +0.0000 | 1.0000 | 0.000 | −7.1ms | 0.2281 |
| A1 纯向量 | +0.0575 | 0.1834 | 0.142 | +10.8ms | 0.0946 |
| A2 融合 | −0.0115 | 0.6743 | −0.030 | **+28.1ms** | **<1e-10** |

> ⚠️ **这张表换过口径，引用时必须带语料版本。** 2306 chunk 旧语料下，
> A1 的延迟差曾是显著的（+37.7ms，p<1e-10），A2 的 MRR Δ 曾是 **+**0.0115。
> 补抓角色页（→3932 chunk）之后两者都变了。两套数字并列在
> `EVALUATION.md` 实验 D 章节；本节此前抄的是**旧表却没标注**，
> 于是同一件事在 ARCHITECTURE 与 README 里读起来方向相反。

结论有两个方向：

1. **检索质量无法区分** —— A2 的效应量 d=−0.030，配对检验连方向都测不出来
   （p=0.6743）。**不能说「bge 更强」**（这条结论曾用「Qwen 组内 A1→A2」的
   检验来支撑，检验对象就错了，详见 `EVALUATION.md` 实验 D 章节）。
2. **Qwen3-Embedding-0.6B 只在融合档（A2）显著更快**（+28.1ms，p<1e-10），
   效应量只有 d=0.052 —— **统计显著 ≠ 实际重要**。
   **A1 的延迟差已经不再显著**（p=0.0946），所以不能说「全面更快」；
   旧语料下那句「A1 快 23%、这是唯一显著差异」正是从上一版口径来的。

A0 五个检索指标 Δ 全为 0.0000（纯 BM25 不依赖 embedding），
证明语料/分块/BM25/评测集确实完全一致。

> **所以默认留在 bge-m3 的真实理由是**：reranker `bge-reranker-v2-m3` 本就是
> bge 系，配套适配；且 embedding 不动可让实验 E 对比 Chat 模型时少一个变量。
> **不是**「质量更好」—— 端到端里 embedding 只占几十毫秒，
> 换模型省下的 30ms 量级在 A3 完整链路（三库合计 867ms）里是噪声级。

---

## 六、Agent 层 —— 手搓 ReAct

**调用点**：`backend/app/agent/react.py`（约 1140 行）、`agent/tools.py`

### 四个工具

| 工具 | 什么时候用 | 调用点 |
|---|---|---|
| `search_kb` | 查游戏内数值、机制、角色信息 | 混合检索 |
| `get_page` | 检索片段不够，需要全文 | **现抓 wiki**（不读向量库），见下 |
| `calculator` | 检索到系数后要算数 | AST 白名单，**不用 eval** |
| `list_games` | 需要确认知识库范围 | 路由 |

### 四道安全阀

| 阀 | 防什么 |
|---|---|
| 步数上限 | 空转 |
| token 预算 | 空转但每步都输出很多 |
| 墙钟超时 | 工具卡死 |
| 单工具超时 | 单个工具拖垮整轮 |

### SearchGuard 熔断

知识库里没有的问题，Agent 会连续空转（实测 7 步 / 30317 token）。
SearchGuard 用「查询指纹去重 + 连续低置信度」把它压到 4 步 / 6514 token。

关键设计：**`search_kb` 的返回体第一行是机器可读状态**
（`status=ok hits=3 top_score=0.912 ...`）。
没有这层，Agent 只看得到一段文本，会把「召回了但都不相关」当成「再搜一次说不定有」。

---

## 七、LLM 调用点 —— 四个基座，其中 Chat 可切换

| 用途 | 基座 | 调用点 | 计费方 |
|---|---|---|---|
| Chat（工具决策 + 生成答案） | DeepSeek ⇄ MiniMax-M2.5 **可切换** | `llm/client.py: chat()` | 对应方 |
| Embedding | SiliconFlow `BAAI/bge-m3` | `llm/client.py: embed()` | SiliconFlow |
| Rerank | SiliconFlow `BAAI/bge-reranker-v2-m3` | `llm/client.py: rerank()` | SiliconFlow |
| Judge（评测打分） | SiliconFlow `Qwen/Qwen3-8B` | `llm/client.py: judge()` | SiliconFlow |

### Chat 基座切换：把 provider 选择做成数据，不是分支

```python
# llm/client.py
CHAT_BACKENDS: ClassVar[dict[str, str]] = {
    "deepseek": "deepseek_chat_model",
    "minimax":   "minimax_chat_model",
}
```

一条链路：`ChatRequest.backend` → `ReActAgent(backend=...)` → `chat(backend=...)`
→ 按名字取 key / base_url。前端下拉框的数据由 `/api/health` 的
`chat_backends` 下发，**只列出配好密钥的**，且不含任何密钥内容。

加一个新基座要改的地方：`.env` 加两个字段 → `CHAT_BACKENDS` 加一行。
Agent、API 路由、前端**都不用动**。这是这一层唯一的价值主张 ——
如果写成 `if backend == "minimax": ... else: ...`，每加一个基座就是一次全链路返工。

> **默认基座为什么是 DeepSeek**（实验 E 的结论）：
> 质量差异**三库均未达统计显著**（Δ=−0.70 / −0.16 / −0.38，
> p=0.0697 / 0.6738 / 0.2831）；成本差异三库一致 —— MiniMax 的
> prompt token ×1.55~2.85、延迟 ×2.56~3.65。此外 MiniMax 出现过一次
> `steps=0` 压根不调工具、直接把系统提示词复述成回答的失效。
> 所以选 DeepSeek 的依据是**延迟与稳定性，不是质量** ——
> 说「DeepSeek 更强」是超出数据支持的。详见 `EVALUATION.md` 实验 E。
>
> ⚠️ 本文件此前写的是**单库旧值**（judge p=0.1667、correctness p=0.0728、
> 延迟 4.08 倍 p=0.0011）。那是单库 15 条时期的结果，实验 E 已扩到三库逐库
> （`data/eval/exp_E_{sr,ys,zzz}.json`），上面才是当前值。

### 协议差异在客户端吸收，上层无感

两个基座都号称 OpenAI 兼容，实际有两处不一致。适配全部收敛在 `llm/client.py`：

| 差异 | 适配 | 不做会怎样 |
|---|---|---|
| MiniMax 的 content 带 `<think>…</think>` | `strip_think()` | DeepSeek 收到当普通文本，下一轮推理被带偏 |
| MiniMax 偶尔把工具调用写成正文 JSON | `parse_inline_tool_calls(content, allowed)` | assistant 消息不带 `tool_calls` → 下一轮 API 400（`role='tool' must follow a message with 'tool_calls'`） |

`allowed` 由调用方传入，也就是**本次请求声明过的工具名**。
这样兜底解析只可能认出"本来就允许调"的工具，不会凭空造出 schema 里不存在的函数名。

### LLMClient 必须是单例，且必须和检索器共用

`get_llm()` 返回进程级单例，`get_retriever()` 内部也用 `get_llm()` 而不是 `LLMClient()`。

这不是洁癖，有两个实测后果：

- **可观测性**：曾经各 new 一个，检索的 embedding 缓存待在另一个池子里，
  `/api/health` 报的命中率永远是 0% —— 侧栏那个数字是假的。
- **命中率**：同一条 query 在 `/retrieve` 命中过、到 `/chat` 又要重新 embedding 一次。

同理 `HybridRetriever` 必须是单例：BM25 索引是实例级缓存，
新建实例等于每个请求重付 1.8s 建索引（实测 `1820→1778→1799ms` vs 单例 `3325→20→16ms`）。

### Judge 必须独立基座

**严禁用被测模型自评** —— self-preference bias 会让所有配置看起来都好。
成对比较时正反各跑一次，抵消位置偏好。

### 关于 KV cache：不用做任何事，它已经在生效

DeepSeek 的上下文缓存在 `usage.prompt_cache_hit_tokens` 里直接可见，
**不需要配置**。实测（`scripts/probe_kv_cache.py`，模拟 Agent 每轮追加 tool 消息）：

| 轮 | prompt tokens | cache hit | 命中率 | 墙钟 |
|---|---|---|---|---|
| 1 | 914 | 768 | 84.0% | 1108ms |
| 2 | 977 | 768 | 78.6% | 1053ms |
| 3 | 1040 | 896 | 86.2% | 825ms |

（`data/eval/probe_kv_cache.json`，完整 5 轮在产物里。）

命中率稳定在 78%–88%，前提是**消息前缀稳定** —— 这一点在本项目天然满足：
Agent 每轮只追加 tool 消息，从不修改历史。

> ⚠️ **两个口径上的坑，引用这张表时别踩**：
>
> 1. 探针的 system 前缀是一段**固定字符串**（`filler * 60`）。只要它在此前
>    任何一次调用里出现过，第 1 轮就会被命中 —— 这份产物第 1 轮已有 768
>    token 命中。所以「第 1 轮必定 0 命中」这个说法不成立。
> 2. **墙钟没有量级上的收益**：5 轮在 651–1191ms 之间波动，看不出稳定下降。
>    本文件此前写的是「墙钟降约 50%」（配 1355 / 808 / 674ms 三个数），
>    这三个数在当前产物里找不到出处，已按产物改写。

**但它省的是费用，不是延迟。** 实测同一模型下首 token 延迟稳定在 100–350ms，
总耗时主要由生成 token 数决定，与上下文长度关系不大 —— 这一条结论不受上面
两个口径问题影响，反而被「命中率很高、墙钟却没降」进一步印证。

### 延迟优化的真正杠杆：流式

| 方式 | 耗时 | 用户体验 |
|---|---|---|
| 非流式 `/api/chat` | 3410ms | 全程干等 |
| 流式 `/api/chat/stream` | 首包 **860ms**，全部 3147ms | 860ms 就有反馈 |

**感知延迟 −75%，总耗时几乎不变。**

> ⚠️ **措辞**：这里**不是「第一个字」**。上游没有 token 级流式，860ms 是
> **首个 SSE 数据包**到达的时间。本文件此前写作「首字 721ms，全部 3927ms，
> 感知延迟 −85%」，三个数都不在产物里 —— 当前源是
> `data/eval/probe_stream_latency.json`（`chat_non_stream_ms` 3410.3 /
> `ttft_ms` 860 / `total_ms` 3147），README 也已就这条措辞做过更正。

所以优化优先级是：流式 > 跳过重排 > 预热 > 缓存 > KV cache（无需动作）。
详见 `EVALUATION.md` 的延迟章节。

### 延迟优化的调用点

| 优化 | 位置 | 作用 |
|---|---|---|
| embedding 缓存（LRU 512） | `llm/client.py: _embed_cache` | 省掉重复 query 的远程调用（实测单条 147ms） |
| rerank 缓存（LRU 256，键含完整候选集） | `llm/client.py: _rerank_cache` | 省掉重复 (query, 候选集) 的重排（实测 449ms） |
| 启动预热 BM25 | `retrieval/hybrid.py: warmup()` | 把 2849ms 冷启动移出首个请求 |
| rerank 候选裁剪 | `retrieval/hybrid.py: _rerank()` | 送 31 条 → 20 条，省约 100ms |

**缓存不影响正确性**：两者都是确定性函数，同输入必然同输出。
但**评测脚本必须在每个 label 前 `clear_caches()`**（`run_eval.py` 已接），
否则后跑的 label 会白捡前一个 label 的缓存，延迟列就不是同条件下的可比数字了。

#### 为什么缓存要分两条路径

`embed()` 里有两条分支，因为两种场景的最优策略相反：

| 输入量 | 走哪条 | 原因 |
|---|---|---|
| ≤ 8 条（检索时的 1 条 query） | 逐条查缓存，未命中的攒成一批 | 命中就完全不发 HTTP |
| > 8 条（入库时的几千条） | `embed_batch()` 串行分批 | 逐条串行会退化成上千次 HTTP，比 16 条/批慢一个数量级 |

**这是我写错过一次的地方** —— 第一版统一走逐条缓存路径，
入库 3932 条变成 3932 次串行请求。分场景之后才既快又不退化。

#### rerank 裁剪是有损的，所以必须验证

我原本假设「top-20 之外的进不了 top-5，裁剪无损」。**实测 15 组对照里 12 组排名变了**，
假设被推翻。于是改用评测集验证 gold 召回是否受损：

| A3 配置 | page_hit | recall@3 | MRR | NDCG | 延迟 |
|---|---|---|---|---|---|
| `rerank_candidates=0`（全量） | 0.966 | 0.966 | 0.966 | 0.966 | 703ms |
| `rerank_candidates=20`（裁剪） | 0.966 | 0.966 | 0.966 | 0.966 | **599ms** |

四项指标逐位相同，延迟降 14.8%。排名变化发生在 gold 页面之外的边缘位置。
传 `rerank_candidates=0` 可随时还原成对照组。

> ⚠️ **这张表是旧语料口径，而且没有出处记录。** 它来自
> `exp_A3_full_k3.json` / `exp_A3_cut20_k3.json`，两个文件都**不带 `_meta`**，
> 追溯不到语料版本；0.966 是 2306 chunk 时期的值。
> 当前语料（3932 chunk）下 sr 的 A3 是 **0.897**，但**没有重跑过裁剪对照** ——
> 所以「四项逐位相同」这个结论目前只建立在旧语料上。
> 要在当前语料上验证裁剪无损，需要重跑一次
> `python scripts/run_eval.py --rerank-candidates 0` 与 `--rerank-candidates 20`。

> **这 14.8% 是冷缓存口径**：`run_eval.py` 在每个 label 前 `llm.clear_caches()`，
> 所以 703 / 599ms 测的是"缓存全空时的一次真实远程重排"。
> 在两组都预热过的热缓存下重测（`bench_latency.py`），
> 中位数 31.7ms → 31.1ms，只有 **-2.0%**
> （`data/eval/bench_latency_rerank_cut.json`）。
>
> 这两个数字都对，取决于缓存状态 —— 裁剪省的是那次真的调用远程 rerank 的钱和时间。
> 引用时必须带上口径，否则"省 14.8%"和"只省 2%"会像自相矛盾。

---

## 八、评测层 —— 指标怎么算才可信

**调用点**：`backend/app/eval/{metrics,judge,cases}.py`

| 指标 | 定义 | 为什么这么算 |
|---|---|---|
| recall@k | gold 页面是否出现在 top-k | 召回能力 |
| MRR | 第一个正确结果的倒数排名 | 排序质量 |
| NDCG | 带位置折损的召回 | 兼顾召回和排序 |
| page_hit | gold 页面是否在结果里 | 比 recall 更贴近"用户看到没有" |

### 六条不可妥协的规则

1. **指标在页面级排名上算**，不绑 chunk_id —— 否则重建索引等于评测集失效。
2. **无 gold_pages 的 case 不计入检索均值** —— 否则算术题以 0 分拉低所有配置。
3. **Judge 独立基座** —— 防 self-preference bias。
4. **每个 label 前清缓存** —— 保证延迟可比。
5. **Agent 与 Judge 两侧的失败都不进能力指标。**
6. **均值与配对检验必须落在同一批样本上。**

第 5、6 条是实测踩出来的（2026-10-13 实验 E）：

- `is_kept()` 早先只判 `error`，而 `error` **只记 Agent 侧异常**。Judge 自己
  超时时 Agent 其实好好跑完了，`error=None`，但 `score()` 已把 0 分兜底写进
  `judge_score` —— 于是这个 0 被当成「模型答得一塌糊涂」进了均值。
  实测那条 `sr-008` 的答案实为一段正常的攻击力计算推导，
  **0 分纯属 Judge 一次 `APITimeoutError`**，却把该档从 4.5714 压到 4.2667。
  判据是 `judge_reason` 的 `judge_error:` 前缀；注意空回答走「空回答」、
  前缀不同，所以不会误伤真实低分与空回答。
- `paired_scores` 当时**硬编码 `not r.get("error")`、绕过了 `is_kept`**，
  于是 aggregate 按 n=14 算、配对检验按 n=15 算。两份数字来自不同样本集，
  Δ 与 p 却被当成同一份数据的结论。这类不自洽**事后极难发现**，
  因为每个数字单独看都「正常」。现在两者共用同一个谓词与同一个 `exclude`。

同类的还有第三条铁律：**配对检验的 Δ 必须与直觉同向**（改进为正）。
曾把 `(基线, 改进档)` 传给 `paired_test`，打出 `Δ=-0.092`，
而改进档实际高 9.2 个点 —— 数字没算错，方向要靠读者自己翻。

违反其中任何一条，得到的数字都是不能写进简历的。

---

## 改代码前先看这里

| 想改什么 | 改哪个文件 | 会连带影响 |
|---|---|---|
| 抓取频率、批量大小 | `ingest/wiki_client.py` | WAF 拦截概率 |
| 清洗规则 | `ingest/template_parser.py` | **必须重建索引**，否则新旧文本不一致 |
| 分块粒度 | `ingest/chunker.py` | **必须重建索引**，且实验要重跑 |
| 检索策略、融合方式 | `retrieval/hybrid.py` | 实验 A 全部数字 |
| 向量库后端 | `store/vector_store.py` + `.env` | 切 `VECTOR_BACKEND=numpy` 前**必须先跑 `rebuild_index.py`** |
| Agent 行为、安全阀 | `agent/react.py` | 步数/token 指标 |
| 工具定义 | `agent/tools.py` | schema 必须符合 DeepSeek 的 `{"type": ...}` 包装要求 |
| 评测集 | `eval/cases.py` | **所有历史实验数字失效，必须重跑** |
| 评测聚合口径 | `scripts/exp_common.py` 的 `is_kept` / `aggregate_rows` | **所有历史实验的 n 与均值口径都可能变**，需重算或标注版本 |
| 评测脚本的 client 生命周期 | `scripts/exp_*.py` 的 `chat_with` | 异常路径下连接池泄漏会**累积成静默挂死**，`close()` 必须在 `finally` |