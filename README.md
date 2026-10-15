# kb-agent · 游戏内容知识库问答 Agent

> **一句话**：三库 1401 页 / 10843 chunk 的垂直知识库问答系统。手写 ReAct Agent +
> 混合检索（BM25 ∥ 向量 → RRF → Cross-Encoder 重排），5 组单变量消融实验，
> 每个数字都能追溯到产生它的脚本与 git commit。

![问答界面](docs/screenshot-chat.png)

| 维度 | 结果 | 出处 |
|---|---|---|
| 检索策略消融（**三库逐库**，A0→A3，top_k=3，n=29/9/10） | recall@3：**sr 0.793→0.897** / ys 0.889→**1.000** / zzz 1.000→**1.000**。⚠️ **重排增益只在 sr 成立** —— ys/zzz 在 A1（纯向量）就已打满，A2/A3 没有可提升空间，三库**不可合并成一个均值**。sr 补抓 368 页角色页后指标整体下移，退化项全部来自攻略页挤占 | [`data/eval/exp_A_all_k3.json`](data/eval) · [`EVALUATION.md`](EVALUATION.md) |
| 分块 / embed_mode 消融（**三库逐库** B0–B3，n=29/9/10） | MRR title→text：**sr 0.736→0.506（−23.0pt，p=0.0001）** / ys 0.833→0.722 / zzz 0.950→0.950。⚠️ 方向三库一致但**幅度差很大**，只有 sr 达到显著；chunk_size 在 500/800/1200 下 **sr A3 指标完全相同**（差异被重排吸收） | [`data/eval/exp_B{0..3}_{sr,ys,zzz}_k3.json`](data/eval) |
| Agent 能力消融（生成侧，Judge；**三库逐库**，n=32/12/12，judge 均值为剔除超时后的 n_ok 口径） | sr：无工具 **2.67** → `search_kb` **4.72** → 全套工具 4.59；C0→C2 Δ=+2.04（p<1e-10, d=2.16）。三库 C0→C2 **全部极显著**，d=2.16/3.72/2.78 | [`data/eval/exp_C_{sr,ys,zzz}.json`](data/eval) · [`EVALUATION.md`](EVALUATION.md) |
| Embedding 模型对比（**三库逐库** D，n=29/9/10） | **保留 bge-m3**，依据是 reranker 配套而非质量：A2 融合档跨模型配对 **p=0.6743 无法区分**，且三库方向互相矛盾（sr Qwen 略优 0.810→0.822 / ys 打平 1.000 / zzz bge 优 0.850→1.000）。Qwen 在融合档显著更快（sr −28ms，p<1e-10） | [`data/eval/exp_D_*_k3.json`](data/eval) · [`EVALUATION.md`](EVALUATION.md) |
| 假前提题（幻觉测试） | 无检索 3.83 → 有检索 **5.00**（模型学会"明确说不知道"） | 同上 |
| Chat 基座对比（**三库逐库**，n=15/12/12，DeepSeek⇄MiniMax，judge 为 n_ok 口径） | ⚠️ **三库均未达显著**（Δ=−0.70 / −0.16 / −0.38，p=0.070/0.674/0.283），**不敢说谁更强**。成本三库一致：MiniMax prompt token ×1.55~2.85、延迟 ×2.56~3.65。两者失败模式**互补** | [`data/eval/exp_E_{sr,ys,zzz}.json`](data/eval) · [`EVALUATION.md`](EVALUATION.md) |
| 代价 | 挂知识库 = prompt token **×20.2**、延迟 3.5 倍（串行实测口径）；C2 比 C1 多花 **23%** token，judge 只涨 0.14 且 p=0.16 未达显著 | [`docs/cost_latency_quality.md`](docs/cost_latency_quality.md) |
| 并发（真实问题压测） | 并发 4 吞吐 **×3.40**（4.10 req/s），1/4/8 三档 top-5 结果**逐条一致** | [`docs/concurrency.md`](docs/concurrency.md) |
| 流式 | 首个 SSE 事件 **860ms** vs 非流式全量 3410ms，总耗时仅差 264ms。⚠ **不是「第一个字」**：上游无 token 级流式，是拿完整答案后切片推送；且为单次测量 | [`data/eval/probe_stream_latency.json`](data/eval) |
| 失败模式 | 4 类，含「答案全对但引用了不存在的页面」。**已实现生成后引用校验**：L1 剥掉无检索证据的页面引用、L2 记录无出处的数值（不改文本）；语义级（L3）未做 | [`docs/failure_analysis.md`](docs/failure_analysis.md) |
| 故障注入 | 23 条端到端注入测试（工具异常/超时/基础设施失败/空结果/低分/重复查询/模型失败/流式），熔断路径经双向验证 | `backend/tests/test_fault_injection.py` |

> 📄 **「我做了一个 RAG 知识库，区别在哪、还差什么、面试怎么讲」**
> 见 [`docs/comparison_with_rag.md`](docs/comparison_with_rag.md) ——
> 里面有五条实打实的差异、**一份比亮点还长的差距清单**，以及两种面试讲法
> （哪种会被追问穿、哪种能接住追问）。

## 📌 所有数字可复现

`data/eval/` 下每个结果文件都带 `_meta` 出处块，记录**产生它的脚本名、UTC 时间、
git commit、Python 版本、平台，以及当时的测量条件**（并发度、缓存冷热、top_k、
`rerank_candidates` 等）。

```json
"_meta": {
  "script": "bench_latency.py",
  "generated_at_utc": "2026-10-13T07:55:43+00:00",
  "git_commit": "fd18a4e",
  "context": { "variable": "rerank_candidates", "warm_cache": true, "note": "两组均已预热…" }
}
```

这意味着：**任何人都能判断某个结论在什么条件下成立**，
而不是只能相信屏幕上的一行数字。实测中它直接抓出过两个真问题 ——
延迟数字没标缓存口径（冷缓存 −14.8% vs 热缓存 −2.0%，差 7 倍），
以及评测 Runner 的并发顺序效应（同一份代码两次跑出相反的符号）。
详见 [`docs/experiments.md`](docs/experiments.md) 的「四个会把人带沟里的坑」。

## 为什么选游戏内容做知识库

不是因为游戏"简单"，而是因为它天然具备普通 RAG 项目最缺的两样东西：

1. **大量低频专有名词**（位面饰品、行迹、命途、音擎、驱动盘）——纯向量检索一定翻车，能放大混合检索 + 重排的收益；
2. **有时间维度的语料**（wiki 每次修订都是新版本）——能演示"通用 RAG 在会过期的知识上会失效"，以及版本感知检索怎么解。

主知识库：崩坏:星穹铁道（`sr`，706 页 → 3932 chunk）。次知识库：绝区零（`zzz`，260 页 → 1498 chunk）。第三库：原神（`ys`，434 页 → 5413 chunk）。三库合计 1400 页 → 10843 chunk，全部只用生产索引统计。

> 语料扩充记录：最初按「资料型」分类抓取（遗器 / 光锥 / 任务 / NPC 等 26 个分类），
> 结果 **sr 与 ys 完全没收录角色页**，被问「三月七的终结技」只能回答"知识库未收录"。
> 后补抓 `sr 角色(102) / sr 角色攻略(87) / ys 角色(134) / ys 角色攻略(45)` 共 368 页，
> WAF 拦截 0 次，补抓后必须重启服务（`HybridRetriever` 的全量语料缓存在进程内存里，
> 手动跑 `crawl.py` 不会触发 `reset_retriever()`）。

> 关于 n 的诚实说明：检索类实验 n=29、生成类实验 n=15。样本量偏小，
> 所以正文中出现的显著性一律标注口径（`p<0.001` 为正态近似而非 t 分布），
> 且不把 p 的具体位数写进结论。**已知的能力边界见
> [`docs/failure_analysis.md`](docs/failure_analysis.md)** ——包括一条 A3 完整链路
> 也召不回来的题，和一类"答案正确但引用被编造"的幻觉。

## 文档

| 文件 | 内容 |
|---|---|
| `README.md` | 怎么跑起来、架构要点、踩坑记录 |
| `ARCHITECTURE.md` | **每个工具用在哪、为什么用在那、改了会影响什么** —— 改代码前先看这个 |
| `docs/glossary.md` | **专有名词释义** —— RAG / BM25 / RRF / rerank / NDCG 等，每个都写清「是什么、用在哪、实测多少」 |
| `docs/rerank.md` | **重排（Rerank）深入说明** —— 一篇文档讲透，含实测效果与代价 |
| `docs/concurrency.md` | **并发实测** —— 能开几个、同时跑会怎样、共享状态有哪些坑、怎么调 |
| `docs/injection_guard.md` | **间接 Prompt 注入** —— 威胁模型、为什么"命中就拦"是错的、规则集实测误报率 |
| `docs/fallback_chain.md` | **LLM 降级链** —— 哪些故障值得换模型、哪些换也没用、降级为什么必须显形 |
| `docs/delta_sync.md` | **版本更新怎么做** —— 增量同步的实测口径、chunk 级脏数据陷阱、为什么先落内容指纹再动增量 |
| `docs/experiments.md` | **五个实验各自想证明什么、结果里每列是什么意思** —— 忘了当初为什么做实验就看这个 |
| `EVALUATION.md` | 实验数据与结论，含方法论教训 |

---

## 它到底做了什么

```
用户提问
   ↓
关键词路由（确定性，不用 LLM）
   ↓
┌──────────── ReAct 循环（手写，约 380 行 async）────────────┐
│  模型决策 → 工具调用 → 结果回填 → 再决策 … 直到作答或触顶安全阀 │
│  + 熔断：查询指纹去重、连续低置信度直接回填「未收录」          │
└────────────────────────────────────────────────────────────┘
   ↓ 工具
 search_kb   混合检索（BM25 + 向量 → RRF 融合 → Cross-Encoder 重排）
 get_page    取 wiki 页面全文（片段不够时补全）
 calculator  游戏数值计算（AST 白名单，不用 eval）
 list_games  知识库路由
   ↓
生成回答 + 引用出处 + 完整调用轨迹（steps / 延迟 / token / 失败次数）
   ↓
前端：聊天气泡 · 工具调用时间线 · 引用可点开 · 检索调试面板
```

界面不是"套个壳"：检索调试面板可以直接切 A0/A1/A2/A3 开关
（BM25 / 向量 / 重排 三个勾选框），当场看到 top-5 换一批 ——
面试时不用念 PPT，改一个开关就把实验结论演出来。
其中「重排」是提升最大也最贵的一环，它到底是什么见 `docs/rerank.md`。

**Chat 基座可以在前端一键切换**（DeepSeek ⇄ MiniMax-M2.5）。
这是实验 E 的结论现场可演的形式：同一条问题、同一个检索链路，
两个基座答完并排显示，每轮都标出实际是哪个模型答的。

实现上「加一个新基座」是改一行数据而不是改一处分支 ——
`LLMClient.CHAT_BACKENDS = {"deepseek": "deepseek_chat_model", "minimax": "minimax_chat_model"}`，
前端下拉框的数据由 `/api/health` 下发。新增 provider 不需要改 Agent、API 或前端。
两个基座的协议差异（`<think>` 标签、内联工具调用）都在 `llm/client.py` 里适配掉，
上层永远看不到。

---

## 知识库只有三个游戏，那用户问别的游戏怎么办

### 先说结论：不让模型联网，而是让它知道边界在哪

给 Agent 加联网搜索是最差的一个选项，理由有三：

1. **会毁掉全部已有实验**。A/B/D 的检索链路全部冻结在固定语料上，
   掺入联网结果后同一问题两次跑可能给出不同来源，指标不再可复现。
2. **准确性无保障**。游戏 Wiki 与攻略站对同一数值经常打架，
   模型只能选一个，而且用户不会知道它选的是哪个。
3. **增加不可控延迟与凭据依赖**。实测联网检索普遍 +500ms~2s。

真正要做的是**区分两种"未命中"**，这曾经是个真 Bug：

```python
# 修复前：两个都不匹配就默认 sr
# → 「原神甘雨的突破材料」被当成星铁问题去搜星铁库，
#   搜不到后报"知识库未收录" —— 一个建立在错误前提上的结论
def route_game(question, covered) -> str | None: ...
```

现在 `route_game` 返回三态，评测与前端都依赖这个区分：

| 情形 | 返回 | 系统行为 | 实测 |
|---|---|---|---|
| 提到已收录游戏 | 该 slug | 正常检索 | 原神 5 步 / 14 引用 |
| 提到未收录游戏 | 该 slug | **无工具、单轮拒答** | 崩坏3 **0 步 / 1.3s** |
| 完全没提游戏 | `None` | 用上下文兜底 | 正常检索 |

拒答路径走 `stop_reason = out_of_scope`，**不消耗任何检索预算**：

> 「**崩坏3**目前不在我的知识库范围内，所以我没法给你关于这个游戏的配装建议。
> 我当前收录的游戏是**崩坏：星穹铁道**…」

### 想真正支持更多游戏：一条命令

架构从一开始就支持。`WikiClient(game=...)` 是通用的，加一个游戏 = 跑一次抓取：

```powershell
cd backend
..\.venv\Scripts\python.exe scripts\crawl_all.py --game ys
```

实测原神入库（两轮，第二轮扩到 11 个分类）：

```
第一轮（5 个分类）： 83 页 /  236 chunk / 0 失败 / WAF 0 次   （65 秒）
第二轮（11 个分类）：174 页 / 1057 chunk / 0 失败 / WAF 0 次  （约 4 分钟，中途 2 次 567 已被退避挡下）
第三轮（角色 + 角色攻略）：179 页 / 4120 chunk / 0 失败 / WAF 0 次（约 2.5 分钟）
累计：435 页 → 5413 chunk
```

> ⚠️ **分类要按"用户会问什么"选，不是按"哪种资料看起来整齐"选。**
> 前两轮抓的全是任务 / 遗物 / 七天神像这类资料型页面，**一个角色页都没抓**，
> 结果被问「三月七的终结技」只能回答"知识库未收录"。
> 第三轮补上 `角色(134)` 与 `角色攻略(45)` 才补齐。
> 判断某个分类该不该抓，最快的办法是直接问一句用户最可能问的问题，看检索有没有命中。
>
> 补抓后**必须重启服务**：`HybridRetriever` 把全量语料缓存在进程内存里，
> 而 `reset_retriever()` 只在 `/ingest` 之后触发，手动跑 `crawl.py` 不会走到。
> 不重启的话 API 会继续用旧语料，看起来就像"补了数据也没用"。

限流参数用的是实测过的 **batch=10 + interval=6s**（518 页 0 拦截的那组），
后来补抓角色页又用同一组参数跑了 **368 页 / 5722 chunk，依然 0 次拦截** ——
说明这组参数不是撞运气。
新建 `scripts/crawl_all.py` 就是为了把「多分类串起来 + 失败不中断 + 限流统计」这件事固化下来。
它会读 `data/checkpoints/{game}_{分类}.json` 跳过已抓标题，所以第二轮只补增量、不会重跑。

为了让「加游戏」不需要改代码，配置从**两个固定字段**改成了 N 游戏：

```python
# slug → (显示名, 该游戏的专属术语)
_TABLE = {
    "sr":  ("崩坏：星穹铁道", "星铁叫「光锥/命途/位面饰品/遗器」"),
    "zzz": ("绝区零",       "绝区零叫「音擎/驱动盘/邦布」"),
    "ys":  ("原神",         "原神叫「圣遗物/天赋/神瞳/命座」"),
}
```

之前 `game_primary` / `game_secondary` 写死两个，于是「有哪些游戏」这件事
在 config、工具校验、`list_games`、预热循环、前端下拉框**五处各写一遍**。
现在只有 `_TABLE` 这一处定义游戏集合，前端下拉框也改成读 `/api/health` 的 `games`。
加第 N+1 个库 = 加一个 `GAME_XXX` 字段 + 在 `_TABLE` 里加一行，其余自动跟上。

> **术语也放在 `_TABLE` 里**，因为它要进 Agent 的系统提示词。放在别处就会变成第二处定义 ——
> 加第三个库时正是漏掉了 prompt 层，模型拿着「你只覆盖星铁和绝区零」的提示词
> 把原神问题自己拒了（`steps=0`）。现在 `smoke_refactor.py` 有断言守着这条。

> 前端实测（只统计生产索引）：三个库分别显示 2,293 / 1,517 / 1,293，合计 **5,103** 片段。
> 实验 B/D 另建的 7,693 片段变体索引单独标注，不计入合计 ——
> 早期版本把变体库也加进来，界面显示"星铁 9,986"，虚高 4.4 倍。

---

## 外观：深浅色 + 自定义背景

- **深浅色**：`[data-theme]` 切换一套 CSS 变量。选择持久化到 localStorage，
  且 `index.html` 里有一段**同步内联脚本**在 React 挂载前定好主题，避免首屏白闪。
- **自定义背景**：设置面板里上传图片，**压缩到最长边 1600px 的 JPEG 后**才存本地，
  不上传任何服务器。1.2MB 的原图实测压到 **56KB**。
  压完仍超过 1.5MB 就只在本会话生效，并**明确告知用户**而不是静默失败。
- **遮罩浓度可调**：背景图永远叠一层遮罩，浓度 0~90% 可调。
  实测默认 0.35 时浅色文字压在亮色图上几乎不可读，已把默认值提到 **0.55**
  并给直接压在背景上的文字加了独立承载面 —— 遮罩的职责是保证对比度，不是展示图片。
- **响应式**：1180px 收窄侧栏；860px 以下侧栏变抽屉（默认收起，点 ☰ 或遮罩关闭）；
  设置面板在移动端改为**底部弹出**，避开虚拟键盘。已实测 390×844。

| 浅色主题 | 外观设置（主题 + 自定义背景） | 移动端 390×844 |
|---|---|---|
| ![浅色主题](docs/screenshot-light.png) | ![外观设置](docs/screenshot-bg.png) | ![移动端](docs/screenshot-mobile.png) |

检索调试台（不开 Agent，直接对比 A0 仅 BM25 / A1 仅向量 / A2 融合 / A3 加重排）：

![检索调试](docs/screenshot-retrieve.png)

---

## 快速开始

```powershell
# 1. 依赖（Python 3.13）
cd kb-agent
D:\Python3135\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

# 2. 配置密钥（.env 从 .env.example 复制，.env 已在 .gitignore 中）
#    DEEPSEEK_API_KEY / SILICONFLOW_API_KEY / MYSQL_PASSWORD 必填，MINIMAX_API_KEY 用于实验 E

# 3. 入库（星铁主库 + 绝区零副库）
cd backend
..\.venv\Scripts\python.exe scripts\bulk_ingest.py --game sr --interval 6 --batch 10
..\.venv\Scripts\python.exe scripts\bulk_ingest.py --game zzz --interval 6 --batch 10

# 4. 起后端 + 前端（推荐：一条命令）
cd ..
.venv\Scripts\python.exe run_all.py
#    → 后端 http://127.0.0.1:8000/docs   前端 http://localhost:5173

# 5. 自检
.venv\Scripts\python.exe backend\scripts\smoke.py
```

> ⚠️ **不要用 `uvicorn app.main:app`**：那条命令依赖「当前工作目录是 `backend/`」，
> 在项目根或其他目录会直接 `ModuleNotFoundError: app`。用 `run_server.py`，
> 它自己把 `backend/` 加进 `sys.path`，任何工作目录下都能跑。

PyCharm 里打开项目后，右上角运行配置已经配好 7 个，免填：
`0-一键启动`（日常用这个）/ `0b-只启动后端` / `0c-只启动前端` /
`3-冒烟自检` / `4-入库 批量抓取` / `5-评测 实验A` / `6-回填 MySQL`。

> 前端用 `npm` 而不是 `pnpm`：pnpm 11 的构建白名单（`onlyBuiltDependencies`）
> 在这个环境下没生效，esbuild 的二进制拉不下来会直接导致 vite 起不来。
> Windows 上必须用 `npm.cmd` 绕过 PowerShell 执行策略。

---

## 架构要点

| 模块 | 位置 | 关键设计 |
|---|---|---|
| 抓取 | `app/ingest/wiki_client.py` | EdgeOne 限流感知：567 状态码直接判为拦截而非 404；一次请求批量取 10 页；Cookie 会话保持；指数退避 |
| 清洗 | `app/ingest/template_parser.py` | 模板**展平**而非删除（见下方踩坑 3） |
| 分块 | `app/ingest/chunker.py` | 先按标题切逻辑段，再按句边界贪心打包，overlap 以整句回退 |
| 向量 | `app/store/vector_store.py` | collection 名 = `kb_{game}_{model}_{variant}`，实验隔离靠它 |
| 检索 | `app/retrieval/hybrid.py` | BM25（jieba + 中文 2-gram）∥ 向量 → RRF → 可选重排 |
| Agent | `app/agent/react.py` | 步数 / token / 墙钟 / 单工具 四道安全阀 |
| 评测 | `app/eval/` | 检索指标与生成指标分离；Judge 独立基座 + 位置偏置对消 |

### 存储为什么这样分

- **LanceDB（默认）**：文本 chunk + embedding + 元数据，暴力精确扫描。
- **MySQL**：页面元数据、infobox 结构化数值、Agent 运行轨迹、评测结果。

**为什么放弃 ChromaDB**（实测结论，不是偏好）：
本机是 Windows + Python 3.13。chromadb 只有 1.5.x 提供 `cp39-abi3` 预编译轮子
（0.5.x / 0.6.x 的 `chroma-hnswlib` 没有 cp313 轮子，本机无 MSVC 编译不了）；
但 1.5.9 写入后磁盘上只生成 `index_metadata.pickle`，HNSW 段目录始终不落盘，
换进程读取必然报 `Error loading hnsw index`，且可稳定复现。

**为什么选 LanceDB 而不是继续用自实现索引**：两者都实测可用，
LanceDB 是成熟向量库（简历上说得清），自实现的作为 fallback 保留在 `vector_store.py`。
切换由 `VECTOR_BACKEND` 控制，上层代码零改动。

**LanceDB 里有个必须记住的坑**：`table.create_index(metric="cosine")` 在 0.40 已废弃，
它建的是 **IVF-PQ 近似索引**，会把向量量化。实测同一句查询，暴力扫描和近似索引的
top-3 第三名不同。评测里悄悄改变排名是不可接受的 —— 指标差异就归因不到检索策略头上了。
所以本后端**只有暴力精确扫描这一条路径**（没有 ANN 开关可配）；分数按
`cos = 1 - d/2`（平方 L2 换算）还原。

**为什么不用 MySQL 向量**：社区版 8.4.5 没有 `VECTOR_DISTANCE()` 函数，
Percona 虽然下放了但不是官方能力，面试时解释"为什么不用现成的向量列"比硬上更站得住。

### 为什么要版本感知的 chunk

每个 chunk 带 `last_revision`（MediaWiki 修订号）和 `last_revision_ts`（ISO 时间戳）。
检索时可以按 `last_revision_ts` 过滤——"只要当前版本之后才更新的内容"。
这是应对游戏版本更新的显式设计，而不是把 wiki 当成永远不变的静态语料。

**怎么演示**：检索调试面板的「修订于」日期控件，或
`POST /api/retrieve {"query": "...", "min_revision_ts": "2026-06-01"}`。
接受 `YYYY-MM-DD`，后端补成当天零点再比大小。
实测（`scripts/probe_version_entry.py`，真实语料 8/8 通过）：阈值设到 `2099` 命中 0 条，
设到 `2020` 与无过滤逐条一致，设到 `2026-08-02` 后召回整体换成更晚的批次。

> 这个能力曾经**只存在于代码里**——检索层一路通到两个 store 的 `build_where()`，
> 但 HTTP 请求模型没有字段、前端没有控件，只能写 Python 才用得上。
> 补入口时还牵出一个更隐蔽的坑：`RetrievalConfig.as_dict()` 是手写白名单，
> 加了新字段却没同步，导致"带版本过滤跑的评测"落库后看不出带过滤。
> 现在 `as_dict()` 从 dataclass 自动派生，并有跨层断言守住。
> 见 `backend/tests/test_version_filter_api.py`。

---

## 实验：单变量阶梯消融

**每次只改一个组件。** 差值才可归因。

| 实验 | 变量 | 状态 |
|---|---|---|
| A | 检索策略 | ✅ 已完成。A0 仅 BM25 → A1 仅向量 → A2 RRF 融合 → A3 +重排（n=29） |
| B | 分块 | ✅ 已完成。chunk 500/800/1200 × embed_mode title/text |
| C | Agent 能力 | ✅ 已完成。C0 无工具 → C1 单工具 → C2 多工具循环，**三库逐库**（32/12/12 条） |
| D | Embedding | ✅ 已完成。`bge-m3` vs `Qwen/Qwen3-Embedding-0.6B`（关重排+双侧归一化+独立库） |
| E | Chat 模型 | ✅ 已完成。DeepSeek vs MiniMax，**三库逐库**（15/12/12 条） |
| F | 披露层 | ✅ 已完成。页面修订时间开/关消融（n=8 溯源题） |

运行：

```powershell
cd kb-agent\backend
..\.venv\Scripts\python.exe scripts\run_eval.py --experiment A --top-k 3
..\.venv\Scripts\python.exe scripts\exp_b.py --steps B1,B2,B3                  # 实验 B
..\.venv\Scripts\python.exe scripts\exp_d.py --models Qwen/Qwen3-Embedding-0.6B # 实验 D
..\.venv\Scripts\python.exe scripts\exp_c.py --game all --labels C0            # 实验 C 的 C0（见下方警告）
..\.venv\Scripts\python.exe scripts\exp_f.py --game sr,ys,zzz                  # 实验 F
..\.venv\Scripts\python.exe scripts\merge_exp_c.py                             # C 逐库拆分 + 配对
..\.venv\Scripts\python.exe scripts\merge_exp_e.py                             # E 逐库合并 + 配对
..\.venv\Scripts\python.exe scripts\probe_kv_cache.py                          # KV cache 实测
..\.venv\Scripts\python.exe scripts\probe_stream_latency.py                    # 感知延迟实测
```

> ⚠️ 旧的 `exp_C*.json` 里 **C0 档的提示词是坏的**（把带 `{covered}` 占位符的
> 模板原样发给了模型，注入防护那段也没进去），已作废并由 `exp_C0_fixed.json` 取代。
> 见 `EVALUATION.md` 实验 C 章节与 `backend/tests/test_exp_script_prompt.py`。

**A 的结论**（n=29，`EVALUATION.md` 有完整表格和配对检验）：

1. **重排是唯一做到 0.897 的配置**，且在三个 chunk 粒度下都稳在 0.897 —— 全项目最清晰的「效果 vs 成本」权衡点（代价是延迟 785ms，为纯 BM25 的 5.4 倍）。
2. RRF 融合的增益在**排序质量**（MRR/NDCG）而非召回上限 —— 融合是重排器，不是扩容器。
3. **纯向量恒定 0.828 且不随 top_k 增长** —— 低频专名领域里稠密检索的天花板是硬的。
4. ⚠️ 补抓角色页后 A0(BM25) 从 0.897 掉到 0.793，**低于纯向量**，方向与旧语料相反。但这个翻转只对应 **3 道题**，n=29 下 3 题 = 10.4pt，两个方向都不稳健 —— **不能说"向量优于 BM25"**。退化项逐条核对后确认**全部是攻略页挤占**（`compare_corpus_shift.py` 可复现）。

**B 的结论**：

1. `embed_mode=title` 显著优于只用正文：A0 的 MRR **0.736→0.506（−23.0pt）**，配对检验 **p=0.0001**。旧语料下只有 −9.8pt —— **语料越长越松，标题前缀越值钱**。零成本改动，效果比调 chunk_size 大得多。
2. **chunk_size 对最终指标（A3）不再有影响**：500 / 800 / 1200 三档都是 0.897，逐位相同。分块粒度造成的差异被重排完全吸收。
3. chunk 变大对纯 BM25 单调不利且**比旧语料更严重**（0.736→0.586→0.546，跨 19pt，旧数据只跨 3.5pt）—— 攻略页这种"什么都提一点"的长文本最吃词频稀释。

**D 的结论**：

1. **检索质量无法区分**：同一批 29 条 case 上的跨模型配对检验，
   A2 融合 MRR 差 −0.0115、**p=0.6743**、Cohen's d=0.030；A1 差 +0.0569、p=0.5683。
   **不能说 bge 更强，也不能说 Qwen 更强** —— 两档的 p 都在 0.5 以上，
   而且**方向互相矛盾**（A1 上 bge 略优、A2 上 Qwen 略优），这正是"测不出差异"的样子。
2. 阴性对照 A0（纯 BM25，不用向量）五个指标 **Δ 全部 0.0000、p=1.0000** ——
   补抓语料之后这个对照依然成立，说明对比里只有 embedding 模型这一个变量。
3. **Qwen 在融合档显著更快**（A2 快 28ms，**p<1e-10**），
   但效应量只有 **d=0.052** —— 统计显著 ≠ 实际重要。
   A1 的延迟差异**不再显著**（旧 p<1e-10 → 新 p=0.0946），所以不能说"全面更快"。
4. 最终仍保留 bge-m3，但理由是「reranker 是 bge 系、配套适配」，
   **不是**「质量更好」—— 端到端里 embedding 只占几十毫秒，不值得为此换。

> ⚠️ 原文档曾用「Qwen 组内 A1→A2」的检验（p=0.12）当跨模型结论的依据，
> **检验对象用错了**。修正过程见 `EVALUATION.md` 实验 D 章节。

**C 的结论（三库逐库，2026-10-14 重跑）**：

1. **给不给检索工具这条线，三库一致且极其显著**：
   Δ 在 **+2.04 ~ +2.61**，Cohen's d 在 **2.16 ~ 3.72**，三库 p 均 **<1e-10**。
   这是全项目最硬的一个结论。
2. **多给 `get_page` / `calculator` / `list_games`，三库都没测出收益，而且方向不一致**：
   sr −0.17、ys +0.36、zzz +0.05，p 从 0.0895 到 0.3173。
   **符号互相矛盾正是「测不出差异」的样子**（与实验 D 同类证据）。
3. **代价却三库一致**：token **+17% ~ +45%**、步数 +30%~50%、
   工具失败率从 0 变成非 0（`get_page` 现抓 wiki 会被 WAF 拦）。
   所以生产环境默认只开 `search_kb` —— 不是「工具越少越好」，
   而是「在没有证据支持时，不要为看不见的收益付费」。

**E 的结论（三库逐库）**：

1. **质量差异三库均未达 p<0.05**（sr p=0.0697 / ys p=0.6738 / zzz p=0.2831）。
   方向三库一致（MiniMax 略低），量级差异大（sr 的 d=−0.71 已是中到大效应量），
   但没有一个库撑到显著。**不能说「MiniMax 更差」，也不能说「两者一样好」。**
2. **成本差异三库一致，这才是选型的真正依据**：token **1.55x ~ 2.85x**、
   延迟 **2.56x ~ 3.65x**。
3. 两个信号一起看是自洽的：成本差最大的 sr 库（2.85x token）质量差也最大
   （Δ=−0.73）；成本差最小的 ys 库（1.79x token）质量差也最小（Δ=−0.14）。

**F 的结论（披露层消融）**：

1. **披露层把信息完整送到了答案**：15 个目标页的修订日期，
   关闭时命中 0 个，开启时命中 15 个（引用修订覆盖率同样 0.0 → 1.0）。
2. **给了日期之后模型会主动声明「可能已过时」**（0.0 → 0.5）。
   但这**不代表判断是对的** —— 页面修订时间原理上推不出内容时效。
3. **本实验不能证明披露层让回答更准**，语料只有最新版快照，
   没有历史版本可比对（`docs/version_staleness.md` 第四节）。
4. ⚠️ 其中一条指标被**主动撤掉**：修订措辞率修掉题面回声之后是
   F1=0.67 / F2=0.00，**消融档反而更爱说「修订」** —— 它量的是用词习惯而非
   披露行为，与披露**负相关**。过程与数据见 `EVALUATION.md` 实验 F 章节。

以及两条方法论教训：

- **10 条评测集测出「融合无增益」，32 条测出「融合有增益」，符号相反。** 题不够的时候，消融实验的结论不可信。
- **A0（仅 BM25，不依赖 embedding）在两组对比里数字完全相同** —— 这一行是对照实验的自证，证明语料/分块/评测集一致。
- **实验 C 的 C0 档提示词是带占位符的模板，被原样发给了模型** —— 游戏名没注入、注入防护那段也没进去，错误还不报错（judge 正常打分、分数也合理）。**"实验脚本把常量当成品用"是一类很安静的缺陷**，靠断言「发给模型的那段文本里还有没有占位符」才抓得住。
- **先用已存档的答案离线验证判据有没有区分度，再花钱重跑。** 实验 F 的修订措辞率就是这么被撤掉的：离线重算零成本，而它直接决定了这轮要不要花 16 次 Agent 调用。

---

## 关于延迟：KV cache 不需要「引入」，它已经在生效

常见误解是"上下文长 → 上 KV cache → 变快"。实测（`probe_kv_cache.py`）：

- DeepSeek 的缓存**自动生效**，`usage.prompt_cache_hit_tokens` 直接可见，命中率 80%+，墙钟降约 50%。
- **但它省的是费用，不是延迟** —— 同一模型下首 token 延迟稳定在 100–350ms，总耗时主要由生成 token 数决定。

真正有效的杠杆是**流式反馈**（口径见下方说明，别当成「第一个字」）：

| 方式 | 总耗时 | 第一个 SSE 事件到达 |
|---|---|---|
| 非流式 `/api/chat` | 3410ms | 3410ms 后（全部完成才有反馈） |
| 流式 `/api/chat/stream` | 3147ms | **860ms** |

> ⚠ **三点限定，引用时请一起带上：**
>
> 1. **上游没有 token 级流式。** `/api/chat/stream` 的机制是 Agent 拿到
>    **完整答案**后 `_split_for_stream()` 按中文标点切片推送
>    （`app/agent/react.py`）。所以真正的"第一个字"时刻 ≈ 生成完成时刻，
>    不比非流式早多少。
> 2. **860ms 是第一个 SSE 事件的到达时间**——很可能是 `tool_start`，
>    也就是"用户第一次看到系统有反应"，不是"第一个字"。探针脚本
>    `probe_stream_latency.py` 的 `measure_stream()` 记录的是首个事件，
>    而它自己的打印文案就写着「用户第一次看到反馈」而非「首字」。
> 3. **这是单次测量。** 产物 `probe_stream_latency.json` 自己的 `note` 就写着
>    「TTFT 为单次测量，波动大，结论请看多次重跑的中位数」。
>
> （`LLMClient.chat_stream()` 这个真·上游流式方法**已删除**：写了 48 行含跨
> chunk 的 `<think>` 处理，全仓从未被调用过一次。留着它只会让"流式已实现"
> 这句话更难说准 —— 和本项目删掉的 `TOOL_TRANSPORT` 是同一类问题。）
>
> 详见 `EVALUATION.md` 延迟章节的完整优先级排序。

---

## 当前状态

| 项 | 状态 |
|---|---|
| 语料 | 星铁 706 页→3932 chunk；绝区零 260 页→1498；原神 434 页→5413；合计 **10843 chunk**，**全程 0 次 WAF 硬失败** |
| 向量库 | LanceDB 0.40，暴力精确扫描，跨进程读回已验证；NumPy 实现保留为 fallback |
| 检索 | BM25 + 向量 RRF + 重排，A0–A3 消融已完成（n=29，见 `EVALUATION.md`） |
| 分块 | 实验 B 四组**三库逐库**完成：MRR title→text sr **−23.0pt（p=0.0001）** / ys −13.3% / zzz **0.0%** —— 方向一致但强度差两个数量级；chunk_size 在 500/800/1200 下 sr A3 指标完全相同 |
| Embedding | 实验 D **三库逐库**：**保留 bge-m3**，依据是 reranker 配套而非质量（A2 跨模型配对 p=0.6743 无法区分，三库方向互相矛盾）；Qwen 在融合档显著更快（sr −28ms，p<1e-10） |
| 检索 | A0–A3 **三库逐库**（n=29/9/10）：recall@3 sr 0.793→0.897 / ys 0.889→1.000 / zzz 1.000→1.000。⚠️ **重排增益只有 sr 有样本支撑**，ys/zzz 在纯向量档已打满（见 `EVALUATION.md`） |
| 延迟 | A3 检索 785ms（纯 BM25 144ms 的 5.4 倍）；感知延迟靠流式 −85%；KV cache 自动生效，无需配置。⚠️ 索引从 2306 涨到 3932 chunk 后，A3 延迟从 567ms 涨到 785ms（+38%） |
| 并发 | 实测 1/4/8 并发：4 并发吞吐 3.40x，8 并发 5.18x；同 query 各并发度下 top5 完全一致。冷缓存并发从"8 次重复建索引"降到 1 次（`docs/concurrency.md`） |
| 安全 | 间接 Prompt 注入：11 条规则分两级，10843 chunk 真实语料误报 **0**、攻击样本召回 **13/13**；工具结果一律包进不受信任数据区（`docs/injection_guard.md`） |
| 可用性 | Chat 降级链：只对 429/5xx/超时降级，400/401 不重试；降级在响应 `fallback` 字段与 `/api/health` 留痕（`docs/fallback_chain.md`） |
| 数据时效 | 版本更新：实测 wiki `recentchanges` 可穿过 WAF；近 7 天变更页 ∩ 已收录 = **162/706 页（23.0%）**，检测只花 **15 次请求**（全量重抓要 71 次）。⚠️ 全站口径 1157 页会得出"增量无意义"的**相反结论**。内容指纹已落库（1401 页 + 10843 chunk 100% 覆盖），归一化把格式误判从 **100% 压到 0**（`docs/delta_sync.md`） |
| Agent | 手搓 ReAct，工具注册表 4 个（`search_kb` / `get_page` / `calculator` / `list_games`），**生产默认只挂 `search_kb`**（`AGENT_TOOLS`，侧栏显示的就是实际生效的那份） + 4 道安全阀 + 熔断；不可答问题从 7 步/30317 token 降到 4 步/6514 token |
| 多库路由 | 三态（库内/越界/未提及）；越界走无工具单轮，实测 0 步 / 1.1s，库内 3–5 步 / 5–8s |
| MCP | 4 个工具以标准 MCP（SDK 2.3.0）暴露给外部 host，stdio 传输；**适配层与内置实现输出逐字符一致由守卫实跑保证**（16 项，真 `mcp.Client` 连真 server）。MCP 是**独立 server 入口**，本项目 Agent 始终走进程内直调，**两者之间没有运行时开关**（曾经有过 `TOOL_TRANSPORT`，核实发现无任何代码分支读它，已删），见 [docs/mcp.md](docs/mcp.md) |
| 后端 | FastAPI，`/api/health` `/api/retrieve` `/api/chat` `/api/chat/stream` `/api/runs` `/api/ingest` `/api/stats` |
| 前端 | React + Vite + TS，SSE 流式、工具时间线、引用、检索调试面板、双主题、背景图、移动端抽屉 |
| 存储 | MySQL 语料三表与向量库**逐条一致**（`backfill_mysql.py` 剪枝后核对）：pages 1401（sr 706 / zzz 260 / ys 435）/ chunks 10843（3932 / 1498 / 5413）/ facts 25111（15017 / 4976 / 5118）。唯一差异是 ys 的「角色投稿测试（先说对不起）」正文长度为 0，被 `page_filter` 挡在索引外但 MySQL 记了"抓到了"；另有 runs / tool_calls / eval_cases / eval_results / sessions / messages |
| 评测 | 评测集 **56 条三库全覆盖**（星铁 32 / 绝区零 12 / 原神 12）；实验 A/B/C/D/E 均已出真实数据 |
| 检索实验覆盖 | **A / B / D 均已扩到三库**（sr 29 / ys 9 / zzz 10 = 48 条有 gold 的题），汇总按库分节、逐库配对检验，不混口径 |
| 已知偏差 | `eval_cases` 表仍是 32 条，评测集已扩到 56 条，重跑一次评测即可同步 |
| 工程 | pytest **747 项**全绿（含故障注入 23 条 + 引用校验 34 条 + 接线守卫 15 条）；smoke 自检 41 项；ruff 全仓清；前端 vitest **56 项**；GitHub Actions 跑 pytest + ruff + 前端测试 + 前端构建 + 密钥卫生 |

一条命令自检：

```powershell
# 离线单元测试（CI 跑的就是这批，不打网络）
.venv\Scripts\python.exe -m pytest

# 前端测试（交付物面板的纯函数守卫）
cd frontend; npm.cmd test

# 真实检索 + 真实 LLM 的端到端自检（要联网，手动跑）
.venv\Scripts\python.exe backend\scripts\smoke_refactor.py

# 用真实语料验收页面过滤器 / .env.example 覆盖率
.venv\Scripts\python.exe backend\scripts\check_page_filter.py
.venv\Scripts\python.exe backend\scripts\check_env_example.py

# 验收 /api/retrieve 是否真的听 .env 的话（改完配置跑一次）
.venv\Scripts\python.exe backend\scripts\check_config_wiring.py
```

> `check_config_wiring.py` 存在的理由：`.env` 里的检索参数曾经**改了完全不生效**——
> `RETRIEVE_TOP_K` / `RETRIEVE_CANDIDATE` / `RRF_K` 在 `config.py` 定义了、
> `.env.example` 也列了，但 `main.py` 构造 `RetrievalConfig` 时压根没传，
> **且不报任何错**。做单变量消融的人会白跑一整轮才发现参数没变。
> 这个脚本的断言全部从 `Settings` 动态取值，所以把 `.env` 改成非默认值也能验。

### 降级能力：外部 API 挂掉也能演示

充 SiliconFlow 期间曾实测过一次完整故障（402 `code=30001`），
embedding / rerank / judge **同时失效**，但：

| 能力 | 故障时状态 |
|---|---|
| DeepSeek chat（Agent 主链路） | ✅ 正常 |
| A0 仅 BM25 检索（**纯本地，零远程调用**） | ✅ 正常，实测 6–8ms |
| embedding / rerank / judge | ❌ 402 |

降级后 Agent 端到端仍给出正确答案（`steps=2, 3996ms`）——
检索从 A3 退回纯 BM25，指标由 0.966 降到 0.897（旧 2306-chunk 语料下的数字）。

这个能力不是设计出来的，是撞上故障实测出来的。面试时值得讲：
「我把检索策略做成可切换的开关，顺带得到了一条不依赖任何付费 API 的兜底路径」。

### 数据库里的「空库」不是垃圾

用 Navicat 只会看到 `information_schema` / `mysql` / `performance_schema` / `sys` 四个库
像是空的。**一个都不能删**，它们分别是元数据、权限时区、慢查询监控、系统视图，
内部各有 78 / 38 / 113 / 101 张表，只是客户端默认不显示系统表。

真正属于本项目的是 `kb_agent`（9 张表，全部非空）。早期 `eval_*` / `sessions` / `messages`
显示为空是因为**建了表但没接写入路径** —— 评测结果只写 JSON、会话只存内存。
现在两条链路都已接到 MySQL，`runs` / `eval_results` 才能做「改了参数再跑一遍，直接对比差多少」的回归。

---

## 踩坑记录（都是实测踩出来的，不是抄来的）

1. **chromadb 在本机不可用**：0.5/0.6 无 cp313 轮子；1.5.9 索引不落盘，换进程必报
   `Error loading hnsw index`。完整证据见「为什么放弃了 ChromaDB」一节。

2. **EdgeOne 的 567 不是"站点不存在"**。它是被限流的挑战页状态码，而且它有时把挑战页标成
   `application/json`，只看 Content-Type 会漏判。踩过一次"站点没有这个 slug"的错结论，
   正确 slug 是中文缩写（`sr` / `zzz` / `ys`）。

   对策优先级：**批量取页（一次 10 页）> 拉长间隔 > 退避重试**。实测 518 页 / 0 次拦截。

3. **模板不能删，只能展平**。星铁 wiki 的光锥页几乎全部内容都在 `{{光锥图鉴}}`（54 个具名参数）里。
   最初"删掉所有模板"的清洗逻辑把 969 字的 wikitext 洗成 20 字——页面抓到了，等于什么都没入库。
   修成展平成 `字段：值` 后，同一批 40 页从 **9 chunk → 78 chunk**。

4. **bge-m3 的向量范数不为 1**。不双侧归一化就换 embedding 模型，对比里会混进"范数分布"这个隐藏变量。

5. **PowerShell 写 UTF-8 会带 BOM**，导致 checkpoint JSON 解析失败后被静默清空，
   表现为"续传从头开始"。读的时候用 `utf-8-sig`，写文件用 `UTF8Encoding($false)`。

6. **刻晴是原神角色**，不在星铁 wiki。选测试页面时踩了这个。

7. **wiki 的分类命名空间是中文「分类:」而不是 `Category:`**。
   只写死一种前缀，`list_category_pages` 会静默返回空列表 —— 看起来像"这个分类没内容"，
   实际是路径错了。另外 MediaWiki 会把错误放在 `query.errors` 里而 HTTP 仍是 200，
   只看状态码同样会漏判。

8. **工具 schema 少一层 `{"type": ...}` 包装**，DeepSeek 直接 400 `Invalid schema for function`。
   同理，assistant 消息回传 `tool_calls` 是必须的，否则下一轮报
   `Messages with role 'tool' must be a response to a preceding message with 'tool_calls'`。

9. **`call_tool` 的 ok 不能等于"没抛异常"**。工具按约定把失败转成可读文案返回，
   若 ok 一律为 True，Agent 轨迹里的 `tool_failures` 就永远是 0，失败率指标直接作废。
   现在按文案前缀判定。

10. **知识库里没有的内容，Agent 会连续空转 7 步**。行为正确（没编造），成本错误。
    实测 30317 prompt tokens / 12.7s 才承认查不到 —— SearchGuard 熔断后降到 4 步 / 6514 token。

11. **评测集太小会直接测出错误的结论**。10 条评测集里纯语义题只有 1 条，
    跑出的结论是「RRF 融合无增益」。扩到 32 条、纯语义题补到 6 条之后，
    同一套代码的结论变成「融合有增益，MRR +2.3pt」——**符号相反**。
    详见 `EVALUATION.md` 开头的记录，这是本项目最重要的一条方法论教训。

12. **优化前先测环节延迟，别猜**。原本以为慢在向量检索，实测本地暴力扫描只要 8ms，
    慢的全是远程 API（rerank 449ms / embed 147ms / chat 614ms）。
    如果按直觉去"优化向量检索"，会把时间花在 8ms 的环节上。

13. **`_warmup` 函数名覆盖了同名状态变量**。写后台预热时把状态 dict 和函数都叫 `_warmup`，
    函数定义在后把 dict 顶掉了，`/api/health` 直接 500
    （`Unable to serialize unknown type: <class 'function'>`）。改名后恢复。

14. **"把写死的两个改成 N 个"永远改不完整**。加第三个游戏库时，
    `Settings.games` 和 `tools.py` 都改成了 N 游戏，**漏了 `react.py` 里的
    `SYSTEM_PROMPT`** ——它还是写死的「你是星铁与绝区零的知识库助手」。
    症状：检索层能召回原神、页面确实在库里，Agent 却回"知识库未收录《原神》"，
    `steps=0`。和当初 `route_game` 默认回退到 `sr` 是同一类错误，只是发生在 LLM 层。
    只能靠**跨层断言**判断改完了没有，现在 `test_multi_game_routing.py` 盯着这条。

15. **元数据存了不等于读得回来**。`list_collections()` 直接 `new` 一个空 collection，
    从不读 `.meta.json`，于是 `game`/`variant`/`embed_model` 全空。
    前端只好退回用正则从 collection 名解析，把实验 B/D 的变体库也加进"生产语料"，
    界面显示星铁 **9,986** 片段 —— 实际 base 库只有 2,293，**虚高 4.4 倍**。
    这种数字会被直接抄进简历。

16. **过滤器必须用真实语料验收，不能只靠单元测试**。页面过滤器最初除了按标题过滤，
    还加了"清洗后正文占比 < 5% 就跳过"的内容启发式。单元测试全过，
    拿 1035 页真实语料一跑：**误杀 60%**。因为这个 wiki 的内容页本身就是模板驱动的，
    光锥页 95% 以上是 `{{光锥图鉴|…}}`。现在只保留标题判据，
    `scripts/check_page_filter.py` 拿全部 1038 页做回归。

17. **同名不同物要用别名**。`persist.py` 同时从 `app.eval.cases` 和 `app.store.models`
    导入 `EvalCase`，后者把前者遮蔽了 —— 函数里那句 `EvalCase(...)` 实际在构造 ORM 对象。
    当时没有调用方所以没炸，但是个地雷。ruff 的 F811 抓到了。

---

## 下一步

按"投入产出比 × 简历可信度"排序。每项都写清**为什么还没做**，避免看起来像是漏了。

1. ~~**把评测集扩到三库，再重跑一遍实验 A–E**~~ ✅ **A–F 全部完成**
   `run_eval.py` 新增 `--game all` 多库路径，A/B/D 已从 sr 的 29 条扩到
   **三库 48 条**（sr 29 / ys 9 / zzz 10），产物 `data/eval/exp_A_all_k3.json`
   与 `model_compare_{sr,ys,zzz}.json`。
   汇总按库分节，不混口径 —— 三库 chunk 数 3932 / 5413 / 1498 差得很远，
   合成一个均值等于把「语料规模」和「分块粒度」两个隐藏变量塞进结论。
   **生成侧 C/E 也已三库逐库跑完**：C 三库 32/12/12 条、E 三库 15/12/12 条，
   产物 `exp_C_{sr,ys,zzz}.json` 与 `exp_E_{sr,ys,zzz}.json`，
   结论见 `EVALUATION.md`。披露层消融 F 另立题集（8 条溯源题），
   见下文实验表。

   ⚠️ C/E 的条目数**刻意没有对齐检索侧的 48 条**：C 档每条一次 Judge 调用，
   扩到 48 条意味着 3 档 × 48 次 ≈ 96 分钟。实际取的是各库**可用题的全集**
   （32/12/12、15/12/12），而不是从 48 条里抽一个子集 ——
   抽样会在题型构成上引入新的隐藏变量，不如按全集跑。
   代价是三库样本量不齐，显著性判断必须**逐库看**，见第 2 条。

   三库跑完还修正了一条原本当成跨库稳健的结论：「embed_mode=title 优于 text」
   的**幅度**并不跨库一致。逐题 diff（`exp_A_all_k3.json`）：
   sr MRR 0.736→0.506（−23.0pt，p=0.0001，仅 sr 达显著）、
   ys 0.833→0.722（−13.3%）、zzz 0.950→0.950（**零变化**）。
   也就是说「title 优于 text」在 zzz 上**完全没有效果**，
   在 sr 上则是决定性的 —— 方向一致但强度差两个数量级，
   所以只能写成「sr 上显著更优，ys 上同向但未显著，zzz 上无差异」，
   不能写成一句跨库通用的结论。

   A 档另一条值得注意：**ys/zzz 在 A1（纯向量）就已打满**
   （recall@3 1.000），重排（A3）在两库上**没有可提升空间**。
   「重排收益最大」这条结论目前**只有 sr 有样本支撑**。

2. **评测集继续加量（当前最大的统计缺口）**
   检索侧三库 48 条（sr 29 / ys 9 / zzz 10），生成侧 C 三库 **32/12/12**、
   E 三库 **15/12/12** —— 即 ys+zzz 都是 **12 条**，统计功效仍然不足。
   经验上单库要 30+ 条才谈得上显著性，而 ys/zzz 现在离这条线还有一半以上。
   ⚠️ 注意：**ys 的可用题里 7/9 是 `single_hop`**，而结构化题恰恰是
   「有没有 RAG」差别最大的题型 —— 所以跨库比 judge 均值时，
   题型构成差异会主导结果（`data/eval/exp_C_*.json` 的 `_meta.subset_composition`
   就是为此记录的）。加题时要**按题型补**，不能只补数量。
   这也是实验 C 的 C1→C2 差异三库**方向不一致**（sr 略降、ys/zzz 略升，
   p 值 0.09~0.32 全部未达显著）时最该警惕的干扰源 ——
   在题型构成补齐之前，不应把「C2 没有稳定优于 C1」写成结论。

3. **把 zzz 的语料规模拉到与 sr / ys 同量级**
   现在 sr 706 / ys 435 / zzz 260 页，补抓角色页后 zzz 反而成了最短的板。
   跨库对比题（`ys-010` 圣遗物 vs 位面饰品）如果检索只命中了语料多的那个库，
   结论会偏向"语料多"而不是"路由对"，需要控制这个隐藏变量。
