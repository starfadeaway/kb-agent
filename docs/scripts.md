# 脚本索引

`backend/scripts/` 下共 51 个脚本。这一页说明**每一个是干什么的、哪些会破坏数据**。

保留一次性探针不是为了"常用"，而是它们记录的是**踩坑过程**——
`probe_lancedb.py` 证明了 ChromaDB 的失败点在「换进程读不回」，
而不是"我听说它不行"。删掉，这些结论就只剩没有出处的传闻。

## ⚠ 破坏性脚本（会丢数据）

跑之前先确认要删什么。`data/raw` 一般不会被这些脚本动，
但索引删掉后重建要花时间和 embedding 费用。

| 脚本 | 作用 | 影响面 |
|---|---|---|
| `reset_collection.py <game> [model]` | 删除指定 collection，重新入库前清场，避免旧 chunk 混进评测 | **删向量索引**，需 `rebuild_index.py` 恢复 |
| `purge_pages.py --apply --page-id N` | 从已建好的索引里**精确删掉某个 page_id 的全部 chunk**（默认 `--dry-run` 只看不删） | **删索引里的 chunk**，删完**必须重启后端**（索引缓存是进程内的） |

> `purge_pages` 故意不做"删 raw"：`data/raw` 是可再生的，而 embedding 不白给。
> 按 `page_id` 定点删除与"重建后自然不存在"等价，且零 embedding 开销。

## Schema 迁移

| 脚本 | 作用 |
|---|---|
| `migrate_content_hash.py [--backfill]` | 给 `pages`/`chunks` **ALTER TABLE 加列**并回填指纹 |

> 为什么必须单独写：`create_all()` 只建**新表**，对已存在的表一列都不会加。
> 「改了 model 就以为生效了」是错的。脚本是幂等的——MySQL 不支持
> `ADD COLUMN IF NOT EXISTS`（那是 MariaDB 语法），只能先查 `information_schema`。

## 数据入库与索引

| 脚本 | 作用 |
|---|---|
| `crawl.py <game>` | 单库抓取（走 `WikiClient`，含 WAF 退避） |
| `crawl_all.py` | 依次建 sr / zzz / ys 三个库 |
| `bulk_ingest.py` | 从 `data/raw` 批量入库并切块 |
| `rebuild_index.py --game sr` | 重建向量索引（`lancedb` 或 `numpy` 后端） |
| `backfill_meta.py` | 回填页面元数据到本地文件 |
| `backfill_mysql.py --game sr` | 把本地缓存语料**全量**回填到 MySQL |

## 评测

| 脚本 | 作用 |
|---|---|
| `run_eval.py` | 检索侧评测，支持 `--game all` |
| `exp_common.py` | **不是可执行脚本**，是实验脚本的公共底座：评测子集、结果聚合、表格打印。抽它的理由不是省代码，是让 A/B/C/D/E 的**统计口径只有一份** |
| `exp_b.py` / `exp_c.py` / `exp_d.py` / `exp_e.py` | 消融实验 B/C/D/E |
| `merge_exp_e.py` | 合并 E 的分档产物 |
| `analyze_model_compare.py` | 模型对比，**显式输出 winner / verdict** |
| `analyze_significance.py` | 配对检验与效应量 |
| `seed_eval_cases.py` | 生成/扩充评测集 |
| `check_eval_gold.py` | 校验 gold 页面是否真在语料里 |

## 一致性自检（pytest 里有等价断言，CI 会跑）

| 脚本 | 作用 |
|---|---|
| `check_config_wiring.py` | 验证 `.env` 的值真的进了运行时配置 |
| `check_env_example.py` | 验证 `.env.example` 覆盖所有配置项 |
| `check_variant_consistency.py` | 验证各实验变体的索引规模一致 |
| `check_page_filter.py` | 版本/页面过滤条件是否按预期生效 |

## 性能与行为探针（一次性，取结论用）

| 脚本 | 得出的结论 |
|---|---|
| `probe_lancedb.py` | LanceDB 必须**换进程**验证读回；ChromaDB 正是挂在这一步 |
| `probe_kv_cache.py` | 前缀缓存命中率 80%+，**省费用不省延迟** |
| `probe_index_warmup.py` | 首次建索引的冷启动成本 |
| `probe_ctx_reuse.py` | retriever 必须复用单例，否则每请求重付 BM25 建索引 |
| `probe_concurrency_consistency.py` | 并发下单例状态是否串 |
| `probe_stream_latency.py` / `probe_run_trace.py` | 流式首字延迟与单次运行轨迹 |
| `probe_delta_sync.py` / `probe_fingerprint_precision.py` | 增量同步收益与指纹精度 |
| `probe_version_entry.py` | 版本过滤：入口层（`/api/retrieve` 的 `min_revision_ts`） |
| `probe_version_filter.py` | 版本过滤：检索层（两个 store 的 `$gte` 条件是否真的生效） |
| `probe_version_story.py` | **把「通用 RAG 在会过期的语料上会失效」量化**：实测 sr 库 518 页横跨 2023-07 → 2026-10，其中 155 页（**30%**）是两年前的内容；用户问当前版本，检索却可能把两年前的条目排进来 |
| `probe_minimax.py` / `probe_minimax_endpoints.py` | MiniMax 端点可用性与延迟 |
| `probe_io.py` | 读写 IO 开销 |
| `bench_latency.py` / `bench_concurrency.py` / `profile_latency.py` | 延迟、并发、火焰图 |

## 诊断与自检

| 脚本 | 作用 |
|---|---|
| `smoke.py` | 端到端冒烟 |
| `smoke_refactor.py` | 重构回归自检：模块可导入、`strip_think`、`parse_inline_tool_calls`、工具超时文案、熔断两条路径一致性。⚠ **只有前 6 项不依赖外部 API**，第 7 项会真调一次检索 → 打 SiliconFlow |
| `scan_corpus_injection.py` | 在真实语料上跑注入规则，报的是**误报率**而不是命中率 |
| `verify_ys_expansion.py` | 原神扩充后的端到端验收（三个新分类必须命中） |
| `compare_corpus_shift.py` | 语料版本前后的检索结果差异 |
| `dump_fields.py` | 打印 wiki 页面字段结构 |
| `watch_experiment.py` | 实验进度监控（判据只看完成数是否增长，CPU=0 不算证据） |

## MCP

| 脚本 | 作用 |
|---|---|
| `mcp_stdio_entry.py` | MCP stdio server 入口（stdout 留给协议） |
| `probe_mcp_stdio.py` | 跨进程探针，验证真 host 能连上并调用 |