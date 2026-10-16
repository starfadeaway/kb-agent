# db/ —— 知识库的 SQL 快照

## 这个文件是什么

`kb_agent.sql` 是 MySQL 里那份知识库的完整快照（约 6.5 MB），由
`backend/scripts/dump_db.py` 从本机 MySQL 导出：

```powershell
.venv\Scripts\python.exe backend\scripts\dump_db.py
```

**为什么要入库**：在此之前，库的 schema 只存在于本机 MySQL 里 ——
`.gitignore` 挡住了 `data/` 与 `.env`，仓库里没有任何「这库长什么样」的记录。
换一台机器、或本机 MySQL 损坏，就要靠读 `backend/app/store/db.py` 反推建表语句，
而数据（1401 页 / 10,843 个文本块 / 25,111 条数值事实）则完全找不回来。

## 库名为什么还叫 kb_agent（不是漏改）

项目在 2026-10 从 `kb-agent` 改名为 `lorekeeper`。那轮改名一共动了
**32 处 / 20 个文件** —— 代码里的常量、文档标题、前端包名、MCP server 名、
localStorage 的 key，全都改了。**唯独数据库名保留 `kb_agent` 没动**，理由三条：

1. **它是内部标识。** 这个名字只出现在 `.env` 的 `MYSQL_DB` 和连接串里，
   不进任何用户可见的界面、日志或 API 响应 —— 改不改对项目对外的名字没有影响。
2. **改名要新建库 + 导数据。** MySQL 不支持直接重命名库，
   而库里已经有 9 张表、1401 页、10,843 个文本块 —— 收益为零，风险不为零。
3. 所以它成了那次改名里**唯一被明确保留的旧名字**。

> **看到 `kb_agent` 不要以为是漏改的。** 这是有意为之，也是那轮改名里
> 唯一一处「改了会亏、不改也对」的地方。将来真要统一，步骤是：
> 建新库 → 导入本文件 → 改 `.env` 的 `MYSQL_DB` → 验证 9 张表行数 → 删旧库。

## 怎么重建

```bash
mysql -u root -p < db/kb_agent.sql
```

脚本里带的是 ``CREATE DATABASE IF NOT EXISTS `kb_agent` `` + ``USE `kb_agent` ``，
所以建出来的库名就是 `kb_agent`，与 `.env` 里的 `MYSQL_DB` 一致，不用再改配置。

## 什么不在这个文件里

- **向量索引**：在 `data/lance/`（15 个集合，约 60 MB 二进制），被 `.gitignore` 挡住。
  需要时用 `backend/scripts/rebuild_index.py` 重新生成 —— 它可以从这个 SQL 里的
  页面正文重建出来。
- **API 密钥**：在 `.env` 里，从不入库。

## 导出时做了什么处理

- `--single-transaction`：一致性快照，不锁表（导出时服务照常可用）。
- `--skip-comments`：抹掉 mysqldump 默认写入的**主机名与 MySQL 版本号**，
  避免把本机环境信息带进公开仓库；文件头换成我们自己的说明。
- 密码通过 `MYSQL_PWD` 环境变量传给 mysqldump，**不落命令行**（命令行参数对其他进程可见）。
