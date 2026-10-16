"""把 MySQL 里的知识库导出成一个可入库的 .sql 快照。

## 为什么要有它

在此之前，库的 schema 只存在于本机 MySQL 里 —— `.gitignore` 挡住 `data/` 与
`.env`，仓库里没有任何「这库长什么样」的记录。换机器或库损坏，就要靠读
`store/db.py` 反推建表语句，而 1401 页 / 10843 chunk / 25111 条数值事实
则完全找不回来。产物见 `db/kb_agent.sql`。

## 库名为什么还叫 kb_agent（不是漏改）

2026-10 那轮从 `kb-agent` 改名 `lorekeeper`，动了 32 处 / 20 个文件
（代码常量、文档标题、前端包名、MCP server 名、localStorage key），
**唯独数据库名保留 `kb_agent`**：它是内部标识、不进任何用户可见的地方；
改名要新建库 + 导数据（MySQL 不支持重命名库），而库里已有 9 张表 ——
收益为零，风险不为零。**看到它不要以为是漏改的。** 详见 `db/README.md`。

## 用法

    .venv\\Scripts\\python.exe backend\\scripts\\dump_db.py
    .venv\\Scripts\\python.exe backend\\scripts\\dump_db.py --out db/other.sql

密码从 `.env` 读，经 `MYSQL_PWD` 环境变量传给 mysqldump —— **不落命令行**。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from shutil import which

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings  # noqa: E402

DEFAULT_OUT = ROOT / "db" / "kb_agent.sql"

HEADER = (
    "-- lorekeeper 知识库快照\n"
    "-- 导出时间：{ts} UTC\n"
    "-- 来源库名：{db}（刻意保留的旧名，原因见 db/README.md）\n"
    "-- 重建：mysql -u root -p < db/kb_agent.sql\n"
    "--\n"
    "-- 只含结构化数据（页面元数据 / 文本块 / 数值事实 / 评测题 / 运行轨迹）。\n"
    "-- 向量索引不在里面 —— 它在 data/lance/ 下且被 .gitignore 挡住，\n"
    "-- 需要时用 backend/scripts/rebuild_index.py 重新生成。\n\n"
)


def find_mysqldump() -> str:
    """优先 PATH，其次本机 MySQL 8.4 的默认安装位置。"""
    exe = which("mysqldump")
    if exe:
        return exe
    fallback = Path(r"D:\MySQL\mysql-8.4.5-winx64\bin\mysqldump.exe")
    if fallback.is_file():
        return str(fallback)
    raise SystemExit("找不到 mysqldump：装 MySQL 客户端，或把它加进 PATH")


def dump(out: Path) -> int:
    s = get_settings()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")

    cmd = [
        find_mysqldump(),
        "-h", s.mysql_host, "-P", str(s.mysql_port), "-u", s.mysql_user,
        "--databases", s.mysql_db,      # 带 CREATE DATABASE / USE，保住库名
        "--single-transaction",         # 一致性快照，不锁表
        "--routines", "--triggers",
        "--default-character-set=utf8mb4",
        "--skip-comments",              # 抹掉 mysqldump 写的主机名与版本号
        "--result-file", str(tmp),
    ]
    env = {**os.environ, "MYSQL_PWD": s.mysql_password}
    print(f"[dump] {s.mysql_user}@{s.mysql_host}:{s.mysql_port}/{s.mysql_db} -> {out}")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("[dump] 失败：", (r.stderr or "").strip()[:400], file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return r.returncode

    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    out.write_text(HEADER.format(ts=ts, db=s.mysql_db) + tmp.read_text(encoding="utf-8"),
                   encoding="utf-8")
    tmp.unlink(missing_ok=True)
    size = out.stat().st_size
    print(f"[dump] 完成：{size:,} 字节（{size / 1024 / 1024:.2f} MB）")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="导出 lorekeeper 的知识库到 .sql")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出路径")
    args = ap.parse_args()
    raise SystemExit(dump(Path(args.out)))


if __name__ == "__main__":
    main()
