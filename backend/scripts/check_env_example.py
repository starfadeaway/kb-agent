"""校验 .env.example 与本机 .env 是否覆盖 config.py 读取的全部环境变量。

新增一个配置项却忘了写进 .env.example 时，症状很隐蔽：
本机能跑（.env 里有），新克隆的人跑不起来，且只在用到那项时才炸。
所以这里从源码里正则抽出所有 _env*("KEY") 调用做双向 diff。

**还要比对本机 `.env`** —— 这是本项目真实栽过的坑：
`.env` 缺了 `GAME_TERTIARY=ys` 和 `AGENT_TOOLS=search_kb`，
但 config.py 里两者都有默认值（`"ys"` / `"search_kb"`），
于是**代码默认值悄悄兜住了，一切照常跑，不报任何错**。
表面症状是"配置文件里只有两个游戏"，实际生效的却是三个 ——
配置文件不再是事实来源，而是历史遗留。
`.env` 里还留着已经从代码删除的 `EMBED_DIM`，同样没人发现。

`_env_int("KEY", 60)` 这类第二参数就是代码默认值：
配置没写 → 用默认值 → 静默兜底。所以"缺失配置"不会炸，只会慢慢漂移。

用法：
    python scripts/check_env_example.py
退出码 1 表示有缺失/多余。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PY = PROJECT_ROOT / "backend" / "app" / "core" / "config.py"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ENV_LOCAL = PROJECT_ROOT / ".env"

# _env / _env_int / _env_float / _env_bool 的第一个位置参数就是变量名
CALL_RE = re.compile(r'_env(?:_int|_float|_bool)?\(\s*"([A-Z0-9_]+)"')
# def 内重复读取的键（如 JUDGE_BASE_URL 或 SILICONFLOW_BASE_URL 兜底）按集合去重


def parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    config_src = CONFIG_PY.read_text(encoding="utf-8")
    code_keys = {m.group(1) for m in CALL_RE.finditer(config_src)}

    example_keys = set(parse_env(ENV_EXAMPLE))

    missing = sorted(code_keys - example_keys)
    extra = sorted(example_keys - code_keys)

    print(f"config.py 读取 {len(code_keys)} 个环境变量")
    print(f".env.example 声明 {len(example_keys)} 个")
    print("-" * 60)

    if missing:
        print(f"缺失 {len(missing)} 个（.env.example 没写）：")
        for k in missing:
            print(f"  - {k}")
    if extra:
        print(f"多余 {len(extra)} 个（代码里没人读）：")
        for k in extra:
            print(f"  - {k}")
    if not missing and not extra:
        print("一致，无缺失无多余。")

    # ---- 本机 .env ----
    local_issues = False
    if ENV_LOCAL.is_file():
        local = parse_env(ENV_LOCAL)
        local_missing = sorted(code_keys - set(local))
        local_extra = sorted(set(local) - code_keys)
        print("-" * 60)
        print(f"本机 .env 声明 {len(local)} 个")
        if local_missing:
            local_issues = True
            print(f"缺失 {len(local_missing)} 个（.env 没写，当前靠 config.py 默认值兜底）：")
            for k in local_missing:
                print(f"  - {k}")
            print("  ↑ 这些键没写进 .env 时不会报错，只会静默用代码默认值。")
        if local_extra:
            local_issues = True
            print(f"多余 {len(local_extra)} 个（代码里已没人读，可删）：")
            for k in local_extra:
                print(f"  - {k}")
        if not local_missing and not local_extra:
            print("本机 .env 与代码读取的键集一致。")
    else:
        print("-" * 60)
        print(f"没有 {ENV_LOCAL.name}（新克隆的仓库属正常，跳过本机校验）")

    return 1 if (missing or extra or local_issues) else 0


if __name__ == "__main__":
    sys.exit(main())