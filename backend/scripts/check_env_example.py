"""校验 .env.example 是否覆盖 config.py 里读取的全部环境变量。

新增一个配置项却忘了写进 .env.example 时，症状很隐蔽：
本机能跑（.env 里有），新克隆的人跑不起来，且只在用到那项时才炸。
所以这里从源码里正则抽出所有 _env*("KEY") 调用做双向 diff。

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

# _env / _env_int / _env_float / _env_bool 的第一个位置参数就是变量名
CALL_RE = re.compile(r'_env(?:_int|_float|_bool)?\(\s*"([A-Z0-9_]+)"')
# def 内重复读取的键（如 JUDGE_BASE_URL 或 SILICONFLOW_BASE_URL 兜底）按集合去重


def main() -> int:
    config_src = CONFIG_PY.read_text(encoding="utf-8")
    code_keys = {m.group(1) for m in CALL_RE.finditer(config_src)}

    example_keys = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        example_keys.add(line.split("=", 1)[0].strip())

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
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())