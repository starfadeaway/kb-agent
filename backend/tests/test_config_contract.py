"""配置契约：`.env.example` 必须覆盖 `config.py` 读取的全部环境变量。

为什么值得单独一组测试：加一个新配置项却忘了写进 `.env.example` 时，
本机能跑（`.env` 里有），**新克隆的人跑不起来**，而且只在用到那项时才炸。
上一轮就漏了 6 个（`GAME_TERTIARY` + 5 个 MiniMax 变量），
结果是别人克隆下来既跑不了实验 E，也没有第三个游戏。
"""

from __future__ import annotations

import re

CONFIG_PY = None
ENV_EXAMPLE = None

_CALL_RE = re.compile(r'_env(?:_int|_float|_bool)?\(\s*"([A-Z0-9_]+)"')


def _config_path():
    from tests.conftest import BACKEND_DIR

    return BACKEND_DIR / "app" / "core" / "config.py"


def _example_path():
    from tests.conftest import PROJECT_ROOT

    return PROJECT_ROOT / ".env.example"


def _code_keys() -> set[str]:
    src = _config_path().read_text(encoding="utf-8")
    return {m.group(1) for m in _CALL_RE.finditer(src)}


def _example_keys() -> set[str]:
    out = set()
    for line in _example_path().read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            out.add(line.split("=", 1)[0].strip())
    return out


def test_env_example_has_no_missing_keys():
    missing = sorted(_code_keys() - _example_keys())
    assert not missing, f".env.example 缺少 {len(missing)} 个变量：{missing}"


def test_env_example_has_no_dead_keys():
    """反过来也要查：写了但代码里没人读 = 改了不生效的假开关。"""
    extra = sorted(_example_keys() - _code_keys())
    assert not extra, f".env.example 声明了代码里没人读的死变量：{extra}"


def test_no_api_key_committed_to_example():
    """示例文件里出现真实密钥是最严重的一种泄漏。"""
    text = _example_path().read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        if "KEY" in line.upper() and "=" in line:
            value = line.split("=", 1)[1].strip()
            assert not value or value.startswith("<"), f"示例里疑似有真密钥：{line}"


def test_games_table_covers_configured_slugs(settings):
    """`_TABLE` 是游戏集合的唯一来源，配置里出现没登记的 slug 必须报错。"""
    for slug in (settings.game_primary, settings.game_secondary, settings.game_tertiary):
        if slug:
            assert slug in settings.games