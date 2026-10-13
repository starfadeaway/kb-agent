"""requirements.txt 必须与真实 import 对得上 —— 两个方向都要抓。

## 为什么两个方向都查

`test_settings_coverage.py` 已经守着「配置项必须真被读到」。
这一条守的是它的孪生兄弟：**声明了 ≠ 实现了 ≠ 装了 ≠ 用了**。

- **装了没用**（正向）：requirements 里躺着一个谁也不 import 的包。
  危害是具体的 —— CI 每次多装一遍、部署镜像变大、而且让人以为
  「这个项目的日志/HTML 解析是走 X 的」。本项目 2026-10 因此删掉了
  loguru / beautifulsoup4 / lxml / python-multipart / pydantic-settings。
- **用了没装**（反向）：代码 import 了一个 requirements 里没有的包。
  危害是本机有、CI 没有 —— 换台机器才发现。

这两个方向单看都像小事，合起来就是「文档与现实脱节」。
项目定位是简历作品，声明必须能对上。

## 白名单的边界

`_TRANSIITIVE` 里每一项都必须写明**由哪个直接依赖带入**，
不写理由的条目不许加 —— 空理由等于给「懒得核实」开后门。

⚠ 有**一个反例必须显式保留**：`cryptography` 看起来也是传递依赖残留，
但 MySQL 8 的默认认证插件 `caching_sha2_password` 在 RSA 交换时会走到
`PyMySQL._auth.sha2_rsa_encrypt()`，而 `_auth.py` 顶层就 import 它。
删掉它 MySQL 连接会在认证阶段炸，报错完全指不到原因。
它被钉在 requirements 里正是为此，`test_cryptography_must_stay` 锁住这件事。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REQUIREMENTS = BACKEND_DIR / "requirements.txt"

#: 扫描范围。**不含 tests/** —— 测试可以 import 任何东西，
#: 那是测试自己的责任，而且 requirements 里 pytest 已经是 dev 依赖。
SCAN_DIRS = ("app", "scripts")


def _req_packages() -> set[str]:
    """requirements.txt 里的包名（已去注释/版本号/extras，统一小写连字符）。"""
    names: set[str] = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[; ]", line)[0].strip()
        if name:
            names.add(name.lower().replace("_", "-"))
    return names


def _imported_top_level() -> set[str]:
    """app/ 与 scripts/ 下所有 `import X` / `from X import` 的顶层模块名。

    用 ast 而不是正则：正则分不清 `import os, sys` 和
    `from a import os`，在注释和字符串里也会误命中。
    """
    mods: set[str] = set()
    for d in SCAN_DIRS:
        for p in (BACKEND_DIR / d).rglob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods.add(node.module.split(".")[0])
    return mods


#: import 名 ≠ pip 名的那几个。
_IMPORT_TO_PIP = {
    "dotenv": "python-dotenv",   # python-dotenv 提供的顶层模块叫 dotenv
    "rank_bm25": "rank-bm25",
}

#: 项目自有包。`scripts/` 下的脚本会互相 import（`from exp_common import ...`、
#: `from crawl import ...`），这些是**同级脚本模块**而不是第三方包 ——
#: 它们靠各脚本顶部的 sys.path bootstrap 解析，不能当外部依赖。
_SCRIPT_MODULES = {p.stem for p in (BACKEND_DIR / "scripts").glob("*.py")}
_PROJECT_LOCAL = {"app", "scripts"} | _SCRIPT_MODULES

#: 传递依赖：代码可能直接 import，但由 requirements 里**已列出的**
#: 直接依赖带入，因此不必（也不该）在 requirements 里重复声明。
#: 每项都要写清楚由谁带入 —— 见模块 docstring 的「白名单的边界」。
_TRANSIITIVE: dict[str, str] = {
    "pymysql": "aiomysql 需要它驱动 MySQL 协议",
    "greenlet": "SQLAlchemy 的 asyncio 支持需要它",
    # cryptography 不在白名单里：它是必须显式钉住的，理由见 docstring。
    "anyio": "httpx / starlette 带入",
    "h11": "httpx / uvicorn 带入",
    "httpcore": "httpx 带入",
    "httptools": "uvicorn[standard] 带入",
    "idna": "httpx 带入",
    "certifi": "httpx 带入",
    "sniffio": "httpx / anyio 带入",
    "click": "uvicorn 带入",
    "watchfiles": "uvicorn[standard] 带入",
    "websockets": "uvicorn[standard] 带入",
    "pyarrow": "lancedb 带入",
    "pydantic_core": "pydantic 带入",
    "annotated_types": "pydantic 带入",
    "typing_extensions": "pydantic / fastapi 带入",
    "distro": "openai 带入",
    "jiter": "openai 带入",
    "tqdm": "openai 带入",
    "referencing": "jsonschema 带入（mcp SDK 依赖）",
    "attrs": "jsonschema 带入（mcp SDK 依赖）",
}
# 早先这里还列过 uvloop / deprecated / rpds / python_dotenv，四项都**已删除**：
# 它们在本环境根本不存在（uvloop 是 Windows 不装），而 app/ 与 scripts/ 里
# 也没有任何地方 import 它们 —— 也就是说这些白名单项从来没起过作用，
# 只是让「白名单里的东西都是真的」这个前提悄悄失效了。
# 由 test_transitive_allowlist_entries_are_real_packages 把住这个前提。


#: 开发工具：扫描范围**不含 tests/**，所以它们天然不会被 app/ scripts/ import。
#: 但仍要显式钉版本 —— 跨 pytest / ruff 大版本换行为会让 CI 结果不可比。
_DEV_ONLY = {
    "pytest": "测试框架，扫描范围不含 tests/",
    "pytest-asyncio": "pytest 插件，扫描范围不含 tests/",
    "ruff": "lint/format，扫描范围不含 tests/",
}

#: **本项目代码不 import、但必须显式声明**的运行时依赖。
#: 共同点：它们由部署环境或第三方包使用，出现在 import 图之外 ——
#: 因此「没被 import」不能推出「没用」。
_RUNTIME_ONLY = {
    "uvicorn": "ASGI 服务器，由启动入口/命令行拉起，本就不出现在业务 import 图里",
    "starlette": (
        "FastAPI 的底层 ASGI 实现，项目不直接 import；"
        "但**必须钉 0.41.x** —— mcp 2.3.0 要求 starlette>=0.48，"
        "不钉死就会被顶到 1.8.0 并让 FastAPI 连构造都失败"
    ),
    "cryptography": (
        "由 PyMySQL._auth 顶层 import，MySQL 8 默认的 caching_sha2_password "
        "认证在 RSA 交换时要用；删掉会在认证阶段炸且报错指不到原因"
    ),
}


def test_every_declared_package_is_actually_imported():
    """装了没用的包不许留在 requirements 里。"""
    declared = _req_packages()
    imported = _imported_top_level()
    pip_of_imported = {
        _IMPORT_TO_PIP.get(m, m).lower().replace("_", "-") for m in imported
    }
    allowed = set(_TRANSIITIVE) | set(_DEV_ONLY) | set(_RUNTIME_ONLY)
    unused = sorted(declared - pip_of_imported - allowed)

    assert not unused, (
        f"这些包在 requirements.txt 里但 app/ 与 scripts/ 都没 import 过：{unused}\n"
        f"要么删掉，要么移进 _TRANSITIVE / _DEV_ONLY / _RUNTIME_ONLY，"
        f"并写清它为什么合理地不出现于 import 图。\n"
        f"注意反过来也可能是真的漏了 import —— 先 grep 确认再删。"
    )


def test_no_module_is_imported_without_being_installable():
    """用了没装的包必须在 requirements 里（或在白名单里说明它是被带入的）。"""
    declared = _req_packages()
    external = _imported_top_level() - set(sys.stdlib_module_names) - _PROJECT_LOCAL
    missing = sorted(
        m for m in external
        if _IMPORT_TO_PIP.get(m, m).lower().replace("_", "-") not in declared
        and m not in _TRANSIITIVE
    )
    assert not missing, (
        f"这些模块在 app/ scripts/ 里被 import，但 requirements.txt 里没有：{missing}\n"
        f"本机能跑是因为它们是别的包的传递依赖，换个环境/CI 就会 ModuleNotFoundError。"
        f"\n修法：要么加进 requirements，要么证明它由已列依赖稳定带入、"
        f"并写进 _TRANSIITIVE 说明由谁带入。"
    )


def test_transitive_allowlist_has_no_empty_reasons():
    """白名单不许有空理由 —— 空理由等于给未来的懒于核实开后门。"""
    empty = [
        k for k, v in list(_TRANSIITIVE.items()) + list(_DEV_ONLY.items())
        + list(_RUNTIME_ONLY.items())
        if not v.strip()
    ]
    assert not empty, f"这些例外项没有写理由：{empty}"


def test_cryptography_must_stay():
    """`cryptography` 看着像残留，但它有真实的运行路径，删了会在认证阶段炸。

    固定断言：`cryptography` 必须留在 requirements.txt。
    若将来确认不再需要 MySQL，请连同这条断言一起删，别默默删包。
    """
    assert "cryptography" in _req_packages(), (
        "cryptography 不能删：MySQL 8 默认的 caching_sha2_password 认证"
        "要走 PyMySQL._auth.sha2_rsa_encrypt()，而 _auth.py 顶层 import 它。"
    )


def test_removed_packages_stay_removed():
    """本轮清掉的死依赖，防回退。

    每个都要写明「为什么当初以为需要、实际上为什么不需要」，
    免得半年后有人看到 FastAPI 就又把 python-multipart 加回来。
    """
    declared = _req_packages()
    removed = {
        "loguru": "全仓用标准库 logging",
        "beautifulsoup4": "wiki 源是 wikitext，走正则清洗，从不解析 HTML",
        "lxml": "同上，bs4 的配套依赖",
        "python-multipart": "FastAPI 只有出现 Form/File 端点时才需要，本项目没有",
        "pydantic-settings": "Settings 是自己写的 frozen dataclass + python-dotenv",
    }
    back = sorted(p for p in removed if p in declared)
    assert not back, f"这些已确认无用的依赖又回来了：{back}"