"""全局配置：所有密钥与开关只从 .env 读取，不接受代码内硬编码。"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM：Chat 基线 DeepSeek，Embedding/Rerank/Judge 走 SiliconFlow ---
    deepseek_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    deepseek_chat_model: str = field(default_factory=lambda: _env("DEEPSEEK_CHAT_MODEL", "deepseek-chat"))

    siliconflow_api_key: str = field(default_factory=lambda: _env("SILICONFLOW_API_KEY"))
    siliconflow_base_url: str = field(default_factory=lambda: _env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))

    embed_model_main: str = field(default_factory=lambda: _env("EMBED_MODEL_MAIN", "BAAI/bge-m3"))
    embed_model_alt: str = field(default_factory=lambda: _env("EMBED_MODEL_ALT", "embo-01"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))

    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "Qwen/Qwen3-8B"))
    judge_base_url: str = field(default_factory=lambda: _env("JUDGE_BASE_URL") or _env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))

    # --- MiniMax：实验 D/E 的对照组（Chat 与 Embedding 各一个）---
    minimax_api_key: str = field(default_factory=lambda: _env("MINIMAX_API_KEY"))
    minimax_base_url: str = field(default_factory=lambda: _env("MINIMAX_BASE_URL", "https://api.minimax.chat/v1"))
    minimax_chat_model: str = field(default_factory=lambda: _env("MINIMAX_CHAT_MODEL", "MiniMax-M2.5"))
    minimax_embed_model: str = field(default_factory=lambda: _env("MINIMAX_EMBED_MODEL", "embo-01"))
    minimax_embed_dim: int = field(default_factory=lambda: _env_int("MINIMAX_EMBED_DIM", 1536))

    # --- MySQL ---
    mysql_host: str = field(default_factory=lambda: _env("MYSQL_HOST", "127.0.0.1"))
    mysql_port: int = field(default_factory=lambda: _env_int("MYSQL_PORT", 3306))
    mysql_user: str = field(default_factory=lambda: _env("MYSQL_USER", "root"))
    mysql_password: str = field(default_factory=lambda: _env("MYSQL_PASSWORD"))
    mysql_db: str = field(default_factory=lambda: _env("MYSQL_DB", "kb_agent"))
    mysql_required: bool = field(default_factory=lambda: _env_bool("MYSQL_REQUIRED", False))

    # --- 向量库 ---
    chroma_path: str = field(default_factory=lambda: _env("CHROMA_PATH", "./data/chroma"))
    chroma_collection_prefix: str = field(default_factory=lambda: _env("CHROMA_COLLECTION_PREFIX", "kb"))
    # lancedb（真实向量库）| numpy（零依赖精确索引）| sqlite（最小演示后端）
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "lancedb"))
    # ANN 近似索引：默认关闭。开启会引入近似误差，改变排名，
    # 而评测必须可复现，所以只在规模大到暴力扫描吃不消时才打开。
    vector_ann: str = field(default_factory=lambda: _env("VECTOR_ANN", "false"))

    # --- 数据源 ---
    wiki_base: str = field(default_factory=lambda: _env("WIKI_BASE", "https://wiki.biligame.com"))
    game_primary: str = field(default_factory=lambda: _env("GAME_PRIMARY", "sr"))
    game_secondary: str = field(default_factory=lambda: _env("GAME_SECONDARY", "zzz"))
    # 第三个及以后的库。原神入库实测：83 页 / 236 chunk / **0 次 WAF 拦截**，
    # 全程 65 秒 —— 加一个游戏确实只要一条抓取命令。
    game_tertiary: str = field(default_factory=lambda: _env("GAME_TERTIARY", "ys"))
    wiki_min_interval: float = field(default_factory=lambda: _env_float("WIKI_MIN_INTERVAL", 2.0))
    wiki_max_retry: int = field(default_factory=lambda: _env_int("WIKI_MAX_RETRY", 4))

    # --- Agent ---
    agent_max_steps: int = field(default_factory=lambda: _env_int("AGENT_MAX_STEPS", 8))
    agent_token_budget: int = field(default_factory=lambda: _env_int("AGENT_TOKEN_BUDGET", 60000))
    agent_wall_timeout: float = field(default_factory=lambda: _env_float("AGENT_WALL_TIMEOUT", 60))
    tool_timeout: float = field(default_factory=lambda: _env_float("TOOL_TIMEOUT", 15))
    concurrency_limit: int = field(default_factory=lambda: _env_int("CONCURRENCY_LIMIT", 4))
    #: 逗号分隔的启用工具名，空 = 全部。
    #: 默认只给 search_kb —— 实验 C 实测多挂 3 个工具没有可测收益（C1 4.60 vs C2 4.47，
    #: p=0.47 未达显著）却要多付 22.6% token。改成全量前请先重跑实验 C。
    agent_tools: str = field(default_factory=lambda: _env("AGENT_TOOLS", "search_kb"))

    @property
    def agent_tool_names(self) -> list[str] | None:
        """""不限具体名称，只返回 None（= 全部工具）"或一份名称列表。"""
        raw = (self.agent_tools or "").strip()
        if not raw or raw.lower() in {"all", "*", "全部"}:
            return None
        return [x.strip() for x in raw.split(",") if x.strip()] or None

    # --- 检索 ---
    retrieve_top_k: int = field(default_factory=lambda: _env_int("RETRIEVE_TOP_K", 5))
    retrieve_candidate: int = field(default_factory=lambda: _env_int("RETRIEVE_CANDIDATE", 20))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))

    @property
    def chroma_abs_path(self) -> Path:
        p = Path(self.chroma_path)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    # slug → (显示名, 该游戏的专属术语)。
    # 术语放在这里是因为它要进 Agent 的系统提示词；如果只留在 prompt 里，
    # 就成了"有哪些游戏"的第二处定义 —— 第三个游戏加进来时必然漏掉。
    _TABLE: ClassVar[dict[str, tuple[str, str]]] = {
        "sr": ("崩坏：星穹铁道", "星铁叫「光锥/命途/位面饰品/遗器」"),
        "zzz": ("绝区零", "绝区零叫「音擎/驱动盘/邦布」"),
        "ys": ("原神", "原神叫「圣遗物/天赋/神瞳/命座」"),
    }

    @property
    def games(self) -> dict[str, str]:
        """已收录游戏 slug → 显示名。

        刻意做成 **N 个**而不是"主库 + 副库"两个字段。
        之前 `game_primary` / `game_secondary` 写死了两个，
        于是每个用到"有哪些游戏"的地方都得手写这对字段 ——
        加第三个库要改 config + tools 校验 + list_games + 预热 + 前端，五处。

        现在只有 `_TABLE` 这一处定义游戏集合。加第 N+1 个库 = 加一个 `GAME_XXX`
        字段再在 `_TABLE` 里加一行，**其余代码自动支持**。
        空的（未配置）会被过滤掉，所以不配就是不存在。
        """
        configured = [self.game_primary, self.game_secondary, self.game_tertiary]
        return {g: self._TABLE[g][0] if g in self._TABLE else g for g in configured if g}

    @property
    def game_terms(self) -> dict[str, str]:
        """已收录游戏 slug → 术语提示，拼进系统提示词用。"""
        return {g: self._TABLE[g][1] for g in self.games if g in self._TABLE}

    @property
    def game_slugs(self) -> list[str]:
        return list(self.games)

    @property
    def mysql_dsn(self) -> str:
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )

    #: repr 里要打码的字段。dataclass 的自动 repr 会把所有字段原样打出来，
    #: 于是任何一个断言失败、任何一个 traceback、任何一次 pytest 输出 fixture
    #: 都会把 API Key 和数据库密码写进日志 —— CI 上就是明文泄漏。
    _SECRET_FIELDS = ("deepseek_api_key", "siliconflow_api_key", "minimax_api_key",
                      "judge_api_key", "mysql_password")

    def __repr__(self) -> str:
        parts = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name in self._SECRET_FIELDS:
                value = f"***({len(value)} chars)" if value else ""
            parts.append(f"{f.name}={value!r}")
        return f"Settings({', '.join(parts)})"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # .env 优先从项目根找，其次 backend/，最后回退当前工作目录
    for candidate in (PROJECT_ROOT / ".env", BACKEND_DIR / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break
    return Settings()