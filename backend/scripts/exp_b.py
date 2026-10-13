"""实验 B：分块参数阶梯消融（单变量）。

设计：
  B0  chunk_size=500  embed_mode=title   ← 基准，复用已有 base 索引，不重建
  B1  chunk_size=800  embed_mode=title   ← 只改 chunk_size
  B2  chunk_size=1200 embed_mode=title   ← 只改 chunk_size（继续阶梯）
  B3  chunk_size=500  embed_mode=text    ← 回到 B0 的 chunk_size，只改 embed_mode

B1/B2 是 chunk_size 的一维阶梯；B3 回到基准再单独动 embed_mode，
这样两组变量的效果不会互相污染 —— 每一步相对上一步只改一个东西。

每组写进独立 variant（collection 名 kb_sr_{model}_{variant}），
所以互不覆盖，也不用跑完就删，可以事后复查。

用法：
    .venv\\Scripts\\python.exe scripts\\exp_b.py --steps B1,B2,B3
    .venv\\Scripts\\python.exe scripts\\exp_b.py --steps B3          # 只跑一组
    .venv\\Scripts\\python.exe scripts\\exp_b.py --skip-rebuild      # 复用已有索引
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
# 解释器路径：先按 Windows 布局找，再试 POSIX 布局，都没有才退回当前解释器。
# 退回时必须提示 —— 否则可能在一个没装项目依赖的解释器里跑完整套实验，
# 最后炸在一个看不出根因的 ImportError 上。
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.is_file():
    _posix = ROOT / ".venv" / "bin" / "python"
    if _posix.is_file():
        PY = _posix
    else:
        PY = Path(sys.executable)
        print(f"[警告] 没找到项目 venv（{ROOT / '.venv'}），改用当前解释器：{PY}")

# 每一步相对 BASE 改了哪个变量，在这里写死，避免跑的时候临时改
STEPS = {
    "B0": {"variant": "base", "chunk_size": 500, "embed_mode": "title", "desc": "基准（已建索引）"},
    "B1": {"variant": "b1", "chunk_size": 800, "embed_mode": "title", "desc": "chunk_size 500→800"},
    "B2": {"variant": "b2", "chunk_size": 1200, "embed_mode": "title", "desc": "chunk_size 800→1200"},
    "B3": {"variant": "b3", "chunk_size": 500, "embed_mode": "text", "desc": "回到 500，embed_mode title→text"},
}


def sh(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)
    p = subprocess.run(cmd, cwd=str(BACKEND), env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    return p.returncode


def is_stale(path, cfg: dict, game: str) -> bool:
    """产物记录的索引 chunk 数与当前 collection 不一致 → 它描述的是另一个索引。

    为什么必须查这个：`--skip-rebuild` 下产物文件不会被刷新，
    重建索引后旧产物仍在原地，汇总时就会把两个时期的数据并排放进一张表。
    这类错误没有任何报错，表格看上去完全正常。

    判据取自产物里的 `_meta.index_chunks`（由 `exp_common.build_meta` 写入）。
    取不到时返回 False —— "能证明不一致才排除"，
    老格式产物没有这个字段，不该把所有步骤都踢掉。
    """
    try:
        blocks = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True
    meta = (blocks[0] or {}).get("_meta") if blocks else None
    recorded = (meta or {}).get("index_chunks")
    if recorded is None:
        return False

    try:
        import sys as _sys
        _sys.path.insert(0, str(BACKEND))
        from app.core.config import get_settings
        from app.store.vector_store import get_vector_store

        s = get_settings()
        col = get_vector_store(s).get_collection(game, s.embed_model_main, cfg["variant"])
        actual = len(col.get(include=[])["ids"])
    except Exception:
        return False
    return int(recorded) != actual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sr")
    ap.add_argument("--steps", default="B1,B2,B3")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--skip-rebuild", action="store_true", help="索引已存在，只跑评测")
    args = ap.parse_args()

    out_dir = ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in args.steps.split(","):
        step = step.strip().upper()
        if step not in STEPS:
            print(f"未知步骤 {step}，跳过")
            continue
        cfg = STEPS[step]

        # ---- 1. 离线重建索引（只读本地缓存，不碰 wiki，不吃 WAF 配额）----
        if not args.skip_rebuild:
            t0 = time.perf_counter()
            rc = sh([
                str(PY), "-X", "utf8", "scripts/rebuild_index.py",
                "--game", args.game,
                "--variant", cfg["variant"],
                "--chunk-size", str(cfg["chunk_size"]),
                "--embed-mode", cfg["embed_mode"],
                "--fresh",
            ], f"{step} 重建索引：{cfg['desc']}")
            if rc != 0:
                print(f"{step} 重建失败(rc={rc})，跳过该步")
                continue
            print(f"[{step}] 重建耗时 {time.perf_counter() - t0:.0f}s")

        # ---- 2. 在这个 variant 上跑 A0/A2/A3 三档检索 ----
        # 只跑这三档：纯向量 A1 在实验 A 里已证明恒定最弱且不随配置变化，
        # 分块实验里再跑一遍不会带来新信息。
        sh([
            str(PY), "-X", "utf8", "scripts/run_eval.py",
            "--experiment", "A",
            "--game", args.game,
            "--top-k", str(args.top_k),
            "--variant", cfg["variant"],
            "--labels", "A0,A2,A3",
            "--no-persist",
            "--out", str(out_dir / f"exp_{step}_{args.game}_k{args.top_k}.json"),
        ], f"{step} 评测：{cfg['desc']}")

    # ---- 汇总 ----
    print(f"\n{'=' * 70}\n  实验 B 汇总（chunk / top_k={args.top_k}）\n{'=' * 70}")
    rows = []
    stale: list[str] = []
    for step, cfg in STEPS.items():
        f = out_dir / f"exp_{step}_{args.game}_k{args.top_k}.json"
        # 原来 B0 被特殊处理去读写死的 `exp_A_k{k}_v2.json`：
        # ① 那个文件名是某一次临时跑法的产物，run_eval 早就改成别的命名了；
        # ② 重跑实验 A 之后 B0 不会跟着更新，于是汇总表里混着两个时期的数据
        #    （本次就发现 B0 的延迟还是旧的 95/147/221/567，而新 A 是 90/156/219/709）。
        # 统一按各步自己的产物读，B0 走 --skip-rebuild 跑评测即可，不必重建 base。
        if not f.is_file():
            if step == "B0":
                print(f"  提示：缺 {f.name}，B0 需要先跑一次 --steps B0 --skip-rebuild")
            else:
                print(f"  提示：缺 {f.name}，跳过 {step}")
            continue
        # 产物比索引还旧 → 这一行的数字来自另一个语料时期。
        #
        # 2026-10-13 又踩了一次：补抓角色页后 base 已到 3932 chunk，
        # 跑 `exp_b.py --skip-rebuild` 时 B0 的产物文件还是补抓前写的，
        # 汇总表里 B0 显示 0.966 而 B1/B2/B3 是新语料数字 ——
        # **一行旧数据混在一张新表里，没有任何报错**。
        #
        # 判据不能用文件 mtime（碰一下就变），要用产物里记的索引 chunk 数：
        # 它与实际 collection 行数一致才说明这份产物描述的是当前索引。
        if is_stale(f, cfg, args.game):
            stale.append(step)
            print(f"  ⚠ {step} 的产物描述的不是当前索引，**已排除出汇总**"
                  f"（跑一次 --steps {step} --skip-rebuild 重评测即可刷新）")
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            if step != "B0" and r["config"].get("variant") != cfg["variant"]:
                continue
            a = r["aggregate"]
            rows.append((step, cfg, r["label"], a))

    print(f"{'步骤':<5}{'chunk':>7}{'embed':>7}{'配置':>6}{'recall@k':>10}{'mrr':>8}{'ndcg':>8}{'延迟ms':>9}")
    for step, cfg, label, a in rows:
        print(f"{step:<5}{cfg['chunk_size']:>7}{cfg['embed_mode']:>7}{label:>6}"
              f"{a.get(f'recall@{args.top_k}', 0):>10.3f}{a['mrr']:>8.3f}{a['ndcg']:>8.3f}"
              f"{a['avg_latency_ms']:>9}")
    print("=" * 70)


if __name__ == "__main__":
    main()