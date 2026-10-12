"""实验 D：embedding 模型对比。

三个前提，缺任何一个结论就无效：

1. **关闭重排** —— bge-reranker-v2-m3 是 bge 系的，对 embo-01 没有意义，
   开着等于给对照组套了一件不适配的外衣。
2. **双侧归一化** —— bge-m3 实测范数不为 1；embo-01 的范数分布又不一样。
   不归一化就换模型，对比里会混进"范数分布"这个隐藏变量。
   （`LLMClient.embed()` 默认 normalize=True，索引和查询两侧都会走归一化。）
3. **独立 collection** —— collection 名带模型名，两套向量不能混在一个库里。

对比时**只跑 A0/A1/A2，不跑 A3** —— 理由同 1。

用法：
    .venv\\Scripts\\python.exe scripts\\exp_d.py --models BAAI/bge-m3,embo-01
    .venv\\Scripts\\python.exe scripts\\exp_d.py --skip-rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.is_file():
    PY = Path(sys.executable)


def sh(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)
    p = subprocess.run(cmd, cwd=str(BACKEND),
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sr")
    ap.add_argument("--models", default="BAAI/bge-m3,embo-01")
    ap.add_argument("--variant", default="d")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--labels", default="A0,A1,A2",
                    help="默认不跑 A3：rerank 是 bge 专用，换模型时开着会引入隐藏变量")
    ap.add_argument("--skip-rebuild", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        slug = model.replace("/", "_")
        variant = f"{args.variant}_{slug}"
        t0 = time.perf_counter()

        if not args.skip_rebuild:
            rc = sh([
                str(PY), "-X", "utf8", "scripts/rebuild_index.py",
                "--game", args.game,
                "--variant", variant,
                "--chunk-size", "500",
                "--embed-mode", "title",
                "--fresh",
            ] + (["--embed-model", model] if model != "BAAI/bge-m3" else []),
                f"重建索引：{model} → collection kb_{args.game}_{slug}_{variant}")
            if rc != 0:
                print(f"{model} 重建失败(rc={rc})，跳过")
                continue
            print(f"[{model}] 重建耗时 {time.perf_counter() - t0:.0f}s")

        sh([
            str(PY), "-X", "utf8", "scripts/run_eval.py",
            "--experiment", "A",
            "--game", args.game,
            "--top-k", str(args.top_k),
            "--variant", variant,
            "--embed-model", model,
            "--labels", args.labels,
            "--no-persist",
            "--out", str(out_dir / f"exp_D_{slug}_{args.game}_k{args.top_k}.json"),
        ], f"评测 {model}（labels={args.labels}，不含重排）")

    # ---- 汇总 ----
    print(f"\n{'=' * 78}\n  实验 D 汇总：embedding 模型对比（不含重排，双侧归一化，top_k={args.top_k}）\n{'=' * 78}")
    print(f"{'模型':<20}{'配置':>6}{'recall@k':>10}{'mrr':>8}{'ndcg':>8}{'延迟ms':>9}{'n':>5}")
    for model in models:
        slug = model.replace("/", "_")
        f = out_dir / f"exp_D_{slug}_{args.game}_k{args.top_k}.json"
        if not f.is_file():
            print(f"{model:<20}{'（无结果）':>20}")
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            a = r["aggregate"]
            print(f"{model:<20}{r['label']:>6}{a.get(f'recall@{args.top_k}', 0):>10.3f}"
                  f"{a['mrr']:>8.3f}{a['ndcg']:>8.3f}{a['avg_latency_ms']:>9}{a['n']:>5}")
    print("=" * 78)
    print("注：两套模型的向量维度不同，collection 完全隔离，未做任何跨模型混算。")


if __name__ == "__main__":
    main()