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
import sys
import time

from exp_runner import BACKEND, PY, ROOT, sh  # noqa: E402

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings  # noqa: E402


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sr")
    # 默认的对照组模型名取自 .env，而不是写死字符串 ——
    # 之前这里硬编码 "BAAI/bge-m3,embo-01"，于是 EMBED_MODEL_MAIN /
    # EMBED_MODEL_ALT / MINIMAX_EMBED_MODEL 改了都不影响这个实验跑哪两个模型。
    ap.add_argument("--models",
                    default=f"{s.embed_model_main},{s.embed_model_alt}",
                    help="逗号分隔；默认取 .env 的 EMBED_MODEL_MAIN / EMBED_MODEL_ALT")
    ap.add_argument("--variant", default="d")
    # 主基座（EMBED_MODEL_MAIN）必须复用生产 `base` 索引，不能套 `d_{slug}` 前缀。
    #
    # 踩过的坑：2026-10-13 重跑时 bge-m3 那组汇总显示"（无结果）"——
    # 它的 variant 被算成 `d_BAAI_bge-m3`，而这个 collection 从来不存在
    # （生产基座叫 `base`），于是查询打在空集合上。
    # 这个对照组本来就有现成数据（实验 A 就是 base + EMBED_MODEL_MAIN），
    # 另建一个同参数 collection 只会让"两个时期的数据"变成"两个索引的数据"。
    ap.add_argument("--main-variant", default="base",
                    help="主基座（EMBED_MODEL_MAIN）复用的 collection variant")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--labels", default="A0,A1,A2",
                    help="默认不跑 A3：rerank 是 bge 专用，换模型时开着会引入隐藏变量")
    ap.add_argument("--skip-rebuild", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"对比模型：{models}")
    print(f"参考维度：{s.minimax_embed_model} 的向量维度为 {s.minimax_embed_dim} "
          f"（换向量模型时务必确认维度一致，否则查询侧会维度不匹配）")
    for model in models:
        slug = model.replace("/", "_")
        # 主基座复用 base，对照组才用 d_{slug}
        variant = args.main_variant if model == s.embed_model_main else f"{args.variant}_{slug}"
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
                # 明确说「这一档没跑成」，并给出最常见的原因。
                # 不写清楚的话，汇总表里那一行只会被读成"这个模型指标差"。
                print(f"⚠️ {model} 索引重建失败(rc={rc})，本档**未跑成**，"
                      f"不参与任何对比结论。常见原因：模型已下线"
                      f"（接口返回 Model does not exist）、额度用尽、网络中断。")
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
    missing: list[str] = []
    for model in models:
        slug = model.replace("/", "_")
        f = out_dir / f"exp_D_{slug}_{args.game}_k{args.top_k}.json"
        if not f.is_file():
            # 「没有产物文件」有两种完全不同的原因，必须分开报：
            #   1. 索引重建失败（模型被下线 / 额度耗尽 / 网络断）—— 没跑成
            #   2. 根本没跑到这一档 —— 同样是没跑成
            # 之前这里统一打「（无结果）」，和「这个模型指标很差」在表里长得一样，
            # 读者会读成后者。2026-10-13 实测踩到：EMBED_MODEL_ALT=embo-01
            # 已被 SiliconFlow 下线，接口返回 `Model does not exist`，
            # 汇总表里 embo-01 一行「（无结果）」，很容易被当成"对比下来 embo-01 很差"。
            missing.append(model)
            print(f"{model:<20}{'（未跑成，见下方说明）':>26}")
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            a = r["aggregate"]
            print(f"{model:<20}{r['label']:>6}{a.get(f'recall@{args.top_k}', 0):>10.3f}"
                  f"{a['mrr']:>8.3f}{a['ndcg']:>8.3f}{a['avg_latency_ms']:>9}{a['n']:>5}")
    print("=" * 78)
    if missing:
        print(f"⚠️ 以下模型本轮**没有出数**（不是指标差，是没跑成）：{', '.join(missing)}")
        print("   常见原因：模型已被下线（接口返回 Model does not exist）、额度用尽、索引重建失败。")
        print("   这些模型**不能**参与任何对比结论 —— 单变量对比要求两档都真的跑过。")
    print("注：两套模型的向量维度不同，collection 完全隔离，未做任何跨模型混算。")


if __name__ == "__main__":
    main()