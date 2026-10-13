"""实测：内容指纹的「归一化」这一步到底值多少钱。

单变量对照：唯一变化的是**哈希前是否归一化**，语料与扰动方式完全相同。

对照组：不归一化，直接对原文取哈希
实验组：先归一化再取哈希（`fingerprint.page_hash`）

扰动方式刻意模拟 wiki 上最常见的**无实质改动**编辑：
换行符从 LF 变 CRLF、行尾多一个空格、连续空行增加。
这些都不携带语义，但会让原文字符串逐位不同。

如果指纹不做归一化，这些页面会被全部判成"内容变了"，
进而触发全页重新 embedding —— 增量相对全量的收益会被直接抹平。

跑法：python backend/scripts/probe_fingerprint_precision.py
产物：data/eval/probe_fingerprint_precision.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.ingest.fingerprint import chunk_hash, normalize_text, page_hash  # noqa: E402

ROOT = BACKEND.parent
RAW = ROOT / "data" / "raw"


def perturb(text: str) -> str:
    """制造"看起来一样、字符不同"的正文。"""
    s = text.replace("\n", "\r\n")
    s = re.sub(r"(?<!\r)\n$", " \r\n", s, count=1)   # 行尾多一个空格
    s = s.replace("\r\n\r\n", "\r\n\r\n\r\n")        # 多一处空行
    return s + " \r\n"


def raw_hash(text: str) -> str:
    """对照组：不归一化，直接哈希原文。"""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    out: dict = {
        "probe": "fingerprint_precision",
        "note": "唯一变量 = 哈希前是否归一化；语料与扰动方式相同",
    }
    games = ["sr", "ys", "zzz"]
    total = 0
    fp_raw = 0     # 不归一化时被误判为"变了"的页数
    fp_norm = 0    # 归一化后仍被误判的页数（理论上为 0）
    per_game: dict = {}

    for g in games:
        d = RAW / g
        if not d.is_dir():
            continue
        metas = sorted(d.glob("*.meta.json"))
        n = r = nrm = 0
        for mp in metas:
            pid = int(mp.stem.split(".")[0])
            meta = json.loads(mp.read_text(encoding="utf-8-sig"))
            title = re.sub(r'[\\/:*?"<>|]', "_", meta.get("title", "")).strip()[:80]
            txt_p = d / f"{pid}_{title}.txt"
            if not txt_p.is_file():
                continue
            text = txt_p.read_text(encoding="utf-8")
            if not text.strip():
                continue
            n += 1
            alt = perturb(text)
            if raw_hash(text) != raw_hash(alt):
                r += 1
            if page_hash(text) != page_hash(alt):
                nrm += 1
        total += n
        fp_raw += r
        fp_norm += nrm
        per_game[g] = {
            "pages": n,
            "false_positive_without_normalize": r,
            "false_positive_with_normalize": nrm,
            "false_positive_rate_without_normalize": round(r / n * 100, 2) if n else 0,
        }
        print(f"  {g}: {n} 页，不归一化误判 {r} 页"
              f"（{r / n * 100:.1f}%）" if n else f"  {g}: 0 页")

    # 格式一致性自检：normalize 必须幂等，否则同一样本两次算出两个指纹
    sample = next((p for p in RAW.glob("*/*.txt")), None)
    s = sample.read_text(encoding="utf-8") if sample else ""
    idem = normalize_text(normalize_text(s)) == normalize_text(s)
    # 空文本安全
    empty_ok = page_hash("") == page_hash("") and chunk_hash("") == chunk_hash("")

    out.update({
        "pages_total": total,
        "false_positive_without_normalize": fp_raw,
        "false_positive_with_normalize": fp_norm,
        "false_positive_rate_without_normalize": round(fp_raw / total * 100, 2) if total else 0,
        "per_game": per_game,
        "normalize_is_idempotent": idem,
        "empty_input_safe": empty_ok,
        "chunk_hash_sample": chunk_hash(s[:500]),
    })

    print("\n-- 判定 --")
    print(f"  样本 {total} 页")
    print(f"  不归一化：{fp_raw} 页被误判为变更（{out['false_positive_rate_without_normalize']}%）")
    print(f"  归一化后：{fp_norm} 页被误判为变更")
    print(f"  normalize 幂等：{idem}")
    print(f"  空输入安全：{empty_ok}")
    if fp_raw and not fp_norm:
        print("  → 归一化把误判率从 "
              f"{out['false_positive_rate_without_normalize']}% 压到 0，"
              "增量的收益保住了")
    elif fp_norm:
        print("  → ⚠️ 归一化后仍有误判，归一化规则漏了某种扰动")

    (ROOT / "data" / "eval" / "probe_fingerprint_precision.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n产物已写 data/eval/probe_fingerprint_precision.json")
    return 0 if not fp_norm else 1


if __name__ == "__main__":
    raise SystemExit(main())
