"""守卫：实验监控脚本的判据必须能区分「在跑」与「卡死」。

## 背景

2026-10-13 晚上，实验 C 跑了 19 分钟，我用「CPU=0 + 无网络连接 + 无输出」
判定它卡死并杀掉。重新用文件重定向跑才发现它**一切正常**。

根因是判据用错了量：

- **CPU=0**：等网络时本来就是 0，与是否卡死无关
- **「没有输出」**：取决于命令写法（`| Select-Object -Last` 会等管道结束才输出），
  与进程状态无关

唯一能区分二者的量是**完成数在最近一段时间内有没有增长**。
这三条守卫锁的就是这个判据，以及「多档实验的进度怎么算」这个易错点。
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "watch_experiment.py"


def _load():
    spec = importlib.util.spec_from_file_location("watch_exp_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["watch_exp_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


#: 与脚本里的 DEFAULT_PATTERN 保持一致（含 label 捕获组）。
#: 独立抄一份是为了让测试真的验证「脚本用的就是这条正则」——
#: 直接 import 脚本的常量的话，正则写错了测试也跟着错。
PAT = re.compile(
    r"(?:\[(?P<label>[A-Za-z0-9]+)\][^\n]*?)?"
    r"\((?P<done>\d+)\s*/\s*(?P<total>\d+)\)"
)


def test_progress_sums_across_labels_not_last_line():
    """多档实验的完成数必须**累加**，不能取最后一条。

    实验 C 有三档，每档的编号都从 1 重新开始。
    早先的直觉写法「取日志里最后一条进度」会得到「C1 跑到第 2 条」，
    而实际已完成 20 + 2 = 22 条 —— 进度被低估一个数量级，
    停滞检测会因此在 C1 刚开始时就误报「卡死」。
    """
    mod = _load()
    lines = [f"  [C0] sr-{i:03d}   score=3.0 ({i}/56)" for i in range(1, 21)]
    lines += [f"  [C1] sr-{i:03d}   score=4.0 ({i}/56)" for i in range(1, 3)]
    done, total = mod.parse_progress("\n".join(lines), PAT)
    assert done == 22, f"多档完成数应累加，实际 {done}（取最后一条会得到 2）"
    assert total == 56


def test_progress_reader_does_not_double_count(tmp_path):
    """增量读取：反复 poll 同一份日志，完成数不能虚增。

    这是本脚本最关键的一条守卫。监控循环每轮都重读日志，
    若对全量求和，完成数会按 1+2+...+n 涨（第 7 轮就从 22 变成 213），
    于是「完成数有没有增长」永远为真 —— 停滞检测彻底失效，
    而监控看起来一切正常（进度只涨不跌）。
    """
    mod = _load()
    log = tmp_path / "exp.txt"
    lines = [f"  [C0] sr-{i:03d}   score=3.0 ({i}/56)" for i in range(1, 21)]
    lines += [f"  [C1] sr-{i:03d}   score=4.0 ({i}/56)" for i in range(1, 3)]
    log.write_text("\n".join(lines), encoding="utf-8")

    reader = mod.ProgressReader(log, PAT)
    first = reader.poll()
    assert first == (22, 56), f"首次应读到全量 22，实际 {first}"

    # 再 poll 两次，日志没有新增内容 —— 计数必须保持不变
    for _ in range(5):
        assert reader.poll() == (22, 56), (
            "日志没有新增内容时计数却变了 —— 说明在重复累加全量，"
            "停滞检测会因此永远为真"
        )

    # 追加 2 条真实的新进度
    with log.open("a", encoding="utf-8") as f:
        f.write("\n  [C1] sr-003   score=4.2 (3/56)")
        f.write("\n  [C1] sr-004   score=4.4 (4/56)")
    assert reader.poll() == (24, 56), "新增的两条应当被计入"


def test_progress_reader_handles_log_truncation(tmp_path):
    """日志被截断重写时要从头读，不能沿用旧偏移。"""
    mod = _load()
    log = tmp_path / "exp.txt"
    log.write_text("\n".join(f"x ({i}/56)" for i in range(1, 30)), encoding="utf-8")
    reader = mod.ProgressReader(log, PAT)
    assert reader.poll() == (29, 56)

    log.write_text("x (1/56)", encoding="utf-8")  # 截断重写
    assert reader.poll() == (1, 56), "截断后应从新内容重新计数"


def test_progress_none_before_any_line():
    """还没有进度行时返回 None，而不是 (0, 0)。"""
    mod = _load()
    assert mod.parse_progress("正在加载 jieba 词典...", PAT) is None
    assert mod.parse_progress("", PAT) is None


def test_stall_detection_uses_growth_not_absolute_value():
    """停滞判据必须是「一段时间内没有增长」，不是「完成数等于某个值」。

    完成数可以是 0（刚开始），也可以长时间停住（真卡死）——
    单点快照上这两种情况完全一样，只有区间增长才能区分。
    """
    mod = _load()
    # 一直停在 3 —— 疑似卡死
    stalled = [(0.0, 3), (60.0, 3), (120.0, 3), (700.0, 3)]
    assert mod.is_stalled(stalled, now=720.0, stall_minutes=10.0) is True

    # 一直在增长 —— 不能误报
    progressing = [(0.0, 1), (60.0, 5), (120.0, 9), (700.0, 40)]
    assert mod.is_stalled(progressing, now=720.0, stall_minutes=10.0) is False


def test_stall_ignores_growth_before_the_window():
    """窗口之前涨过不算数 —— 判据问的是「最近有没有动」。

    曾经从 1 涨到 40、然后卡在 40 十分钟的实验，
    单看历史会以为「它动过所以没卡死」。
    """
    mod = _load()
    hist = [(0.0, 1), (60.0, 40), (300.0, 40), (700.0, 40), (800.0, 40)]
    assert mod.is_stalled(hist, now=800.0, stall_minutes=10.0) is True


def test_stall_needs_at_least_two_samples():
    """样本不足时不能判死 —— 刚启动时只有一个点。"""
    mod = _load()
    assert mod.is_stalled([(0.0, 0)], now=1000.0, stall_minutes=1.0) is False


def test_script_does_not_treat_cpu_as_evidence():
    """脚本里不得把 CPU 占用当作卡死判据。

    这正是本轮踩的坑：CPU=0 在等网络时是常态。
    脚本必须明确说明这一点，避免后人又加回去。
    """
    src = SCRIPT.read_text(encoding="utf-8")
    assert "CPU" in src, "脚本应显式说明 CPU 不是判据"
    assert "不能作为卡死证据" in src, (
        "脚本必须写明「CPU=0 不能作为卡死证据」，"
        "否则下一个人会再犯同样的误判"
    )
    # 不得真的去读 CPU
    assert "cpu_percent" not in src.lower(), "不应采集 CPU 作为判据"
    assert "times(" not in src, "不应读取进程 CPU 时间"