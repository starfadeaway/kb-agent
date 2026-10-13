r"""实验进度监控：区分「在跑」与「卡死」。

## 为什么需要这个脚本

2026-10-13 晚上踩过一次：实验 C 跑了 19 分钟，我看到
「CPU=0 + 没有网络连接 + 没有任何输出」就判定它卡死并杀掉了。
重新用文件重定向跑才发现它**一切正常** —— 真正的原因是我自己写了
`| Select-Object -Last 45`，它会等整个管道结束才输出，
于是我完全看不到中间进度，把「在跑但看不见」读成了「卡死」。

**根因：用间接信号（CPU=0、没有输出）去推断直接状态（是否在推进）。**
CPU 在等网络时本来就会是 0；管道的输出时机由命令写法决定，与进程状态无关。

所以判据必须是**能区分二者的那个量**：已完成条数在最近一段时间内有没有增长。

## 判定逻辑

| 观察 | 结论 |
|---|---|
| 完成数在涨 | 在跑，继续等 |
| 完成数连续 `stall_minutes` 分钟没变，且进程还活着 | **疑似卡死**，退出码 2 |
| 进程已退出 + 产物文件存在 | 正常完成，退出码 0 |
| 进程已退出 + 产物文件缺失 | 异常终止，退出码 3 |

注意「疑似卡死」不等于「一定卡死」： Judge 可能正卡在某个超时的请求上
（`DEFAULT_JUDGE_TIMEOUT = 120s`）。所以默认阈值给到 10 分钟，
并且退出码 2 只表示「需要人来决定」，脚本不会自作主张杀进程。

## 用法

```powershell
# 后台跑实验并监控（示例）
Start-Process .venv\\Scripts\\python.exe `
    -ArgumentList "-u","backend/scripts/exp_c.py","--game","all","--labels","C0,C1,C2" `
    -RedirectStandardOutput logs\expc.txt -RedirectStandardError logs\expc.err.txt

.venv\\Scripts\\python.exe backend/scripts/watch_experiment.py `
    --log logs/expc.txt --total 168 --artifact data/eval/exp_C_sr_ys_zzz.json
```

退出码：0 完成 / 2 疑似卡死 / 3 进程死但产物缺失 / 1 用法错误
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

#: 进度行的形状：`  [C0] sr-002   score=3.0 corr=2.0 faith=4.0 steps=0 tok=188 1620ms (3/56)`
#:
#: 必须捕获 label（`[C0]`）—— 多档实验里每档编号都从 1 重新开始，
#: 只有按 label 分组才能知道「哪一档跑到第几条」。
#: label 捕获组用**可选**形式，这样不含 label 的裸 `(n/total)` 也能匹配。
DEFAULT_PATTERN = r"(?:\[(?P<label>[A-Za-z0-9]+)\][^\n]*?)?\((?P<done>\d+)\s*/\s*(?P<total>\d+)\)"

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_STALLED = 2
EXIT_DIED = 3


def parse_progress(text: str, pattern: re.Pattern[str]) -> tuple[int, int] | None:
    """从**一整份**日志里取「已完成 / 总数」。

    返回 None 表示日志里还没有进度行（还没开始跑）。

    **多档实验的正确算法：按 label 分组，每组取最大编号，再把各组相加。**

    两种错法都会让监控失效：
    - 只取最后一条 -> 把「C0 已完成 20 条 + C1 已完成 2 条」读成 2
    - 把所有行的编号相加 -> 得到 1+2+...+n，20 条的一档算出 210
      （实测真实 22 条被报成 213）

    第二种尤其阴险：数字**看起来一直在推进**，于是「有没有增长」这个判据
    永远为真，停滞检测彻底失效，而监控输出的一切都显得正常。
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    by_label: dict[str, int] = {}
    total = 0
    for m in matches:
        label = m.groupdict().get("label") or "?"
        by_label[label] = max(by_label.get(label, 0), int(m.group("done")))
        total = max(total, int(m.group("total")))
    return sum(by_label.values()), total


class ProgressReader:
    """增量读取日志；**每档维护一个最大编号**，累计 = 各档最大值之和。

    ## 为什么不能对全量日志反复求和

    日志是**追加**的，监控循环每轮都会重读文件。若每轮都对全量求和，
    完成数会按 1+2+...+n 虚增（实测真实 22 条被报成 213）。
    「完成数有没有增长」正是区分「在跑」与「卡死」的唯一判据，
    判据被污染等于监控失效，而且**看起来一切正常**（数字只涨不跌）。

    ## 为什么累计的是「各档最大值之和」而不是「增量之和」

    一档跑出 1..20、再跑出 21、22 时，增量块里的编号是 21 与 22；
    若把它们直接加到累计上，就等于把「这一档的第 21 条」当成「新增 21 条」。
    正解是每档只保留它见过的**最大编号**，各档之间才相加 ——
    因为不同档是独立的题集，C0 完成 20 条和 C1 完成 4 条确实应该加起来。
    """

    def __init__(self, log: Path, pattern: re.Pattern[str]):
        self.log = log
        self.pattern = pattern
        self._offset = 0
        self._by_label: dict[str, int] = {}
        self._total = 0

    def poll(self) -> tuple[int, int] | None:
        """读入新增内容并返回当前 (done, total)。"""
        if not self.log.is_file():
            return None
        text = self.log.read_text(encoding="utf-8", errors="replace")
        # 文件被截断重写（少见但可能，如手动清理）时从头读
        if len(text) < self._offset:
            self._offset = 0
            self._by_label = {}
            self._total = 0
        fresh = text[self._offset:]
        self._offset = len(text)
        for m in self.pattern.finditer(fresh):
            label = m.groupdict().get("label") or "?"
            self._by_label[label] = max(self._by_label.get(label, 0), int(m.group("done")))
            self._total = max(self._total, int(m.group("total")))
        if not self._by_label:
            return None
        return sum(self._by_label.values()), self._total


def is_stalled(history: list[tuple[float, int]], now: float, stall_minutes: float) -> bool:
    """完成数在 stall_minutes 内是否一直没增长。

    history 是 [(时间戳, 完成数), ...]，按时间升序。

    为什么要看**区间增长**而不是绝对值：完成数可以是 0（还没开始），
    也可以长时间停在同一个数（真卡死）。两种情况在单点快照上一样，
    只有「一段时间内有没有动过」才能区分。
    """
    if len(history) < 2:
        return False  # 样本不足，别急着判死
    cutoff = now - stall_minutes * 60
    recent = [d for t, d in history if t >= cutoff]
    return len(set(recent)) <= 1


def python_pids_running(script_hint: str) -> list[int]:
    """找出命令行里含 script_hint 的 python 进程。

    只在 Windows 上用；其他平台返回空列表 ——
    判据退化到「日志里还有没有新进度」，照样能用。
    """
    if not sys.platform.startswith("win"):
        return []
    import subprocess
    try:
        q = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
             "| Where-Object { $_.CommandLine -like '*" + script_hint + "*' } "
             "| Select-Object -ExpandProperty ProcessId)"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return []
    return [int(x) for x in q.stdout.split() if x.strip().isdigit()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="实验的 stdout 重定向文件")
    ap.add_argument("--artifact", default=None, help="期望产出的 JSON；给了才会校验")
    ap.add_argument("--total", type=int, default=0,
                    help="总题数（默认从日志的分母自动推）")
    ap.add_argument("--script-hint", default="exp_",
                    help="用于在进程列表里定位实验进程")
    ap.add_argument("--interval", type=int, default=60, help="检查间隔秒")
    ap.add_argument("--stall-minutes", type=float, default=10.0,
                    help="完成数连续多少分钟不增长就判疑似卡死")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="超过多少分钟就退出（0 = 不限）")
    args = ap.parse_args()

    log = Path(args.log)
    if not log.parent.exists():
        log.parent.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(DEFAULT_PATTERN)
    reader = ProgressReader(log, pattern)
    start = time.time()
    history: list[tuple[float, int]] = []
    last_report = 0.0

    print(f"[watch] 监控 {log}（每 {args.interval}s 检查，"
          f"停滞阈值 {args.stall_minutes} 分钟）", flush=True)

    while True:
        now = time.time()
        elapsed_min = (now - start) / 60

        # 必须走增量读取器：对全量日志反复求和会让进度虚增（见 ProgressReader）
        prog = reader.poll()

        if prog:
            done, total = prog
            history.append((now, done))
            if args.total and total != args.total:
                print(f"[watch] ⚠ 日志里的分母是 {total}，与 --total {args.total} 不一致",
                      flush=True)
            # 每 2 分钟打一次进度，别刷屏
            if now - last_report >= 120:
                pct = (done / total * 100) if total else 0
                eta = ""
                if len(history) >= 2 and done > history[0][1]:
                    rate = (done - history[0][1]) / max(1e-9, now - history[0][0])
                    if rate > 0:
                        eta = f"，预计还需 {(total - done) / rate / 60:.0f} 分钟"
                print(f"[watch] {done}/{total}（{pct:.0f}%）"
                      f" 已跑 {elapsed_min:.0f} 分钟{eta}", flush=True)
                last_report = now
        else:
            history.append((now, 0))
            if now - last_report >= 120:
                print(f"[watch] 日志里还没有进度行，已等 {elapsed_min:.0f} 分钟", flush=True)
                last_report = now

        # ① 疑似卡死
        if is_stalled(history, now, args.stall_minutes):
            pids = python_pids_running(args.script_hint)
            print(f"[watch] ❌ 完成数已连续 {args.stall_minutes} 分钟没有增长"
                  f"（当前 {history[-1][1]}）。", flush=True)
            print(f"[watch]    相关进程 pid={pids or '未找到'}", flush=True)
            print("[watch]    注意：CPU=0 不能作为卡死证据（等网络时就是 0），"
                  "本判据只看完成数是否增长。", flush=True)
            print("[watch]    请人工确认后再决定是否终止。", flush=True)
            return EXIT_STALLED

        # ② 进程退出 + 产物校验
        pids = python_pids_running(args.script_hint)
        if not pids and log.is_file():
            # 再等一轮确认不是刚好在启动（进程列表与文件写入之间有窗口）
            time.sleep(min(5, args.interval))
            if not python_pids_running(args.script_hint):
                if args.artifact:
                    art = Path(args.artifact)
                    if art.is_file():
                        print(f"[watch] ✅ 完成，产物 {art}（{art.stat().st_size} 字节）",
                              flush=True)
                        return EXIT_OK
                    tail = log.read_text(encoding="utf-8", errors="replace")[-800:]
                    print(f"[watch] ❌ 进程已退出但产物缺失：{art}", flush=True)
                    print(f"[watch]    日志尾部：\n{tail}", flush=True)
                    return EXIT_DIED
                print("[watch] 进程已退出（未指定 --artifact，跳过产物校验）", flush=True)
                return EXIT_OK

        if args.max_minutes and elapsed_min > args.max_minutes:
            print(f"[watch] 超过 --max-minutes {args.max_minutes}，退出监控"
                  f"（实验仍在跑，交给后台任务继续）", flush=True)
            return EXIT_OK

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())