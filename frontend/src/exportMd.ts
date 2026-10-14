/**
 * Markdown 导出。**问答气泡与交付物面板共用同一套逻辑**。
 *
 * ## 为什么不各自写一份
 *
 * 交付物那边已经踩过一次：文件名直接用 `run_id`，下载下来是
 * `4022ebe9d6c74a29.md` —— 对人类毫无意义，多下几次就分不清是哪一份。
 * 现在两边共用 `fileStem`，包括 Windows 非法字符清洗那条规则：
 * 浏览器不会替你转义 `\/:*?"<>|`，而标题里出现「/」极常见
 * （「三月七•存护/攻略」），非法字符会让下载静默失败或被系统改写。
 *
 * ## 时间戳用本地时间、无时区后缀
 *
 * 与后端 `deliverable._stamp_now()` 同口径。产物是**单次运行的快照**，
 * 不参与跨时区对比，带 `+08:00` 反而让人以为可以拿去做时间序列分析。
 * 需要机器可读的那份（`_meta`、实验产物）走 ISO，不走这里。
 */

import type { Ref } from './types'

/**
 * 标题 → 下载文件名（不含扩展名）。
 *
 * 保留 run_id 作为**后缀**而不是前缀：它是同一份产物的唯一标识，
 * 排查时对得上 `_meta.run_id` 与产物里的 run 字段。
 *
 * 导出是因为这段逻辑改过一次（原本用 run_id 当文件名，
 * 下载下来是 `4022ebe9d6c74a29.md`），纯函数 + 有测试才不会再退回去。
 */
export function fileStem(topic: string, runId: string): string {
  const cleaned = (topic || '')
    .replace(/[\\/:*?"<>|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  // 超长标题会让下载对话框难读，且各平台的文件名长度上限不一。
  const stem = cleaned.slice(0, 32).trim()
  return `${stem || '交付物'}-${runId.slice(0, 8)}`
}

/** 触发浏览器下载。文件名由调用方给全（含扩展名）。 */
export function download(name: string, text: string, type: string) {
  const blob = new Blob([text], { type })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}

/** `2026-10-14 18:37:53` —— 与后端交付物头部一致 */
export function stampNow(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
  )
}

/** Markdown 表格里的单元格：竖线与换行会破坏表格结构，必须转义。 */
function cell(s: string): string {
  return String(s ?? '').replace(/\|/g, '\\|').replace(/\n/g, ' ')
}

/**
 * 一轮问答 → Markdown。
 *
 * 刻意把**工具过程与运行指标一起导出**，而不只是答案正文：
 * 这份产物的价值就在于「答案 + 它是怎么来的 + 花了多少」，
 * 只导出正文会让它退化成一段无法核对的话。
 */
export function turnToMarkdown(
  turn: {
    question: string
    game: string
    gameLabel: string
    backend: string
    backendLabel: string
    answer: string
    at: string
    outOfScope?: string | null
    tools?: Array<{ name: string; ok?: boolean; ms?: number; chars?: number; running?: boolean }>
    refs: Ref[]
    done?: {
      steps: number
      tool_failures: number
      stop_reason: string
      elapsed_ms: number
      prompt_tokens: number
      completion_tokens: number
      run_id: string
    } | null
  },
  stamp = stampNow(),
): string {
  const out: string[] = []

  out.push(`# ${turn.gameLabel} · 问答`)
  out.push('')
  out.push(
    `> 生成于 ${stamp}　·　知识库：${turn.game}　·　模型：${turn.backendLabel}` +
      (turn.done ? `　·　耗时 ${turn.done.elapsed_ms}ms` : ''),
  )
  out.push('')
  out.push('## 问题')
  out.push('')
  out.push(turn.question)
  out.push('')

  if (turn.outOfScope) {
    out.push('> **本轮为越界拒答**：该游戏未收录，以下是边界说明而非检索结果。')
    out.push('')
  }

  out.push('## 回答')
  out.push('')
  out.push(turn.answer || '（本次没有输出内容）')
  out.push('')

  if (turn.tools?.length) {
    out.push('## 检索过程')
    out.push('')
    out.push('| # | 工具 | 结果 | 耗时 | 返回字符 |')
    out.push('|---|---|---|---|---|')
    turn.tools.forEach((t, i) => {
      const status = t.running ? '进行中' : t.ok === false ? '失败' : '完成'
      out.push(
        `| ${i + 1} | ${cell(t.name)} | ${status} | ${t.ms ?? '—'}ms | ${t.chars ?? '—'} |`,
      )
    })
    out.push('')
  }

  if (turn.refs.length) {
    const known = turn.refs.map((r) => r.last_revision_ts).filter(Boolean)
    const span = known.length
      ? known.length === 1
        ? known[0]
        : `${known.slice().sort()[0]} ~ ${known.slice().sort()[known.length - 1]}`
      : '未知'

    out.push('## 出处')
    out.push('')
    out.push(
      `> 「页面修订」是 wiki 页面最后一次被编辑的时间，**不是数据生效日期** —— ` +
        `本页两年没被动过，通常意味着 wiki 认为它不需要改，而不是内容已过时。`,
    )
    out.push('')
    out.push(`页面修订区间：${span}`)
    out.push('')
    out.push('| 页面 | 页面修订 | 链接 |')
    out.push('|---|---|---|')
    turn.refs.forEach((r) => {
      const name = r.url.replace(/^https?:\/\/wiki\.biligame\.com\//, '')
      out.push(`| ${cell(name)} | ${r.last_revision_ts || '未知'} | ${r.url} |`)
    })
    out.push('')
  }

  if (turn.done) {
    out.push('## 运行指标')
    out.push('')
    out.push('| 指标 | 值 |')
    out.push('|---|---|')
    out.push(`| 模型 | ${cell(turn.backendLabel)} |`)
    out.push(`| 步数 | ${turn.done.steps} |`)
    out.push(`| 工具失败 | ${turn.done.tool_failures} |`)
    out.push(`| 终止原因 | ${cell(turn.done.stop_reason)} |`)
    out.push(`| 耗时 | ${turn.done.elapsed_ms}ms |`)
    out.push(`| prompt token | ${turn.done.prompt_tokens} |`)
    out.push(`| completion token | ${turn.done.completion_tokens} |`)
    out.push(`| Run | ${turn.done.run_id} |`)
    out.push('')
  }

  return out.join('\n')
}

/** 导出文件名：与交付物同一套规则 */
export function turnFileName(turn: { question: string; gameLabel?: string; done?: { run_id: string } | null }): string {
  // 用「问题首句」当标题比用整句问题好：整句常有 30+ 字，截断后会失去辨识度。
  // 末位再兜一层库名 —— 问题是空串时（不该发生，但守卫测得到）仍要有可辨识的名字。
  const head = turn.question.trim().split(/[？?。\n]/)[0] || turn.gameLabel || '问答'
  return `${fileStem(head, turn.done?.run_id ?? 'unknown')}.md`
}