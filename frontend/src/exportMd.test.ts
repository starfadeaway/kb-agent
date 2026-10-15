/**
 * 问答导出 Markdown 的守卫。
 *
 * 三个必须守住的地方：
 *
 * 1. **出处必须带页面修订，且措辞不能撒谎**。
 *    「页面修订」是 wiki 页面最后一次被编辑的时间，不是数据生效日期。
 *    导出的 .md 会被人拿去做核对，措辞错了比不写更糟 —— 它会被人信。
 *    禁用词与后端 `test_revision_disclosure.py` 保持同一份清单。
 *
 * 2. **运行指标必须一起导出**。
 *    只导出答案正文，这份文件就退化成一段无法核对的话 —— 而这个项目的
 *    全部价值就在「答案 + 怎么来的 + 花了多少」。指标一列都不能少。
 *
 * 3. **表格单元格必须转义竖线**。
 *    页面标题里出现 `|` 会把整张表冲散，而表一散，读者就看不出哪些修订时间
 *    属于哪个页面 —— 那比没有出处更糟。
 */
import { describe, expect, it } from 'vitest'

import { pageName, revisionSpan, stampNow, turnFileName, turnToMarkdown } from './exportMd'

/** 与 backend/tests/test_revision_disclosure.py 同一份清单 */
const FORBIDDEN_FRESHNESS = [
  '数据截至',
  '数据更新于',
  '截至今天',
  '最新版本数据',
  '当前版本数据',
]

function baseTurn(over: Partial<Parameters<typeof turnToMarkdown>[0]> = {}) {
  return {
    question: '太空封印站的两件套效果是什么？',
    game: 'sr',
    gameLabel: '崩坏：星穹铁道',
    backend: 'deepseek',
    backendLabel: 'DeepSeek',
    answer: '**太空封印站**是限定位面饰品套装。',
    at: '18:37',
    outOfScope: null,
    tools: [],
    refs: [],
    done: null,
    ...over,
  }
}

describe('turnToMarkdown', () => {
  it('包含问题与回答正文', () => {
    const md = turnToMarkdown(baseTurn())
    expect(md).toContain('太空封印站的两件套效果是什么？')
    expect(md).toContain('**太空封印站**是限定位面饰品套装。')
  })

  it('运行指标一项都不少', () => {
    const md = turnToMarkdown(
      baseTurn({
        done: {
          steps: 4,
          tool_failures: 1,
          stop_reason: 'final',
          elapsed_ms: 8123,
          prompt_tokens: 12345,
          completion_tokens: 678,
          run_id: 'abcdef1234567890',
        },
      }),
    )
    for (const needle of [
      '## 运行指标',
      '步数 | 4',
      '工具失败 | 1',
      '终止原因 | final',
      '耗时 | 8123ms',
      'prompt token | 12345',
      'completion token | 678',
      'abcdef1234567890',
    ]) {
      expect(md).toContain(needle)
    }
  })

  it('工具过程按顺序导出，含失败状态', () => {
    const md = turnToMarkdown(
      baseTurn({
        tools: [
          { name: 'search_kb', ok: true, ms: 675, chars: 1200 },
          { name: 'get_page', ok: false, ms: 120, chars: 0 },
        ],
      }),
    )
    expect(md).toContain('## 检索过程')
    expect(md).toContain('| 1 | search_kb | 完成 | 675ms | 1200 |')
    expect(md).toContain('| 2 | get_page | 失败 | 120ms | 0 |')
  })

  it('出处表带页面修订，且给出区间', () => {
    const md = turnToMarkdown(
      baseTurn({
        refs: [
          {
            url: 'https://wiki.biligame.com/sr/三月七',
            title: '三月七',
            last_revision_ts: '2026-02-13',
          },
          {
            url: 'https://wiki.biligame.com/sr/光锥/冷冽',
            title: '冷冽',
            last_revision_ts: '2025-11-02',
          },
        ],
      }),
    )
    expect(md).toContain('## 出处')
    expect(md).toContain('2025-11-02 ~ 2026-02-13')
    expect(md).toContain('三月七 | 2026-02-13')
  })

  it('禁止把页面修订说成数据时效（禁用词清单）', () => {
    const md = turnToMarkdown(
      baseTurn({
        refs: [
          {
            url: 'https://wiki.biligame.com/sr/三月七',
            title: '三月七',
            last_revision_ts: '2026-02-13',
          },
        ],
      }),
    )
    for (const bad of FORBIDDEN_FRESHNESS) {
      expect(md).not.toContain(bad)
    }
    // 反向断言：必须真的写了「页面修订」这三个字，
    // 否则上面那句 not.toContain 只是因为压根没写日期。
    expect(md).toContain('页面修订')
  })

  it('标题里的竖线被转义，表格不被冲散', () => {
    const md = turnToMarkdown(
      baseTurn({
        refs: [
          {
            url: 'https://wiki.biligame.com/sr/A|B',
            title: 'A|B',
            last_revision_ts: '2026-01-01',
          },
        ],
      }),
    )
    expect(md).toContain('A\\|B')
    // 出处表头与分隔行都还在，说明表格结构没被这一行破坏
    expect(md).toContain('| 页面 | 页面修订 | 链接 |')
    expect(md).toContain('|---|---|---|')
  })

  it('越界拒答单独标注，不能被当成检索结果', () => {
    const md = turnToMarkdown(baseTurn({ outOfScope: 'hhzr', answer: '知识库未收录该游戏。' }))
    expect(md).toContain('越界拒答')
  })

  it('无引用时不出处段，无工具时不出现检索过程', () => {
    const md = turnToMarkdown(baseTurn())
    expect(md).not.toContain('## 出处')
    expect(md).not.toContain('## 检索过程')
    expect(md).not.toContain('## 运行指标')
  })

  it('时间戳格式为本地时间且无时区后缀', () => {
    const md = turnToMarkdown(baseTurn(), '2026-10-14 18:37:53')
    expect(md).toContain('生成于 2026-10-14 18:37:53')
    // 带 +08:00 会让人以为可以拿去做跨时区对比，而它只是单次运行快照
    expect(md).not.toMatch(/生成于.*[+-]\d{2}:\d{2}/)
  })
})

describe('turnFileName', () => {
  it('用问题首句当标题，去掉结尾标点', () => {
    const name = turnFileName({
      question: '太空封印站的两件套效果是什么？',
      gameLabel: '崩坏：星穹铁道',
      done: { run_id: 'abcdef1234567890' },
    })
    expect(name).toBe('太空封印站的两件套效果是什么-abcdef12.md')
    expect(name).not.toContain('？')
  })

  it('缺 run 时不留空后缀', () => {
    const name = turnFileName({ question: '邦布等级怎么提升', gameLabel: '绝区零', done: null })
    expect(name).toBe('邦布等级怎么提升-unknown.md')
  })

  it('文件名不含 Windows 非法字符', () => {
    const name = turnFileName({
      question: '三月七/存护:攻略?',
      gameLabel: '崩坏：星穹铁道',
      done: { run_id: 'abcdef1234567890' },
    })
    for (const ch of ['\\', ':', '*', '?', '"', '<', '>', '|']) {
      expect(name).not.toContain(ch)
    }
  })
})

describe('revisionSpan / pageName', () => {
  // 这两个此前在 ChatPanel 与 exportMd 里各写一份。收口之后要锁住
  // 「空态由调用方决定」这条边界 —— 面板要空串、导出要「未知」，
  // 一旦谁把文案塞进函数体，另一边就被迫改行为。
  it('空数组返回空串，不替调用方决定兜底文案', () => {
    expect(revisionSpan([])).toBe('')
    expect(revisionSpan([undefined, undefined])).toBe('')
  })

  it('单个原样、多个取最早到最晚（内部会排序）', () => {
    expect(revisionSpan(['2026-02-13'])).toBe('2026-02-13')
    expect(revisionSpan(['2026-03-16', '2025-01-01', '2026-02-13'])).toBe(
      '2025-01-01 ~ 2026-03-16',
    )
  })

  it('wiki 域名前缀被剥掉，非 wiki URL 原样保留', () => {
    expect(pageName('https://wiki.biligame.com/sr/太空封印站')).toBe('太空封印站')
    expect(pageName('https://example.com/x')).toBe('https://example.com/x')
  })
})

describe('stampNow', () => {
  it('格式为 yyyy-mm-dd hh:mm:ss', () => {
    expect(stampNow()).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/)
  })
})