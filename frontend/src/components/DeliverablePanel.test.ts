/**
 * 交付物面板的下载文件名守卫。
 *
 * `fileStem` / `download` 现在住在 `../exportMd`，与问答导出的 Markdown 共用 ——
 * 两边各写一份的话，交付物这边「用 run_id 当文件名」的历史 bug 会被复制一遍。
 *
 * 为什么需要这个守卫：这个面板改过「看起来没问题但用户能直接看到」的问题 ——
 * 下载文件名从 `4022ebe9d6c74a29.md`（run_id）改成 `三月七养成指南-c924fee1.md`。
 * 只靠浏览器手测的话，改回去不会有任何东西报警。
 *
 * 这里只测纯函数。渲染与网络交互不覆盖：它们需要 jsdom + fetch mock，
 * 收益不足以支撑引入那套依赖，而纯函数才是真正出过 bug 的位置。
 */
import { describe, expect, it } from 'vitest'

import { fileStem } from '../exportMd'

describe('fileStem', () => {
  it('用标题而不是 run_id 当主体，run_id 只留后 8 位', () => {
    // 这是改动本身：原来直接返回 run_id，下载下来是一串十六进制。
    const got = fileStem('三月七养成指南', 'c924fee1f42f41c2')
    expect(got.startsWith('三月七养成指南')).toBe(true)
    expect(got.endsWith('-c924fee1')).toBe(true)
  })

  it('清掉 Windows 文件名非法字符', () => {
    // 「三月七•存护/攻略」里的 / 在 Windows 上会让下载静默失败。
    // 只测「结果里没有 /」不够 —— 还要保证它被替换掉而不是整条被丢弃。
    const got = fileStem('三月七•存护/攻略', 'abcdef1234567890')
    expect(got).not.toContain('/')
    expect(got).toContain('三月七•存护')
    expect(got).toContain('攻略')
  })

  it('覆盖全部 9 个非法字符', () => {
    const got = fileStem('a\\b:c*d?e"f<g>h|i', '1234567890abcdef')
    for (const ch of ['\\', ':', '*', '?', '"', '<', '>', '|']) {
      expect(got).not.toContain(ch)
    }
  })

  it('连续空白压成一个，不留下首尾空格', () => {
    const got = fileStem('  三月七   养成   指南  ', 'abcdef1234567890')
    expect(got).toBe('三月七 养成 指南-abcdef12')
  })

  it('超长标题截到 32 字', () => {
    const got = fileStem('甲'.repeat(60), 'abcdef1234567890')
    const stem = got.slice(0, got.lastIndexOf('-'))
    expect(stem.length).toBe(32)
  })

  it('空标题回落到「交付物」而不是空文件名', () => {
    // 全是非法字符时 cleaned 会变成空串；若不回落，下载对话框里
    // 会出现一个只剩 `-abcdef12` 的文件名。
    expect(fileStem('', 'abcdef1234567890')).toBe('交付物-abcdef12')
    expect(fileStem('///', 'abcdef1234567890')).toBe('交付物-abcdef12')
  })

  it('run_id 短于 8 位时不再补齐', () => {
    // 后缀是 slice 不是 pad：短 id 不该被填成 8 位而看起来像另一个 id。
    expect(fileStem('标题', 'abc')).toBe('标题-abc')
  })

  it('同一 run 两次调用结果稳定（可重复下载同名）', () => {
    const a = fileStem('三月七养成指南', 'c924fee1f42f41c2')
    const b = fileStem('三月七养成指南', 'c924fee1f42f41c2')
    expect(a).toBe(b)
  })

  it('不同 run 同标题不会撞名', () => {
    // 后缀存在的意义就在这里：同一题面跑两次必须能区分开。
    expect(fileStem('三月七养成指南', 'aaaa1111')).not.toBe(
      fileStem('三月七养成指南', 'bbbb2222'),
    )
  })
})