/**
 * 配色单一真相源的守卫。
 *
 * ## 为什么单独开一个文件
 *
 * 上一轮把配色从暖色改成深靛 + 青之后，**漏了一处**：
 * `index.html` 的 meta theme-color 和 `prefs.ts` 的 applyTheme 里
 * 都写死了旧配色（#0e1116 / #f6f7f9）。后果是浏览器地址栏、
 * 移动端 Safari 的配色条仍然是旧配色。
 *
 * 这个缺陷极其隐蔽，因为它**不会被任何自动检查抓到**：
 * - tsc 通过 —— 两处都是合法字符串；
 * - vitest 通过 —— 没有测试读它；
 * - 构建通过 —— CSS 里没写 meta 的值；
 * - 后端 665 条测试全绿 —— 前端和后端无关。
 *
 * 只有肉眼盯浏览器才看得出来。所以必须有一条断言把它钉住。
 *
 * ## 这条测试的写法
 *
 * 它**真的去读 styles.css**，解析出深色/浅色两套主题的 --bg，
 * 再与 index.html 的 meta 初值比对。如果只是断言
 * `expect(meta).toBe('#080d14')`，那只是把同一个魔法数字抄了三遍 ——
 * 下次换配色三处一起错，测试照样绿。
 */
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const HERE = dirname(fileURLToPath(import.meta.url))
const root = resolve(HERE, '..')
const css = readFileSync(resolve(root, 'src/styles.css'), 'utf-8')
const html = readFileSync(resolve(root, 'index.html'), 'utf-8')

/** 从 CSS 里取出某个主题块中的 --bg 值。 */
function bgOf(block: 'light' | 'dark'): string {
  // ⚠️ 锚点必须带 `{`，且要跳过所有注释。
  // 曾经写成 `indexOf("[data-theme='dark']")`，结果匹配到了**文件头注释里
  // 提到这个属性名的文字**，浅色块被切成了负区间，测试直接报「找不到 --bg」。
  const stripComments = css.replace(/\/\*[\s\S]*?\*\//g, '')
  const rootAt = stripComments.indexOf(':root {')
  const darkAt = stripComments.indexOf("[data-theme='dark'] {")
  if (rootAt < 0 || darkAt < 0 || darkAt < rootAt) {
    throw new Error('styles.css 里找不到 :root { 或 [data-theme=\'dark\'] { 锚点')
  }
  const body = block === 'light' ? stripComments.slice(rootAt, darkAt) : stripComments.slice(darkAt)
  const m = body.match(/--bg:\s*(#[0-9a-fA-F]{3,8})\s*;/)
  if (!m) throw new Error(`styles.css 的 ${block} 主题里找不到 --bg`)
  return m[1]
}

describe('theme-color 与 CSS 变量保持一致', () => {
  const darkBg = bgOf('dark')
  const lightBg = bgOf('light')

  it('index.html 的 meta 初值等于深色主题的 --bg', () => {
    // html 标签默认 data-theme="dark"，首帧就是深色，所以初值必须是深色的 --bg
    const meta = html.match(/<meta name="theme-color" content="([^"]+)"/)
    expect(meta, 'index.html 里找不到 meta theme-color').toBeTruthy()
    expect(meta![1]).toBe(darkBg)
  })

  it('两套主题的 --bg 不相同（否则这段守卫就是空转）', () => {
    // 自检：万一解析逻辑坏了，两边都读成 undefined，这两条断言会一起假绿
    expect(darkBg).not.toBe(lightBg)
    expect(darkBg).toMatch(/^#[0-9a-fA-F]{6}$/)
    expect(lightBg).toMatch(/^#[0-9a-fA-F]{6}$/)
  })

  it('内联脚本与 prefs.ts 都不再写死 theme-color 颜色值', () => {
    // 写死过一次并且真的漂了。现在两处都从 CSS 变量读，
    // 所以「旧的写死色值」不该再出现在任何地方。
    //
    // 断言方式：源码里不得出现任何 hex 颜色字面量赋给 meta。
    const offenders: string[] = []
    const inline = html.slice(html.indexOf('<script>'), html.indexOf('</script>'))
    // `content', '#xxxxxx'` 形态的写死赋值
    for (const m of inline.matchAll(/setAttribute\(\s*['"]content['"]\s*,\s*['"]#[0-9a-fA-F]+['"]/g)) {
      offenders.push(`index.html 内联脚本: ${m[0]}`)
    }
    const prefs = readFileSync(resolve(root, 'src/prefs.ts'), 'utf-8')
    for (const m of prefs.matchAll(/setAttribute\(\s*['"]content['"]\s*,\s*['"]#[0-9a-fA-F]+['"]/g)) {
      offenders.push(`prefs.ts: ${m[0]}`)
    }
    expect(offenders).toEqual([])
  })

  it('applyTheme 从 --bg 读，而不是按主题名硬编码', () => {
    // 这条比上一条更强：即使将来有人「好心」写回 if (dark) '#xxx'，
    // 上面的正则抓不到（形式不同），但这条会因为 applyTheme 里
    // 出现 getPropertyValue 而通过、缺少 getPropertyValue 而失败。
    const prefs = readFileSync(resolve(root, 'src/prefs.ts'), 'utf-8')
    const fn = prefs.slice(prefs.indexOf('export function applyTheme'))
    const body = fn.slice(0, fn.indexOf('\n}'))
    expect(body, 'applyTheme 没有从 CSS 变量读 --bg，theme-color 会再次漂移').toContain(
      "getPropertyValue('--bg')",
    )
  })
})