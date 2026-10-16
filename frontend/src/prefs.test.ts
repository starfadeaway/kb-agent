/**
 * 外观偏好（prefs.ts）的守卫。
 *
 * 这个模块的价值全在**边界处理**上：隐私模式、脏数据、以及首屏那一帧。
 * 三件事都出过或差点出问题，所以这里逐条钉住。
 */
import { beforeEach, describe, expect, it } from 'vitest'

import { BG_MAX_BYTES, dataUrlBytes, loadBg, readStoredTheme, saveBg, clearBg } from './prefs'

/**
 * 最小 localStorage 替身。
 *
 * 不引 jsdom：这个模块只用到 get/set/remove 三个方法，
 * 为了三个方法拖进一整套 DOM 模拟不划算（对比：rendering 层的测试才需要 jsdom）。
 * `throwOn` 用来模拟隐私模式 —— 那正是文档里说的「隐蔽且只在特定环境复现」的失败。
 */
function makeStorage(initial: Record<string, string> = {}, throwOn?: 'get' | 'set' | 'remove') {
  const map = new Map(Object.entries(initial))
  return {
    getItem: (k: string) => {
      if (throwOn === 'get') throw new Error('SecurityError: 隐私模式下不可读')
      return map.has(k) ? map.get(k)! : null
    },
    setItem: (k: string, v: string) => {
      if (throwOn === 'set') throw new Error('QuotaExceededError')
      map.set(k, v)
    },
    removeItem: (k: string) => {
      if (throwOn === 'remove') throw new Error('SecurityError')
      map.delete(k)
    },
    _dump: () => Object.fromEntries(map),
  }
}

function installStorage(initial: Record<string, string> = {}, throwOn?: 'get' | 'set' | 'remove') {
  const s = makeStorage(initial, throwOn)
  ;(globalThis as Record<string, unknown>).localStorage = s
  return s
}

const THEME_KEY = 'lorekeeper.theme'
const BG_OPACITY_KEY = 'lorekeeper.bg-opacity'

describe('readStoredTheme', () => {
  beforeEach(() => {
    ;(globalThis as Record<string, unknown>).window = {
      matchMedia: (_query: string) => ({ matches: false }),
    }
  })

  it('用户显式选择优先于一切', () => {
    installStorage({ [THEME_KEY]: 'light' })
    expect(readStoredTheme()).toBe('light')
    installStorage({ [THEME_KEY]: 'dark' })
    expect(readStoredTheme()).toBe('dark')
  })

  it('没存过时看系统偏好', () => {
    installStorage()
    ;(globalThis as Record<string, unknown>).window = {
      matchMedia: () => ({ matches: true }), // 系统是浅色
    }
    expect(readStoredTheme()).toBe('light')
  })

  it('存了非法值必须回落，不能把界面染成一个不存在的第三主题', () => {
    // 这条最容易漏：'blue' / '' / 'true' 都曾经可能进过 localStorage
    installStorage({ [THEME_KEY]: 'blue' })
    expect(readStoredTheme()).toBe('dark')
    installStorage({ [THEME_KEY]: '' })
    expect(readStoredTheme()).toBe('dark')
  })

  it('localStorage 抛异常时返回默认值而不是让应用起不来', () => {
    // 隐私模式。这是模块 docstring 点名的失败：不 try/catch 整个应用直接白屏。
    installStorage({}, 'get')
    expect(readStoredTheme()).toBe('dark')
  })

  it('没有 window.matchMedia 时（老浏览器）不崩', () => {
    installStorage()
    ;(globalThis as Record<string, unknown>).window = {}
    expect(readStoredTheme()).toBe('dark')
  })
})

describe('loadBg 遮罩浓度', () => {
  it('默认值 0.55', () => {
    installStorage()
    expect(loadBg().opacity).toBe(0.55)
  })

  it('0 是合法取值：用户明确拉满透明，不能被当成「没设」', () => {
    // 真实缺陷：原来写的是 `Number(x) || 0.55`，
    // 而 `0 || 0.55 === 0.55` —— 用户选了「完全透明」，
    // 刷新后遮罩悄悄回到 0.55，界面上没有任何地方解释为什么。
    installStorage({ [BG_OPACITY_KEY]: '0' })
    expect(loadBg().opacity).toBe(0)
  })

  it('正常区间内的值原样返回', () => {
    installStorage({ [BG_OPACITY_KEY]: '0.35' })
    expect(loadBg().opacity).toBe(0.35)
    installStorage({ [BG_OPACITY_KEY]: '1' })
    expect(loadBg().opacity).toBe(1)
  })

  it('脏值回落到默认而不是变成 NaN 或越界数', () => {
    // NaN 会让 style 的 opacity 失效，遮罩直接消失 —— 正文压在亮背景上不可读。
    installStorage({ [BG_OPACITY_KEY]: 'abc' })
    expect(loadBg().opacity).toBe(0.55)
    installStorage({ [BG_OPACITY_KEY]: '' })
    expect(loadBg().opacity).toBe(0.55)
    // 越界值夹到 0–1
    installStorage({ [BG_OPACITY_KEY]: '5' })
    expect(loadBg().opacity).toBe(1)
    installStorage({ [BG_OPACITY_KEY]: '-2' })
    expect(loadBg().opacity).toBe(0)
  })
})

describe('saveBg / clearBg', () => {
  it('保存后能读回，包括浓度 0', () => {
    installStorage()
    expect(saveBg({ image: 'data:image/jpeg;base64,AAA', opacity: 0 })).toBe(true)
    expect(loadBg()).toEqual({ image: 'data:image/jpeg;base64,AAA', opacity: 0 })
  })

  it('写入失败时返回 false 而不是假装成功', () => {
    // 调用方靠这个返回值提示用户「这张图太大，本会话有效但存不下」
    installStorage({}, 'set')
    expect(saveBg({ image: 'x', opacity: 0.5 })).toBe(false)
  })

  it('读失败时返回默认结构而不是抛', () => {
    installStorage({}, 'get')
    expect(loadBg()).toEqual({ image: '', opacity: 0.55 })
  })

  it('清除后两个键都不剩', () => {
    const s = installStorage()
    saveBg({ image: 'abc', opacity: 0.3 })
    clearBg()
    expect(s._dump()).toEqual({})
  })

  it('localStorage 全挂时 save/load/clear 都不抛', () => {
    // 这三个函数出现在启动路径上，任何一个抛都会让首屏白屏
    installStorage({}, 'set')
    expect(() => saveBg({ image: 'x', opacity: 0.5 })).not.toThrow()
    installStorage({}, 'remove')
    expect(() => clearBg()).not.toThrow()
  })
})

describe('dataUrlBytes', () => {
  it('base64 每 4 字符 = 3 字节', () => {
    // 'AAAA' = 3 字节
    expect(dataUrlBytes('data:image/jpeg;base64,AAAA')).toBe(3)
    expect(dataUrlBytes('data:image/jpeg;base64,AAAAAAAA')).toBe(6)
  })

  it('没有逗号时退化为字符串长度', () => {
    // 纯文本输入（用户粘贴而不是上传）不该算出负数或 NaN
    expect(dataUrlBytes('abc')).toBe(3)
  })

  it('空串返回 0', () => {
    expect(dataUrlBytes('')).toBe(0)
  })

  it('上限常量存在且为 1.5MB 量级', () => {
    expect(BG_MAX_BYTES).toBeGreaterThan(1024 * 1024)
    expect(BG_MAX_BYTES).toBeLessThan(2 * 1024 * 1024)
  })
})