/**
 * 外观偏好：主题 + 自定义背景。
 *
 * 抽成独立模块而不是塞在 App 里，有两个理由：
 *  1. **首屏防闪烁**：主题必须在 React 挂载**之前**写进 DOM，
 *     所以读取逻辑必须能被 index.html 里的内联脚本复用（见 readStoredTheme）。
 *     放在组件里就只能等 useEffect，那时已经白闪一下了。
 *  2. **localStorage 的坑**：隐私模式下写 localStorage 会抛异常，
 *     不 try/catch 的话整个应用起不来 —— 这个失败很隐蔽且只在特定环境复现。
 */

export type Theme = 'light' | 'dark'

const THEME_KEY = 'kb-agent.theme'
const BG_KEY = 'kb-agent.bg'
const BG_OPACITY_KEY = 'kb-agent.bg-opacity'

/** 安全读写：隐私模式 / 配额满 / 禁用存储时都不能让应用崩掉 */
function safeGet(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function safeSet(key: string, value: string): boolean {
  try {
    localStorage.setItem(key, value)
    return true
  } catch {
    return false
  }
}

function safeRemove(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch {
    /* 忽略 */
  }
}

/**
 * 决定初始主题。顺序：**用户显式选择 > 系统偏好 > 深色**。
 * 深色放最后而不是默认，因为这个界面主要在演示时投屏，深色更稳。
 */
export function readStoredTheme(): Theme {
  const saved = safeGet(THEME_KEY)
  if (saved === 'light' || saved === 'dark') return saved
  if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: light)').matches) {
    return 'light'
  }
  return 'dark'
}

export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute('data-theme', theme)
  // theme-color（浏览器地址栏 / 移动端 Safari 的配色）**从 CSS 变量读**，
  // 不在这里写死颜色值。
  //
  // 写死过一次并且真的漂了：改成深靛配色后 CSS 的 --bg 已经是 #080d14，
  // 而这里的 meta 还停在旧配色的 #0e1116 —— 功能全对、测试全绿，
  // 只有浏览器地址栏露馅。改配色时它不会被任何构建步骤或 tsc 抓到。
  //
  // 从 CSS 读之后，CSS 是唯一的真相源：以后换配色只会影响 styles.css 一处。
  const meta = document.querySelector('meta[name="theme-color"]')
  if (!meta) return
  const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()
  // 读不到（极老的浏览器 / CSS 未加载）就保持 meta 原值，不要写成空串 ——
  // content="" 会让浏览器退回默认色，比用一个略旧的固定色更糟。
  if (bg) meta.setAttribute('content', bg)
}

export function saveTheme(theme: Theme): void {
  safeSet(THEME_KEY, theme)
}

export interface BgSettings {
  /** dataURL；空字符串表示不启用 */
  image: string
  /** 0–1，遮罩浓度。图片再好看也不能盖到影响读文字 */
  opacity: number
}

export function loadBg(): BgSettings {
  const raw = safeGet(BG_OPACITY_KEY)
  // 空串要**单独**判成「没设过」，不能直接丢给 Number()：
  // `Number('')` 是 0 而不是 NaN，所以走 isFinite 分支会被当成
  // 「用户选了完全透明」—— 而空串其实只可能是没设过（或被清空）。
  // 结果是第一次进设置面板就看到遮罩是 0，背景图直接糊住正文。
  const parsed = raw === null || raw.trim() === '' ? NaN : Number(raw)
  return {
    image: safeGet(BG_KEY) ?? '',
    // 默认 0.55 而不是看起来更"通透"的 0.35：
    // 实测 0.35 时，一行浅色文字压在亮色渐变背景上几乎不可读。
    // 遮罩的职责是**保证文字对比度**，不是展示图片 —— 图片透过一层
    // 依然看得见就已经达到目的了。
    //
    // ⚠️ 这里必须用 Number.isFinite 而不是 `Number(x) || 0.55`：
    // 后者把 **0 当成"没有值"** —— 用户在设置里明确把遮罩拉到 0
    // （"我想清楚看到背景图"），刷新后会被悄悄拉回 0.55，
    // 而界面上没有任何地方解释为什么。0 是一个合法且有意义的取值。
    // Number.isFinite 同时挡住了 NaN（存了 'abc'）和 Infinity/负数等脏值。
    opacity: Number.isFinite(parsed) ? clampOpacity(parsed) : 0.55,
  }
}

/** 遮罩浓度夹到 0–1。存进 localStorage 的值不该越界，但脏数据会。 */
function clampOpacity(v: number): number {
  return Math.min(1, Math.max(0, v))
}

export function saveBg(settings: BgSettings): boolean {
  const ok = safeSet(BG_KEY, settings.image)
  safeSet(BG_OPACITY_KEY, String(settings.opacity))
  return ok
}

export function clearBg(): void {
  safeRemove(BG_KEY)
  safeRemove(BG_OPACITY_KEY)
}

/**
 * 压缩上传的图片。
 *
 * 直接把原图塞进 localStorage 是行不通的：一张手机照片 3–8MB，
 * 而 localStorage 总配额通常只有 5MB 且**和同域所有键共享**，
 * 超了会直接抛异常把设置写不进去（而且往往不报错，只是静默失败）。
 *
 * 所以这里先缩到最长边 1600px 再转 JPEG —— 对背景图来说分辨率再高也看不出来，
 * 体积却能降到一两百 KB。
 */
export async function compressImage(file: File, maxEdge = 1600, quality = 0.82): Promise<string> {
  const bitmap = await createImageBitmap(file)
  const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height))
  const w = Math.round(bitmap.width * scale)
  const h = Math.round(bitmap.height * scale)

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('无法创建画布（浏览器不支持 canvas）')
  ctx.drawImage(bitmap, 0, 0, w, h)
  bitmap.close?.()
  return canvas.toDataURL('image/jpeg', quality)
}

/** 超过这个体积就只在本会话生效，不写 localStorage */
export const BG_MAX_BYTES = 1.5 * 1024 * 1024

export function dataUrlBytes(dataUrl: string): number {
  const i = dataUrl.indexOf(',')
  if (i < 0) return dataUrl.length
  // base64 每 4 字符 = 3 字节
  return Math.round(((dataUrl.length - i - 1) * 3) / 4)
}
