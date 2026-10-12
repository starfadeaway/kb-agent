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
  const meta = document.querySelector('meta[name="theme-color"]')
  if (meta) meta.setAttribute('content', theme === 'dark' ? '#0e1116' : '#f6f7f9')
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
  return {
    image: safeGet(BG_KEY) ?? '',
    // 默认 0.55 而不是看起来更"通透"的 0.35：
    // 实测 0.35 时，一行浅色文字压在亮色渐变背景上几乎不可读。
    // 遮罩的职责是**保证文字对比度**，不是展示图片 —— 图片透过一层
    // 依然看得见就已经达到目的了。
    opacity: Number(safeGet(BG_OPACITY_KEY) ?? '0.55') || 0.55,
  }
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
