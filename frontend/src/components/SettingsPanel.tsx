import { useRef, useState } from 'react'
import {
  BG_MAX_BYTES,
  applyTheme,
  clearBg,
  compressImage,
  dataUrlBytes,
  saveBg,
  saveTheme,
  type BgSettings,
  type Theme,
} from '../prefs'

interface Props {
  theme: Theme
  bg: BgSettings
  /** 传入 setTheme（而不是 toggle）——面板里点「浅色」是明确指定，不该是"翻转" */
  onTheme: (t: Theme) => void
  onBg: (b: BgSettings) => void
  onClose: () => void
}

/**
 * 外观设置面板。
 *
 * 背景图走「压缩 → 存 localStorage」而不是直接存原图：
 * 手机照片 3–8MB 远超 localStorage 配额（通常 5MB 且全域共享）。
 * 压缩后通常 100–300KB。压完还是太大就只在本会话生效并明确告诉用户，
 * 而不是静默失败 —— 静默失败会让人以为"功能坏了"。
 */
export function SettingsPanel({ theme, bg, onTheme, onBg, onClose }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [notice, setNotice] = useState('')

  const pick = async (file: File | undefined) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setNotice('请选择图片文件')
      return
    }
    try {
      const dataUrl = await compressImage(file)
      const bytes = dataUrlBytes(dataUrl)
      const next = { ...bg, image: dataUrl }
      onBg(next)
      if (bytes > BG_MAX_BYTES) {
        setNotice(
          `图片约 ${(bytes / 1024 / 1024).toFixed(1)}MB，超过 ${(
            BG_MAX_BYTES / 1024 / 1024
          ).toFixed(1)}MB，只在本次会话生效，刷新后需重新设置。`,
        )
      } else {
        const ok = saveBg(next)
        setNotice(ok ? '已保存到本地' : '浏览器拒绝写入本地存储，仅本次会话生效')
      }
    } catch (e) {
      setNotice(`处理失败：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const remove = () => {
    onBg({ ...bg, image: '' })
    clearBg()
    setNotice('已清除背景')
  }

  return (
    <div className="settings" role="dialog" aria-label="外观设置">
      <h3>外观</h3>

      <div className="settings-section">
        <span className="settings-label">主题</span>
        <div className="segmented">
          <button
            className={theme === 'light' ? 'seg active' : 'seg'}
            onClick={() => {
              applyTheme('light')
              saveTheme('light')
              onTheme('light')
            }}
          >
            浅色
          </button>
          <button
            className={theme === 'dark' ? 'seg active' : 'seg'}
            onClick={() => {
              applyTheme('dark')
              saveTheme('dark')
              onTheme('dark')
            }}
          >
            深色
          </button>
        </div>
      </div>

      <div className="settings-section">
        <span className="settings-label">自定义背景</span>
        <div
          className="preview"
          style={bg.image ? { backgroundImage: `url(${bg.image})` } : undefined}
        >
          {!bg.image && '未设置'}
        </div>
        <div className="settings-actions">
          <button className="ghost mini" onClick={() => fileRef.current?.click()}>
            选择图片
          </button>
          {bg.image && (
            <button className="ghost mini" onClick={remove}>
              清除
            </button>
          )}
        </div>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          onChange={(e) => void pick(e.target.files?.[0])}
        />
        <span className="settings-hint">
          图片会先缩到最长边 1600px 再存到本地，不会上传到任何服务器。
        </span>
      </div>

      {bg.image && (
        <div className="settings-section">
          <span className="settings-label">遮罩浓度 {(bg.opacity * 100).toFixed(0)}%</span>
          <input
            type="range"
            min={0}
            max={0.9}
            step={0.05}
            value={bg.opacity}
            onChange={(e) => {
              const next = { ...bg, opacity: Number(e.target.value) }
              onBg(next)
              saveBg(next)
            }}
          />
          <span className="settings-hint">
            调高可提升文字对比度。背景图不该让人读不清正文。
          </span>
        </div>
      )}

      {notice && <span className="settings-hint">{notice}</span>}

      <div className="settings-actions">
        <button className="ghost mini" onClick={onClose}>
          完成
        </button>
      </div>
    </div>
  )
}
