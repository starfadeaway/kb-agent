import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getHealth, streamChat } from './api'
import type { Health, StreamEvent, Turn } from './types'
import { ChatPanel } from './components/ChatPanel'
import { DeliverablePanel } from './components/DeliverablePanel'
import { Sidebar, type TabId } from './components/Sidebar'
import { RetrievePanel } from './components/RetrievePanel'
import { SettingsPanel } from './components/SettingsPanel'
import { applyTheme, loadBg, readStoredTheme, saveTheme, type BgSettings, type Theme } from './prefs'

export default function App() {
  const [tab, setTab] = useState<TabId>('chat')
  const [health, setHealth] = useState<Health | null>(null)
  const [turns, setTurns] = useState<Turn[]>([])
  const [busy, setBusy] = useState(false)
  // 当前选中的 Chat 基座。默认 deepseek —— /api/health 没回来时也能先问。
  const [backend, setBackend] = useState('deepseek')
  const [theme, setTheme] = useState<Theme>(() => readStoredTheme())
  const [bg, setBg] = useState<BgSettings>(() => loadBg())
  const [showSettings, setShowSettings] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  // 主题必须在这里**再应用一次**，不能只在 useState 初始化时读取。
  // index.html 的内联脚本只负责首屏防闪烁，之后 React 每次挂载都走不到那段代码 ——
  // HMR 热更新、路由重挂载之后，DOM 上的 data-theme 会停留在上一次的值，
  // 出现「state 说浅色、界面还是深色」的错位。实测踩到过。
  useEffect(() => {
    applyTheme(theme)
  }, [theme])

  // 背景图层独立于内容渲染：不塞进 .app 里，
  // 否则 z-index 会被 position:relative 的祖先困住，且会跟着内容一起滚动。
  const bgStyle = useMemo(
    () =>
      ({
        '--bg-scrim': bg.image ? String(bg.opacity) : '0',
        '--bg-blur': bg.image ? '2px' : '0px',
      }) as React.CSSProperties,
    [bg.image, bg.opacity],
  )

  /** 每轮问答结束后刷新一次：缓存命中率、预热状态这类数字不刷新就永远是加载时的快照 */
  const refreshHealth = useCallback(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  const send = useCallback(
    async (question: string, game: string, model: string) => {
      if (busy) return
      setBusy(true)
      const controller = new AbortController()
      abortRef.current = controller

      const turn: Turn = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        question,
        game,
        backend: model,
        answer: '',
        tools: [],
        refs: [],
        at: new Date().toLocaleTimeString('zh-CN', {
          hour: '2-digit',
          minute: '2-digit',
          hour12: false,
        }),
        done: null,
        error: null,
        outOfScope: null,
      }
      setTurns((prev) => [...prev, turn])

      const patch = (fn: (t: Turn) => Turn) =>
        setTurns((prev) => prev.map((t) => (t.id === turn.id ? fn(t) : t)))

      try {
        for await (const ev of streamChat({ question, game, backend: model }, controller.signal)) {
          handleEvent(ev, patch)
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        patch((t) => ({ ...t, error: controller.signal.aborted ? '已取消' : msg }))
      } finally {
        setBusy(false)
        abortRef.current = null
        refreshHealth()
      }
    },
    [busy, refreshHealth],
  )

  const stop = useCallback(() => abortRef.current?.abort(), [])
  const reset = useCallback(() => setTurns([]), [])
  const toggleTheme = useCallback(() => {
    setTheme((cur) => {
      const next: Theme = cur === 'dark' ? 'light' : 'dark'
      applyTheme(next)
      saveTheme(next)
      return next
    })
  }, [])

  /** 设置面板里点的是明确的主题，不是"翻转" */
  const setThemeExplicit = useCallback((t: Theme) => {
    applyTheme(t)
    saveTheme(t)
    setTheme(t)
  }, [])

  return (
    <div className="app" style={bgStyle}>
      {bg.image && <div className="bg-layer" style={{ backgroundImage: `url(${bg.image})` }} />}
      <div className="bg-scrim" />

      {drawerOpen && <div className="scrim" onClick={() => setDrawerOpen(false)} />}

      <Sidebar
        health={health}
        tab={tab}
        onTab={(t) => {
          setTab(t)
          setDrawerOpen(false)
        }}
        open={drawerOpen}
        theme={theme}
        onToggleTheme={toggleTheme}
        onOpenSettings={() => setShowSettings((v) => !v)}
      />

      <main className="main">
        {/* 移动端顶栏：侧栏收成抽屉后，这里是唯一的常驻入口 */}
        <div className="mobile-bar">
          <button className="icon-btn" onClick={() => setDrawerOpen((v) => !v)} aria-label="菜单">
            ☰
          </button>
          <span className="brand-title">kb-agent</span>
          <span style={{ flex: 1 }} />
          <button
            className="icon-btn"
            onClick={() => setShowSettings((v) => !v)}
            aria-label="外观设置"
          >
            ⚙
          </button>
          <button
            className="icon-btn"
            onClick={toggleTheme}
            aria-label="切换主题"
            title={theme === 'dark' ? '切换到浅色' : '切换到深色'}
          >
            {theme === 'dark' ? '☀' : '☾'}
          </button>
        </div>

        {showSettings && (
          <SettingsPanel
            theme={theme}
            bg={bg}
            onTheme={setThemeExplicit}
            onBg={setBg}
            onClose={() => setShowSettings(false)}
          />
        )}

        {tab === 'chat' ? (
          <ChatPanel
            turns={turns}
            busy={busy}
            health={health}
            backend={backend}
            onBackend={setBackend}
            theme={theme}
            onToggleTheme={toggleTheme}
            settingsOpen={showSettings}
            onOpenSettings={() => setShowSettings((v) => !v)}
            onSend={send}
            onStop={stop}
            onReset={reset}
          />
        ) : tab === 'deliverable' ? (
          <DeliverablePanel health={health} />
        ) : (
          <RetrievePanel health={health} onOpenSettings={() => setShowSettings((v) => !v)} />
        )}
      </main>
    </div>
  )
}

/**
 * SSE 事件 → 轮次状态。
 *
 * 这里刻意把「工具过程」当成一等公民展示：
 * 用户看到的是「它先去检索了太空封印站，耗时 675ms，命中 3 条」，
 * 而不是干等一个黑框。这正是 Agent 类产品和「调一下模型 API」的区别。
 */
function handleEvent(
  ev: StreamEvent,
  patch: (fn: (t: Turn) => Turn) => void,
): void {
  if (ev.event === 'tool_start') {
    patch((t) => ({
      ...t,
      tools: [...t.tools, { name: ev.data.name, args: ev.data.args, running: true }],
    }))
    return
  }
  if (ev.event === 'tool_end') {
    // 一次 assistant 消息可能带多个 tool_calls，tool_end 会连续到达；
    // 总是收敛到最后一个仍在 running 的工具，避免并行调用时串行错位。
    patch((t) => {
      const idx = t.tools.map((x, i) => (x.running ? i : -1)).filter((i) => i >= 0).pop()
      if (idx === undefined) return t
      return {
        ...t,
        tools: t.tools.map((tool, i) =>
          i === idx
            ? { ...tool, running: false, ok: ev.data.ok, ms: ev.data.ms, chars: ev.data.chars }
            : tool,
        ),
      }
    })
    return
  }
  if (ev.event === 'token') {
    patch((t) => ({ ...t, answer: t.answer + ev.data.text }))
    return
  }
  if (ev.event === 'out_of_scope') {
    patch((t) => ({ ...t, outOfScope: ev.data.game }))
    return
  }
  if (ev.event === 'error') {
    patch((t) => ({ ...t, error: ev.data.message }))
    return
  }
  if (ev.event === 'done') {
    patch((t) => ({
      ...t,
      // 用后端回传的实际基座覆盖本地记录：请求可能被网关改写，
      // 界面上显示的必须是真的在跑的那个。
      backend: ev.data.backend ?? t.backend,
      refs: ev.data.refs ?? [],
      outOfScope: ev.data.out_of_scope ?? t.outOfScope,
      // **必须在这里回填 answer。** 越界拒答那条路径不发任何 token 事件，
      // 答案只存在于 done 里 —— 漏了这一行，气泡会因为 `!t.answer`
      // 永远停在「思考中…」，哪怕 done 早就到了（实测踩过）。
      // 用 `t.answer ||` 而不是直接赋值：正常问答的答案来自 token 逐段追加，
      // 那是更细粒度的来源，不能被 done 里被截断过的整段覆盖掉。
      answer: t.answer || ev.data.answer || '',
      done: ev.data,
    }))
  }
}
