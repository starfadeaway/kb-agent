import { useEffect, useMemo, useRef, useState } from 'react'
import type { Health, Ref, Turn } from '../types'
import type { BgSettings, Theme } from '../prefs'
import { download, turnFileName, turnToMarkdown } from '../exportMd'
import { ToolTimeline } from './ToolTimeline'
import { SettingsPanel } from './SettingsPanel'

/**
 * 引用修订时间汇总。
 *
 * ## 这里曾经有一版「陈旧告警」，被实测推翻了
 *
 * 最初的设计是按 12 个月为界分三档（正常 / 部分来源较早 / 来源均较早），
 * 超过阈值就标红色。**那是错的**，因为它把两个几乎解耦的东西当成了一个：
 *
 * - 页面最后编辑时间 = 整页最后一次被动的时刻
 * - 这条内容的新鲜度 = 这条内容上一次被验证的时刻
 *
 * 实测（`scripts/probe_revision_clustering.py`，sr 706 页）：
 * **Top12 个日期就占了全部页面的 40.5%**（2026-02-13 一天改了 63 页），
 * 说明 wiki 是**版本更新驱动的批量维护**，不是零散编辑。
 *
 * 于是「页面很久没动」最可能的解释是「wiki 认为它不需要改」，
 * 而不是「内容悄悄过期了」。拿修订时间去推断内容过时，
 * 是一个**数据支撑不了的推断** —— 而误导性警示比不警示更糟。
 *
 * 所以现在只陈述事实：这段区间是多少。判断留给用户。
 */
function revisionStats(refs: Ref[]) {
  const dates = refs.map((r) => r.last_revision_ts).filter((d): d is string => !!d)
  const missing = refs.length - dates.length
  if (!dates.length) {
    return {
      known: false,
      span: '',
      title:
        '这些来源没有记录页面修订时间。' +
        '注意：即使有修订时间，也只说明「这一页最近一次被编辑是什么时候」，' +
        '**不能**推出某条数值是否仍然正确 —— wiki 的批量维护会让这两件事脱钩。',
    }
  }
  const sorted = [...dates].sort()
  const span = sorted.length === 1 ? sorted[0] : `${sorted[0]} ~ ${sorted[sorted.length - 1]}`
  return {
    known: true,
    span,
    title:
      '「页面修订」是 wiki 页面最后一次被编辑的时间，**不是数据生效日期**。' +
      '本页两年没被动过，通常意味着 wiki 认为它不需要改，而不是内容已过时。' +
      (missing > 0 ? `另有 ${missing} 条来源没有修订时间。` : ''),
  }
}

function pageName(url: string): string {
  return url.replace(/^https?:\/\/wiki\.biligame\.com\//, '')
}

/**
 * 气泡右下角的元信息行。
 *
 * 三件事各归其位，不要混：
 *   回答于   —— **本机时钟**，说的是「什么时候问的」，与内容新旧无关；
 *   页面修订 —— **溯源信息**，说的是「这条记录是什么时候录的」；
 *   来源     —— 可点开核对，这是用户唯一能验证 AI 的手段。
 *
 * 刻意**没有**陈旧告警：修订时间推不出内容是否仍正确（见 revisionStats 的
 * docstring），用它标红等于制造无法支撑的焦虑。
 */
function MetaLine({ turn }: { turn: Turn }) {
  const s = revisionStats(turn.refs)
  return (
    <div className="meta-line">
      <span className="meta-when">回答于 {turn.at}</span>

      <span className="meta-rev" title={s.title}>
        {s.known ? `页面修订 ${s.span}` : '页面修订未知'}
      </span>

      <span className="meta-src">
        {turn.refs.map((r) => (
          <a
            key={r.url}
            href={r.url}
            target="_blank"
            rel="noreferrer"
            title={
              r.last_revision_ts
                ? `${pageName(r.url)} · 页面最后修订 ${r.last_revision_ts}`
                : `${pageName(r.url)} · 页面修订时间未知`
            }
          >
            {pageName(r.url)}
          </a>
        ))}
      </span>
    </div>
  )
}

/** 样例问题覆盖三个库，方便演示"加一个游戏 = 多个体系都能查" */
const SAMPLES = [
  '位面饰品太空封印站的两件套效果是什么？',
  '圣遗物套装有哪些效果？',
  '绝区零的邦布等级怎么提升？',
  '太空封印站给攻击力1000的角色加多少攻击力？请给出计算过程。',
  '崩坏3的雷电女王怎么配装？',
]

/** 基座 id → 展示名。显示模型全名才能看出「换的到底是哪个模型」。 */
const BACKEND_LABELS: Record<string, string> = {
  deepseek: 'DeepSeek',
  minimax: 'MiniMax',
}

interface Props {
  turns: Turn[]
  busy: boolean
  health: Health | null
  backend: string
  onBackend: (b: string) => void
  theme: Theme
  onToggleTheme: () => void
  /** 设置面板里点的是明确的主题，与 toggle 语义不同 */
  setTheme: (t: Theme) => void
  onOpenSettings: () => void
  settingsOpen: boolean
  /** 背景设置由 App 持有（背景图层在那里渲染），必须传进来而不是各存一份 */
  bg: BgSettings
  onBg: (b: BgSettings) => void
  onSend: (question: string, game: string, backend: string) => void
  onStop: () => void
  onReset: () => void
}

export function ChatPanel({
  turns,
  busy,
  health,
  backend,
  onBackend,
  theme,
  onToggleTheme,
  setTheme,
  onOpenSettings,
  settingsOpen,
  bg,
  onBg,
  onSend,
  onStop,
  onReset,
}: Props) {
  const [input, setInput] = useState('')
  const [game, setGame] = useState('sr')
  const scrollRef = useRef<HTMLDivElement>(null)
  const taRef = useRef<HTMLTextAreaElement>(null)

  // 游戏列表完全由后端下发 —— 前端不再写死「只有星铁和绝区零」，
  // 加第四个库时这里一行都不用改。
  const games = useMemo(() => Object.entries(health?.games ?? { sr: '崩坏：星穹铁道' }), [health])
  const uncovered = health?.known_uncovered ?? {}

  // 库刚加载完时，如果当前选的 game 已经不存在了，纠正到第一个
  useEffect(() => {
    if (games.length && !games.some(([slug]) => slug === game)) setGame(games[0][0])
  }, [games, game])

  // 只有配好密钥的基座才出现在切换器里 —— 点了就报错不如不显示
  const backends = (health?.chat_backends ?? []).filter((b) => b.configured)
  const active = backends.find((b) => b.id === backend) ?? backends[0]
  const activeGame = games.find(([slug]) => slug === game)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [turns])

  // 输入框随内容增高，但设上限，超出后内部滚动
  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`
  }, [input])

  const submit = () => {
    const q = input.trim()
    if (!q || busy || !active) return
    onSend(q, game, active.id)
    setInput('')
  }

  return (
    <div className="chat">
      <header className="chat-head">
        <div className="chat-title">
          <h1>知识库问答</h1>
          <p className="muted">ReAct Agent 先检索再作答，工具调用过程全程可见</p>
        </div>

        <div className="head-actions">
          <Segmented
            value={game}
            onChange={setGame}
            options={games.map(([slug, label]) => ({ id: slug, label }))}
          />

          {backends.length > 1 && (
            <div className="model-picker">
              <span className="picker-label">模型</span>
              <Segmented
                value={active?.id ?? backend}
                onChange={onBackend}
                options={backends.map((b) => ({
                  id: b.id,
                  label: BACKEND_LABELS[b.id] ?? b.id,
                  title: b.model,
                }))}
              />
            </div>
          )}

          <button className="ghost" onClick={onReset} disabled={turns.length === 0}>
            清空
          </button>
          {busy && (
            <button className="danger" onClick={onStop}>
              中断
            </button>
          )}
          <button className="icon-btn" onClick={onToggleTheme} aria-label="切换主题">
            {theme === 'dark' ? '☀' : '☾'}
          </button>
          <button
            className={settingsOpen ? 'icon-btn active' : 'icon-btn'}
            onClick={onOpenSettings}
            aria-label="外观设置"
          >
            ⚙
          </button>
        </div>
      </header>

      {settingsOpen && (
        <SettingsPanel
          theme={theme}
          bg={bg}
          onTheme={setTheme}
          onBg={onBg}
          onClose={onOpenSettings}
        />
      )}

      <div className="messages" ref={scrollRef}>
        {turns.length === 0 && (
          <div className="empty">
            <div className="empty-hint">
              知识库覆盖 {games.map(([, l]) => l).join(' / ')}。
              {Object.keys(uncovered).length > 0
                ? `问点别的游戏（比如 ${Object.values(uncovered)[0]}）会得到明确的边界说明，而不是编一个答案。`
                : '试一个问题：'}
            </div>
            <div className="samples">
              {SAMPLES.map((s) => (
                <button
                  key={s}
                  className="sample"
                  onClick={() => onSend(s, game, active?.id ?? 'deepseek')}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((t) => (
          <div key={t.id} className="turn">
            <div className="q">
              <span
                className="badge"
                data-scope={t.outOfScope ? 'out' : undefined}
                title={t.outOfScope ? '该游戏未收录，本轮为越界拒答' : undefined}
              >
                {t.outOfScope ? (uncovered[t.outOfScope] ?? t.outOfScope) : gameLabel(t.game, health)}
              </span>
              <span>{t.question}</span>
            </div>

            <ToolTimeline tools={t.tools} />

            <div className="a">
              {t.answer ? (
                <>
                  <Markdownish text={t.answer} />
                  <div className="a-actions">
                    <CopyButton text={t.answer} />
                    <DownloadTurn turn={t} gameLabel={gameLabel(t.game, health)} />
                  </div>
                </>
              ) : t.error ? null : !t.done ? (
                // 「思考中」必须同时要求 **流还没结束**。
                // 只判 `!t.answer` 是不够的：越界拒答不产生 token 事件，
                // 答案只在 done 里，所以会出现「done 到了却还在转圈」。
                // 拿 done 当兜底判据，任何将来「不靠 token 送答案」的路径
                // 都不会再把界面永久卡在加载态。
                <span className="muted thinking">
                  <span className="spinner" /> 思考中…
                </span>
              ) : (
                <span className="muted">（本次没有输出内容）</span>
              )}
              {t.error && (
                <div className="error">
                  出错了：{t.error}
                  <button className="ghost mini" onClick={() => onSend(t.question, t.game, t.backend)}>
                    重试
                  </button>
                </div>
              )}
            </div>

            {t.refs.length > 0 && (
              <MetaLine turn={t} />
            )}

            {t.done && (
              <div className="metrics">
                <Stat label="模型" value={BACKEND_LABELS[t.backend] ?? t.backend} />
                <Stat label="步数" value={t.done.steps} />
                <Stat
                  label="工具失败"
                  value={t.done.tool_failures}
                  tone={t.done.tool_failures ? 'bad' : 'ok'}
                />
                <Stat label="耗时" value={`${t.done.elapsed_ms}ms`} />
                <Stat label="Token" value={`${t.done.prompt_tokens}/${t.done.completion_tokens}`} />
                <Stat
                  label="终止"
                  value={t.done.stop_reason}
                  tone={t.done.stop_reason === 'out_of_scope' ? 'warn' : undefined}
                />
                <Stat label="Run" value={t.done.run_id.slice(0, 8)} />
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="composer">
        <textarea
          ref={taRef}
          value={input}
          placeholder={`问点${activeGame?.[1] ?? '游戏'}内的事，或问一个未收录的游戏看看边界处理`}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
          rows={2}
        />
        <div className="composer-side">
          <button onClick={submit} disabled={busy || !input.trim() || !active}>
            {busy ? '回答中…' : '发送'}
          </button>
          <span className="kbd-hint">
            <kbd>Enter</kbd> 发送 · <kbd>Shift</kbd>+<kbd>Enter</kbd> 换行
          </span>
        </div>
      </div>
    </div>
  )
}

function gameLabel(slug: string, health: Health | null): string {
  return health?.games?.[slug] ?? slug
}

/** 分段控件。比原生 select 更能一眼看出「现在有几个选项、选的是哪个」。 */
function Segmented({
  value,
  onChange,
  options,
}: {
  value: string
  onChange: (v: string) => void
  options: Array<{ id: string; label: string; title?: string }>
}) {
  return (
    <div className="segmented" role="group">
      {options.map((o) => (
        <button
          key={o.id}
          title={o.title ?? o.label}
          className={o.id === value ? 'seg active' : 'seg'}
          onClick={() => onChange(o.id)}
          aria-pressed={o.id === value}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/**
 * 导出这一轮问答为 Markdown。
 *
 * 导出的**不只是答案**：工具过程、出处（含页面修订）、运行指标一起写进去。
 * 只导出正文的话，这份 .md 拿出去没法核对 —— 而「能不能核对」正是这个项目
 * 想证明的事。
 */
function DownloadTurn({ turn, gameLabel }: { turn: Turn; gameLabel: string }) {
  const [done, setDone] = useState(false)
  return (
    <button
      className="ghost mini"
      onClick={() => {
        const md = turnToMarkdown({
          question: turn.question,
          game: turn.game,
          gameLabel,
          backend: turn.backend,
          backendLabel: BACKEND_LABELS[turn.backend] ?? turn.backend,
          answer: turn.answer,
          at: turn.at,
          outOfScope: turn.outOfScope,
          tools: turn.tools,
          refs: turn.refs,
          done: turn.done,
        })
        download(turnFileName(turn), md, 'text/markdown;charset=utf-8')
        setDone(true)
        setTimeout(() => setDone(false), 1600)
      }}
      title="导出这一轮的答案、检索过程、出处与运行指标为 Markdown"
    >
      {done ? '已导出' : '导出 Markdown'}
    </button>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      className="ghost mini copy"
      onClick={() => {
        navigator.clipboard?.writeText(text).then(
          () => {
            setCopied(true)
            setTimeout(() => setCopied(false), 1600)
          },
          () => setCopied(false),
        )
      }}
    >
      {copied ? '已复制' : '复制'}
    </button>
  )
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string
  value: string | number
  tone?: 'ok' | 'bad' | 'warn'
}) {
  return (
    <span className="stat" data-tone={tone}>
      <em>{label}</em>
      {value}
    </span>
  )
}

/** 极简 Markdown 渲染：只处理 **粗体**、行内代码、列表和标题，不引第三方库。 */
function Markdownish({ text }: { text: string }) {
  const blocks = text.split('\n')
  return (
    <>
      {blocks.map((line, i) => {
        const t = line.trim()
        if (!t) return <div key={i} style={{ height: 8 }} />
        // 表格分隔行（|---|---|）不是内容，别渲染出来
        if (/^\|?[\s|:-]+\|?$/.test(t) && t.includes('-')) return null
        // Markdown 标题。切换基座后才发现必须处理：DeepSeek 习惯用粗体小标题，
        // MiniMax 直接吐 "### xxx"，不处理就会把 ### 原样显示给用户。
        const heading = t.match(/^(#{1,6})\s+(.*)$/)
        if (heading) {
          return (
            <div key={i} className="md-h" data-level={Math.min(heading[1].length, 4)}>
              {inline(heading[2])}
            </div>
          )
        }
        if (/^[-*]\s+/.test(t) || /^\d+\.\s+/.test(t)) {
          return (
            <div key={i} className="li">
              {inline(t.replace(/^[-*]\s+|^\d+\.\s+/, ''))}
            </div>
          )
        }
        if (t.startsWith('|')) {
          return (
            <div key={i} className="row-line">
              {inline(t.replace(/^\||\|$/g, ''))}
            </div>
          )
        }
        return (
          <div key={i} className="para">
            {inline(t)}
          </div>
        )
      })}
    </>
  )
}

function inline(s: string) {
  const parts = s.split(/(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\])/g)
  return parts.map((p, i) => {
    if (p.startsWith('**') && p.endsWith('**')) return <strong key={i}>{p.slice(2, -2)}</strong>
    if (p.startsWith('`') && p.endsWith('`')) return <code key={i}>{p.slice(1, -1)}</code>
    if (p.startsWith('[') && p.endsWith(']')) return <em key={i} className="cite">{p}</em>
    return <span key={i}>{p}</span>
  })
}
