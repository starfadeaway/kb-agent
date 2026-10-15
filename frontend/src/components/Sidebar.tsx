import type { Health } from '../types'
import type { Theme } from '../prefs'

/** 侧栏三个页签。加新页签时这里和 App.tsx 的 useState 类型要同时改。 */
export type TabId = 'chat' | 'retrieve' | 'deliverable'

interface Props {
  health: Health | null
  tab: TabId
  onTab: (t: TabId) => void
  open?: boolean
  theme?: Theme
  onToggleTheme?: () => void
  onOpenSettings?: () => void
}

export function Sidebar({
  health,
  tab,
  onTab,
  open = true,
  theme = 'dark',
  onToggleTheme,
  onOpenSettings,
}: Props) {
  const collections = health?.collections ?? []
  const mysqlOk = health?.mysql.state === 'ready'
  const warmup = typeof health?.warmup?.state === 'string' ? health?.warmup?.state : 'unknown'
  const embed = health?.cache?.embed
  const rerank = health?.cache?.rerank
  // 命中率低不是故障，但预热失败是 —— 分开标，别混成一个红点
  const warmupBad = warmup.startsWith('failed')

  /**
   * 只统计**生产索引**：embed_model = 当前生产模型 且 variant = base。
   *
   * 实验 B/D 会在同一游戏下建出 b1/b2/b3 和 qwen 等变体库，把它们一并加进来
   * 会让"星铁 9,986 片段"这种数字出现 —— 实际生产 base 库只有 3932。
   * 这种数字会被拿去写简历，必须是真实的。
   */
  const isProduction = (c: (typeof collections)[number]) =>
    (c.variant ?? 'base') === 'base' && c.embed_model === health?.embed_model
  const production = collections.filter(isProduction)

  const chunks = production.reduce((s, c) => s + (c.count ?? 0), 0)
  const experimentChunks = collections
    .filter((c) => !isProduction(c))
    .reduce((s, c) => s + (c.count ?? 0), 0)

  // 每个游戏的片段数。优先用后端下发的 game 字段；
  // 它来自索引自己的 .meta.json（落盘的是 game/embed_model/variant），
  // 拿不到才退回从 collection 名里正则解析。
  const perGame = new Map<string, number>()
  for (const c of production) {
    const m = /^kb_([^_]+)_/.exec(c.name)
    const game = c.game ?? m?.[1]
    if (game) perGame.set(game, (perGame.get(game) ?? 0) + (c.count ?? 0))
  }

  return (
    <aside className="sidebar" data-open={open}>
      <div className="brand">
        <div className="brand-title">kb-agent</div>
        <div className="brand-sub">游戏内容知识库问答</div>
      </div>

      <nav className="nav">
        <button className={tab === 'chat' ? 'nav-item active' : 'nav-item'} onClick={() => onTab('chat')}>
          问答
        </button>
        <button
          className={tab === 'retrieve' ? 'nav-item active' : 'nav-item'}
          onClick={() => onTab('retrieve')}
        >
          检索调试
        </button>
        <button
          className={tab === 'deliverable' ? 'nav-item active' : 'nav-item'}
          onClick={() => onTab('deliverable')}
        >
          交付物
        </button>
      </nav>

      {health && (
        <div className="meta">
          <div className="group-title">知识库</div>
          {Object.entries(health.games).map(([slug, label]) => (
            <Row
              key={slug}
              label={label}
              value={perGame.has(slug) ? `${(perGame.get(slug) ?? 0).toLocaleString()} 片段` : '未建立'}
              ok={perGame.has(slug)}
              title={`slug: ${slug}`}
            />
          ))}
          <Row label="合计" value={`${chunks.toLocaleString()} 片段`} />
          {experimentChunks > 0 && (
            <div className="settings-hint" style={{ marginTop: 4 }}>
              另有 {experimentChunks.toLocaleString()} 片段来自实验变体索引（实验 B/D），
              不计入上面的合计。
            </div>
          )}
          {Object.keys(health.known_uncovered ?? {}).length > 0 && (
            <div className="settings-hint" style={{ marginTop: 4 }}>
              可按需扩展：抓一条命令即可加新游戏，检索层零改动。
            </div>
          )}
        </div>
      )}

      <div className="meta">
        <div className="group-title">运行环境</div>
        <Row
          label="索引预热"
          value={warmup === 'ready' ? '就绪' : warmup === 'running' ? '构建中' : warmup}
          ok={warmup === 'ready'}
          tone={warmupBad ? 'bad' : undefined}
          title="BM25 索引在启动时后台构建，避免第一个用户等 2.8s"
        />
        <Row label="Embedding" value={health?.embed_model ?? '—'} />
        <Row label="Rerank" value={health?.rerank_model ?? '—'} />
        <Row
          label="MySQL"
          value={health ? (mysqlOk ? '已连接' : '降级模式') : '未知'}
          ok={mysqlOk}
          title={health?.mysql.detail}
        />
        <Row
          label="并发上限"
          value={health ? `${health.concurrency_limit} 个请求` : '—'}
          title="超出后排队，不会因同时来多个请求而卡死"
        />
      </div>

      {health && embed && rerank && (
        <div className="meta">
          <div className="group-title">缓存命中率</div>
          <Row
            label="Embedding"
            value={`${(embed.hit_rate * 100).toFixed(0)}% · ${embed.size}`}
            title={`命中 ${embed.hits} / 未命中 ${embed.misses}`}
          />
          <Row
            label="Rerank"
            value={`${(rerank.hit_rate * 100).toFixed(0)}% · ${rerank.size}`}
            title={`命中 ${rerank.hits} / 未命中 ${rerank.misses}`}
          />
        </div>
      )}

      {health && (
        <div className="tools">
          <div className="group-title">Agent 工具</div>
          <div className="chips">
            {health.tools.map((t) => (
              <span key={t} className="chip">
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      <div style={{ marginTop: 'auto', display: 'flex', gap: 8, paddingTop: 12 }}>
        <button
          className="icon-btn"
          onClick={onToggleTheme}
          aria-label="切换主题"
          title={theme === 'dark' ? '切换到浅色' : '切换到深色'}
        >
          {theme === 'dark' ? '☀' : '☾'}
        </button>
        <button
          className="icon-btn"
          onClick={onOpenSettings}
          aria-label="外观设置"
          title="外观设置：主题与自定义背景"
        >
          ⚙
        </button>
      </div>
    </aside>
  )
}

function Row({
  label,
  value,
  ok,
  title,
  tone,
}: {
  label: string
  value: string
  ok?: boolean
  title?: string
  tone?: 'bad'
}) {
  // tone 优先于 ok：预热失败时不能因为"取到了值"就显示成正常色
  const color = tone === 'bad' ? 'var(--bad)' : ok ? 'var(--ok)' : undefined
  return (
    <div className="row" title={title}>
      <span className="row-label">{label}</span>
      <span className="row-value" style={color ? { color } : undefined}>
        {value}
      </span>
    </div>
  )
}
