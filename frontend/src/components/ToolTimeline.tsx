import type { ToolCallView } from '../types'

const TOOL_LABEL: Record<string, string> = {
  search_kb: '检索知识库',
  get_page: '抓取页面',
  calculator: '计算',
  list_games: '查询知识库列表',
}

/**
 * 工具调用时间线。
 * 这是整个界面最想让人看到的东西：Agent 不是黑箱，
 * 每一步用了什么工具、传了什么参数、多久返回、成没成功，全部摊开。
 */
export function ToolTimeline({ tools }: { tools: ToolCallView[] }) {
  if (tools.length === 0) return null
  return (
    <div className="tools-timeline">
      {tools.map((t, i) => (
        <div key={i} className="tl-item" data-state={t.running ? 'running' : t.ok ? 'ok' : 'fail'}>
          <span className="tl-dot" />
          <span className="tl-name">{TOOL_LABEL[t.name] ?? t.name}</span>
          <code className="tl-args">{summarize(t.args)}</code>
          <span className="tl-meta">
            {t.running ? (
              <span className="spinner" />
            ) : (
              <>
                {t.ms}ms{t.chars ? ` · ${t.chars} 字符` : ''}
              </>
            )}
          </span>
        </div>
      ))}
    </div>
  )
}

function summarize(args: Record<string, unknown>): string {
  const keys = Object.keys(args)
  if (keys.length === 0) return '()'
  const main = keys
    .map((k) => {
      const v = String(args[k] ?? '')
      return `${k}=${v.length > 34 ? `${v.slice(0, 34)}…` : v}`
    })
    .join('  ')
  return `{ ${main} }`
}
