/** 与后端 FastAPI 的类型契约。改了后端就改这里，前后端类型对不上是这类项目最常见的返工来源。 */

export interface ChatBackend {
  id: string
  model: string
  configured: boolean
}

/** 可切换的 Chat 基座。由 /api/health 下发 —— 后端加 provider 只改后端，前端不用动。 */
export interface ChatBackend {
  id: string
  model: string
  configured: boolean
}

export interface Health {
  ok: boolean
  mysql: { state: string; detail: string }
  llm_configured: { deepseek: boolean; siliconflow: boolean; judge: boolean }
  chat_backends: ChatBackend[]
  embed_model: string
  rerank_model: string
  /** slug → 显示名。由后端下发，前端不再写死游戏列表 */
  games: Record<string, string>
  /** 知道存在但未收录的游戏，让「暂未收录」的提示有据可依 */
  known_uncovered: Record<string, string>
  collections: Array<{ name: string; count: number; game?: string; embed_model?: string; variant?: string }>
  tools: string[]
  warmup: Record<string, unknown>
  cache: Record<string, { size: number; hits: number; misses: number; hit_rate: number }>
  concurrency_limit: number
}

export interface RetrieveHit {
  rank: number
  chunk_id: string
  score: number
  source: string
  page_title: string
  section: string
  text: string
  url: string
  last_revision_ts: string
}

export interface RetrieveResponse {
  query: string
  label: string
  config: Record<string, unknown>
  hits: RetrieveHit[]
}

/** SSE 事件：与后端 agent/react.py 里 yield 的 event 名一一对应 */
export type StreamEvent =
  | { event: 'tool_start'; data: { name: string; args: Record<string, unknown> } }
  | { event: 'tool_end'; data: { name: string; ok: boolean; ms: number; chars: number } }
  | { event: 'token'; data: { text: string } }
  | { event: 'limit'; data: { reason: string } }
  /** 路由判定为「用户问的是未收录的游戏」，接下来会直接给边界说明 */
  | { event: 'out_of_scope'; data: { game: string } }
  | { event: 'error'; data: { message: string } }
  | {
      event: 'done'
      data: {
        run_id: string
        game: string
        /** 本轮实际使用的基座，由后端回传 —— 不靠前端自己记，避免猜错 */
        backend?: string
        answer: string
        steps: number
        tool_failures: number
        stop_reason: string
        elapsed_ms: number
        prompt_tokens: number
        completion_tokens: number
        guard?: Record<string, number>
        refs: Ref[]
        persisted?: unknown
        /** 被拒答时告知是哪个未收录游戏 */
        out_of_scope?: string | null
        trace: Array<{ step: number; tool: string; args: Record<string, unknown>; ok: boolean; ms: number }>
      }
    }

export interface RunRecord {
  run_id: string
  game: string
  steps: number
  tool_failures: number
  stop_reason: string
  latency_ms: number
  prompt_tokens: number
  completion_tokens: number
  question: string
  created_at: string | null
  tool_calls?: Array<{
    step: number
    tool: string
    args: Record<string, unknown> | null
    ok: boolean
    ms: number
    chars?: number
    error?: string | null
  }>
}

/** 一次工具调用在前端的展示态 */
export interface ToolCallView {
  name: string
  args: Record<string, unknown>
  running: boolean
  ok?: boolean
  ms?: number
  chars?: number
}

export interface Ref {
  url: string
  /**
   * 该来源所在 wiki 页面**最后一次编辑**的日期（YYYY-MM-DD）。
   *
   * 注意语义：这是「页面最后修订」，**不是**「数据截至」——
   * 一页 2020 年建的 wiki 昨天改了个错别字，它也是昨天。
   * 所以 UI 上只能写「页面修订」，不能写「数据更新于」，
   * 否则就是用一个更隐蔽的误导替换掉原本的沉默。
   *
   * 可能为空：老数据或某些 chunk 的 metadata 里没有这个字段。
   */
  last_revision_ts?: string
}

/** 一轮问答 */
export interface Turn {
  id: string
  question: string
  game: string
  /** 发问时选中的基座。done 事件回来后会被后端回传的实际值覆盖。 */
  backend: string
  answer: string
  tools: ToolCallView[]
  refs: Ref[]
  /** 发出这一轮的本地时间（HH:MM），用于气泡右下角的「回答于」 */
  at: string
  done: Extract<StreamEvent, { event: 'done' }>['data'] | null
  error: string | null
  /** 命中的未收录游戏；非空时这轮是「越界拒答」而不是检索未命中 */
  outOfScope: string | null
}
