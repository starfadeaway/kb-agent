import type {
  DeliverableResponse,
  Health,
  RetrieveResponse,
  RunRecord,
  StreamEvent,
} from './types'

const BASE = '/api'

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}

export const getHealth = () => json<Health>('/health')

export const getStats = () => json<{ collections: unknown[]; mysql: { state: string } }>('/stats')

export const getRuns = (limit = 20) => json<RunRecord[]>(`/runs?limit=${limit}`)

export const getRun = (runId: string) => json<RunRecord>(`/runs/${runId}`)

export const retrieve = (body: {
  query: string
  game: string
  top_k: number
  use_bm25: boolean
  use_vector: boolean
  use_rerank: boolean
  /** 版本感知：只要这个日期之后修订过的内容（YYYY-MM-DD）。留空 = 不过滤 */
  min_revision_ts?: string
  /** 只在这些页面里检索；留空 = 全库 */
  page_titles?: string[]
}) => json<RetrieveResponse>('/retrieve', { method: 'POST', body: JSON.stringify(body) })

/**
 * 交付物任务。**这是个慢接口**（实测 6 个子任务约 30-60s），
 * 所以超时必须放宽，而且**不能**拿它做轮询。
 * 超时不足的表现是前端拿到一个没有原因的错误 —— 那比慢更难查。
 */
export const postDeliverable = (body: { question: string; game: string }) =>
  json<DeliverableResponse>('/deliverable', { method: 'POST', body: JSON.stringify(body) })

/**
 * 流式问答。SSE 不能用 fetch + await 直接读，必须手动解析 \n\n 分隔的帧。
 * 这里显式处理 UTF-8 分片：中文常被切在多字节字符中间，
 * 直接 new TextDecoder().decode(chunk) 会产生乱码，必须用 {stream:true}。
 */
export async function* streamChat(
  body: { question: string; game?: string; backend?: string },
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`SSE 连接失败: ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let idx: number
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const frame = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      const evLine = frame.split('\n').find((l) => l.startsWith('event: '))
      const dataLine = frame.split('\n').find((l) => l.startsWith('data: '))
      if (!evLine || !dataLine) continue
      try {
        yield {
          event: evLine.slice(7).trim() as StreamEvent['event'],
          data: JSON.parse(dataLine.slice(6)),
        } as StreamEvent
      } catch {
        /* 帧不完整就跳过，等下一帧 */
      }
    }
  }
}

/**
 * `/api/health` 还没回来（或后端没下发 games）时的兜底游戏列表。
 *
 * 单独提出来是因为它此前在三个组件里各写了一份，**其中一份已经漂了**
 * （`DeliverablePanel` 少了「崩坏：」前缀）—— 表现为同一款游戏在两个面板里
 * 显示两个名字。后端一旦下发就用后端的，这里只兜首帧。
 */
export const FALLBACK_GAMES: Record<string, string> = { sr: '崩坏：星穹铁道' }
