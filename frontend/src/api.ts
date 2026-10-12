import type { Health, RetrieveResponse, RunRecord, StreamEvent } from './types'

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
}) => json<RetrieveResponse>('/retrieve', { method: 'POST', body: JSON.stringify(body) })

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
