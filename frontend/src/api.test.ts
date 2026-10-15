/**
 * API 层的守卫，重点是 **SSE 分帧**。
 *
 * ## 为什么 SSE 值得单独测
 *
 * `streamChat` 手写了解析：fetch 的每个 chunk 边界和 SSE 的帧边界
 * **完全无关**，一个 `data:` 帧可能横跨多个 chunk，也可能几个帧挤在一个 chunk 里。
 * 这里出错的症状特别恶心：
 *
 * - 中文被切在多字节中间 → 界面出现「三月七」变成「三月�」
 * - 半截 JSON → 解析失败被 `catch` 静默吞掉 → **界面永远停在「思考中」**
 *
 * 第二条是这个项目修过的真实 bug（越界拒答卡「思考中」，fcbcaca），
 * 根因就是「漏掉一个事件界面就永远等下去」。
 * 所以这里测的不是「能不能解析」，而是**每种畸形输入都不会静默丢事件**。
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { streamChat } from './api'

/**
 * 用真实的分块边界造一个假 SSE 响应。
 *
 * 关键：`chunks` 是**按字节**切的，不是按字符串 ——
 * 中文 UTF-8 每字 3 字节，按字符切就没有测试意义了。
 */
function sseResponse(chunks: Uint8Array[], ok = true, status = 200): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(c)
      controller.close()
    },
  })
  return {
    ok,
    status,
    body,
  } as unknown as Response
}

/** 把字符串按字节数切开（会真的切在多字节字符中间）。 */
function splitBytes(s: string, size: number): Uint8Array[] {
  const enc = new TextEncoder()
  const all = enc.encode(s)
  const out: Uint8Array[] = []
  for (let i = 0; i < all.length; i += size) out.push(all.slice(i, i + size))
  return out
}

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
}

async function collect(gen: AsyncGenerator<unknown>): Promise<unknown[]> {
  const out: unknown[] = []
  for await (const v of gen) out.push(v)
  return out
}

afterEach(() => {
  vi.unstubAllGlobals()
})

function stubFetch(resp: Response) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => resp),
  )
}

describe('streamChat 分帧', () => {
  it('完整的一帧能被解析', async () => {
    const sse = frame('token', { text: '你好' })
    stubFetch(sseResponse(splitBytes(sse, 1024)))
    const evs = await collect(streamChat({ question: 'q', game: 'sr' }))
    expect(evs).toEqual([{ event: 'token', data: { text: '你好' } }])
  })

  it('一帧被切成 1 字节一块也能正确拼回来', async () => {
    // 逐字节喂：任何一处忘记用 {stream:true} 或忘记维护 buffer，
    // 这里都会炸成乱码或半截 JSON
    const sse = frame('token', { text: '三月七的命途是巡猎' }) + frame('done', { steps: 3 })
    stubFetch(sseResponse(splitBytes(sse, 1)))
    const evs = (await collect(streamChat({ question: 'q' }))) as Array<{
      event: string
      data: unknown
    }>
    expect(evs).toHaveLength(2)
    expect(evs[0].data).toEqual({ text: '三月七的命途是巡猎' })
    expect(evs[1].event).toBe('done')
  })

  it('中文被切在多字节中间时不能乱码', async () => {
    // 这条是本文件存在的核心理由。逐字节喂中文最容易暴露
    // 「忘了 {stream:true}」—— 那会让 decode 产生替换字符 U+FFFD。
    const sse = frame('token', { text: '角色「三月七」命途为巡猎。' })
    stubFetch(sseResponse(splitBytes(sse, 1)))
    const evs = (await collect(streamChat({ question: 'q' }))) as Array<{
      data: { text: string }
    }>
    expect(evs[0].data.text).toBe('角色「三月七」命途为巡猎。')
    expect(evs[0].data.text).not.toContain('�')
  })

  it('多个帧挤在一个 chunk 里不会漏', async () => {
    const sse = frame('token', { text: 'a' }) + frame('token', { text: 'b' }) + frame('done', { steps: 1 })
    stubFetch(sseResponse(splitBytes(sse, 9999)))
    const evs = await collect(streamChat({ question: 'q' }))
    expect(evs).toHaveLength(3)
  })

  it('缺 event: 行的帧被跳过而不抛异常', async () => {
    // 注释帧（`: keep-alive`）就是这种形状，不能让它把整条流带崩
    const sse = ': keep-alive\n\n' + frame('token', { text: 'x' })
    stubFetch(sseResponse(splitBytes(sse, 4096)))
    const evs = await collect(streamChat({ question: 'q' }))
    expect(evs).toEqual([{ event: 'token', data: { text: 'x' } }])
  })

  it('JSON 解析失败的帧被跳过，后续帧仍然送达', async () => {
    // ⚠️ 这条锁的是「静默丢事件」这个行为：解析失败只丢当前帧，
    // 不能中断整个流。流一断，界面就永远停在「思考中」。
    const sse =
      'event: token\ndata: {坏掉的 json\n\n' + frame('token', { text: '后面这条必须送达' })
    stubFetch(sseResponse(splitBytes(sse, 4096)))
    const evs = (await collect(streamChat({ question: 'q' }))) as Array<{
      data: { text: string }
    }>
    expect(evs).toHaveLength(1)
    expect(evs[0].data.text).toBe('后面这条必须送达')
  })

  it('末尾没有以 \\n\\n 收尾的残帧不会产出半截事件', async () => {
    // 流被中断时最后通常有一截残帧。它不该被当成完整事件吐出来。
    const sse = frame('token', { text: '完整' }) + 'event: token\ndata: {"text":"残'
    stubFetch(sseResponse(splitBytes(sse, 4096)))
    const evs = await collect(streamChat({ question: 'q' }))
    expect(evs).toHaveLength(1)
    expect((evs[0] as { data: { text: string } }).data.text).toBe('完整')
  })

  it('空流直接结束，不抛异常', async () => {
    stubFetch(sseResponse([]))
    expect(await collect(streamChat({ question: 'q' }))).toEqual([])
  })
})

describe('streamChat 失败路径', () => {
  it('HTTP 非 2xx 抛错，错误里带状态码', async () => {
    stubFetch(sseResponse([], false, 503))
    await expect(collect(streamChat({ question: 'q' }))).rejects.toThrow(/503/)
  })

  it('响应没有 body 时抛错而不是静默结束', async () => {
    // 静默结束的表现是「一问就答不出来，但界面不报错」——
    // 用户只会以为是模型没说话。这种必须抛。
    stubFetch({ ok: true, status: 200, body: null } as unknown as Response)
    await expect(collect(streamChat({ question: 'q' }))).rejects.toThrow(/SSE/)
  })

  it('signal 传得下去（取消后 fetch 收到它）', async () => {
    const ctrl = new AbortController()
    // 显式标注两个入参：vi.fn(async () => ...) 的 calls 会被推断成 `[]`，
    // 断言时就拿不到第二个参数（signal 正是从这儿传的）。
    const spy = vi.fn((_url: RequestInfo | URL, _init?: RequestInit) =>
      Promise.resolve(sseResponse([])),
    )
    vi.stubGlobal('fetch', spy)
    await collect(streamChat({ question: 'q' }, ctrl.signal))
    expect(spy.mock.calls[0][1]?.signal).toBe(ctrl.signal)
  })

  it('请求体带上 question / game / backend', async () => {
    const spy = vi.fn((_url: RequestInfo | URL, _init?: RequestInit) =>
      Promise.resolve(sseResponse([])),
    )
    vi.stubGlobal('fetch', spy)
    await collect(streamChat({ question: '三月七', game: 'sr', backend: 'minimax' }))
    const init = spy.mock.calls[0][1] as RequestInit
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({
      question: '三月七',
      game: 'sr',
      backend: 'minimax',
    })
  })
})