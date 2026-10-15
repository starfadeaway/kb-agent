import { useEffect, useMemo, useState } from 'react'
import { retrieve } from '../api'
import type { Health, RetrieveHit } from '../types'

interface Result {
  query: string
  label: string
  hits: RetrieveHit[]
  ms: number
}

interface Props {
  health: Health | null
  onOpenSettings: () => void
}

/**
 * 检索调试台：不开 Agent，直接调 /retrieve 对比不同检索策略。
 * 这是把「实验 A 的结论」变成可以当场演示的东西 ——
 * 面试时不用念 PPT，改一个开关就能看到 top-5 换一批。
 */
export function RetrievePanel({ health, onOpenSettings }: Props) {
  const [query, setQuery] = useState('位面饰品太空封印站的两件套效果是什么')
  const [game, setGame] = useState('sr')
  const [topK, setTopK] = useState(5)
  const [useBm25, setUseBm25] = useState(true)
  const [useVector, setUseVector] = useState(true)
  const [useRerank, setUseRerank] = useState(true)
  const [sinceDate, setSinceDate] = useState('')
  const [result, setResult] = useState<Result | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 游戏列表同样由后端下发，不在前端写死
  const games = useMemo(() => Object.entries(health?.games ?? { sr: '崩坏：星穹铁道' }), [health])
  useEffect(() => {
    if (games.length && !games.some(([slug]) => slug === game)) setGame(games[0][0])
  }, [games, game])

  const run = async () => {
    if (!query.trim() || busy) return
    setBusy(true)
    setError(null)
    const t0 = performance.now()
    try {
      const r = await retrieve({
        query: query.trim(),
        game,
        top_k: topK,
        use_bm25: useBm25,
        use_vector: useVector,
        use_rerank: useRerank,
        min_revision_ts: sinceDate || undefined,
      })
      setResult({ ...r, ms: Math.round(performance.now() - t0) })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="retrieve">
      <header className="chat-head">
        <div>
          <h1>检索调试</h1>
          <p className="muted">切换开关即时对比 A0 仅BM25 / A1 仅向量 / A2 融合 / A3 加重排</p>
        </div>
        <div className="head-actions">
          <button className="icon-btn" onClick={onOpenSettings} aria-label="外观设置">
            ⚙
          </button>
        </div>
      </header>

      <div className="controls">
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="输入检索问题" />
        <select value={game} onChange={(e) => setGame(e.target.value)}>
          {games.map(([slug, label]) => (
            <option key={slug} value={slug}>
              {label}
            </option>
          ))}
        </select>
        <label>
          top_k
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
        </label>
        <label className="switch" title="稀疏检索：按词面匹配。星铁的「位面饰品」「行迹」这类低频专名靠它才召得回。">
          <input type="checkbox" checked={useBm25} onChange={(e) => setUseBm25(e.target.checked)} />
          BM25
        </label>
        <label className="switch" title="稠密检索：按语义相似。问题里没出现道具名时（如「攻击力不够还想更快」）只有它管用。">
          <input type="checkbox" checked={useVector} onChange={(e) => setUseVector(e.target.checked)} />
          向量
        </label>
        <label
          className="switch"
          title="重排（Cross-Encoder）：把候选逐条和问题一起送进模型重新打分。recall@3 从 0.828 拉到 0.897，但平均检索延迟从 272ms 涨到 784ms（单条探针口径下是 37ms → 763ms）—— 效果与代价最大的取舍。详见 docs/rerank.md"
        >
          <input type="checkbox" checked={useRerank} onChange={(e) => setUseRerank(e.target.checked)} />
          重排
        </label>
        <label
          className="since"
          title="版本感知检索：只保留在这个日期之后被修订过的页面。wiki 每次改动都是新版本，这个开关能直接看到某个结论是不是最近才改的。"
        >
          修订于
          <input
            type="date"
            value={sinceDate}
            onChange={(e) => setSinceDate(e.target.value)}
          />
          {sinceDate && (
            <button className="link-clear" onClick={() => setSinceDate('')} title="清除版本过滤">
              ✕
            </button>
          )}
        </label>
        <button onClick={run} disabled={busy}>
          {busy ? '检索中…' : '检索'}
        </button>
      </div>

      <div className="preset-help muted">
        {useBm25 && !useVector && 'A0 仅 BM25：只用词面匹配，最快，低频专名能中'}
        {!useBm25 && useVector && 'A1 仅向量：只用语义，问题里没道具名时才用得上'}
        {useBm25 && useVector && !useRerank && 'A2 融合：BM25 + 向量取并集后按 RRF 合并排名，兼顾两者'}
        {useBm25 && useVector && useRerank && 'A3 融合+重排：把候选和问题一起送 Cross-Encoder 精排，精度最高也最慢'}
        {!useBm25 && !useVector && '未启用任何检索'}
      </div>

      {error && <div className="error">检索失败：{error}</div>}

      {result && (
        <>
          <div className="summary">
            配置 <code>{result.label}</code> · 命中 {result.hits.length} 条 · 端到端 {result.ms}ms
            {sinceDate && (
              <span className="muted"> · 仅看 {sinceDate} 之后修订的内容</span>
            )}
            <span className="muted">
              {' '}
              （{useBm25 && !useVector && '仅BM25'}
              {!useBm25 && useVector && '仅向量'}
              {useBm25 && useVector && !useRerank && 'RRF融合'}
              {useBm25 && useVector && useRerank && '融合+重排'}
              {!useBm25 && !useVector && '无检索'})
            </span>
          </div>
          <div className="hits">
            {result.hits.map((h) => (
              <div key={h.chunk_id} className="hit">
                <div className="hit-head">
                  <span className="rank">#{h.rank}</span>
                  <span className="score" title="相似度 / 重排相关度">
                    {h.score.toFixed(4)}
                  </span>
                  <span className="src">{h.source}</span>
                  <a href={h.url} target="_blank" rel="noreferrer">
                    {h.page_title}
                  </a>
                  <span className="section">{h.section}</span>
                </div>
                <div className="hit-text">{h.text.slice(0, 220)}</div>
                <div className="hit-foot">
                  <code>{h.chunk_id}</code>
                  {h.last_revision_ts && <span className="muted">修订 {h.last_revision_ts}</span>}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
