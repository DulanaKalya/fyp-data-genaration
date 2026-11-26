import { useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, useApi, type Job } from '../lib/api'
import { Badge, Button, Card, Empty, ErrorBox, LogBox, Spinner } from '../components/ui'
import { HORIZON_COLORS, SLICE_COLOR, SLICE_LABEL, fmt } from '../lib/theme'

const SLICES = ['embb', 'urllc', 'mmtc']
const MODEL_ORDER = ['predp', 'lstm', 'gru', 'transformer', 'arima']

export default function PrensComparison() {
  const { data: C, reload, error } = useApi<any>('/prens')
  const { data: jobs } = useApi<Job[]>('/jobs', { kind: 'prens' }, 3000)
  const job = jobs?.[jobs.length - 1]
  const [err, setErr] = useState<string | null>(null)
  const [wasRunning, setWasRunning] = useState(false)
  useEffect(() => {
    if (job?.running) setWasRunning(true)
    if (wasRunning && job && !job.running) { setWasRunning(false); reload() }
  }, [job?.running])

  const refresh = async () => {
    setErr(null)
    try { await api.post('/prens/refresh', {}); setWasRunning(true) } catch (e: any) { setErr(e.message) }
  }

  if (error) return <ErrorBox error={error} />
  if (!C) return <Spinner />
  const header = (
    <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
      <div className="text-sm text-slate-500">
        PreNS: Wu et al., <i>IEEE Trans. Cognitive Communications and Networking</i>, 2026 — Table III
        {C.available && <> · updated {new Date(C.updated * 1000).toLocaleString()}</>}
      </div>
      <Button variant="secondary" onClick={refresh} disabled={job?.running}>
        <RefreshCw className={`h-4 w-4 ${job?.running ? 'animate-spin' : ''}`} /> {job?.running ? 'Re-evaluating…' : 'Refresh with latest E1 runs'}
      </Button>
    </div>
  )
  if (!C.available) return <>{header}<ErrorBox error={err} /><Empty>No comparison yet — press Refresh (needs finished E1 runs).</Empty></>

  const H: number[] = C.horizons
  const P = C.prens
  const labels: Record<string, string> = {}
  Object.entries(P).forEach(([label, p]: any) => { labels[p.key] = label })
  const ours = C.ours
  const sw = C.same_window ?? {}
  const keys = MODEL_ORDER.filter((k) => labels[k])

  // bar chart rows
  const bars = keys.map((k) => {
    const o: any = { model: labels[k].replace(' (attention Bi-LSTM)', ''), prens: P[labels[k]].r2 }
    H.forEach((h) => { o[`h${h}`] = (k === 'arima' ? sw.arima?.[h] : ours[k]?.[h]?.r2) ?? null })
    return o
  })

  // findings (computed, so they stay true after a refresh)
  const deep = ['predp', 'lstm', 'gru', 'transformer'].filter((k) => sw[k])
  const bestAt = (h: number) => Object.keys(sw).reduce((a, b) => (sw[b][h] > sw[a][h] ? b : a))
  const spread = (h: number) => deep.length ? Math.max(...deep.map((k) => sw[k][h])) - Math.min(...deep.map((k) => sw[k][h])) : 0
  const meanOver = (k: string) => H.reduce((a, h) => a + sw[k][h], 0) / H.length
  const ranking = Object.keys(sw).filter((k) => labels[k]).sort((a, b) => meanOver(b) - meanOver(a))
  const prensRank = Object.entries(P).sort((a: any, b: any) => b[1].r2 - a[1].r2)

  return (
    <div className="space-y-5">
      {header}
      <ErrorBox error={err} />
      {job?.running && <LogBox text={job.log_tail} />}

      <Card title="Setup" subtitle="The two studies predict at very different time scales — compare rankings and relative gains, not absolute numbers">
        <table className="w-full text-sm">
          <thead><tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="py-2 w-40"></th><th>PreNS paper</th><th>This work</th></tr></thead>
          <tbody>
            {['data', 'sampling', 'input', 'horizon', 'hardware'].map((k) => (
              <tr key={k} className="border-b border-slate-100 last:border-0">
                <td className="py-2 font-medium capitalize text-slate-600">{k}</td>
                <td className="text-slate-700">{C.setup.prens[k]}</td>
                <td className="font-medium text-slate-900">{C.setup.ours[k]}</td>
              </tr>))}
            <tr><td className="py-2 font-medium text-slate-600">Test data</td><td className="text-slate-500">not stated</td>
              <td className="font-medium text-slate-900">{C.n_windows.toLocaleString()} windows · 18 h of unseen traces</td></tr>
          </tbody>
        </table>
      </Card>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Card title="R² — PreNS (200 ms) vs this work (5 / 10 / 15 s)" subtitle="Mean over the three slices · dotted lines = persistence at each horizon">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={bars} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
              <CartesianGrid stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey="model" tick={{ fontSize: 11 }} interval={0} />
              <YAxis domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip formatter={(v: any) => (typeof v === 'number' ? v.toFixed(3) : 'running')} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="prens" name="PreNS (200 ms)" fill="#94a3b8" radius={[3, 3, 0, 0]} isAnimationActive={false} />
              {H.map((h, i) => <Bar key={h} dataKey={`h${h}`} name={`this work ${h} s`} fill={HORIZON_COLORS[i]} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
              {sw.persistence && H.map((h, i) => <ReferenceLine key={h} y={sw.persistence[h]} stroke={HORIZON_COLORS[i]} strokeDasharray="3 3" />)}
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Key findings" subtitle="Computed from the current results">
          <ul className="space-y-3 text-sm text-slate-700">
            <li>🏆 <b>{labels[ranking[0]] ?? ranking[0]}</b> has the best mean R² over 5–15 s ({fmt(meanOver(ranking[0]), 3)}) — PreNS also ranks PreDP first.</li>
            <li>📈 Best model per horizon (same windows): {H.map((h) => <span key={h} className="mr-2"><Badge color="blue">{h} s</Badge> {labels[bestAt(h)] ?? bestAt(h)}</span>)}</li>
            <li>⚖️ The deep models are within <b>{fmt(Math.max(...H.map(spread)), 3)}</b> R² of each other (PreNS: up to 0.12 between PreDP and UT-LSTM) — with one input series per slice and second-scale dynamics, model choice matters little.</li>
            {sw.persistence && <li>⏱️ Over persistence, PreDP gains <b>+{fmt(sw.predp?.[H[H.length - 1]] - sw.persistence[H[H.length - 1]], 3)}</b> R² at {H[H.length - 1]} s but only +{fmt(sw.predp?.[H[0]] - sw.persistence[H[0]], 3)} at {H[0]} s — prediction pays off at the longer horizons the control loop needs.</li>}
            {sw.arima && <li>📉 ARIMA is far stronger than in PreNS (R² {fmt(meanOver('arima'), 2)} vs 0.13): 1 s data is smoother than 100 ms data, which favours linear models.</li>}
            <li>🎯 URLLC is the hardest slice in both studies.</li>
          </ul>
        </Card>
      </div>

      <Card title="Accuracy table" subtitle="RMSE/MAE in z-score units (PreNS values fit this scale: RMSE ≈ √(1−R²)); this work = mean over slices">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="py-2">Model</th>
                <th className="border-l border-slate-100 pl-3 text-center" colSpan={3}>PreNS (200 ms)</th>
                {H.map((h) => <th key={h} className="border-l border-slate-100 pl-3 text-center" colSpan={3}>This work · {h} s</th>)}
              </tr>
              <tr className="border-b border-slate-200 text-left text-[11px] text-slate-400">
                <th></th>{[0, ...H].map((h) => ['R²', 'RMSE', 'MAE'].map((m, j) => <th key={`${h}${m}`} className={j === 0 ? 'border-l border-slate-100 pl-3' : ''}>{m}</th>))}
              </tr>
            </thead>
            <tbody>
              {keys.map((k) => (
                <tr key={k} className="border-b border-slate-100 last:border-0">
                  <td className="py-2 font-medium">{labels[k]}</td>
                  <td className="border-l border-slate-100 pl-3 font-semibold text-slate-600">{fmt(P[labels[k]].r2, 3)}</td>
                  <td className="text-slate-500">{fmt(P[labels[k]].rmse, 3)}</td><td className="text-slate-500">{fmt(P[labels[k]].mae, 3)}</td>
                  {H.map((h) => ours[k]?.[h] ? [
                    <td key={`${h}a`} className="border-l border-slate-100 pl-3 font-semibold">{fmt(ours[k][h].r2, 3)}</td>,
                    <td key={`${h}b`}>{fmt(ours[k][h].rmse_z, 3)}</td>, <td key={`${h}c`}>{fmt(ours[k][h].mae_z, 3)}</td>,
                  ] : <td key={h} colSpan={3} className="border-l border-slate-100 pl-3 text-xs text-slate-400">training…</td>)}
                </tr>))}
              {ours.persistence && (
                <tr className="bg-slate-50/60">
                  <td className="py-2 italic text-slate-600">Persistence ŷ(t+h)=y(t)</td>
                  <td className="border-l border-slate-100 pl-3 text-slate-400" colSpan={3}>not reported</td>
                  {H.map((h) => [
                    <td key={`${h}a`} className="border-l border-slate-100 pl-3">{fmt(ours.persistence[h].r2, 3)}</td>,
                    <td key={`${h}b`}>{fmt(ours.persistence[h].rmse_z, 3)}</td>, <td key={`${h}c`}>{fmt(ours.persistence[h].mae_z, 3)}</td>])}
                </tr>)}
            </tbody>
          </table>
        </div>
        {C.arima_windows > 0 && <p className="mt-2 text-xs text-slate-500">ARIMA(2,0,1) is evaluated on a random {C.arima_windows}-window subset; see the same-window table for a fair comparison.</p>}
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Same-window comparison" subtitle={`All methods on the same ${C.arima_windows} random test windows — mean R² (best in bold)`}>
          <table className="w-full text-sm">
            <thead><tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-500"><th className="py-2">Model</th>{H.map((h) => <th key={h}>{h} s</th>)}</tr></thead>
            <tbody>{[...keys, 'persistence', 'ma10'].filter((k) => sw[k]).map((k) => (
              <tr key={k} className="border-b border-slate-100 last:border-0">
                <td className="py-1.5">{labels[k] ?? k}</td>
                {H.map((h) => {
                  const best = Math.max(...Object.values(sw).map((v: any) => v[h]))
                  return <td key={h} className={Math.abs(sw[k][h] - best) < 1e-9 ? 'font-bold text-indigo-700' : ''}>{fmt(sw[k][h], 3)}</td>
                })}
              </tr>))}</tbody>
          </table>
        </Card>
        <Card title="Ranking" subtitle="PreNS by R² at 200 ms · this work by mean R² over 5–15 s (same windows)">
          <table className="w-full text-sm">
            <thead><tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-500"><th className="py-2 w-12">#</th><th>PreNS</th><th>This work</th></tr></thead>
            <tbody>{prensRank.map(([label, p]: any, i) => (
              <tr key={label} className="border-b border-slate-100 last:border-0">
                <td className="py-1.5 font-semibold text-slate-400">{i + 1}</td>
                <td>{label} <span className="text-xs text-slate-400">({p.r2.toFixed(3)})</span></td>
                <td>{ranking[i] ? <>{labels[ranking[i]]} <span className="text-xs text-slate-400">({meanOver(ranking[i]).toFixed(3)})</span></> : <span className="text-slate-400">—</span>}</td>
              </tr>))}</tbody>
          </table>
        </Card>
      </div>

      <Card title="Per-slice R² (this work)">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-500">
                <th className="py-2">Model</th>{SLICES.map((s) => <th key={s} colSpan={H.length} className="border-l border-slate-100 pl-3 text-center" style={{ color: SLICE_COLOR[s] }}>{SLICE_LABEL[s]}</th>)}
              </tr>
              <tr className="border-b border-slate-200 text-left text-[11px] text-slate-400"><th></th>
                {SLICES.map((s) => H.map((h, j) => <th key={s + h} className={j === 0 ? 'border-l border-slate-100 pl-3' : ''}>{h} s</th>))}</tr>
            </thead>
            <tbody>{[...keys, 'persistence'].filter((k) => ours[k]).map((k) => (
              <tr key={k} className="border-b border-slate-100 last:border-0">
                <td className="py-1.5">{labels[k] ?? k}</td>
                {SLICES.map((s) => H.map((h, j) => <td key={s + h} className={j === 0 ? 'border-l border-slate-100 pl-3' : ''}>{fmt(ours[k][h].per_slice_r2[s], 3)}</td>))}
              </tr>))}</tbody>
          </table>
        </div>
      </Card>

      <Card title="Cost" subtitle="Our training: 84 h of data, GPU shared by several jobs — not comparable to PreNS's single-job RTX 4090 times">
        <table className="w-full text-sm">
          <thead><tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-500">
            <th className="py-2">Model</th><th>Params PreNS</th><th>Params ours</th><th>Train min PreNS</th><th>Train min ours (3 slices)</th>
            <th>Infer ms PreNS</th><th>Infer ms ours</th></tr></thead>
          <tbody>{keys.map((k) => {
            const p = P[labels[k]]; const c = C.cost[k]
            return (
              <tr key={k} className="border-b border-slate-100 last:border-0">
                <td className="py-1.5">{labels[k]}</td><td>{p.params.toLocaleString()}</td><td>{c ? c.params_per_slice_model.toLocaleString() : '—'}</td>
                <td>{p.train_min}</td><td>{c ? c.train_min_total : '—'}</td><td>{p.infer_ms}</td><td className="font-semibold">{c ? c.infer_ms_batch1 : '—'}</td>
              </tr>)
          })}</tbody>
        </table>
      </Card>
    </div>
  )
}
