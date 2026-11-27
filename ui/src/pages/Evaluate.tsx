import { useEffect, useMemo, useState } from 'react'
import { BarChart3, Play } from 'lucide-react'
import { api, useApi, SLICES, type Job, type Run } from '../lib/api'
import { Badge, Button, Card, Empty, ErrorBox, Field, LogBox, NumberInput, PageHeader, Select, Spinner, TextInput } from '../components/ui'
import { HorizonPicker, hValid, range, H_MIN, H_MAX } from '../components/HorizonPicker'
import { XYChart } from '../components/charts'
import { Tabs } from '../components/ui'
import PrensComparison from './PrensComparison'
import { SLICE_COLOR, SLICE_LABEL, fmt, methodColor } from '../lib/theme'

export default function EvaluatePage() {
  const { data: runs } = useApi<Run[]>('/runs')
  const { data: reports, reload } = useApi<{ tag: string }[]>('/reports')
  const [report, setReport] = useState<string | null>(null)
  const [tab, setTab] = useState<'reports' | 'prens'>(new URLSearchParams(window.location.search).get('tab') === 'prens' ? 'prens' : 'reports')
  useEffect(() => { if (!report && reports?.length) setReport(reports[0].tag) }, [reports])
  return (
    <>
      <PageHeader icon={<BarChart3 className="h-5 w-5" />} title="Evaluate & compare" subtitle="Same test windows for every method · persistence, moving average and ARIMA baselines" />
      <Tabs value={tab} onChange={setTab} tabs={[{ value: 'reports', label: 'Evaluations' }, { value: 'prens', label: 'PreNS paper comparison' }]} />
      {tab === 'prens' ? <PrensComparison /> : <>
      <NewEval runs={runs?.filter((r) => r.status === 'done') ?? null} onFinished={(tag) => { reload(); setReport(tag) }} />
      <div className="mt-6">
        {!reports ? <Spinner /> : !reports.length ? <Empty>No evaluation reports yet — run one above.</Empty> : (
          <>
            <div className="mb-4 flex items-end gap-3">
              <Field label="Report" className="w-[28rem]"><Select value={report ?? reports[0].tag} onChange={setReport} options={reports.map((r) => r.tag)} /></Field>
            </div>
            {report && <Report tag={report} />}
          </>
        )}
      </div>
      </>}
    </>
  )
}

function NewEval({ runs, onFinished }: { runs: Run[] | null; onFinished: (tag: string) => void }) {
  const [sel, setSel] = useState<string[]>([])
  const [split, setSplit] = useState('test')
  const [arima, setArima] = useState(2000)
  const [H, setH] = useState([5, 10, 15])
  const [tag, setTag] = useState('')
  const [ens, setEns] = useState({ ensemble: false, blend: false })
  const [err, setErr] = useState<string | null>(null)
  const { data: jobs } = useApi<Job[]>('/jobs', { kind: 'eval' }, 2500)
  const last = jobs?.[jobs.length - 1]
  const [watching, setWatching] = useState<string | null>(null)
  useEffect(() => {
    if (watching && last && last.name === watching && !last.running) { onFinished(last.meta?.out ?? watching); setWatching(null) }
  }, [last?.running])
  useEffect(() => { if (runs?.length && !sel.length) setSel([runs[0].id]) }, [runs])
  useEffect(() => {
    setTag(sel.length ? `ui_${sel.map((s) => s.split('/').pop()).join('_vs_')}`.slice(0, 70).replace(/[^A-Za-z0-9_\-]/g, '_') : '')
  }, [sel.join(',')])
  const selRuns = runs?.filter((r) => sel.includes(r.id)) ?? []
  const mixedDir = new Set(selRuns.map((r) => r.direction ?? 'dl')).size > 1
  const mixedTarget = new Set(selRuns.map((r) => r.target ?? 'point')).size > 1
  const allMode = sel.length > 0 && runs?.filter((r) => sel.includes(r.id)).every((r) => r.horizon_mode === 'all')
  const allowed = allMode ? range(H_MIN, H_MAX) : (runs?.find((r) => r.id === sel[0])?.model_horizons ?? range(H_MIN, H_MAX))
  const run = async () => {
    setErr(null)
    try {
      await api.post('/evaluate', { runs: sel, split, horizons: H, arima_n: arima, tag, ...ens })
      setWatching(tag)
    } catch (e: any) { setErr(e.message) }
  }
  if (!runs) return <Spinner />
  return (
    <Card title="New evaluation" subtitle="Runs evaluate.py in the background (≈1–3 min with ARIMA)">
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <div className="mb-2 text-xs font-medium text-slate-600">Runs to compare ({sel.length} selected)</div>
          <div className="max-h-64 space-y-1 overflow-y-auto rounded-xl border border-slate-200 p-2">
            {runs.map((r) => (
              <label key={r.id} className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-sm hover:bg-slate-50">
                <input type="checkbox" checked={sel.includes(r.id)} onChange={(e) => setSel(e.target.checked ? [...sel, r.id] : sel.filter((x) => x !== r.id))} />
                <span className="font-mono text-xs">{r.id}</span>
                <span className="ml-auto text-xs text-slate-400">{(r.direction ?? 'dl').toUpperCase()} · {r.model} · {r.horizon_mode}{r.target && r.target !== 'point' ? ` · ${r.target}` : ''}</span>
              </label>
            ))}
          </div>
        </div>
        <div className="space-y-3">
          <HorizonPicker value={H} onChange={setH} allowed={allowed} />
          <div className="grid grid-cols-3 gap-3">
            <Field label="Split"><Select value={split} onChange={setSplit} options={['test', 'ood', 'val']} format={(v) => v === 'ood' ? 'ood (unseen ranges)' : v} /></Field>
            <Field label="ARIMA windows" hint="0 = skip"><NumberInput value={arima} min={0} max={20000} step={500} onChange={setArima} /></Field>
            <Field label="Report name"><TextInput value={tag} onChange={setTag} /></Field>
          </div>
          <div className="flex flex-wrap gap-4 text-sm text-slate-600">
            <label className="flex items-center gap-2"><input type="checkbox" checked={ens.ensemble} onChange={(e) => setEns({ ...ens, ensemble: e.target.checked })} />
              Ensemble <span className="text-xs text-slate-400">(mean of the selected runs)</span></label>
            <label className="flex items-center gap-2"><input type="checkbox" checked={ens.blend} onChange={(e) => setEns({ ...ens, blend: e.target.checked })} />
              Blend with persistence <span className="text-xs text-slate-400">(weights fitted on validation)</span></label>
          </div>
          {mixedDir && <p className="text-xs font-medium text-rose-600">The selected runs mix DL and UL — DL and UL are separate PRB budgets, evaluate them separately.</p>}
          {mixedTarget && <p className="text-xs font-medium text-rose-600">The selected runs predict different targets (point / average / peak) — pick runs with the same target.</p>}
          <ErrorBox error={err} />
          <Button onClick={run} disabled={!sel.length || !hValid(H) || !tag || !!last?.running || mixedDir || mixedTarget}><Play className="h-4 w-4" /> Run evaluation</Button>
          {last && (last.running || watching) && (
            <div>
              <div className="mb-1 text-sm">{last.name} {last.running ? <Badge color="blue">running</Badge> : <Badge color="green">finished</Badge>}</div>
              <LogBox text={last.log_tail} />
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}

function Report({ tag }: { tag: string }) {
  const { data: R, error } = useApi<any>('/reports/detail', { tag })
  const [metric, setMetric] = useState<'r2' | 'rmse'>('r2')
  const curves = useMemo(() => {
    if (!R) return null
    const out: Record<string, any[]> = {}
    for (const s of SLICES) {
      out[s] = R.horizons.map((h: number) => {
        const o: any = { h }
        for (const m of Object.keys(R.metrics)) if (R.metrics[m][s]) o[m] = R.metrics[m][s][String(h)]?.[metric]
        return o
      })
    }
    return out
  }, [R, metric])
  if (error) return <ErrorBox error={error} />
  if (!R || !curves) return <Spinner />
  const methods = Object.keys(R.metrics)
  const Hs: number[] = R.horizons_selected
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-500">
        <Badge color="blue">split: {R.split}</Badge>
        <span>{R.n_windows.toLocaleString()} windows</span>
        {R.arima_windows > 0 && <span>· ARIMA on a random {R.arima_windows}-window subset (skill vs persistence on the same subset)</span>}
      </div>
      <div className="grid gap-5 lg:grid-cols-3">
        {SLICES.map((s) => (
          <Card key={s} title={<span style={{ color: SLICE_COLOR[s] }}>{SLICE_LABEL[s]}</span>}
            actions={s === 'embb' ? (
              <select className="rounded-md border border-slate-200 px-2 py-1 text-xs" value={metric} onChange={(e) => setMetric(e.target.value as any)}>
                <option value="r2">R²</option><option value="rmse">RMSE (PRB)</option>
              </select>) : undefined}>
            <XYChart data={curves[s]} xKey="h" xLabel="horizon (s)" yLabel={metric === 'r2' ? 'R²' : 'RMSE'} height={240} shade={[5, 15]}
              series={methods.filter((m) => R.metrics[m][s]).map((m, i) => ({
                key: m, name: m.split('/').pop()!, color: methodColor(m, i),
                dash: ['persistence', 'ma10', 'arima'].includes(m) ? '5 4' : undefined, width: ['persistence', 'ma10', 'arima'].includes(m) ? 1.5 : 2.2,
              }))} />
          </Card>
        ))}
      </div>
      {SLICES.map((s) => (
        <Card key={s} title={<span><span style={{ color: SLICE_COLOR[s] }}>{SLICE_LABEL[s]}</span> — requested PRBs</span>}
          subtitle="Skill = % RMSE reduction vs persistence (positive = better)">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-2 pr-3">Method</th>
                  {Hs.map((h) => <th key={h} colSpan={3} className="border-l border-slate-100 pl-3 text-center">{h} s</th>)}
                </tr>
                <tr className="border-b border-slate-200 text-left text-[11px] text-slate-400">
                  <th></th>{Hs.map((h) => [<th key={h + 'a'} className="border-l border-slate-100 pl-3">RMSE</th>, <th key={h + 'b'}>R²</th>, <th key={h + 'c'}>skill</th>])}
                </tr>
              </thead>
              <tbody>
                {methods.filter((m) => R.metrics[m][s]).map((m) => (
                  <tr key={m} className="border-b border-slate-100 last:border-0">
                    <td className="py-1.5 pr-3 font-mono text-xs">{m}</td>
                    {Hs.map((h) => {
                      const r = m === 'arima' && R.arima_subset_metrics?.arima ? { ...R.arima_subset_metrics.arima[s][String(h)], skill_vs_persistence: R.metrics[m][s][String(h)].skill_vs_persistence } : R.metrics[m][s][String(h)]
                      const sk = r.skill_vs_persistence
                      return [
                        <td key={h + 'a'} className="border-l border-slate-100 pl-3">{fmt(r.rmse)}</td>,
                        <td key={h + 'b'} className="font-semibold">{fmt(r.r2, 3)}</td>,
                        <td key={h + 'c'} className={sk == null ? 'text-slate-300' : sk >= 0 ? 'text-emerald-600' : 'text-rose-600'}>
                          {sk == null ? '–' : `${sk >= 0 ? '+' : ''}${(100 * sk).toFixed(1)}%`}</td>,
                      ]
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ))}
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Cost" subtitle="From training (shared GPU inflates times when jobs run in parallel)">
          <table className="w-full text-sm">
            <thead><tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-500"><th className="py-2">Run</th><th>Params</th><th>Train min</th><th>Infer ms</th><th>GPU MB</th></tr></thead>
            <tbody>{Object.entries(R.cost).map(([n, c]: any) => (
              <tr key={n} className="border-b border-slate-100 last:border-0"><td className="py-1.5 font-mono text-xs">{n}</td><td>{c.params_per_slice_model.toLocaleString()}</td>
                <td>{c.train_min_total}</td><td>{c.infer_ms_batch1}</td><td>{c.peak_gpu_mb}</td></tr>))}</tbody>
          </table>
        </Card>
        <Card title="RMSE by scenario" subtitle={`${methods[0]} vs persistence at ${Hs[Hs.length - 1]} s (PRBs)`}>
          <table className="w-full text-sm">
            <thead><tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-500"><th className="py-2">Scenario</th>{Object.keys(R.scenario_rmse).map((s) => <th key={s} style={{ color: SLICE_COLOR[s] }}>{SLICE_LABEL[s]}</th>)}</tr></thead>
            <tbody>{Object.keys(Object.values(R.scenario_rmse)[0] as any).map((sc) => (
              <tr key={sc} className="border-b border-slate-100 last:border-0"><td className="py-1.5">{sc}</td>
                {Object.keys(R.scenario_rmse).map((s) => {
                  const v = R.scenario_rmse[s][sc][String(Hs[Hs.length - 1])]
                  return <td key={s}><b>{fmt(v.model, 1)}</b> <span className="text-xs text-slate-400">/ {fmt(v.persistence, 1)}</span></td>
                })}</tr>))}</tbody>
          </table>
        </Card>
      </div>
      {R.images?.includes('predictions_vs_actual.png') && (
        <Card title="Predictions vs actual" subtitle={`First ${R.split ?? 'test'} trace, first 15 min — each prediction is drawn at the time it is for (made h s earlier)`}>
          <img className="w-full" src={api.url('/reports/image', { tag, name: 'predictions_vs_actual.png' })} />
        </Card>
      )}
    </div>
  )
}
