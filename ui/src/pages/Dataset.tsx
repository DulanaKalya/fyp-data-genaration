import { useEffect, useMemo, useState } from 'react'
import { CheckCircle2, Database, Play, XCircle } from 'lucide-react'
import { api, useApi, SLICES, type Dataset, type Job } from '../lib/api'
import { Badge, Button, Card, ErrorBox, Field, LogBox, NumberInput, PageHeader, Select, Spinner, Stat, Tabs, TextInput } from '../components/ui'
import { ScenarioLegend, StackedChart, TimeChart, XYChart } from '../components/charts'
import { SLICE_COLOR, SLICE_LABEL, fmt } from '../lib/theme'

export default function DatasetPage() {
  const [tab, setTab] = useState<'explore' | 'generate'>('explore')
  const { data: datasets, reload } = useApi<Dataset[]>('/datasets')
  const [data, setData] = useState('data/raw')
  useEffect(() => {
    if (datasets?.length && !datasets.find((d) => d.path === data)) setData(datasets[0].path)
  }, [datasets])
  const ds = datasets?.find((d) => d.path === data)
  return (
    <>
      <PageHeader icon={<Database className="h-5 w-5" />} title="Dataset" subtitle="Testbed-matched traffic simulator · 1 s rows · requested vs allocated PRBs" />
      <Tabs value={tab} onChange={setTab} tabs={[{ value: 'explore', label: 'Explore' }, { value: 'generate', label: 'Generate new dataset' }]} />
      {tab === 'explore' ? (
        !datasets ? <Spinner /> : (
          <>
            <div className="mb-5 flex items-end gap-4">
              <Field label="Dataset" className="w-64">
                <Select value={data} onChange={setData} options={datasets.map((d) => d.path)} />
              </Field>
              {ds && <span className="pb-2 text-sm text-slate-500">sim v{ds.sim_version} · {ds.trace_hours} h per trace · {ds.total_prb} PRBs</span>}
            </div>
            {ds && (
              <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
                {Object.entries(ds.splits).map(([s, n]) => (
                  <Stat key={s} label={s} value={`${ds.hours[s]} h`} hint={`${n} traces · ${(ds.hours[s] * 3600).toLocaleString()} rows`} />
                ))}
              </div>
            )}
            <Checks data={data} />
            <TraceViewer data={data} ds={ds} />
          </>
        )
      ) : <Generate onDone={reload} />}
    </>
  )
}

function Checks({ data }: { data: string }) {
  const { data: s } = useApi<any>('/datasets/summary', { data })
  if (!s) return <Spinner />
  if (!s.available) return <Card title="Dataset checks" className="mb-6"><p className="text-sm text-slate-500">No check report for this dataset.</p></Card>
  const lags = Array.from({ length: 30 }, (_, i) => i + 1)
  const acf = lags.map((L) => ({ lag: L, ...Object.fromEntries(SLICES.map((sl) => [sl, s.acf_train[sl][String(L)]])) }))
  const pers = Array.from({ length: 15 }, (_, i) => i + 1).map((h) => ({ h, ...Object.fromEntries(SLICES.map((sl) => [sl, s.persistence_r2_test[sl][String(h)]])) }))
  const series = SLICES.map((sl) => ({ key: sl, name: SLICE_LABEL[sl], color: SLICE_COLOR[sl] }))
  return (
    <div className="mb-6 grid gap-5 lg:grid-cols-3">
      <Card title="Quality checks" subtitle="PLAN.md Phase 1.8">
        <ul className="space-y-2.5">
          {Object.entries(s.checks).map(([k, v]) => (
            <li key={k} className="flex items-center gap-2 text-sm">
              {v ? <CheckCircle2 className="h-4 w-4 text-emerald-500" /> : <XCircle className="h-4 w-4 text-rose-500" />}
              <span className="font-mono text-xs text-slate-600">{k}</span>
            </li>
          ))}
        </ul>
        <div className="mt-5 border-t border-slate-100 pt-4">
          <div className="mb-2 text-xs font-medium text-slate-500">Requested PRBs (train)</div>
          <table className="w-full text-sm">
            <thead><tr className="text-left text-xs text-slate-400"><th>slice</th><th>mean</th><th>p95</th><th>max</th></tr></thead>
            <tbody>{SLICES.map((sl) => (
              <tr key={sl}><td className="py-1 font-medium" style={{ color: SLICE_COLOR[sl] }}>{SLICE_LABEL[sl]}</td>
                <td>{fmt(s.stats_train[sl].mean, 1)}</td><td>{fmt(s.stats_train[sl]['95%'], 1)}</td><td>{fmt(s.stats_train[sl].max, 1)}</td></tr>
            ))}</tbody>
          </table>
          <div className="mt-3 text-xs text-slate-500">Over capacity (total &gt; 162): <b>{(100 * s.frac_over_capacity_train).toFixed(1)}%</b> of seconds</div>
        </div>
      </Card>
      <Card title="Autocorrelation" subtitle="Demand still correlated 5–15 s ahead → predictable">
        <XYChart data={acf} xKey="lag" series={series} xLabel="lag (s)" yLabel="ACF" shade={[5, 15]} height={250} />
      </Card>
      <Card title="Persistence baseline R²" subtitle="ŷ(t+h) = y(t) on test — the bar to beat">
        <XYChart data={pers} xKey="h" series={series} xLabel="horizon (s)" yLabel="R²" shade={[5, 15]} height={250} />
      </Card>
    </div>
  )
}

function TraceViewer({ data, ds }: { data: string; ds?: Dataset }) {
  const { data: splits } = useApi<Record<string, string[]>>('/datasets/traces', { data })
  const [split, setSplit] = useState('test')
  const [tid, setTid] = useState('test_000')
  const [start, setStart] = useState(0)
  const [len, setLen] = useState(15)
  useEffect(() => {
    if (splits && !splits[split]?.includes(tid)) setTid(splits[split]?.[0] ?? '')
  }, [splits, split])
  const { data: tr, loading, error } = useApi<any>(tid ? '/trace' : null, { data, id: tid, start: start * 60, end: (start + len) * 60 })
  const rows = useMemo(() => {
    if (!tr) return []
    return tr.t.map((t: number, i: number) => {
      const o: any = { x: t / 60, total: tr.total_req[i] }
      for (const s of SLICES) {
        o[`${s}_mean`] = tr.slices[s].req_mean[i]
        o[`${s}_max`] = tr.slices[s].req_max[i]
        o[`${s}_alloc`] = tr.slices[s].granted[i]
      }
      return o
    })
  }, [tr])
  const totalMin = ds ? ds.trace_hours * 60 : 60
  return (
    <Card title="Trace viewer" subtitle="Requested (solid) vs allocated (dashed) PRBs per slice" actions={<ScenarioLegend />}>
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Field label="Split"><Select value={split} onChange={setSplit} options={splits ? Object.keys(splits) : [split]} /></Field>
        <Field label="Trace"><Select value={tid} onChange={setTid} options={splits?.[split] ?? [tid]} /></Field>
        <Field label="Start (min)"><NumberInput value={start} onChange={(v) => setStart(Math.max(0, Math.min(v, totalMin - 1)))} min={0} max={totalMin - 1} /></Field>
        <Field label="Length (min)"><NumberInput value={len} onChange={(v) => setLen(Math.max(1, Math.min(v, totalMin)))} min={1} max={totalMin} /></Field>
      </div>
      <ErrorBox error={error} />
      {loading && !tr ? <Spinner /> : tr && (
        <div className="space-y-2">
          {SLICES.map((s) => (
            <div key={s}>
              <div className="mb-1 flex items-center gap-2 text-sm font-semibold" style={{ color: SLICE_COLOR[s] }}>{SLICE_LABEL[s]}</div>
              <TimeChart data={rows} segments={tr.segments} height={190} series={[
                { key: `${s}_max`, name: 'requested (1 s max)', color: SLICE_COLOR[s], width: 0.8, opacity: 0.3 },
                { key: `${s}_mean`, name: 'requested (target)', color: SLICE_COLOR[s], width: 1.6 },
                { key: `${s}_alloc`, name: 'allocated', color: '#0f172a', width: 1, dash: '4 3' },
              ]} />
            </div>
          ))}
          <div>
            <div className="mb-1 text-sm font-semibold text-slate-700">All slices — stacked allocation vs total requested</div>
            <StackedChart data={rows} segments={tr.segments} capacity={ds?.total_prb ?? 162}
              series={SLICES.map((s) => ({ key: `${s}_alloc`, name: `${SLICE_LABEL[s]} allocated`, color: SLICE_COLOR[s] }))}
              total={{ key: 'total', name: 'total requested', color: '#7c3aed' }} />
          </div>
          {tr.params && (
            <details className="mt-2 text-sm">
              <summary className="cursor-pointer text-slate-500">Trace parameters (domain-randomization draw)</summary>
              <div className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-xs text-slate-600 md:grid-cols-4">
                {Object.entries(tr.params).map(([k, v]) => <div key={k}>{k}: <b>{Number(v).toFixed(2)}</b></div>)}
              </div>
            </details>
          )}
        </div>
      )}
    </Card>
  )
}

function Generate({ onDone }: { onDone: () => void }) {
  const [out, setOut] = useState('data/raw_new')
  const [hours, setHours] = useState(1)
  const [seed, setSeed] = useState(20261126)
  const [n, setN] = useState<Record<string, number>>({ train: 84, val: 18, test: 18, ood: 10 })
  const [err, setErr] = useState<string | null>(null)
  const { data: jobs } = useApi<Job[]>('/jobs', { kind: 'dataset' }, 2000)
  const total = Object.values(n).reduce((a, b) => a + b, 0) * hours
  const anyRunning = jobs?.some((j) => j.running)
  useEffect(() => { if (jobs && !anyRunning) onDone() }, [anyRunning])
  const start = async () => {
    setErr(null)
    try { await api.post('/datasets/generate', { out, trace_hours: hours, seed, splits: n }) } catch (e: any) { setErr(e.message) }
  }
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <Card title="New dataset" subtitle="Uses configs/sim.yaml · runs the quality checks afterwards">
        <div className="grid grid-cols-3 gap-3">
          <Field label="Output folder" hint="must be inside data/, not data/raw"><TextInput value={out} onChange={setOut} /></Field>
          <Field label="Hours per trace"><NumberInput value={hours} onChange={setHours} min={0.25} max={24} step={0.25} /></Field>
          <Field label="Base seed"><NumberInput value={seed} onChange={setSeed} /></Field>
        </div>
        <div className="mt-3 grid grid-cols-4 gap-3">
          {Object.keys(n).map((s) => (
            <Field key={s} label={`${s} traces`}><NumberInput value={n[s]} min={0} onChange={(v) => setN({ ...n, [s]: Math.max(0, v) })} /></Field>
          ))}
        </div>
        <p className="mt-3 text-sm text-slate-500">Total <b>{total} h</b> of 1 s data (train {n.train * hours} h). About 15 s per 130 h.</p>
        <ErrorBox error={err} />
        <Button className="mt-4" onClick={start} disabled={anyRunning}><Play className="h-4 w-4" /> Generate</Button>
      </Card>
      <Card title="Generation jobs">
        {!jobs?.length ? <p className="text-sm text-slate-500">No jobs yet.</p> : [...jobs].reverse().slice(0, 3).map((j) => (
          <div key={j.id} className="mb-4">
            <div className="mb-1.5 flex items-center gap-2 text-sm font-medium">{j.name} {j.running ? <Badge color="blue">running</Badge> : <Badge color="green">finished</Badge>}</div>
            <LogBox text={j.log_tail} />
          </div>
        ))}
      </Card>
    </div>
  )
}
