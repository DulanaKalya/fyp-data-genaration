import { useEffect, useMemo, useRef, useState } from 'react'
import { Download, Pause, Play, Sparkles, Upload } from 'lucide-react'
import { CartesianGrid, ComposedChart, Legend, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts'
import { api, useApi, type Dataset, type Run } from '../lib/api'
import { Badge, Button, Card, Empty, ErrorBox, Field, NumberInput, PageHeader, Segmented, Select, Spinner, Stat, Tabs } from '../components/ui'
import { HorizonPicker, hValid, range, H_MIN, H_MAX } from '../components/HorizonPicker'
import { ScenarioLegend, TimeChart, XYChart } from '../components/charts'
import { HORIZON_COLORS, SLICE_COLOR, SLICE_LABEL, confColor, fmt } from '../lib/theme'

export default function InferencePage() {
  const { data: allRuns } = useApi<Run[]>('/runs')
  const runs = allRuns?.filter((r) => r.status === 'done') ?? null
  const { data: datasets } = useApi<Dataset[]>('/datasets')
  const [runId, setRunId] = useState<string | null>(null)
  const run = runs?.find((r) => r.id === runId)
  useEffect(() => {
    if (runs?.length && !runId) {
      const pref = runs.find((r) => r.horizon_mode === 'all' && r.slices.length === 3) ?? runs[0]
      setRunId(pref.id)
    }
  }, [runs])
  const allowed = run ? (run.horizon_mode === 'all' ? range(H_MIN, H_MAX).filter((h) => run.model_horizons.includes(h)) : run.model_horizons) : range(5, 15)
  const [H, setH] = useState([5, 10, 15])
  useEffect(() => { if (run) setH(run.horizons_selected.every((h) => allowed.includes(h)) ? run.horizons_selected : allowed.slice(0, 3)) }, [run?.id])

  const [src, setSrc] = useState<'dataset' | 'upload'>('dataset')
  const [data, setData] = useState('data/raw')
  const { data: splits } = useApi<Record<string, string[]>>(src === 'dataset' ? '/datasets/traces' : null, { data })
  const [split, setSplit] = useState('test')
  const [tid, setTid] = useState('test_000')
  const [upload, setUpload] = useState<{ id: string; rows: number } | null>(null)
  const [upErr, setUpErr] = useState<string | null>(null)
  useEffect(() => { if (splits && !splits[split]?.includes(tid)) setTid(splits[split]?.[0] ?? '') }, [splits, split])
  const source = src === 'dataset' ? { data, id: tid } : upload ? { data: 'upload', id: upload.id } : null

  const [tab, setTab] = useState<'forecast' | 'replay' | 'export'>('forecast')
  const ready = run && source && hValid(H)

  const onFile = async (f: File | undefined) => {
    if (!f) return
    setUpErr(null)
    try { const r = await api.upload<any>('/infer/upload', f); setUpload({ id: r.id, rows: r.rows }) } catch (e: any) { setUpErr(e.message) }
  }

  return (
    <>
      <PageHeader icon={<Sparkles className="h-5 w-5" />} title="Inference" subtitle="rApp view: forecasts at t + H1/H2/H3 and the A1 prediction packages sent to the xApp" />
      {!runs ? <Spinner /> : !runs.length ? <Empty>No finished runs yet — train one first.</Empty> : (
        <div className="mb-6 grid gap-5 lg:grid-cols-2">
          <Card title="Model & horizons">
            <Field label="Trained run"><Select value={runId ?? runs[0].id} onChange={setRunId} options={runs.map((r) => r.id)} /></Field>
            {run && (
              <div className="mt-2 flex flex-wrap gap-2 text-xs">
                <Badge color="blue">{run.model}</Badge><Badge>{run.features}</Badge><Badge>{run.horizon_mode}</Badge>
                <Badge>lookback {run.lookback} s</Badge><Badge>{run.train_hours} h train</Badge>
                <Badge>{run.slices.map((s) => SLICE_LABEL[s]).join(' · ')}</Badge>
              </div>
            )}
            <div className="mt-4"><HorizonPicker value={H} onChange={setH} allowed={allowed} /></div>
            {run?.horizon_mode === 'all' && <p className="mt-2 text-xs text-slate-500">This model was trained for every horizon 1–15 s — switching H1/H2/H3 needs no retraining.</p>}
          </Card>
          <Card title="Input traffic">
            <Segmented value={src} onChange={setSrc} options={[{ value: 'dataset', label: 'Dataset trace' }, { value: 'upload', label: 'Upload KPI file' }]} />
            {src === 'dataset' ? (
              <div className="mt-4 grid grid-cols-3 gap-3">
                <Field label="Dataset"><Select value={data} onChange={setData} options={datasets?.map((d) => d.path) ?? [data]} /></Field>
                <Field label="Split"><Select value={split} onChange={setSplit} options={splits ? Object.keys(splits) : [split]} /></Field>
                <Field label="Trace"><Select value={tid} onChange={setTid} options={splits?.[split] ?? [tid]} /></Field>
              </div>
            ) : (
              <div className="mt-4">
                <label className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border-2 border-dashed border-slate-300 px-4 py-6 text-sm text-slate-500 hover:border-indigo-400 hover:text-indigo-600">
                  <Upload className="h-4 w-4" /> {upload ? `Loaded ${upload.id} (${upload.rows} rows) — click to replace` : 'CSV or Parquet with the same 1 s columns (e.g. embb_req_prb_mean …)'}
                  <input type="file" accept=".csv,.parquet" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
                </label>
                <div className="mt-2"><ErrorBox error={upErr} /></div>
              </div>
            )}
          </Card>
        </div>
      )}
      {ready && (
        <>
          <Tabs value={tab} onChange={setTab} tabs={[
            { value: 'forecast', label: 'Forecast vs actual' },
            { value: 'replay', label: 'Replay at time t' },
            { value: 'export', label: 'Export A1 packages' },
          ]} />
          {tab === 'forecast' && <Forecast run={run!} source={source!} H={H} />}
          {tab === 'replay' && <Replay run={run!} source={source!} H={H} />}
          {tab === 'export' && <Export run={run!} source={source!} H={H} />}
        </>
      )}
    </>
  )
}

type Src = { data: string; id: string }

function Forecast({ run, source, H }: { run: Run; source: Src; H: number[] }) {
  const [start, setStart] = useState(10)
  const [len, setLen] = useState(10)
  const { data: f, loading, error } = useApi<any>('/infer/forecast', { run: run.id, ...source, h: H, start: start * 60, end: (start + len) * 60 })
  // Use the horizons this response was computed for: while a new request loads, `f` still holds the old one.
  const FH: number[] = f?.horizons ?? H
  const rows = useMemo(() => {
    if (!f) return []
    return f.t.map((t: number, i: number) => {
      const o: any = { x: t / 60 }
      for (const s of Object.keys(f.slices)) {
        o[`${s}_actual`] = f.slices[s].actual[i]
        FH.forEach((h) => { o[`${s}_h${h}`] = f.slices[s][`h${h}`]?.[i]; o[`${s}_c${h}`] = f.slices[s][`c${h}`]?.[i] })
      }
      return o
    })
  }, [f])
  const maxMin = f ? Math.floor(f.length / 60) : 60
  return (
    <div className="space-y-5">
      <Card title="Forecasts drawn at their target time" subtitle="Each predicted value is plotted at t + h, so it lines up with the demand it predicts" actions={<ScenarioLegend />}>
        <div className="mb-4 grid max-w-md grid-cols-2 gap-3">
          <Field label="Start (min)"><NumberInput value={start} min={0} max={maxMin - 1} onChange={(v) => setStart(Math.max(0, Math.min(v, maxMin - 1)))} /></Field>
          <Field label="Length (min)"><NumberInput value={len} min={1} max={maxMin} onChange={(v) => setLen(Math.max(1, Math.min(v, maxMin)))} /></Field>
        </div>
        <ErrorBox error={error} />
        {loading && !f ? <Spinner label="Running the model…" /> : f && Object.keys(f.slices).map((s) => (
          <div key={s} className="mb-3">
            <div className="mb-1 text-sm font-semibold" style={{ color: SLICE_COLOR[s] }}>{SLICE_LABEL[s]}</div>
            <TimeChart data={rows} segments={f.segments} height={220} series={[
              { key: `${s}_actual`, name: 'actual', color: SLICE_COLOR[s], width: 2.2 },
              ...FH.map((h, k) => ({ key: `${s}_h${h}`, name: `predicted ${h} s earlier`, color: HORIZON_COLORS[k], width: 1.3, dash: k === 0 ? undefined : k === 1 ? '6 3' : '2 3' })),
            ]} />
            {f.has_confidence && (
              <TimeChart data={rows} segments={f.segments} height={120} yLabel="confidence" series={
                FH.map((h, k) => ({ key: `${s}_c${h}`, name: `C ${h} s`, color: HORIZON_COLORS[k], width: 1.2, dash: k === 0 ? undefined : k === 1 ? '6 3' : '2 3' }))} />
            )}
          </div>
        ))}
        {f && !f.has_confidence && <p className="text-xs text-slate-500">This run has no confidence calibration yet — see the Confidence card below.</p>}
      </Card>
      <ConfidenceCard run={run} />
      {f && (
        <Card title="Accuracy on the whole trace" subtitle="R² of the model vs persistence (ŷ(t+h) = y(t))">
          <div className="grid gap-4 md:grid-cols-3">
            {Object.keys(f.metrics).map((s) => (
              <div key={s} className="rounded-xl border border-slate-200 p-4">
                <div className="mb-2 font-semibold" style={{ color: SLICE_COLOR[s] }}>{SLICE_LABEL[s]}</div>
                <table className="w-full text-sm">
                  <thead><tr className="text-left text-xs text-slate-400"><th>h</th><th>RMSE</th><th>R²</th><th>persist.</th></tr></thead>
                  <tbody>{FH.map((h) => (
                    <tr key={h}><td className="py-0.5">{h} s</td><td>{fmt(f.metrics[s][String(h)]?.rmse)}</td>
                      <td className="font-semibold">{fmt(f.metrics[s][String(h)]?.r2, 3)}</td>
                      <td className="text-slate-500">{fmt(f.metrics[s].persistence?.[String(h)], 3)}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

function Replay({ run, source, H }: { run: Run; source: Src; H: number[] }) {
  const { data: tr } = useApi<any>('/trace', { ...source, start: 0, end: 1 })
  const T = tr?.length ?? 3600
  const [t, setT] = useState(600)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(4)
  const timer = useRef<number | null>(null)
  useEffect(() => {
    if (!playing) return
    timer.current = window.setInterval(() => setT((v) => (v + 1 >= T ? v : v + 1)), 1000 / speed)
    return () => { if (timer.current) clearInterval(timer.current) }
  }, [playing, speed, T])
  const { data: r, error } = useApi<any>('/infer/replay', { run: run.id, ...source, h: H, t })
  return (
    <div className="space-y-5">
      <Card title="What the rApp sees at time t" subtitle="Blue band = model input (lookback). Diamonds = forecasts at t + H. Dotted = what actually happened.">
        <div className="mb-4 flex flex-wrap items-end gap-4">
          <Button variant={playing ? 'secondary' : 'primary'} onClick={() => setPlaying(!playing)}>
            {playing ? <><Pause className="h-4 w-4" /> Pause</> : <><Play className="h-4 w-4" /> Play</>}
          </Button>
          <Field label="Speed" className="w-28"><Select value={speed} onChange={setSpeed} options={[1, 2, 4, 8]} format={(v) => `${v}×`} /></Field>
          <div className="min-w-[280px] flex-1">
            <div className="mb-1 flex justify-between text-xs text-slate-500"><span>t = <b className="text-slate-800">{t} s</b> ({(t / 60).toFixed(1)} min)</span>
              {r && <Badge color={r.scenario === 'stable' ? 'slate' : 'amber'}>{r.scenario}</Badge>}</div>
            <input type="range" min={run.lookback - 1} max={T - 1} value={t} onChange={(e) => { setPlaying(false); setT(Number(e.target.value)) }} className="w-full" />
          </div>
        </div>
        <ErrorBox error={error} />
        {r && (
          <div className="grid gap-4 lg:grid-cols-3">
            {Object.keys(r.slices).map((s) => <ReplayChart key={s} s={s} d={r.slices[s]} lookback={r.lookback} H={r.horizons ?? H} />)}
          </div>
        )}
      </Card>
      {r && (
        <Card title={`A1 packages produced at t = ${r.t} s`} subtitle="P = (predicted demand D̂, confidence C, target time T) per slice and horizon — the xApp uses a prediction only if C ≥ C_min and it is still fresh">
          <table className="w-full text-sm">
            <thead><tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-2">Slice</th><th>Horizon</th><th>Target time</th><th>Predicted PRBs</th><th>Actual PRBs</th><th>Error</th><th>Confidence</th></tr></thead>
            <tbody>{r.packages.map((p: any, i: number) => (
              <tr key={i} className="border-b border-slate-100 last:border-0">
                <td className="py-1.5 font-semibold" style={{ color: SLICE_COLOR[p.slice] }}>{SLICE_LABEL[p.slice]}</td>
                <td>{p.horizon_s} s</td><td>{p.target_time} s</td><td className="font-semibold">{fmt(p.pred_prb, 1)}</td>
                <td className="text-slate-500">{fmt(p.actual_prb, 1)}</td>
                <td className="text-slate-500">{p.actual_prb == null ? '–' : (p.pred_prb - p.actual_prb).toFixed(1)}</td>
                <td><ConfBar c={p.confidence} /></td>
              </tr>))}</tbody>
          </table>
        </Card>
      )}
    </div>
  )
}

function ReplayChart({ s, d, lookback, H }: { s: string; d: any; lookback: number; H: number[] }) {
  const rows = [
    ...d.past.map((p: any) => ({ x: p.x, past: p.y })),
    ...d.future.map((p: any) => ({ x: p.x, future: p.y })),
  ]
  const last = d.past[d.past.length - 1]
  if (last) rows.push({ x: 0, future: last.y })
  rows.sort((a: any, b: any) => a.x - b.x)
  const fc = d.forecast.map((p: any, k: number) => ({ x: p.x, fc: p.y, k }))
  return (
    <div className="rounded-xl border border-slate-200 p-3">
      <div className="mb-1 text-sm font-semibold" style={{ color: SLICE_COLOR[s] }}>{SLICE_LABEL[s]}</div>
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={rows} margin={{ top: 6, right: 8, left: -10, bottom: 0 }}>
          <CartesianGrid stroke="#f1f5f9" />
          <ReferenceArea x1={-lookback + 1} x2={0} fill="#bfdbfe" fillOpacity={0.35} strokeOpacity={0} />
          <XAxis dataKey="x" type="number" domain={[-60, 16]} ticks={[-60, -30, 0, 5, 10, 15]} />
          <YAxis />
          <ReferenceLine x={0} stroke="#0f172a" label={{ value: 't', position: 'top', fontSize: 11 }} />
          <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v: any) => (typeof v === 'number' ? v.toFixed(1) : v)} labelFormatter={(v) => `t ${Number(v) >= 0 ? '+' : ''}${v} s`} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line dataKey="past" name="observed" stroke={SLICE_COLOR[s]} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
          <Line dataKey="future" name="actual future" stroke="#64748b" strokeDasharray="3 3" dot={false} isAnimationActive={false} connectNulls />
          <Scatter data={fc} dataKey="fc" name="forecast" fill="#f59e0b" shape={(props: any) => {
            const c = HORIZON_COLORS[props.payload.k]
            return <path d={`M${props.cx},${props.cy - 7}L${props.cx + 7},${props.cy}L${props.cx},${props.cy + 7}L${props.cx - 7},${props.cy}Z`} fill={c} stroke="#fff" strokeWidth={1.5} />
          }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="mt-1 flex justify-center gap-3 text-xs text-slate-500">
        {d.forecast.map((p: any, k: number) => (
          <span key={k}><span style={{ color: HORIZON_COLORS[k] }}>◆</span> +{H[k]} s: <b>{fmt(p.y, 1)}</b>
            {p.c != null && <span className="ml-1 font-semibold" style={{ color: confColor(p.c) }}>C {p.c.toFixed(2)}</span>}</span>))}
      </div>
    </div>
  )
}

function Export({ run, source, H }: { run: Run; source: Src; H: number[] }) {
  const { data: tr } = useApi<any>('/trace', { ...source, start: 0, end: 1 })
  const T = tr?.length ?? 3600
  const [from, setFrom] = useState(600)
  const [n, setN] = useState(600)
  const [truth, setTruth] = useState(false)
  const params = { run: run.id, ...source, h: H, start: from, seconds: n, with_truth: truth }
  const [preview, setPreview] = useState<string>('')
  useEffect(() => {
    fetch(api.url('/infer/export', { ...params, seconds: 1 })).then((r) => r.text()).then(setPreview).catch(() => setPreview(''))
  }, [run.id, source.id, H.join(','), from, truth])
  const count = Math.max(0, Math.min(n, T - from)) * run.slices.length * 3
  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
      <Card title="Export settings" subtitle="JSON Lines — one package per slice × horizon × second">
        <div className="grid grid-cols-2 gap-3">
          <Field label="From second"><NumberInput value={from} min={run.lookback - 1} max={T - 1} onChange={setFrom} /></Field>
          <Field label="Number of seconds"><NumberInput value={n} min={1} max={T} onChange={setN} /></Field>
        </div>
        <label className="mt-3 flex items-center gap-2 text-sm text-slate-600">
          <input type="checkbox" checked={truth} onChange={(e) => setTruth(e.target.checked)} /> Include actual_prb (for checking)
        </label>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <Stat label="Packages" value={count.toLocaleString()} />
          <Stat label="Horizons" value={H.join(' / ') + ' s'} />
        </div>
        <a href={api.url('/infer/export', params)} download>
          <Button className="mt-4 w-full"><Download className="h-4 w-4" /> Download .jsonl</Button>
        </a>
      </Card>
      <Card title="Preview — packages at the first second">
        <pre className="overflow-x-auto rounded-lg bg-slate-900 p-4 font-mono text-[11px] leading-relaxed text-emerald-200">{preview || '…'}</pre>
      </Card>
    </div>
  )
}

function ConfBar({ c }: { c: number | null }) {
  if (c == null) return <span className="text-xs text-slate-400">n/a</span>
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-20 overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full" style={{ width: `${Math.round(100 * c)}%`, background: confColor(c) }} />
      </div>
      <span className="text-xs font-semibold" style={{ color: confColor(c) }}>{c.toFixed(2)}</span>
    </div>
  )
}

function ConfidenceCard({ run }: { run: Run }) {
  const { data: det, reload } = useApi<any>('/runs/detail', { id: run.id })
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const { data: jobs } = useApi<any[]>(busy ? '/jobs' : null, { kind: 'calibrate' }, 2000)
  useEffect(() => {
    const j = jobs?.filter((x) => x.name === run.id).pop()
    if (busy && j && !j.running) { setBusy(false); reload() }
  }, [jobs])
  const cal = det?.confidence
  const start = async () => {
    setErr(null)
    try { await api.post('/confidence/calibrate', { run: run.id }); setBusy(true) } catch (e: any) { setErr(e.message) }
  }
  return (
    <Card title="Prediction confidence"
      subtitle="C = 2^(−recent RMSE / τ) from the last 30 matured predictions · τ = median on validation · C = 0.5 ≈ typical accuracy">
      {!cal ? (
        <div className="flex items-center gap-4">
          <p className="text-sm text-slate-500">Not calibrated yet. Calibration fits τ on the validation split and checks it on test (≈1 min, CPU).</p>
          <Button onClick={start} disabled={busy}>{busy ? 'Calibrating…' : 'Calibrate confidence'}</Button>
          <ErrorBox error={err} />
        </div>
      ) : (
        <div className="grid gap-5 lg:grid-cols-3">
          {Object.entries(cal.calibration_test).map(([s, per]: any) => {
            const hs = Object.keys(per)
            const data = per[hs[0]].bins.map((_: any, i: number) => {
              const o: any = { i }
              hs.forEach((h) => { const b = per[h].bins[i]; if (b) { o[`c${h}`] = b.c_mean; o[`m${h}`] = b.mae } })
              return o
            })
            const rows = hs.flatMap((h) => per[h].bins.map((b: any) => ({ c: Number(b.c_mean.toFixed(3)), [`h${h}`]: b.mae })))
              .sort((a: any, b: any) => a.c - b.c)
            return (
              <div key={s} className="rounded-xl border border-slate-200 p-4">
                <div className="mb-1 font-semibold" style={{ color: SLICE_COLOR[s] }}>{SLICE_LABEL[s]}</div>
                <XYChart data={rows} xKey="c" xLabel="confidence (bin mean)" yLabel="mean |error| PRB" height={180}
                  series={hs.map((h, k) => ({ key: `h${h}`, name: `${h} s`, color: HORIZON_COLORS[k % 3] }))} />
                <table className="mt-2 w-full text-xs">
                  <thead><tr className="text-left text-slate-400"><th>h</th><th>MAE C≥0.5</th><th>MAE C&lt;0.5</th><th>ρ(C,|e|)</th></tr></thead>
                  <tbody>{hs.map((h) => (
                    <tr key={h}><td className="py-0.5">{h} s</td>
                      <td className="font-semibold text-emerald-700">{fmt(per[h].mae_high_conf)}</td>
                      <td className="font-semibold text-rose-700">{fmt(per[h].mae_low_conf)}</td>
                      <td>{fmt(per[h].spearman_conf_vs_abs_error)}</td></tr>))}</tbody>
                </table>
                {data && null}
              </div>
            )
          })}
        </div>
      )}
      {cal && <p className="mt-3 text-xs text-slate-500">Test split: when confidence is high the actual error is lower (negative ρ). The xApp can ignore predictions with C below a threshold C_min and fall back to reactive control.</p>}
    </Card>
  )
}
