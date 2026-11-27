import { useEffect, useMemo, useState } from 'react'
import { BrainCircuit, ChevronDown, Rocket } from 'lucide-react'
import { api, useApi, type Dataset, type Direction, type Run } from '../lib/api'
import { Button, Card, Empty, ErrorBox, Field, LogBox, NumberInput, PageHeader, Segmented, Select, Spinner, StatusBadge, TextInput } from '../components/ui'
import { HorizonPicker, hValid, H_MIN, H_MAX } from '../components/HorizonPicker'
import { XYChart } from '../components/charts'
import { SLICE_COLOR, SLICE_LABEL, fmt, methodColor } from '../lib/theme'

const MODELS = [
  { value: 'predp', label: 'PreDP — attention Bi-LSTM (PreNS)' },
  { value: 'lstm', label: 'UT-LSTM (2 layers)' },
  { value: 'gru', label: 'UT-GRU (2 layers)' },
  { value: 'transformer', label: 'UT-Transformer (2 layers)' },
]

function stamp() {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`
}

export default function TrainPage() {
  const { data: datasets } = useApi<Dataset[]>('/datasets')
  const { data: runs, reload: reloadRuns } = useApi<Run[]>('/runs', undefined, 5000)
  const [H, setH] = useState([5, 10, 15])
  const [mode, setMode] = useState<'all' | 'dedicated'>('all')
  const [data, setData] = useState('data/raw')
  const [model, setModel] = useState('predp')
  const [features, setFeatures] = useState<'uni' | 'multi' | 'pair'>('uni')
  const [direction, setDirection] = useState<Direction>('dl')
  const [traces, setTraces] = useState(84)
  const [adv, setAdv] = useState(false)
  const [tgt, setTgt] = useState({ target: 'point', loss: 'mse', quantile: 0.9, target_mode: 'level' })
  const [hp, setHp] = useState({ lookback: 30, hidden: 64, lr: 0.001, batch_size: 32, max_epochs: 200, patience: 20, median_filter: true, seed: 0 })
  const [name, setName] = useState('')
  const [nameTouched, setNameTouched] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)

  const ds = datasets?.find((d) => d.path === data)
  const hasUl = !!ds?.directions?.includes('ul')
  useEffect(() => { if (!hasUl) { setDirection('dl'); if (features === 'pair') setFeatures('uni') } }, [hasUl])
  useEffect(() => { if (ds) setTraces((t) => Math.min(t, ds.splits.train) || ds.splits.train) }, [ds?.path])
  const autoName = `ui/${stamp()}_${model}_${direction === 'ul' ? 'ul_' : ''}${features}_${mode}_${H.join('-')}${tgt.target !== 'point' ? '_' + tgt.target : ''}${tgt.loss === 'quantile' ? '_q' + Math.round(tgt.quantile * 100) : ''}`
  useEffect(() => { if (!nameTouched) setName(autoName) }, [autoName, nameTouched])

  useEffect(() => {
    if (!selected && runs?.length) setSelected((runs.find((r) => r.status === 'running') ?? runs[0]).id)
  }, [runs])

  const start = async () => {
    setErr(null)
    try {
      await api.post('/train', { run_name: name, data, horizons: H, horizon_mode: mode, model, features, train_traces: traces, direction, ...hp, ...tgt })
      setSelected(name)
      setNameTouched(false)
      setTimeout(reloadRuns, 1500)
    } catch (e: any) { setErr(e.message) }
  }

  return (
    <>
      <PageHeader icon={<BrainCircuit className="h-5 w-5" />} title="Train" subtitle="Pick the prediction horizons and train a model — progress updates live" />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="space-y-5">
          <Card title="1 · Prediction horizons" subtitle={`Each between ${H_MIN} and ${H_MAX} s`}>
            <HorizonPicker value={H} onChange={setH} />
            <div className="mt-4">
              <Segmented value={mode} onChange={setMode} options={[{ value: 'all', label: 'All horizons (1–15 s)' }, { value: 'dedicated', label: 'Dedicated (3 outputs)' }]} />
              <p className="mt-2 text-xs text-slate-500">
                {mode === 'all'
                  ? 'One model predicts every horizon from 1 to 15 s — you can switch H1/H2/H3 at inference without retraining.'
                  : 'The model outputs only the three chosen horizons, as in the PreNS paper.'}
              </p>
            </div>
          </Card>
          <Card title="2 · What to predict" subtitle="PreNS predicts the demand at exactly t + h">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Target" className="col-span-2">
                <Segmented value={tgt.target} onChange={(v) => setTgt({ ...tgt, target: v })} options={[
                  { value: 'point', label: 'Demand at t + h' }, { value: 'mean', label: 'Average over next h s' }, { value: 'max', label: 'Peak over next h s' }]} />
              </Field>
              <Field label="Loss" hint={tgt.loss === 'quantile' ? `predicts the ${Math.round(tgt.quantile * 100)}th percentile — errs on the safe side` : 'mean squared error (PreNS)'}>
                <Select value={tgt.loss} onChange={(v) => setTgt({ ...tgt, loss: v })} options={['mse', 'quantile']} format={(v) => v === 'mse' ? 'MSE' : 'Quantile (pinball)'} />
              </Field>
              {tgt.loss === 'quantile'
                ? <Field label="Quantile"><Select value={tgt.quantile} onChange={(v) => setTgt({ ...tgt, quantile: v })} options={[0.5, 0.75, 0.8, 0.9, 0.95]} /></Field>
                : <Field label="Output" hint={tgt.target_mode === 'residual' ? 'change from the last value (level-invariant)' : 'absolute PRBs'}>
                    <Select value={tgt.target_mode} onChange={(v) => setTgt({ ...tgt, target_mode: v })} options={['level', 'residual']} />
                  </Field>}
            </div>
            <p className="mt-2 text-xs text-slate-500">
              {tgt.target === 'point' ? 'Same task as PreNS.' : tgt.target === 'mean' ? 'The PRBs to reserve on average for the next h seconds — smoother and easier to predict.' : 'The PRBs needed to cover the busiest second in the next h seconds — what an SLA needs.'}
            </p>
          </Card>
          <Card title="3 · Data & model">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Dataset"><Select value={data} onChange={setData} options={datasets?.map((d) => d.path) ?? [data]} /></Field>
              <Field label="Direction" className="col-span-2"
                hint={hasUl ? (direction === 'ul' ? 'uplink requested PRBs (separate UL model per slice)' : 'downlink requested PRBs')
                  : 'this dataset is DL only — choose data/raw_v2 for uplink'}>
                <Segmented value={direction} onChange={(v) => hasUl && setDirection(v)}
                  options={[{ value: 'dl', label: 'Downlink (DL)' }, { value: 'ul', label: hasUl ? 'Uplink (UL)' : 'Uplink (UL) — n/a' }]} />
              </Field>
              <Field label="Features" hint={features === 'uni' ? "each slice's own demand (PreNS)" : features === 'pair'
                ? 'the slice’s DL + UL demand (E14: does the other direction help?)' : 'KPIs of all slices, one joint model'}>
                <Select value={features} onChange={setFeatures}
                  options={(hasUl ? ['uni', 'multi', 'pair'] : ['uni', 'multi']) as ('uni' | 'multi' | 'pair')[]} />
              </Field>
              <Field label="Model" className="col-span-2">
                <Select value={model} onChange={setModel} options={MODELS.map((m) => m.value)} format={(v) => MODELS.find((m) => m.value === v)!.label} />
              </Field>
              <Field label={`Training traces — ${traces * (ds?.trace_hours ?? 1)} h`} className="col-span-2">
                <input type="range" min={1} max={ds?.splits.train ?? 84} value={traces} onChange={(e) => setTraces(Number(e.target.value))} className="w-full" />
              </Field>
            </div>
            <button className="mt-3 flex items-center gap-1 text-sm font-medium text-slate-600 hover:text-slate-900" onClick={() => setAdv(!adv)}>
              <ChevronDown className={`h-4 w-4 transition ${adv ? 'rotate-180' : ''}`} /> Hyperparameters <span className="text-xs font-normal text-slate-400">(defaults = PreNS Table II)</span>
            </button>
            {adv && (
              <div className="mt-3 grid grid-cols-3 gap-3">
                <Field label="Lookback (s)"><NumberInput value={hp.lookback} min={5} max={240} onChange={(v) => setHp({ ...hp, lookback: v })} /></Field>
                <Field label="Hidden units"><Select value={hp.hidden} onChange={(v) => setHp({ ...hp, hidden: v })} options={[32, 64, 128]} /></Field>
                <Field label="Learning rate"><Select value={hp.lr} onChange={(v) => setHp({ ...hp, lr: v })} options={[0.0005, 0.001, 0.002]} /></Field>
                <Field label="Batch size"><Select value={hp.batch_size} onChange={(v) => setHp({ ...hp, batch_size: v })} options={[16, 32, 64, 128, 256]} /></Field>
                <Field label="Max epochs"><NumberInput value={hp.max_epochs} min={1} onChange={(v) => setHp({ ...hp, max_epochs: v })} /></Field>
                <Field label="Early-stop patience"><NumberInput value={hp.patience} min={1} onChange={(v) => setHp({ ...hp, patience: v })} /></Field>
                <Field label="Seed"><NumberInput value={hp.seed} onChange={(v) => setHp({ ...hp, seed: v })} /></Field>
                <label className="col-span-2 flex items-center gap-2 self-end pb-2 text-sm text-slate-600">
                  <input type="checkbox" checked={hp.median_filter} onChange={(e) => setHp({ ...hp, median_filter: e.target.checked })} />
                  Causal median filter on inputs
                </label>
              </div>
            )}
          </Card>
          <Card title="4 · Start">
            <Field label="Run name"><TextInput value={name} onChange={(v) => { setName(v); setNameTouched(true) }} /></Field>
            <div className="mt-3"><ErrorBox error={err} /></div>
            <Button className="mt-3 w-full" onClick={start} disabled={!hValid(H) || !name}><Rocket className="h-4 w-4" /> Start training</Button>
          </Card>
        </div>
        <Monitor runs={runs} selected={selected} onSelect={setSelected} />
      </div>
    </>
  )
}

function Monitor({ runs, selected, onSelect }: { runs: Run[] | null; selected: string | null; onSelect: (id: string) => void }) {
  const sel = runs?.find((r) => r.id === selected)
  const { data: det } = useApi<any>(selected ? '/runs/detail' : null, { id: selected ?? '' }, sel?.status === 'running' ? 3000 : undefined)
  const lossRows = useMemo(() => {
    if (!det?.train_log) return { rows: [], groups: [] as string[] }
    const groups = Object.keys(det.train_log)
    const maxEp = Math.max(...groups.map((g) => det.train_log[g].length))
    const rows = Array.from({ length: maxEp }, (_, i) => {
      const o: any = { epoch: i + 1 }
      for (const g of groups) {
        const r = det.train_log[g][i]
        if (r) { o[`${g}_train`] = r.train_loss; o[`${g}_val`] = r.val_loss }
      }
      return o
    })
    return { rows, groups }
  }, [det])

  if (!runs) return <Spinner />
  return (
    <div className="space-y-5">
      <Card title="Training monitor" actions={sel && <StatusBadge status={sel.status} />}>
        {!runs.length ? <Empty>No runs yet.</Empty> : (
          <>
            <Field label="Run">
              <Select value={selected ?? runs[0].id} onChange={onSelect} options={runs.map((r) => r.id)}
                format={(id) => { const r = runs.find((x) => x.id === id)!; return `${id}${r.status === 'running' ? '  ⏳' : r.status === 'failed' ? '  ✕' : ''}` }} />
            </Field>
            {det && (
              <div className="mt-4">
                <div className="mb-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-500">
                  <span>model <b className="text-slate-700">{det.config.model}</b></span>
                  <span>features <b className="text-slate-700">{det.config.features}</b></span>
                  <span>mode <b className="text-slate-700">{det.config.horizon_mode}</b></span>
                  <span>horizons <b className="text-slate-700">{det.config.horizons_selected.join('/')} s</b></span>
                  <span>train <b className="text-slate-700">{det.config.train_hours} h</b></span>
                  <span>lookback <b className="text-slate-700">{det.config.lookback} s</b></span>
                </div>
                {det.status === 'running' && det.progress?.epoch && (
                  <div className="mb-3 rounded-lg bg-indigo-50 px-3 py-2 text-sm text-indigo-800">
                    Training <b>{det.progress.group}</b> — epoch {det.progress.epoch} (early stopping usually ends at 25–60)
                  </div>
                )}
                {lossRows.rows.length > 0 ? (
                  <XYChart data={lossRows.rows} xKey="epoch" xLabel="epoch" yLabel="MSE (normalized)" height={260}
                    series={lossRows.groups.flatMap((g, i) => [
                      { key: `${g}_val`, name: `${SLICE_LABEL[g] ?? g} val`, color: SLICE_COLOR[g] ?? methodColor(g, i), width: 2 },
                      { key: `${g}_train`, name: `${SLICE_LABEL[g] ?? g} train`, color: SLICE_COLOR[g] ?? methodColor(g, i), width: 1, dash: '4 3' },
                    ])} />
                ) : <Empty>Waiting for the first epoch…</Empty>}
              </div>
            )}
          </>
        )}
      </Card>
      {det?.metrics && <ResultTable det={det} />}
      {det?.log_tail && <Card title="Log"><LogBox text={det.log_tail} /></Card>}
    </div>
  )
}

function ResultTable({ det }: { det: any }) {
  const H: number[] = det.config.horizons_selected
  const cost = Object.values(det.metrics.cost)[0] as any
  return (
    <Card title="Test results" subtitle="Requested PRBs · model vs persistence (ŷ(t+h) = y(t))">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="py-2">Slice</th><th>h</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Persist. R²</th><th>Skill</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(det.metrics.test).flatMap(([s, mh]: any) => H.map((h, i) => {
            const r = mh[String(h)]
            const skill = 1 - r.model_prb.rmse / r.persistence_prb.rmse
            return (
              <tr key={s + h} className="border-b border-slate-100 last:border-0">
                <td className="py-1.5 font-semibold" style={{ color: SLICE_COLOR[s] }}>{i === 0 ? SLICE_LABEL[s] : ''}</td>
                <td>{h} s</td><td>{fmt(r.model_prb.rmse)}</td><td>{fmt(r.model_prb.mae)}</td>
                <td className="font-semibold">{fmt(r.model_prb.r2, 3)}</td><td className="text-slate-500">{fmt(r.persistence_prb.r2, 3)}</td>
                <td className={skill >= 0 ? 'text-emerald-600' : 'text-rose-600'}>{skill >= 0 ? '+' : ''}{(100 * skill).toFixed(1)}%</td>
              </tr>
            )
          }))}
        </tbody>
      </table>
      {cost && <p className="mt-3 text-xs text-slate-500">{cost.params.toLocaleString()} params per slice model · {cost.epochs} epochs · inference {cost.infer_ms_batch1} ms (batch 1)</p>}
    </Card>
  )
}
