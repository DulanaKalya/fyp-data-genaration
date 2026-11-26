import { Link } from 'react-router-dom'
import { ArrowRight, BarChart3, BrainCircuit, Database, LayoutDashboard, Sparkles } from 'lucide-react'
import { useApi, type Run } from '../lib/api'
import { Card, PageHeader, Stat, StatusBadge, Spinner } from '../components/ui'
import { SLICE_COLOR, SLICE_LABEL } from '../lib/theme'

const STEPS = [
  { to: '/dataset', icon: Database, title: '1 · Dataset', text: 'Generate simulator traces matched to the testbed and inspect requested vs allocated PRBs.' },
  { to: '/train', icon: BrainCircuit, title: '2 · Train', text: 'Choose horizons H1 < H2 < H3 (1–15 s) and train the attention Bi-LSTM or a baseline.' },
  { to: '/evaluate', icon: BarChart3, title: '3 · Evaluate', text: 'Compare models with persistence, moving average and ARIMA per slice and horizon.' },
  { to: '/inference', icon: Sparkles, title: '4 · Inference', text: 'Replay traffic, watch forecasts at t+H and export A1 prediction packages for the xApp.' },
]

export default function Overview() {
  const { data: ov } = useApi<any>('/overview', undefined, 5000)
  const { data: runs } = useApi<Run[]>('/runs', undefined, 8000)
  return (
    <>
      <PageHeader icon={<LayoutDashboard className="h-5 w-5" />} title="Predictive network slice demand"
        subtitle="Attention Bi-LSTM (PreDP) forecasting of per-slice PRB demand at 1 s resolution" />
      <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
        <Stat label="Datasets" value={ov?.datasets ?? '–'} />
        <Stat label="Trained runs" value={ov?.runs ?? '–'} />
        <Stat label="Training now" value={ov?.running ?? '–'} accent="#4f46e5" />
        <Stat label="Eval reports" value={ov?.reports ?? '–'} />
      </div>
      <div className="mb-6 grid gap-4 md:grid-cols-4">
        {STEPS.map(({ to, icon: Icon, title, text }) => (
          <Link key={to} to={to} className="group rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:border-indigo-300 hover:shadow-md">
            <Icon className="h-5 w-5 text-indigo-600" />
            <div className="mt-3 font-semibold text-slate-900">{title}</div>
            <p className="mt-1 text-sm text-slate-500">{text}</p>
            <div className="mt-3 flex items-center gap-1 text-sm font-medium text-indigo-600 opacity-0 transition group-hover:opacity-100">Open <ArrowRight className="h-4 w-4" /></div>
          </Link>
        ))}
      </div>
      <Card title="Runs" subtitle="Test R² at the run's selected horizons (higher is better)">
        {!runs ? <Spinner /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-2 pr-4">Run</th><th className="pr-4">Status</th><th className="pr-4">Model</th>
                  <th className="pr-4">Mode</th><th className="pr-4">Horizons</th><th className="pr-4">Train h</th><th>R² per slice</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id} className="border-b border-slate-100 last:border-0">
                    <td className="py-2.5 pr-4 font-mono text-xs text-slate-700">{r.id}</td>
                    <td className="pr-4"><StatusBadge status={r.status} /></td>
                    <td className="pr-4">{r.model} <span className="text-slate-400">· {r.features}</span></td>
                    <td className="pr-4">{r.horizon_mode}</td>
                    <td className="pr-4">{r.horizons_selected.join(' / ')} s</td>
                    <td className="pr-4">{r.train_hours}</td>
                    <td className="space-x-3 text-xs">
                      {r.headline ? Object.entries(r.headline).map(([s, v]) => (
                        <span key={s}><b style={{ color: SLICE_COLOR[s] }}>{SLICE_LABEL[s]}</b> {Object.values(v).map((x) => x?.toFixed(2)).join(' / ')}</span>
                      )) : <span className="text-slate-400">–</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  )
}
