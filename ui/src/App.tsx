import { NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary'
import { BarChart3, BrainCircuit, Database, LayoutDashboard, Radio, Sparkles } from 'lucide-react'
import { useApi } from './lib/api'
import Overview from './pages/Overview'
import DatasetPage from './pages/Dataset'
import TrainPage from './pages/Train'
import EvaluatePage from './pages/Evaluate'
import InferencePage from './pages/Inference'

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/dataset', label: 'Dataset', icon: Database },
  { to: '/train', label: 'Train', icon: BrainCircuit },
  { to: '/evaluate', label: 'Evaluate', icon: BarChart3 },
  { to: '/inference', label: 'Inference', icon: Sparkles },
]

export default function App() {
  const location = useLocation()
  const { data: health } = useApi<any>('/health')
  const { data: ov } = useApi<any>('/overview', undefined, 5000)
  return (
    <div className="flex h-full">
      <aside className="flex w-60 shrink-0 flex-col bg-slate-900 text-slate-300">
        <div className="flex items-center gap-2.5 px-5 py-5">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-indigo-500 text-white"><Radio className="h-5 w-5" /></div>
          <div>
            <div className="text-sm font-semibold text-white">Slice Demand</div>
            <div className="text-[11px] text-slate-400">Predictive 5G · PreDP</div>
          </div>
        </div>
        <nav className="flex-1 space-y-1 px-3">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end}
              className={({ isActive }) => `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition ${isActive ? 'bg-slate-800 text-white' : 'hover:bg-slate-800/60 hover:text-white'}`}>
              <Icon className="h-4 w-4" /> {label}
              {label === 'Train' && ov?.running > 0 && (
                <span className="ml-auto rounded-full bg-indigo-500 px-1.5 text-[10px] font-semibold text-white">{ov.running}</span>
              )}
            </NavLink>
          ))}
        </nav>
        <div className="space-y-1 border-t border-slate-800 px-5 py-4 text-[11px] text-slate-400">
          <div>n78 · 60 MHz · 30 kHz · 162 PRBs</div>
          <div>3 eMBB · 3 URLLC · 3 mMTC UEs</div>
          <div className="flex items-center gap-1.5 pt-1">
            <span className={`h-2 w-2 rounded-full ${health ? 'bg-emerald-400' : 'bg-rose-400'}`} />
            {health ? (health.gpu ? health.gpu.replace('NVIDIA GeForce ', '') : 'CPU only') : 'API offline'}
          </div>
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-7xl px-8 py-8">
          <ErrorBoundary resetKey={location.pathname}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/dataset" element={<DatasetPage />} />
            <Route path="/train" element={<TrainPage />} />
            <Route path="/evaluate" element={<EvaluatePage />} />
            <Route path="/inference" element={<InferencePage />} />
          </Routes>
          </ErrorBoundary>
        </div>
      </main>
    </div>
  )
}
