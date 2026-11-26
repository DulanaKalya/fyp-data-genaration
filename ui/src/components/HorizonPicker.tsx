import { Timer } from 'lucide-react'
import { HORIZON_COLORS } from '../lib/theme'

/** Three horizon selectors with H1 < H2 < H3. `allowed` limits the choices (e.g. a dedicated model's horizons). */
/** System-wide horizon range (seconds). 'all'-mode models output every horizon in it. */
export const H_MIN = 1
export const H_MAX = 15

export function HorizonPicker({ value, onChange, allowed = range(H_MIN, H_MAX) }:
  { value: number[]; onChange: (h: number[]) => void; allowed?: number[] }) {
  const valid = value[0] < value[1] && value[1] < value[2]
  const set = (i: number, v: number) => {
    const n = [...value]
    n[i] = v
    onChange(n)
  }
  return (
    <div>
      <div className="grid grid-cols-3 gap-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="rounded-xl border border-slate-200 bg-slate-50/60 p-3">
            <div className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold" style={{ color: HORIZON_COLORS[i] }}>
              <Timer className="h-3.5 w-3.5" /> H{i + 1}
            </div>
            <div className="flex items-baseline gap-1">
              <select value={value[i]} onChange={(e) => set(i, Number(e.target.value))}
                className="w-full rounded-lg border-0 bg-white px-2 py-1.5 text-lg font-semibold text-slate-900 ring-1 ring-inset ring-slate-300 outline-none focus:ring-2 focus:ring-indigo-500">
                {allowed.map((h) => <option key={h} value={h}>{h}</option>)}
              </select>
              <span className="text-sm text-slate-500">s</span>
            </div>
          </div>
        ))}
      </div>
      {!valid && <p className="mt-2 text-xs font-medium text-rose-600">Horizons must satisfy H1 &lt; H2 &lt; H3.</p>}
    </div>
  )
}

export function range(a: number, b: number) {
  return Array.from({ length: b - a + 1 }, (_, i) => a + i)
}

export const hValid = (h: number[]) => h.length === 3 && h[0] < h[1] && h[1] < h[2]
