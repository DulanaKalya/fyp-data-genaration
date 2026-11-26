export const SLICE_COLOR: Record<string, string> = {
  embb: '#2563eb',
  urllc: '#dc2626',
  mmtc: '#16a34a',
}

export const SLICE_LABEL: Record<string, string> = {
  embb: 'eMBB',
  urllc: 'URLLC',
  mmtc: 'mMTC',
}

export const SCENARIO_COLOR: Record<string, string> = {
  stable: 'transparent',
  ramp_up: '#86efac',
  ramp_down: '#fdba74',
  burst: '#fca5a5',
  ue_churn: '#c4b5fd',
  contention: '#ef4444',
  sla_conflict: '#7c3aed',
}

export const SCENARIOS = Object.keys(SCENARIO_COLOR)

export const HORIZON_COLORS = ['#f59e0b', '#8b5cf6', '#0891b2']

export const METHOD_COLORS: Record<string, string> = {
  persistence: '#0f172a',
  ma10: '#94a3b8',
  arima: '#a16207',
}

const PALETTE = ['#4f46e5', '#db2777', '#0d9488', '#ea580c', '#2563eb', '#65a30d', '#9333ea']
export function methodColor(name: string, i: number) {
  return METHOD_COLORS[name] ?? PALETTE[i % PALETTE.length]
}

export const fmt = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? '–' : v.toFixed(d)

/** Colour for a confidence value in [0, 1]: red (low) → amber → green (high). */
export function confColor(c: number | null | undefined) {
  if (c === null || c === undefined) return '#94a3b8'
  if (c >= 0.6) return '#16a34a'
  if (c >= 0.4) return '#d97706'
  return '#dc2626'
}
