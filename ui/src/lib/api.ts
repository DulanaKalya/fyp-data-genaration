import { useCallback, useEffect, useRef, useState } from 'react'

export type Params = Record<string, string | number | boolean | (string | number)[] | undefined | null>

export function qs(params?: Params): string {
  if (!params) return ''
  const u = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue
    if (Array.isArray(v)) v.forEach((x) => u.append(k, String(x)))
    else u.append(k, String(v))
  }
  const s = u.toString()
  return s ? `?${s}` : ''
}

async function handle(res: Response) {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      if (j?.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch { /* ignore */ }
    throw new Error(msg)
  }
  return res.json()
}

export const api = {
  get: <T = any>(path: string, params?: Params): Promise<T> => fetch(`/api${path}${qs(params)}`).then(handle),
  post: <T = any>(path: string, body: unknown): Promise<T> =>
    fetch(`/api${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(handle),
  upload: <T = any>(path: string, file: File): Promise<T> => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch(`/api${path}`, { method: 'POST', body: fd }).then(handle)
  },
  url: (path: string, params?: Params) => `/api${path}${qs(params)}`,
}

/** Fetch `path` whenever it or `params` change; optionally poll every `pollMs`. `path=null` disables. */
export function useApi<T = any>(path: string | null, params?: Params, pollMs?: number) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const key = path ? path + qs(params) : null
  const keyRef = useRef(key)
  keyRef.current = key

  const load = useCallback(async (silent = false) => {
    if (!key) return
    if (!silent) setLoading(true)
    try {
      const d = await fetch(`/api${key}`).then(handle)
      if (keyRef.current === key) { setData(d); setError(null) }
    } catch (e: any) {
      if (keyRef.current === key) setError(e.message)
    } finally {
      if (!silent) setLoading(false)
    }
  }, [key])

  useEffect(() => { if (key) load(); else setData(null) }, [key, load])
  useEffect(() => {
    if (!key || !pollMs) return
    const id = setInterval(() => load(true), pollMs)
    return () => clearInterval(id)
  }, [key, pollMs, load])

  return { data, error, loading, reload: () => load() }
}

// ---------------- types (loose; mirror api/server.py) ----------------
export type Slice = 'embb' | 'urllc' | 'mmtc'
export const SLICES: Slice[] = ['embb', 'urllc', 'mmtc']

export interface Dataset {
  path: string; sim_version: string; trace_hours: number; total_prb: number
  splits: Record<string, number>; hours: Record<string, number>
}

export interface Run {
  id: string; status: 'done' | 'running' | 'failed'; model: string; features: string
  horizon_mode: 'all' | 'dedicated'; horizons_selected: number[]; model_horizons: number[]
  lookback: number; train_hours: number; hidden: number; lr: number; slices: Slice[]
  created: string; data: string; headline?: Record<string, Record<string, number>>; has_confidence?: boolean
  default?: boolean; target?: 'point' | 'mean' | 'max'; loss?: 'mse' | 'quantile'; quantile?: number | null; target_mode?: 'level' | 'residual'
}

export interface Segment { start: number; end: number; scenario: string }

export interface Job {
  id: string; kind: string; name: string; pid: number; running: boolean; log_tail: string; started: number
  meta?: Record<string, any>
}
