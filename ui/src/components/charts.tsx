import { CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, Area, AreaChart } from 'recharts'
import type { Segment } from '../lib/api'
import { SCENARIO_COLOR, SCENARIOS } from '../lib/theme'

export interface SeriesDef {
  key: string
  name: string
  color: string
  width?: number
  dash?: string
  opacity?: number
}

function scenarioAreas(segments: Segment[] | undefined, scale: number) {
  return (segments ?? []).filter((s) => s.scenario !== 'stable').map((s, i) => (
    <ReferenceArea key={`sc${i}`} x1={s.start * scale} x2={s.end * scale} fill={SCENARIO_COLOR[s.scenario]}
      fillOpacity={0.22} strokeOpacity={0} ifOverflow="hidden" />
  ))
}

const tooltipStyle = { fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }

export function TimeChart({ data, series, segments, height = 220, xKey = 'x', xLabel = 'time (min)', yLabel = 'PRBs',
  capacity, scale = 1 / 60, xFormatter }:
  { data: any[]; series: SeriesDef[]; segments?: Segment[]; height?: number; xKey?: string; xLabel?: string; yLabel?: string;
    capacity?: number; scale?: number; xFormatter?: (v: number) => string }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 12, left: 0, bottom: 4 }}>
        <CartesianGrid stroke="#f1f5f9" />
        {scenarioAreas(segments, scale)}
        <XAxis dataKey={xKey} type="number" domain={['dataMin', 'dataMax']} tickFormatter={xFormatter ?? ((v) => Number(v).toFixed(1))}
          label={{ value: xLabel, position: 'insideBottomRight', offset: -2, fontSize: 11, fill: '#94a3b8' }} height={32} />
        <YAxis width={44} label={{ value: yLabel, angle: -90, position: 'insideLeft', fontSize: 11, fill: '#94a3b8' }} />
        <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => (xFormatter ? xFormatter(Number(v)) : `${Number(v).toFixed(2)} min`)}
          formatter={(v: any) => (typeof v === 'number' ? v.toFixed(2) : v)} />
        <Legend wrapperStyle={{ fontSize: 12 }} iconType="plainline" />
        {capacity && <ReferenceLine y={capacity} stroke="#0f172a" strokeDasharray="4 4" label={{ value: `${capacity} PRBs`, fontSize: 11, position: 'right' }} />}
        {series.map((s) => (
          <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={s.width ?? 1.6}
            strokeDasharray={s.dash} strokeOpacity={s.opacity ?? 1} dot={false} isAnimationActive={false} connectNulls={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

export function StackedChart({ data, series, total, capacity, segments, height = 260 }:
  { data: any[]; series: SeriesDef[]; total: SeriesDef; capacity: number; segments?: Segment[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 6, right: 12, left: 0, bottom: 4 }}>
        <CartesianGrid stroke="#f1f5f9" />
        {scenarioAreas(segments, 1 / 60)}
        <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v) => Number(v).toFixed(1)}
          label={{ value: 'time (min)', position: 'insideBottomRight', offset: -2, fontSize: 11, fill: '#94a3b8' }} height={32} />
        <YAxis width={44} />
        <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => `${Number(v).toFixed(2)} min`}
          formatter={(v: any) => (typeof v === 'number' ? v.toFixed(1) : v)} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((s) => (
          <Area key={s.key} dataKey={s.key} name={s.name} stackId="a" stroke={s.color} fill={s.color} fillOpacity={0.55}
            strokeWidth={0.5} isAnimationActive={false} />
        ))}
        <Area dataKey={total.key} name={total.name} stroke={total.color} fill="none" strokeWidth={1.6} isAnimationActive={false} />
        <ReferenceLine y={capacity} stroke="#0f172a" strokeDasharray="4 4" label={{ value: `capacity ${capacity}`, fontSize: 11, position: 'insideTopRight' }} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

const smartTick = (v: number) => {
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a < 0.01) return v.toExponential(1)
  if (a < 1) return v.toFixed(2)
  return a < 100 ? v.toFixed(1) : v.toFixed(0)
}

export function XYChart({ data, series, xKey, height = 240, xLabel, yLabel, yDomain, shade }:
  { data: any[]; series: SeriesDef[]; xKey: string; height?: number; xLabel?: string; yLabel?: string;
    yDomain?: [number | string, number | string]; shade?: [number, number] }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 12, left: 0, bottom: 4 }}>
        <CartesianGrid stroke="#f1f5f9" />
        {shade && <ReferenceArea x1={shade[0]} x2={shade[1]} fill="#e0e7ff" fillOpacity={0.45} strokeOpacity={0} />}
        <XAxis dataKey={xKey} type="number" domain={['dataMin', 'dataMax']} allowDecimals={false}
          label={{ value: xLabel, position: 'insideBottomRight', offset: -2, fontSize: 11, fill: '#94a3b8' }} height={32} />
        <YAxis width={54} domain={yDomain ?? ['auto', 'auto']} tickFormatter={(v) => smartTick(Number(v))}
          label={{ value: yLabel, angle: -90, position: 'insideLeft', fontSize: 11, fill: '#94a3b8' }} />
        <ReferenceLine y={0} stroke="#cbd5e1" />
        <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => (typeof v === 'number' ? (Math.abs(v) < 0.01 ? v.toExponential(3) : v.toFixed(3)) : v)} />
        <Legend wrapperStyle={{ fontSize: 12 }} iconType="plainline" />
        {series.map((s) => (
          <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash}
            dot={{ r: 2.5 }} isAnimationActive={false} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

export function ScenarioLegend() {
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs text-slate-500">
      <span className="mr-1">Background = scenario:</span>
      {SCENARIOS.map((s) => (
        <span key={s} className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-1.5 py-0.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: s === 'stable' ? '#fff' : SCENARIO_COLOR[s], border: '1px solid #cbd5e1' }} />
          {s}
        </span>
      ))}
    </div>
  )
}
