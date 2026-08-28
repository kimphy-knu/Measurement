import { useMemo } from 'react';
import type { FFTResult } from '../types';
import { decimateMinMax } from '../lib/dsp';

// Lightweight FFT sparkline drawn as one inline SVG path. Recharts' per-chart
// ResponsiveContainer sets up a ResizeObserver and an async measure pass; the
// Assemble grid rendered one per file, so N files meant N observers mounting at
// once and a multi-second stall. This draws directly into a fixed viewBox that
// the browser scales for free — no measurement, no per-point React elements.
export function MiniFFT({ data, freqMax, peaks, showPeaks }: {
  data: FFTResult[];                       // full FFT (frequency-ascending)
  freqMax: number | 'auto';
  peaks?: { frequency: number; amplitude: number }[];
  showPeaks?: boolean;
}) {
  const W = 300, H = 100;
  const xMax = freqMax === 'auto'
    ? (data.length ? data[data.length - 1].frequency : 1)
    : freqMax;
  // Clip to the visible [0, xMax] window BEFORE decimating. The spectrum runs to
  // Nyquist but the chart only shows a low-frequency slice, so decimating the
  // whole array left almost no samples in view (the sparkline collapsed to one
  // point). Binary-search the cutoff, then decimate just the visible slice.
  const view = useMemo(() => {
    if (!(xMax > 0) || data.length === 0 || data[0].frequency > xMax) return null;
    let l = 0, r = data.length - 1, cut = 0;
    while (l <= r) { const m = (l + r) >> 1; if (data[m].frequency <= xMax) { cut = m; l = m + 1; } else r = m - 1; }
    const pts = decimateMinMax(data.slice(0, cut + 1), 200, p => p.amplitude);
    let yMax = 0;
    for (const p of pts) if (p.amplitude > yMax) yMax = p.amplitude;
    return { pts, yMax: yMax > 0 ? yMax : 1 };
  }, [data, xMax]);
  if (!view) return null;
  const { pts, yMax } = view;
  const sx = (f: number) => (f / xMax) * W;
  const sy = (a: number) => H - (a / yMax) * H;
  let line = '', lastX = 0;
  for (const p of pts) {
    lastX = sx(p.frequency);
    line += (line ? 'L' : 'M') + lastX.toFixed(1) + ',' + sy(p.amplitude).toFixed(1);
  }
  const area = line ? `${line}L${lastX.toFixed(1)},${H}L0,${H}Z` : '';
  const vpeaks = showPeaks ? (peaks || []).filter(p => p.frequency <= xMax) : [];
  return (
    <div className="relative w-full h-full">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="100%" preserveAspectRatio="none">
        {area && <path d={area} fill="#F27D26" fillOpacity={0.3} stroke="none" />}
        {line && <path d={line} fill="none" stroke="#F27D26" strokeWidth={1} vectorEffect="non-scaling-stroke" />}
      </svg>
      {vpeaks.map((p, i) => (
        <span key={i}>
          <span className="absolute rounded-full bg-white border border-[#F27D26] pointer-events-none"
            style={{ left: `${(p.frequency / xMax) * 100}%`, top: `${(1 - p.amplitude / yMax) * 100}%`, width: 4, height: 4, transform: 'translate(-50%,-50%)' }} />
          <span className="absolute font-mono pointer-events-none whitespace-nowrap"
            style={{ left: `${(p.frequency / xMax) * 100}%`, top: `${(1 - p.amplitude / yMax) * 100}%`, transform: 'translate(-50%,-140%)', fontSize: '8px', color: '#fff' }}>
            {p.frequency.toFixed(1)}
          </span>
        </span>
      ))}
    </div>
  );
}
