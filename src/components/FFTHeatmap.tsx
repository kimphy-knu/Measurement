import { useEffect, useMemo, useRef, useState } from 'react';
import type { FFTResult } from '../types';

/**
 * FFT spectra of many files as one image: a row per file (sorted by step/BLM),
 * frequency across, colour = amplitude. With two dozen files the overlaid
 * line chart turns into a knot; here a frequency branch that moves with angle
 * reads as a continuous ridge. Drawn on a canvas so dozens of rows × hundreds
 * of bins cost nothing compared with SVG lines.
 */

interface Row { x: number | string; fileName: string; fft: FFTResult[]; peaks?: FFTResult[] }

// Perceptually uniform, colour-blind safe (viridis), sampled at 9 stops.
const VIRIDIS = ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725'];
function viridis(t: number): [number, number, number] {
  const v = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(v));
  const f = v - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  const c = (h: string, k: number) => parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16);
  return [0, 1, 2].map(k => Math.round(c(a, k) + f * (c(b, k) - c(a, k)))) as [number, number, number];
}

export function FFTHeatmap({ rows, fMax, normalizeRows, showPeaks, rowLabel, onPick }: {
  rows: Row[];
  fMax: number;
  normalizeRows: boolean;
  showPeaks: boolean;
  rowLabel: string;
  // Click to pick: the clicked row (same object that was passed in) and frequency.
  onPick?: (row: Row, F: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ row: number; f: number; amp: number; px: number; py: number } | null>(null);
  const BINS = 600;
  const sorted = useMemo(
    () => [...rows].sort((a, b) => Number(a.x) - Number(b.x) || String(a.x).localeCompare(String(b.x))),
    [rows],
  );

  // rows × bins amplitude matrix (max inside each bin, so narrow peaks survive)
  const grid = useMemo(() => {
    const g = sorted.map(r => {
      const out = new Float64Array(BINS);
      for (const p of r.fft) {
        if (p.frequency < 0 || p.frequency > fMax) continue;
        const k = Math.min(BINS - 1, Math.floor((p.frequency / fMax) * BINS));
        if (p.amplitude > out[k]) out[k] = p.amplitude;
      }
      return out;
    });
    let gmax = 0;
    for (const r of g) for (const v of r) if (v > gmax) gmax = v;
    return { g, gmax };
  }, [sorted, fMax]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const n = grid.g.length;
    cv.width = BINS;
    cv.height = Math.max(1, n);
    const img = ctx.createImageData(BINS, Math.max(1, n));
    grid.g.forEach((row, r) => {
      let rmax = 0;
      for (const v of row) if (v > rmax) rmax = v;
      const scale = normalizeRows ? rmax : grid.gmax;
      for (let k = 0; k < BINS; k++) {
        // sqrt keeps weak branches visible next to a dominant one
        const [R, G, B] = viridis(scale > 0 ? Math.sqrt(row[k] / scale) : 0);
        const o = (r * BINS + k) * 4;
        img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
      }
    });
    ctx.putImageData(img, 0, 0);
  }, [grid, normalizeRows]);

  const rowH = Math.max(10, Math.min(28, Math.floor(520 / Math.max(1, sorted.length))));
  const height = rowH * sorted.length;
  const ticks = Array.from({ length: 6 }, (_, i) => (fMax * i) / 5);

  return (
    <div className="w-full">
      <div className="flex">
        {/* row labels */}
        <div className="shrink-0 pr-2 text-right text-[11px] tabular-nums text-[#6e6e73]" style={{ width: 72 }}>
          <div className="text-[10px] text-[#86868b] -mt-5 h-5">{rowLabel}</div>
          {sorted.map((r, i) => (
            <div key={r.fileName} style={{ height: rowH, lineHeight: `${rowH}px` }}
              className={`truncate ${hover?.row === i ? 'text-[#1d1d1f] font-semibold' : ''}`} title={r.fileName}>
              {r.x}
            </div>
          ))}
        </div>
        <div className="relative flex-1 min-w-0">
          <canvas
            ref={canvasRef}
            className={`w-full rounded-md [image-rendering:pixelated] ${onPick ? 'cursor-crosshair' : ''}`}
            style={{ height }}
            onClick={() => { if (onPick && hover && sorted[hover.row]) onPick(sorted[hover.row], hover.f); }}
            onMouseLeave={() => setHover(null)}
            onMouseMove={e => {
              const rect = e.currentTarget.getBoundingClientRect();
              const fx = (e.clientX - rect.left) / rect.width;
              const row = Math.min(sorted.length - 1, Math.max(0, Math.floor((e.clientY - rect.top) / rowH)));
              const k = Math.min(BINS - 1, Math.max(0, Math.floor(fx * BINS)));
              setHover({ row, f: fx * fMax, amp: grid.g[row]?.[k] ?? 0, px: e.clientX - rect.left, py: e.clientY - rect.top });
            }}
          />
          {/* top 3 peaks per row only — every detected peak turns the map into confetti */}
          {showPeaks && sorted.map((r, i) => [...(r.peaks ?? [])].filter(p => p.frequency <= fMax).sort((a, b) => b.amplitude - a.amplitude).slice(0, 3).map(p => (
            <span key={`${r.fileName}-${p.frequency}`} className="pointer-events-none absolute h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white bg-[#ff3b30]"
              style={{ left: `${(p.frequency / fMax) * 100}%`, top: i * rowH + rowH / 2 }} />
          )))}
          {hover && (
            <div className="pointer-events-none absolute z-10 rounded-lg bg-white/95 px-2.5 py-1.5 text-[11px] tabular-nums shadow-[0_8px_24px_rgba(0,0,0,.12)]"
              style={{ left: Math.min(hover.px + 12, 9999), top: hover.py + 12 }}>
              <div className="font-semibold">{rowLabel} {sorted[hover.row]?.x}</div>
              <div>F ≈ {hover.f.toFixed(0)} T · A = {hover.amp.toExponential(2)}</div>
              {onPick && <div className="text-[#0071e3]">클릭하면 근처 봉우리를 목록에 추가</div>}
              <div className="text-[#86868b] max-w-[220px] truncate">{sorted[hover.row]?.fileName}</div>
            </div>
          )}
          <div className="relative mt-1 h-4 text-[10px] tabular-nums text-[#6e6e73]">
            {ticks.map(t => (
              <span key={t} className="absolute -translate-x-1/2" style={{ left: `${(t / fMax) * 100}%` }}>{t.toFixed(0)}</span>
            ))}
          </div>
          <div className="text-center text-[11px] text-[#6e6e73]">Frequency (T)</div>
        </div>
      </div>
      <div className="mt-3 flex items-center justify-end gap-2 text-[10px] text-[#86868b]">
        <span>낮음</span>
        <span className="h-2 w-40 rounded-full" style={{ background: `linear-gradient(90deg, ${VIRIDIS.join(',')})` }} />
        <span>높음 {normalizeRows ? '(행마다 정규화, √)' : '(전체 기준, √)'}</span>
      </div>
    </div>
  );
}
