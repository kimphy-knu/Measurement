import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { FFTResult } from '../types';

/**
 * FFT amplitude as a filled contour map on true axes: θ across (to scale, so
 * uneven rotator steps don't distort a branch), F up — the same axes as an
 * F(θ) plot, so the picked peaks sit on the map as points of the angle
 * dependence. Files measured at the same angle are averaged into one column;
 * between measured angles the map is interpolated linearly, gaps wider than
 * 3× the typical step are left blank. Below the per-file verification limit
 * (too few cycles in the 1/B window) the map is faded and hatched, and left
 * out of the normalisation so the leftover background there doesn't swamp it.
 */

export interface ContourRow { angle: number; fileName: string; fft: FFTResult[]; fLo: number }
export interface ContourPeak { id: string; angle: number; F: number; status: 'unchecked' | 'confirmed' | 'rejected'; manual?: boolean; title?: string; fill?: string }

const BINS = 600;
const LEVELS = 14;
// ColorBrewer Blues: light = weak, dark = strong; reads cleanly on white and
// leaves the warm colours free for the peak markers
const BLUES = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'];
function blues(t: number): [number, number, number] {
  const v = Math.min(1, Math.max(0, t)) * (BLUES.length - 1);
  const i = Math.min(BLUES.length - 2, Math.floor(v));
  const f = v - i;
  const c = (h: string, k: number) => parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16);
  return [0, 1, 2].map(k => Math.round(c(BLUES[i], k) + f * (c(BLUES[i + 1], k) - c(BLUES[i], k)))) as [number, number, number];
}
const LEVEL_RGB = Array.from({ length: LEVELS }, (_, l) => blues(l / (LEVELS - 1)));

const MARKER: Record<ContourPeak['status'], { fill: string; label: string }> = {
  unchecked: { fill: '#ff9500', label: '미확인' },
  confirmed: { fill: '#34c759', label: '확정' },
  rejected: { fill: '#ffffff', label: '제외' },
};

function niceTicks(lo: number, hi: number, n = 6) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

const M = { l: 56, r: 16, t: 12, b: 44 };

export function AngleContour({ rows, fMax, normalize, peaks, height = 460, legend, onPick }: {
  rows: ContourRow[];
  fMax: number;
  normalize: boolean;
  peaks: ContourPeak[];
  height?: number;
  // replaces the marker legend (for peaks coloured by `fill`)
  legend?: ReactNode;
  // click: every file measured at the nearest angle, and the frequency
  onPick?: (fileNames: string[], F: number) => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(800);
  const [hover, setHover] = useState<{ x: number; y: number; theta: number; F: number } | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(e => setWidth(Math.max(320, Math.round(e[0].contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // one column per measured angle: mean of the files' binned spectra
  const groups = useMemo(() => {
    const byAngle = new Map<string, ContourRow[]>();
    for (const r of rows) {
      const k = r.angle.toFixed(2);
      if (!byAngle.has(k)) byAngle.set(k, []);
      byAngle.get(k)!.push(r);
    }
    const gs = [...byAngle.values()].map(rs => {
      const col = new Float64Array(BINS);
      for (const r of rs) {
        const b = new Float64Array(BINS);
        const hit = new Uint8Array(BINS);
        for (const p of r.fft) {
          if (p.frequency < 0 || p.frequency > fMax) continue;
          const k = Math.min(BINS - 1, Math.floor((p.frequency / fMax) * BINS));
          if (p.amplitude > b[k]) b[k] = p.amplitude;
          hit[k] = 1;
        }
        // spectra coarser than a bin would leave empty stripes
        for (let k = 1; k < BINS; k++) if (!hit[k]) b[k] = b[k - 1];
        for (let k = 0; k < BINS; k++) col[k] += b[k] / rs.length;
      }
      const kLo = Math.min(BINS, Math.ceil((Math.max(...rs.map(r => r.fLo)) / fMax) * BINS));
      let max = 0;
      for (let k = kLo; k < BINS; k++) if (col[k] > max) max = col[k];
      return { angle: rs[0].angle, files: rs.map(r => r.fileName), col, kLo, max };
    }).sort((a, b) => a.angle - b.angle);
    const gmax = Math.max(0, ...gs.map(g => g.max));
    const steps = gs.slice(1).map((g, i) => g.angle - gs[i].angle).sort((a, b) => a - b);
    const step = steps.length ? steps[steps.length >> 1] || 1 : 5;
    return { gs, gmax, step };
  }, [rows, fMax]);

  const { gs, gmax, step } = groups;
  const thLo = gs.length ? gs[0].angle - step / 2 : 0;
  const thHi = gs.length ? gs[gs.length - 1].angle + step / 2 : 1;
  const pw = width - M.l - M.r, ph = height - M.t - M.b;
  const xOf = (th: number) => ((th - thLo) / (thHi - thLo)) * pw;
  const yOf = (F: number) => (1 - F / fMax) * ph;

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !gs.length) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = Math.max(1, Math.round(pw * dpr)), H = Math.max(1, Math.round(ph * dpr));
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    // normalised √amplitude per group (scale set above the verification limit;
    // below it the leftover background may exceed 1, so clamp)
    const vals = gs.map(g => {
      const s = normalize ? g.max : gmax;
      return Float64Array.from(g.col, v => (s > 0 ? Math.min(1, Math.sqrt(v / s)) : NaN));
    });
    const lev = new Int16Array(W * H).fill(-1);
    const dim = new Uint8Array(W * H);
    const colv = new Float64Array(BINS);
    let gi = 0;
    for (let x = 0; x < W; x++) {
      const th = thLo + ((x + 0.5) / W) * (thHi - thLo);
      while (gi < gs.length - 2 && gs[gi + 1].angle < th) gi++;
      // bracket [a, b] and weight; single nearest column outside / across wide gaps
      let a = gi, b = gi, t = 0;
      if (gs.length > 1 && th > gs[gi].angle && gi + 1 < gs.length) {
        const gap = gs[gi + 1].angle - gs[gi].angle;
        if (th >= gs[gi + 1].angle) { a = b = gi + 1; }
        else if (gap <= 3 * step) { b = gi + 1; t = (th - gs[gi].angle) / gap; }
        else {
          const near = th - gs[gi].angle < gs[gi + 1].angle - th ? gi : gi + 1;
          if (Math.abs(th - gs[near].angle) > step / 2) continue;
          a = b = near;
        }
      } else if (Math.abs(th - gs[gi].angle) > step / 2 + 1e-9) continue;
      const va = vals[a], vb = vals[b];
      for (let k = 0; k < BINS; k++) colv[k] = va[k] + t * (vb[k] - va[k]);
      const kLo = Math.max(gs[a].kLo, gs[b].kLo);
      for (let y = 0; y < H; y++) {
        const kf = (1 - (y + 0.5) / H) * BINS - 0.5;
        const k0 = Math.max(0, Math.min(BINS - 2, Math.floor(kf)));
        const f = Math.min(1, Math.max(0, kf - k0));
        const v = colv[k0] + f * (colv[k0 + 1] - colv[k0]);
        if (Number.isNaN(v)) continue;
        lev[y * W + x] = Math.min(LEVELS - 1, Math.floor(v * LEVELS));
        if (kf < kLo) dim[y * W + x] = 1;
      }
    }
    const img = ctx.createImageData(W, H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, o = i * 4, l = lev[i];
      if (l < 0) { img.data[o] = 236; img.data[o + 1] = 236; img.data[o + 2] = 240; img.data[o + 3] = 255; continue; }
      let [R, G, B] = LEVEL_RGB[l];
      // contour line where the level steps to a neighbour
      const edge = (x + 1 < W && lev[i + 1] >= 0 && lev[i + 1] !== l) || (y + 1 < H && lev[i + W] >= 0 && lev[i + W] !== l);
      if (edge) { R *= 0.82; G *= 0.82; B *= 0.86; }
      // too few cycles to verify: faded and hatched, but still visible
      if (dim[i]) {
        const g = ((x + y) / dpr) % 8 < 1.2 ? 200 : 236;
        R = 0.4 * R + 0.6 * g; G = 0.4 * G + 0.6 * g; B = 0.4 * B + 0.6 * (g + 4);
      }
      img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [gs, gmax, step, normalize, pw, ph, thLo, thHi]);

  const nearestGroup = (th: number) => {
    let best = gs[0];
    for (const g of gs) if (Math.abs(g.angle - th) < Math.abs(best.angle - th)) best = g;
    return best;
  };
  const hg = hover ? nearestGroup(hover.theta) : null;
  const hk = hover ? Math.min(BINS - 1, Math.max(0, Math.floor((hover.F / fMax) * BINS))) : 0;

  if (!gs.length) return null;
  return (
    <div ref={wrapRef} className="w-full select-none">
      <div className="relative" style={{ height }}>
        <canvas ref={canvasRef} className={`absolute rounded-md ${onPick ? 'cursor-crosshair' : ''}`}
          style={{ left: M.l, top: M.t, width: pw, height: ph }}
          onMouseLeave={() => setHover(null)}
          onMouseMove={e => {
            const r = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - r.left, y = e.clientY - r.top;
            setHover({ x, y, theta: thLo + (x / r.width) * (thHi - thLo), F: (1 - y / r.height) * fMax });
          }}
          onClick={() => { if (onPick && hover && hg && Math.abs(hg.angle - hover.theta) <= Math.max(step, 1)) onPick(hg.files, hover.F); }} />
        <svg className="pointer-events-none absolute inset-0" width={width} height={height}>
          <g transform={`translate(${M.l},${M.t})`}>
            <rect width={pw} height={ph} fill="none" stroke="#d2d2d7" rx={6} />
            {niceTicks(0, fMax).map(F => (
              <g key={`f${F}`} transform={`translate(0,${yOf(F)})`}>
                <line x1={-4} x2={0} stroke="#86868b" />
                <text x={-8} dy="0.32em" textAnchor="end" fontSize={10} fill="#6e6e73">{F}</text>
              </g>
            ))}
            {niceTicks(thLo, thHi, 8).map(th => (
              <g key={`t${th}`} transform={`translate(${xOf(th)},${ph})`}>
                <line y1={0} y2={4} stroke="#86868b" />
                <text y={16} textAnchor="middle" fontSize={10} fill="#6e6e73">{th}°</text>
              </g>
            ))}
            {/* where data actually exist */}
            {gs.map(g => <line key={`m${g.angle}`} x1={xOf(g.angle)} x2={xOf(g.angle)} y1={ph - 5} y2={ph} stroke="#1d1d1f" strokeOpacity={0.55} />)}
            <text x={pw / 2} y={ph + 36} textAnchor="middle" fontSize={11} fill="#6e6e73">θ (°)</text>
            <text transform={`translate(${-42},${ph / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill="#6e6e73">F (T)</text>
            {hover && <>
              <line x1={hover.x} x2={hover.x} y1={0} y2={ph} stroke="#1d1d1f" strokeOpacity={0.25} strokeDasharray="3 3" />
              <line x1={0} x2={pw} y1={hover.y} y2={hover.y} stroke="#1d1d1f" strokeOpacity={0.25} strokeDasharray="3 3" />
            </>}
            {peaks.filter(p => p.F <= fMax && p.angle >= thLo && p.angle <= thHi).map(p => (
              <circle key={p.id} cx={xOf(p.angle)} cy={yOf(p.F)} r={4.5} fill={p.fill ?? MARKER[p.status].fill}
                stroke={p.manual ? '#1d1d1f' : p.status === 'rejected' ? '#86868b' : '#ffffff'} strokeWidth={1.5}>
                <title>{p.title}</title>
              </circle>
            ))}
          </g>
        </svg>
        {hover && hg && (
          <div className="pointer-events-none absolute z-10 rounded-lg bg-white/95 px-2.5 py-1.5 text-[11px] tabular-nums shadow-[0_8px_24px_rgba(0,0,0,.12)]"
            style={{ left: Math.min(M.l + hover.x + 14, width - 200), top: M.t + hover.y + 14 }}>
            <div className="font-semibold">θ {hover.theta.toFixed(1)}° · F {hover.F.toFixed(0)} T</div>
            <div className="text-[#6e6e73]">가장 가까운 측정 {hg.angle}° ({hg.files.length}개 파일) · A = {hg.col[hk].toExponential(2)}</div>
            {onPick && <div className="text-[#0071e3]">클릭하면 이 각도의 근처 봉우리 추가</div>}
          </div>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-3 text-[11px] text-[#6e6e73]" style={{ paddingLeft: M.l }}>
        <div className="flex flex-wrap items-center gap-3">
          {legend ?? (Object.keys(MARKER) as ContourPeak['status'][]).map(s => (
            <span key={s} className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full border" style={{ background: MARKER[s].fill, borderColor: s === 'rejected' ? '#86868b' : '#fff' }} />{MARKER[s].label}
            </span>
          ))}
          {!legend && <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full border-[1.5px] border-[#1d1d1f] bg-[#ff9500]" />직접 선택</span>}
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-3 rounded-sm bg-[repeating-linear-gradient(45deg,#c8c8cc_0_1px,#ececf0_1px_5px)]" />빗금 = 진동 6회 미만 (B 하한을 낮추면 줄어듦)</span>
          <span className="flex items-center gap-1.5"><span className="h-2 w-px bg-[#1d1d1f]" />측정 각도</span>
        </div>
        <div className="flex items-center gap-2">
          <span>약</span>
          <span className="h-2 w-36 rounded-full" style={{ background: `linear-gradient(90deg, ${BLUES.join(',')})` }} />
          <span>강 {normalize ? '(각도마다 정규화, √A)' : '(전체 기준, √A)'}</span>
        </div>
      </div>
    </div>
  );
}
