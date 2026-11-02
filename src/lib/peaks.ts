import type { AnalysisResult } from '../types';

// FFT resolution 1/Δ(1/B) of the processed window
export function resolutionOf(r: AnalysisResult) {
  let lo = Infinity, hi = -Infinity;
  for (const d of r.plotData) if (Number.isFinite(d.invB)) { lo = Math.min(lo, d.invB); hi = Math.max(hi, d.invB); }
  return hi > lo ? 1 / (hi - lo) : NaN;
}

// lowest frequency that still has MIN_CYCLES oscillations in the 1/B window
export const MIN_CYCLES = 6;
export const fLoOf = (r: AnalysisResult) => MIN_CYCLES * resolutionOf(r);

// Spectral maxima of one file, judged the way the contour map shows them:
// height relative to the strongest peak of that file in the usable range
// [max(fMin, 6·resolution), fMax] ≥ relMin, and a prominence of at least 10 %
// of it (so a shoulder on a slope isn't a peak). A median-based SNR doesn't
// work here — when an oscillation fills much of the window the median is
// signal, not noise. Below 6·resolution there are too few cycles to verify.
// Maxima closer than one resolution are one peak (zero-padding ripples).
export function thresholdPeaks(r: AnalysisResult, opt: { relMin: number; fMin: number; fMax: number; maxPerFile: number }) {
  const fr = r.fftResults;
  const res = resolutionOf(r);
  if (!fr.length || !Number.isFinite(res)) return [];
  const lo = Math.max(opt.fMin, MIN_CYCLES * res);
  let i0 = -1, i1 = -1, top = 0;
  for (let i = 0; i < fr.length; i++) {
    if (fr[i].frequency < lo || fr[i].frequency > opt.fMax) continue;
    if (i0 < 0) i0 = i;
    i1 = i;
    top = Math.max(top, fr[i].amplitude);
  }
  if (i1 - i0 < 2 || !(top > 0)) return [];
  const prominence = (i: number) => {
    const a = fr[i].amplitude;
    let lmin = a, rmin = a;
    for (let l = i - 1; l >= i0 && fr[l].amplitude <= a; l--) lmin = Math.min(lmin, fr[l].amplitude);
    for (let k = i + 1; k <= i1 && fr[k].amplitude <= a; k++) rmin = Math.min(rmin, fr[k].amplitude);
    return a - Math.max(lmin, rmin);
  };
  const cand: { frequency: number; amplitude: number; rel: number }[] = [];
  for (let i = i0 + 1; i < i1; i++) {
    const p = fr[i];
    if (p.amplitude >= fr[i - 1].amplitude && p.amplitude > fr[i + 1].amplitude
      && p.amplitude >= opt.relMin * top && prominence(i) >= 0.1 * top)
      cand.push({ ...p, rel: p.amplitude / top });
  }
  cand.sort((a, b) => b.amplitude - a.amplitude);
  const out: typeof cand = [];
  for (const c of cand) {
    if (out.some(o => Math.abs(o.frequency - c.frequency) < res)) continue;
    out.push(c);
    if (out.length >= opt.maxPerFile) break;
  }
  return out;
}
