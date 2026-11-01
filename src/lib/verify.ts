// Automatic "is this FFT peak a real quantum oscillation?" check — the IFFT
// tab's by-eye verification turned into numbers. Same method and thresholds
// as the batch scan used on the 2026 Aug NHMFL data (analysis/qo_scan.py),
// calibrated there on synthetic signals: a true 1/B oscillation gives a local
// frequency drift ≈ 0.005 and the Dingle slope it was built with; B-periodic
// (time-like) pickup drifts ≈ 0.12 and its envelope shrinks with B.
//
//   1. resample ΔR onto a uniform 1/B grid, FFT
//   2. keep only the positive-frequency bins within ±max(1.5·resolution, 8 %·F)
//      of F, double them and inverse-FFT → analytic signal of that component
//      (band-limited Hilbert transform, no extra pass needed)
//   3. phase → local frequency in three 1/B sub-windows → drift = spread / F
//   4. |analytic| → ln-envelope slope vs 1/B (negative = grows with B)

export type Verdict = 'real' | 'doubtful' | 'artifact' | 'lowF';

export interface PeakCheck {
  F: number;
  cycles: number;      // oscillation periods across the 1/B window
  snr: number;         // peak amplitude / median spectrum
  drift: number;       // NaN when there are too few cycles to measure
  envSlope: number;
  varFrac: number;     // share of ΔR variance this component explains
  verdict: Verdict;
  reasons: string[];
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  real: '진짜 진동',
  doubtful: '불확실',
  artifact: '인공 신호 의심',
  lowF: '저주파 · 검증 불가',
};

function fft(re: Float64Array, im: Float64Array, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

function linSlope(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 3) return NaN;
  const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  return sxx > 0 ? sxy / sxx : NaN;
}

export interface UniformSeries { x: Float64Array; y: Float64Array; dx: number }

// ΔR(1/B) → uniform grid (power of two, 1024–16384 points), linear interpolation.
export function toUniform(series: { invB: number; val: number }[]): UniformSeries | null {
  const s = series.filter(p => Number.isFinite(p.invB) && Number.isFinite(p.val)).sort((a, b) => a.invB - b.invB);
  if (s.length < 32) return null;
  const N = Math.min(16384, Math.max(1024, 1 << Math.ceil(Math.log2(s.length))));
  const lo = s[0].invB, hi = s[s.length - 1].invB;
  if (!(hi > lo)) return null;
  const dx = (hi - lo) / (N - 1);
  const x = new Float64Array(N), y = new Float64Array(N);
  let j = 0;
  for (let k = 0; k < N; k++) {
    const xv = lo + k * dx;
    while (j < s.length - 2 && s[j + 1].invB < xv) j++;
    const a = s[j], b = s[j + 1];
    const t = b.invB === a.invB ? 0 : (xv - a.invB) / (b.invB - a.invB);
    x[k] = xv; y[k] = a.val + Math.min(1, Math.max(0, t)) * (b.val - a.val);
  }
  return { x, y, dx };
}

// The band-limited component itself (for plotting next to ΔR): real part of
// the analytic signal and its envelope, on the uniform 1/B grid.
export function reconstruct(u: UniformSeries, F: number): { x: number; val: number; comp: number; env: number }[] {
  const N = u.y.length;
  const res = 1 / (u.x[N - 1] - u.x[0]);
  const re = Float64Array.from(u.y), im = new Float64Array(N);
  const mean = re.reduce((a, b) => a + b, 0) / N;
  for (let i = 0; i < N; i++) re[i] -= mean;
  fft(re, im);
  const df = 1 / (N * u.dx);
  const hw = Math.max(1.5 * res, 0.08 * F);
  const zr = new Float64Array(N), zi = new Float64Array(N);
  for (let k = 1; k < N / 2; k++) {
    const f = k * df;
    if (f >= F - hw && f <= F + hw) { zr[k] = 2 * re[k]; zi[k] = 2 * im[k]; }
  }
  fft(zr, zi, true);
  const out: { x: number; val: number; comp: number; env: number }[] = [];
  const stride = Math.max(1, Math.floor(N / 1500));
  for (let i = 0; i < N; i += stride) out.push({ x: u.x[i], val: u.y[i] - mean, comp: zr[i], env: Math.hypot(zr[i], zi[i]) });
  return out;
}

export function checkPeak(u: UniformSeries, F: number): PeakCheck {
  const N = u.y.length;
  const span = u.x[N - 1] - u.x[0];
  const res = 1 / span;
  const cycles = F * span;
  const re = Float64Array.from(u.y), im = new Float64Array(N);
  const mean = re.reduce((a, b) => a + b, 0) / N;
  for (let i = 0; i < N; i++) re[i] -= mean;
  const total = re.reduce((a, b) => a + b * b, 0);
  fft(re, im);
  const df = 1 / (N * u.dx);
  const amp = (k: number) => Math.hypot(re[k], im[k]);
  // SNR against the median of the spectrum above 4 cycles (below is leftover background)
  const floorBins: number[] = [];
  for (let k = Math.ceil((4 * res) / df); k < N / 2; k++) floorBins.push(amp(k));
  floorBins.sort((a, b) => a - b);
  const floor = floorBins[floorBins.length >> 1] || NaN;
  const kF = Math.round(F / df);
  let peak = 0;
  for (let k = Math.max(1, kF - 2); k <= Math.min(N / 2 - 1, kF + 2); k++) peak = Math.max(peak, amp(k));
  const snr = peak / floor;

  // analytic signal of the band around F
  const hw = Math.max(1.5 * res, 0.08 * F);
  const zr = new Float64Array(N), zi = new Float64Array(N);
  for (let k = 1; k < N / 2; k++) {
    const f = k * df;
    if (f >= F - hw && f <= F + hw) { zr[k] = 2 * re[k]; zi[k] = 2 * im[k]; }
  }
  fft(zr, zi, true);
  let compPow = 0;
  for (let i = 0; i < N; i++) compPow += zr[i] * zr[i];

  const edge = Math.floor(N / 10);
  const xs: number[] = [], ph: number[] = [], lenv: number[] = [];
  let prev = 0, offset = 0;
  for (let i = edge; i < N - edge; i++) {
    let p = Math.atan2(zi[i], zr[i]);
    if (i > edge) {
      const d = p - prev;
      if (d > Math.PI) offset -= 2 * Math.PI; else if (d < -Math.PI) offset += 2 * Math.PI;
    }
    prev = p; p += offset;
    xs.push(u.x[i]); ph.push(p); lenv.push(Math.log(Math.hypot(zr[i], zi[i]) + 1e-300));
  }
  let drift = NaN;
  if (cycles >= 6) {
    const third = Math.floor(xs.length / 3);
    const fl = [0, 1, 2].map(t => Math.abs(linSlope(xs.slice(t * third, (t + 1) * third), ph.slice(t * third, (t + 1) * third))) / (2 * Math.PI));
    drift = (Math.max(...fl) - Math.min(...fl)) / F;
  }
  const envSlope = linSlope(xs, lenv);

  // The local-frequency measurement itself gets noisier with fewer cycles per
  // third of the window, so both drift limits relax as cycles drop. Calibrated
  // on synthetic signals: a clean oscillation drifts ~1 % at 24 cycles but
  // 6–7 % at 8 cycles (the 15.9–31.5 T default window at F ≈ 265 T), while
  // B-periodic pickup drifts ~12 % even at 27 cycles.
  const realLim = Math.max(0.03, 0.7 / cycles);
  // Under ~12 cycles the resolution (1/Δ(1/B)) is coarse enough that a nearby
  // second frequency falls inside the band and their beating alone kicks the
  // drift up (seen on NbIrTe4 at 15.9–31.5 T, where the same peaks pass with a
  // wider window). There, only a gross drift is called an artifact.
  const artLim = cycles < 12 ? 0.3 : Math.max(0.08, 1.2 / cycles);
  const reasons: string[] = [];
  let verdict: Verdict;
  if (cycles < 6) {
    verdict = 'lowF';
    reasons.push(`창 안 진동 ${cycles.toFixed(1)}회 — 6회 미만이라 1/B 주기성을 잴 수 없음`);
  } else if (drift > artLim) {
    verdict = 'artifact';
    reasons.push(`1/B 에서 주파수가 ${(drift * 100).toFixed(0)}% 흐름 (> ${(artLim * 100).toFixed(0)}%, B·시간 주기 잡음의 특징)`);
    if (envSlope > 0) reasons.push('진폭도 B 가 클수록 줄어듦');
  } else {
    // a rising envelope alone is not decisive: over a short high-field window
    // Dingle damping is weak and the Hilbert edges dominate the slope
    const pass = [snr >= 10, drift < realLim, envSlope < 0];
    verdict = pass.every(Boolean) ? 'real' : 'doubtful';
    reasons.push(`drift ${(drift * 100).toFixed(1)}%`, `진폭 기울기 ${envSlope.toFixed(1)}`, `SNR ${snr.toFixed(1)}`);
    if (snr < 10) reasons.push('SNR 이 낮음 (< 10)');
    if (drift >= realLim) reasons.push(`주파수 흔들림이 다소 큼 (≥ ${(realLim * 100).toFixed(1)}%)`);
    if (envSlope >= 0) reasons.push('진폭이 B 와 함께 커지지 않음');
  }
  if (cycles >= 6 && cycles < 12) reasons.push('B 하한을 낮춰 창을 넓히면 판정이 더 정확해짐');
  return { F, cycles, snr, drift, envSlope, varFrac: total > 0 ? compPow / total : NaN, verdict, reasons };
}
