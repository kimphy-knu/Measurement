import type { LandauExtremum, LandauPoint, FanResult, LandauFilterResult } from '../types';
import { buildOlsModel, olsSolve, linearInterpUniform, getWindowCached, getFftInstance } from './dsp';
import { dropQuadratureStrict } from './quadrature';

export function movingAverage(a: number[], win: number): number[] {
  if (win <= 1) return a;
  const n = a.length, half = Math.floor(win / 2), out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(n - 1, i + half); j++) { s += a[j]; c++; }
    out[i] = s / c;
  }
  return out;
}

// Field above which quantum oscillations become visible. Detrend R(B) with a
// wide moving average, track the local oscillation envelope, and return the B
// where it first rises well above the low-field baseline — so the auto ROI
// covers the oscillatory (branch-bearing) region rather than the flat low field.
export function oscillationOnsetB(bRaw: number[], rRaw: number[]): number | null {
  const n = bRaw.length;
  if (n < 40) return null;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => bRaw[a] - bRaw[b]);
  const b: number[] = new Array(n), r: number[] = new Array(n);
  for (let i = 0; i < n; i++) { b[i] = bRaw[order[i]]; r[i] = rRaw[order[i]]; }
  if (!isFinite(b[0]) || !isFinite(b[n - 1]) || b[n - 1] <= b[0]) return null;

  // Cubic polynomial background (fit on normalized field for conditioning).
  // Unlike a moving average this has no edge bias, so the low-field residual
  // isn't spuriously inflated — critical since the onset is at low field.
  const span = b[n - 1] - b[0];
  const X = new Float64Array(n * 4), y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const u = (b[i] - b[0]) / span;
    X[i * 4] = 1; X[i * 4 + 1] = u; X[i * 4 + 2] = u * u; X[i * 4 + 3] = u * u * u;
    y[i] = r[i];
  }
  const model = buildOlsModel(X, n, 4);
  const resid = new Array<number>(n);
  if (model) {
    const beta = olsSolve(model, y);
    for (let i = 0; i < n; i++) {
      const u = (b[i] - b[0]) / span;
      resid[i] = r[i] - (beta[0] + beta[1] * u + beta[2] * u * u + beta[3] * u * u * u);
    }
  } else {
    const bg = movingAverage(r, Math.max(11, (Math.floor(n * 0.12) | 1)));
    for (let i = 0; i < n; i++) resid[i] = r[i] - bg[i];
  }

  const envWin = Math.max(5, Math.floor(n * 0.03));
  const env = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let j = Math.max(0, i - envWin); j <= Math.min(n - 1, i + envWin); j++) {
      const a = Math.abs(resid[j]); if (a > m) m = a;
    }
    env[i] = m;
  }
  let maxEnv = 0; for (const e of env) if (e > maxEnv) maxEnv = e;
  if (!(maxEnv > 0)) return null;

  // baseline = median envelope over the lowest-field quarter (weakest region)
  const lowCount = Math.max(5, Math.floor(n * 0.25));
  const lowSorted = env.slice(0, lowCount).sort((a, b) => a - b);
  const base = lowSorted[lowSorted.length >> 1];
  const thr = base + 0.2 * (maxEnv - base);
  for (let i = 0; i < n; i++) if (env[i] >= thr) return b[i];
  return b[0];
}

// Rough "will the FFT find anything here?" score, used to pre-select which files
// go into the batch. Cubic-detrends R(B) over the high-field part of the sweep,
// resamples the residual uniformly in 1/B, and compares the tallest spectral
// line against the noise floor immediately beside it. Calibrated on real sweeps
// (T1.7K_*deg_Bsweep.dat): a lock-in channel with visible oscillation scores
// ~19, the same channel on a non-oscillating angle ~4.
export const OSC_SCORE_MIN = 8;

// The field window: everything above 0.6·B_max. A wide 0.5–16 T sweep drops its
// oscillation-free low-field half (which would otherwise dominate the 1/B axis
// and alias the signal away), while a narrow 6–9 T high-field sweep keeps every
// point — the clipping to the data minimum makes the rule adapt to both.
export const OSC_WINDOW_FRAC = 0.6;
// Bins 0–3 hold whatever the cubic background left behind plus any slow drift;
// counting them makes a drifting thermometer outscore a real orbit.
export const OSC_MIN_BIN = 4;

export function oscillationQuality(bRaw: number[], rRaw: number[]): number {
  const nAll = bRaw.length;
  if (nAll < 64) return 0;
  const order = Array.from({ length: nAll }, (_, i) => i).sort((a, b) => bRaw[a] - bRaw[b]);
  const b: number[] = [], r: number[] = [];
  for (let i = 0; i < nAll; i++) {
    const bv = bRaw[order[i]], rv = rRaw[order[i]];
    if (isFinite(bv) && isFinite(rv) && bv > 0) { b.push(bv); r.push(rv); }
  }
  const m = b.length;
  if (m < 64) return 0;

  let s0 = 0;
  const loB = Math.max(b[0], OSC_WINDOW_FRAC * b[m - 1]);
  while (s0 < m && b[s0] < loB) s0++;
  if (m - s0 < 64) s0 = 0;
  const n = m - s0;
  const span = b[m - 1] - b[s0];
  if (!(span > 0)) return 0;

  const X = new Float64Array(n * 4), y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const u = (b[s0 + i] - b[s0]) / span;
    X[i * 4] = 1; X[i * 4 + 1] = u; X[i * 4 + 2] = u * u; X[i * 4 + 3] = u * u * u;
    y[i] = r[s0 + i];
  }
  const model = buildOlsModel(X, n, 4);
  if (!model) return 0;
  const beta = olsSolve(model, y);

  const pts: { invB: number; deltaR: number }[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const u = (b[s0 + i] - b[s0]) / span;
    pts[i] = {
      invB: 1 / b[s0 + i],
      deltaR: r[s0 + i] - (beta[0] + beta[1] * u + beta[2] * u * u + beta[3] * u * u * u),
    };
  }
  // A residual that is nothing but rounding error (a constant channel, or a
  // background the cubic reproduces exactly) carries no oscillation no matter
  // how peaked its spectrum looks — reject it before the FFT sees it.
  let sResid = 0, sSignal = 0;
  for (let i = 0; i < n; i++) { sResid += pts[i].deltaR * pts[i].deltaR; sSignal += y[i] * y[i]; }
  if (Math.sqrt(sResid / n) < 1e-9 * Math.sqrt(sSignal / n)) return 0;

  pts.sort((p, q) => p.invB - q.invB);
  const minInv = pts[0].invB, maxInv = pts[n - 1].invB;
  if (!(maxInv > minInv)) return 0;

  const N = 256;
  const grid = linearInterpUniform(pts, minInv, (maxInv - minInv) / (N - 1), N);
  const win = getWindowCached('hanning', N);
  const fft = getFftInstance(N);
  const input = fft.createComplexArray();
  const out = fft.createComplexArray();
  for (let i = 0; i < N; i++) { input[2 * i] = grid[i] * win[i]; input[2 * i + 1] = 0; }
  fft.transform(out, input);

  // Tallest line measured against the floor immediately beside it, skipping the
  // Hann mainlobe around the peak. A *local* floor is what separates a real orbit
  // from a smooth leftover background: the leftover's spectrum decays smoothly,
  // so the bins next to its maximum are nearly as tall, while a genuine
  // oscillation line sits on bare noise.
  const half = N >> 1;
  const amp = (i: number) => Math.hypot(out[2 * i], out[2 * i + 1]);
  const amps: number[] = [];
  for (let i = OSC_MIN_BIN; i < half; i++) amps.push(amp(i));
  if (amps.length < 8) return 0;
  let max = 0, imax = OSC_MIN_BIN;
  for (let i = OSC_MIN_BIN; i < half; i++) { const a = amp(i); if (a > max) { max = a; imax = i; } }
  const near: number[] = [];
  for (let i = Math.max(OSC_MIN_BIN, imax - 20); i < Math.min(half, imax + 21); i++) {
    if (Math.abs(i - imax) <= 3) continue;
    near.push(amp(i));
  }
  const pool = near.length >= 8 ? near : amps;
  pool.sort((x, z) => x - z);
  const med = pool[pool.length >> 1];
  return med > 0 ? max / med : 0;
}

// Angle tagged in the file name. An explicit "<number>deg" wins over the first
// number on the line, because real names put other quantities first:
// "T1.7K_-175deg_Bsweep.dat" is -175°, not 1.7°. Names like "26down.csv" that
// carry nothing but the angle still resolve through the fallback.
export function angleFromFileName(name: string): number | null {
  const dot = name.lastIndexOf('.');
  const base = dot === -1 ? name : name.substring(0, dot);
  const tagged = base.match(/([-+]?[0-9]*\.?[0-9]+)\s*(?:deg|°)/i);
  if (tagged) return parseFloat(tagged[1]);
  const first = base.match(/[-+]?[0-9]*\.?[0-9]+/);
  const v = first ? parseFloat(first[0]) : null;
  // Untagged numbers are only believed inside a plausible angle range, so a
  // date-stamped name ("20260824_1.006.txt") falls through to the step index
  // instead of labelling the sweep 20260824°.
  return v != null && isFinite(v) && Math.abs(v) <= 360 ? v : null;
}

// The field column, by name. Same rule everywhere so the loader, the analysis
// and the exports can never disagree about which column is B.
export function guessFieldColumn(keys: string[]): string {
  return keys.find(k => {
    const low = k.toLowerCase();
    return (low === 'b' || low.startsWith('b ') || low.startsWith('b_') || low.includes('field') || low.includes('magnetic')) &&
      !low.includes('blm') && !low.includes('angle') && !low.includes('position');
  }) || keys.find(k => {
    const low = k.toLowerCase();
    return low.includes('b') && !low.includes('blm') && !low.includes('angle') && !low.includes('position');
  }) || keys[0];
}

// A sweep that ramps up and back down holds two branches in one file. Every
// analysis path here sorts by B, which interleaves them into a zigzag and
// destroys the oscillation, so such a file is split into one entry per
// direction. Magnet jitter must not trigger this: the decision is made on how
// far the field actually travelled each way, not on counting reversals.
export function splitSweepDirections(rows: any[], bCol: string): { dir: 'up' | 'down'; rows: any[] }[] {
  const n = rows.length;
  if (n < 200) return [];
  const b = new Float64Array(n);
  for (let i = 0; i < n; i++) b[i] = parseFloat(String(rows[i][bCol]));

  let upSpan = 0, downSpan = 0;
  for (let i = 1; i < n; i++) {
    const d = b[i] - b[i - 1];
    if (!isFinite(d)) continue;
    if (d > 0) upSpan += d; else downSpan -= d;
  }
  const travelled = Math.max(upSpan, downSpan);
  if (!(travelled > 0) || Math.min(upSpan, downSpan) / travelled < 0.3) return [];

  // Longest monotonic run per direction, judged on a 5-point trend so a noisy
  // step doesn't chop the ramp into fragments.
  const W = 5;
  const runs: { from: number; to: number; dir: number }[] = [];
  let dir = 0, start = 0;
  for (let i = W; i < n; i++) {
    const d = Math.sign(b[i] - b[i - W]);
    if (d === 0 || !isFinite(d)) continue;
    if (dir === 0) { dir = d; continue; }
    if (d !== dir) { runs.push({ from: start, to: i - W, dir }); start = i - W; dir = d; }
  }
  runs.push({ from: start, to: n - 1, dir: dir || 1 });

  const longest = (want: number) => runs
    .filter(r => r.dir === want)
    .reduce<{ from: number; to: number; dir: number } | null>(
      (best, r) => (!best || r.to - r.from > best.to - best.from ? r : best), null);
  const minRun = Math.max(100, Math.floor(n * 0.15));
  const out: { dir: 'up' | 'down'; rows: any[] }[] = [];
  for (const [want, label] of [[1, 'up'], [-1, 'down']] as [number, 'up' | 'down'][]) {
    const r = longest(want);
    if (!r || r.to - r.from + 1 < minRun) continue;
    // Trim the flat plateau the magnet sits on while it turns around: those rows
    // repeat one field value and would pile up on a single point in 1/B.
    let { from, to } = r;
    while (to > from && !(Math.abs(b[to] - b[to - 1]) > 0)) to--;
    while (from < to && !(Math.abs(b[from + 1] - b[from]) > 0)) from++;
    out.push({ dir: label, rows: rows.slice(from, to + 1) });
  }
  return out.length >= 2 ? out : [];
}

// Instrument housekeeping, not sample signal. These drift smoothly (a
// thermometer reading outscores a real orbit by 10× if you let it in) or are
// held constant, so they never take part in the oscillation score.
// A lone "T" name part is a thermometer ("Prob_T_006", "SORB_T_006", "T_VTI"):
// those drift smoothly and outscore a real orbit, so they must never be picked
// as the signal channel.
export const NON_SIGNAL_COL = /(elapsed|time|temp|current|field|magnet|angle|position|blm|step)|(^|[\s_-])t([\s_-]|$)/i;

// A file's real signal columns, best oscillation first. Raw instrument files
// carry a dozen channels and most of them are all-NaN placeholders for unused
// lock-ins, so both the score and the initial column selection are driven from
// what actually holds data rather than from a name guess.
export function fileSignalColumns(
  file: { columns: string[]; data: any[] },
  bCol: string,
): { col: string; score: number }[] {
  const candidates = dropQuadratureStrict(
    file.columns.filter(c => c !== bCol && !NON_SIGNAL_COL.test(c)),
    file.columns,
  );
  const ranked: { col: string; score: number }[] = [];
  for (const col of candidates) {
    const bv: number[] = [], rv: number[] = [];
    for (const row of file.data) {
      const bVal = parseFloat(String(row[bCol]));
      const rVal = parseFloat(String(row[col]));
      if (isFinite(bVal) && isFinite(rVal)) { bv.push(bVal); rv.push(rVal); }
    }
    if (bv.length < 64) continue;                 // all-NaN / unused channel
    ranked.push({ col, score: oscillationQuality(bv, rv) });
  }
  return ranked.sort((a, b) => b.score - a.score);
}

export function fileOscillationScore(
  file: { columns: string[]; data: any[] },
  bCol: string,
): { score: number; col: string | null } {
  const best = fileSignalColumns(file, bCol)[0];
  return { score: best?.score ?? 0, col: best?.col ?? null };
}

// Alternating maxima/minima of an oscillatory series, filtered by prominence
// relative to neighbouring extrema (rejects noise wiggles near zero-crossings).
export function detectExtrema(
  series: { invB: number; b: number; val: number }[],
  smoothWin: number,
  prominenceFrac: number,
): LandauExtremum[] {
  const n = series.length;
  if (n < 5) return [];
  const raw = series.map(s => s.val);
  const val = movingAverage(raw, smoothWin);
  let lo = Infinity, hi = -Infinity;
  for (const v of val) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const minProm = prominenceFrac * ((hi - lo) || 1);

  const w = 2;
  const cand: LandauExtremum[] = [];
  for (let i = w; i < n - w; i++) {
    let isMax = true, isMin = true;
    for (let k = -w; k <= w; k++) {
      if (k === 0) continue;
      if (val[i] <= val[i + k]) isMax = false;
      if (val[i] >= val[i + k]) isMin = false;
    }
    if (isMax || isMin) cand.push({ invB: series[i].invB, b: series[i].b, val: val[i], type: isMax ? 'max' : 'min' });
  }

  // Enforce alternation (keep the more-extreme of any same-type run)
  const alternate = (list: LandauExtremum[]): LandauExtremum[] => {
    const out: LandauExtremum[] = [];
    for (const e of list) {
      const last = out[out.length - 1];
      if (!last || last.type !== e.type) out.push(e);
      else if ((e.type === 'max' && e.val > last.val) || (e.type === 'min' && e.val < last.val)) out[out.length - 1] = e;
    }
    return out;
  };
  let alt = alternate(cand);
  // Prominence: drop extrema too close to their neighbours, then re-alternate
  alt = alt.filter((e, i) => {
    const prev = alt[i - 1], next = alt[i + 1];
    if (!prev && !next) return true;
    let prom = Infinity;
    if (prev) prom = Math.min(prom, Math.abs(e.val - prev.val));
    if (next) prom = Math.min(prom, Math.abs(e.val - next.val));
    return prom >= minProm;
  });
  return alternate(alt);
}

export function linearFit(xs: number[], ys: number[]): { slope: number; intercept: number; r2: number } | null {
  const n = xs.length;
  if (n < 2) return null;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
  const d = n * sxx - sx * sx;
  if (Math.abs(d) < 1e-300) return null;
  const slope = (n * sxy - sx * sy) / d;
  const intercept = (sy - slope * sx) / n;
  const my = sy / n;
  let ssTot = 0, ssRes = 0;
  for (let i = 0; i < n; i++) { const yh = slope * xs[i] + intercept; ssTot += (ys[i] - my) ** 2; ssRes += (ys[i] - yh) ** 2; }
  return { slope, intercept, r2: ssTot > 0 ? 1 - ssRes / ssTot : 1 };
}

export function computeLandauFan(
  series: { invB: number; b: number; val: number }[],
  opts: { smoothWin: number; prominenceFrac: number; integerAt: 'max' | 'min'; n0: number; useBoth: boolean },
): FanResult {
  const extrema = detectExtrema(series, opts.smoothWin, opts.prominenceFrac);
  const used = opts.useBoth ? extrema : extrema.filter(e => e.type === opts.integerAt);
  const sorted = [...used].sort((a, b) => a.invB - b.invB); // low n (high field) first
  const points: LandauPoint[] = [];
  if (opts.useBoth) {
    // integer index on the chosen extremum type, half-integer on the other
    const startOffset = sorted.length > 0 && sorted[0].type === opts.integerAt ? 0 : -0.5;
    sorted.forEach((e, k) => points.push({ ...e, n: opts.n0 + startOffset + 0.5 * k }));
  } else {
    sorted.forEach((e, k) => points.push({ ...e, n: opts.n0 + k }));
  }
  const fit = points.length >= 2 ? linearFit(points.map(p => p.invB), points.map(p => p.n)) : null;
  return { extrema, points, slope: fit?.slope ?? NaN, intercept: fit?.intercept ?? NaN, r2: fit?.r2 ?? NaN };
}

// Standard Landau-fan preprocessing: FFT the oscillation in 1/B, keep only a
// chosen frequency band (smooth raised-cosine edges to limit ringing), inverse-
// transform to recover that single component, and index its extrema. Returns
// the spectrum (for the band picker) and the reconstructed 1/B-domain signal.
export function landauSpectrumAndFilter(
  series: { invB: number; b: number; val: number }[],
  band: { lo: number; hi: number } | null,
  edgeFrac: number,
  applyFilter: boolean,
): LandauFilterResult | null {
  const n0 = series.length;
  if (n0 < 8) return null;
  const minInvB = series[0].invB, maxInvB = series[n0 - 1].invB;
  const span = maxInvB - minInvB;
  if (!(span > 0)) return null;
  let N = 1; while (N < n0) N <<= 1; N = Math.min(N, 8192);
  const step = span / (N - 1);
  const uni = linearInterpUniform(series.map(s => ({ invB: s.invB, deltaR: s.val })), minInvB, step, N);
  let mean = 0; for (let i = 0; i < N; i++) mean += uni[i]; mean /= N;

  const fft = getFftInstance(N);
  const input = fft.createComplexArray();
  const out = fft.createComplexArray();
  for (let i = 0; i < N; i++) { input[2 * i] = uni[i] - mean; input[2 * i + 1] = 0; }
  fft.transform(out, input);

  const df = 1 / (N * step);         // frequency step, in Tesla
  const half = N >> 1;
  const spectrum: { frequency: number; amplitude: number }[] = new Array(half);
  let peakF = 0, peakA = 0;
  for (let k = 0; k < half; k++) {
    const re = out[2 * k], im = out[2 * k + 1];
    const amp = Math.sqrt(re * re + im * im);
    spectrum[k] = { frequency: k * df, amplitude: amp };
    if (k > 1 && amp > peakA) { peakA = amp; peakF = k * df; }
  }

  let lo: number, hi: number;
  if (band && band.hi > band.lo && band.lo >= 0) { lo = band.lo; hi = band.hi; }
  else { lo = peakF * 0.7; hi = peakF * 1.3; }

  let recon: { invB: number; b: number; val: number }[];
  let filtered = false;
  if (applyFilter && hi > lo && peakF > 0) {
    const w = Math.max(df, (hi - lo) * edgeFrac);
    for (let k = 0; k <= half; k++) {
      const f = k * df;
      let m: number;
      if (f <= lo - w || f >= hi + w) m = 0;
      else if (f >= lo && f <= hi) m = 1;
      else if (f < lo) m = 0.5 * (1 + Math.cos(Math.PI * (lo - f) / w));
      else m = 0.5 * (1 + Math.cos(Math.PI * (f - hi) / w));
      out[2 * k] *= m; out[2 * k + 1] *= m;
      if (k > 0 && k < half) { const kk = N - k; out[2 * kk] *= m; out[2 * kk + 1] *= m; }
    }
    const rec = fft.createComplexArray();
    fft.inverseTransform(rec, out);
    recon = new Array(N);
    for (let i = 0; i < N; i++) {
      const x = minInvB + i * step;
      recon[i] = { invB: x, b: x !== 0 ? 1 / x : 0, val: rec[2 * i] };
    }
    filtered = true;
  } else {
    recon = series;
  }
  return { spectrum, recon, band: { lo, hi }, peakF, filtered };
}
