import FFT from 'fft.js';
import type { FFTResult, OlsModel, FitResult } from '../types';

// Cached window arrays — same (type, N) always returns the same Float64Array
const _windowCache = new Map<string, Float64Array>();
export function getWindowCached(type: string, N: number): Float64Array {
  const key = `${type}-${N}`;
  if (_windowCache.has(key)) return _windowCache.get(key)!;
  const win = new Float64Array(N);
  const M = N - 1;
  switch (type) {
    case 'hanning':
      for (let i = 0; i < N; i++) win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / M));
      break;
    case 'hamming':
      for (let i = 0; i < N; i++) win[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / M);
      break;
    case 'bartlett':
      for (let i = 0; i < N; i++) win[i] = (2 / M) * (M / 2 - Math.abs(i - M / 2));
      break;
    case 'blackman':
      for (let i = 0; i < N; i++) win[i] = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / M) + 0.08 * Math.cos((4 * Math.PI * i) / M);
      break;
    case 'kaiser': {
      const beta = 14;
      const i0 = (x: number) => {
        let sum = 1, fact = 1;
        for (let k = 1; k < 20; k++) { fact *= k; sum += Math.pow(Math.pow(x / 2, k) / fact, 2); }
        return sum;
      };
      const i0beta = i0(beta);
      for (let i = 0; i < N; i++) {
        const term = beta * Math.sqrt(1 - Math.pow((2 * i) / M - 1, 2));
        win[i] = i0(term) / i0beta;
      }
      break;
    }
    default:
      win.fill(1.0);
  }
  _windowCache.set(key, win);
  return win;
}

// Cached FFT instances — reuse object for same size
const _fftCache = new Map<number, InstanceType<typeof FFT>>();
export function getFftInstance(size: number): InstanceType<typeof FFT> {
  if (!_fftCache.has(size)) _fftCache.set(size, new FFT(size));
  return _fftCache.get(size)!;
}

// O(n) sliding-window spike removal (avoids per-element array slicing)
export function applySpikeRemovalOpt(data: number[], kernel: number, threshold: number): number[] {
  const result = [...data];
  const half = Math.floor(kernel / 2);
  const buf: number[] = new Array(kernel);
  for (let i = half; i < data.length - half; i++) {
    for (let k = 0; k < kernel; k++) buf[k] = data[i - half + k];
    buf.sort((a, b) => a - b);
    const median = kernel % 2 === 0
      ? (buf[kernel / 2 - 1] + buf[kernel / 2]) / 2
      : buf[Math.floor(kernel / 2)];
    if (Math.abs(data[i] / (median || 1e-9)) > threshold) result[i] = median;
  }
  return result;
}

// O(n) smoothing via prefix-sum (was O(n*w))
export function applySmoothingOpt(data: number[], windowLen: number): number[] {
  const result = new Array<number>(data.length);
  const half = Math.floor(windowLen / 2);
  const prefix = new Float64Array(data.length + 1);
  for (let i = 0; i < data.length; i++) prefix[i + 1] = prefix[i] + data[i];
  for (let i = 0; i < data.length; i++) {
    const lo = Math.max(0, i - half);
    const hi = Math.min(data.length - 1, i + half);
    result[i] = (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1);
  }
  return result;
}

// Segmented-regression changepoint: the field beyond which a trend's slope
// changes (e.g. R(B) turns over and starts decreasing). Fits two independent
// least-squares lines on either side of every candidate breakpoint — O(1)
// per candidate via prefix sums, O(n) overall — and returns the breakpoint
// that minimizes total squared error. Judging each candidate by how well a
// whole segment fits (rather than point-to-point behavior, the way a
// derivative does) is what makes this tolerant of noisy sweeps.
export function findChangepoint(
  points: { b: number; val: number }[],
  minSegFrac = 0.1,
  minSegAbs = 5,
): {
  breakB: number; breakIdx: number;
  leftSlope: number; leftIntercept: number;
  rightSlope: number; rightIntercept: number;
  totalSSE: number; singleSSE: number; improvement: number;
} | null {
  const n = points.length;
  const minSeg = Math.max(minSegAbs, Math.floor(n * minSegFrac));
  if (n < minSeg * 2 + 1) return null;

  const px = new Float64Array(n + 1), py = new Float64Array(n + 1);
  const pxx = new Float64Array(n + 1), pxy = new Float64Array(n + 1), pyy = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const x = points[i].b, y = points[i].val;
    px[i + 1] = px[i] + x;
    py[i + 1] = py[i] + y;
    pxx[i + 1] = pxx[i] + x * x;
    pxy[i + 1] = pxy[i] + x * y;
    pyy[i + 1] = pyy[i] + y * y;
  }

  // Closed-form OLS line over the half-open range [lo, hi) via prefix sums.
  const fitRange = (lo: number, hi: number) => {
    const m = hi - lo;
    const Sx = px[hi] - px[lo], Sy = py[hi] - py[lo];
    const Sxx = pxx[hi] - pxx[lo], Sxy = pxy[hi] - pxy[lo], Syy = pyy[hi] - pyy[lo];
    const denom = m * Sxx - Sx * Sx;
    if (!(Math.abs(denom) > 1e-12)) return null;
    const slope = (m * Sxy - Sx * Sy) / denom;
    const intercept = (Sy - slope * Sx) / m;
    const sse = Math.max(0, Syy - slope * Sxy - intercept * Sy);
    return { slope, intercept, sse };
  };

  const whole = fitRange(0, n);
  if (!whole) return null;

  let best: { idx: number; sse: number; left: NonNullable<ReturnType<typeof fitRange>>; right: NonNullable<ReturnType<typeof fitRange>> } | null = null;
  for (let k = minSeg; k <= n - minSeg; k++) {
    const left = fitRange(0, k);
    const right = fitRange(k, n);
    if (!left || !right) continue;
    const total = left.sse + right.sse;
    if (!best || total < best.sse) best = { idx: k, sse: total, left, right };
  }
  if (!best) return null;

  return {
    breakB: points[best.idx].b, breakIdx: best.idx,
    leftSlope: best.left.slope, leftIntercept: best.left.intercept,
    rightSlope: best.right.slope, rightIntercept: best.right.intercept,
    totalSSE: best.sse, singleSSE: whole.sse,
    improvement: whole.sse > 0 ? 1 - best.sse / whole.sse : 0,
  };
}

// Multiple changepoints via binary segmentation: repeatedly split whichever
// pending segment currently offers the largest ABSOLUTE squared-error
// reduction, then re-consider its two halves as new candidates, until
// `maxBreaks` breaks are found or no remaining segment clears
// `minImprovement`. A single global findChangepoint() can be dominated by
// one sharp, narrow feature (e.g. a small hump) even when a broader, gentler
// bend elsewhere is the one that matters — this surfaces both, ranked by how
// much of the data's variance each actually explains, so a real second bend
// doesn't stay hidden behind a bigger local one.
export function findChangepoints(
  points: { b: number; val: number }[],
  maxBreaks = 4,
  minSegFrac = 0.1,
  minSegAbs = 5,
  minImprovement = 0.02,
): Array<NonNullable<ReturnType<typeof findChangepoint>> & { absReduction: number }> {
  type Candidate = NonNullable<ReturnType<typeof findChangepoint>> & { absReduction: number; lo: number; hi: number };
  const results: Candidate[] = [];

  const bestSplitFor = (lo: number, hi: number): Candidate | null => {
    const cp = findChangepoint(points.slice(lo, hi), minSegFrac, minSegAbs);
    if (!cp || cp.improvement < minImprovement) return null;
    return { ...cp, breakIdx: lo + cp.breakIdx, absReduction: cp.singleSSE - cp.totalSSE, lo, hi };
  };

  let pending: Candidate[] = [];
  const first = bestSplitFor(0, points.length);
  if (first) pending.push(first);

  while (results.length < maxBreaks && pending.length > 0) {
    pending.sort((a, b) => b.absReduction - a.absReduction);
    const picked = pending.shift()!;
    results.push(picked);
    const left = bestSplitFor(picked.lo, picked.breakIdx);
    const right = bestSplitFor(picked.breakIdx, picked.hi);
    if (left) pending.push(left);
    if (right) pending.push(right);
  }

  return results.sort((a, b) => a.breakB - b.breakB);
}

// Where a curve first sustains a departure from an established baseline line
// — used to find the *start* of a bend, as distinct from findChangepoint's
// breakB (the two-line model's best-fit meeting point, which for a gradual,
// smoothly-curving transition sits closer to its middle than its beginning).
// `referencePts` (typically the pre-transition segment) sets the noise floor
// σ; `scanPts` (typically the whole search range) is walked in order for the
// first sustained departure beyond `thresholdSigma·σ`.
export function findBaselineOnset(
  scanPts: { b: number; val: number }[],
  baseline: (b: number) => number,
  referencePts: { b: number; val: number }[],
  thresholdSigma = 4,
  consecutiveNeeded = 5,
): number | null {
  if (referencePts.length === 0) return null;
  const refResiduals = referencePts.map(p => p.val - baseline(p.b));
  const mean = refResiduals.reduce((a, b) => a + b, 0) / refResiduals.length;
  const sigma = Math.sqrt(refResiduals.reduce((a, r) => a + (r - mean) ** 2, 0) / refResiduals.length);
  if (!(sigma > 0)) return null;

  const cutoff = thresholdSigma * sigma;
  const absResiduals = scanPts.map(p => Math.abs(p.val - baseline(p.b)));
  const smoothed = applySmoothingOpt(absResiduals, 7);

  let streak = 0, streakStartIdx = -1;
  for (let i = 0; i < smoothed.length; i++) {
    if (smoothed[i] > cutoff) {
      if (streak === 0) streakStartIdx = i;
      streak++;
      if (streak >= consecutiveNeeded) return scanPts[streakStartIdx].b;
    } else {
      streak = 0;
      streakStartIdx = -1;
    }
  }
  return null;
}

// Single-pass linear interpolation onto uniform grid (replaces 1024× sortedIndexBy)
export function linearInterpUniform(
  sorted: { invB: number; deltaR: number }[],
  minInvB: number,
  step: number,
  N: number
): Float64Array {
  const out = new Float64Array(N);
  const m = sorted.length;
  let ptr = 0;
  for (let i = 0; i < N; i++) {
    const target = minInvB + i * step;
    while (ptr < m - 2 && sorted[ptr + 1].invB <= target) ptr++;
    if (target <= sorted[0].invB) {
      out[i] = sorted[0].deltaR;
    } else if (target >= sorted[m - 1].invB) {
      out[i] = sorted[m - 1].deltaR;
    } else {
      const p1 = sorted[ptr];
      const p2 = sorted[ptr + 1];
      const t = (target - p1.invB) / (p2.invB - p1.invB);
      out[i] = p1.deltaR + t * (p2.deltaR - p1.deltaR);
    }
  }
  return out;
}

// Envelope-preserving decimation for chart rendering: bucket the points and
// keep the min and max of each bucket, in x order. Unlike stride sampling this
// cannot hide a spike or flatten an oscillation, so the drawn line keeps the
// same visual envelope as the full series.
export function decimateMinMax<T>(pts: T[], maxPoints: number, yOf: (p: T) => number): T[] {
  const n = pts.length;
  if (n <= maxPoints || maxPoints < 4) return pts;
  const buckets = Math.floor(maxPoints / 2);
  const out: T[] = [];
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor((b * n) / buckets);
    const end = Math.min(n, Math.floor(((b + 1) * n) / buckets));
    if (end <= start) continue;
    let iMin = start, iMax = start;
    let vMin = yOf(pts[start]), vMax = vMin;
    for (let i = start + 1; i < end; i++) {
      const v = yOf(pts[i]);
      if (v < vMin) { vMin = v; iMin = i; }
      else if (v > vMax) { vMax = v; iMax = i; }
    }
    const lo = Math.min(iMin, iMax), hi = Math.max(iMin, iMax);
    out.push(pts[lo]);
    if (hi !== lo) out.push(pts[hi]);
  }
  return out;
}

// Downsample an FFT result array to at most maxPoints for chart rendering
export function downsampleFFT(arr: FFTResult[], maxPoints: number): FFTResult[] {
  if (arr.length <= maxPoints) return arr;
  const step = arr.length / maxPoints;
  const out: FFTResult[] = new Array(maxPoints);
  for (let i = 0; i < maxPoints; i++) out[i] = arr[Math.round(i * step)];
  return out;
}

// ── OLS with a cached design-matrix factorization ─────────────────────────
// Every angular fit in the field scans shares the same x-values (the BLM
// angles), so the design matrix is identical at each field step — only y
// changes. Factorizing XᵀX once turns each step into a matvec plus two
// triangular solves: O(n·p) instead of O(n·p² + p³) with a fresh LU.

// Cholesky of XᵀX. Returns null when the normal equations are singular
// (degenerate angle set) — callers skip that fit, as the old LU path did.
export function buildOlsModel(X: Float64Array, n: number, p: number): OlsModel | null {
  const L = new Float64Array(p * p);
  const A = new Float64Array(p * p);
  for (let j = 0; j < p; j++) {
    for (let k = j; k < p; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i * p + j] * X[i * p + k];
      A[j * p + k] = s;
      A[k * p + j] = s;
    }
  }
  for (let j = 0; j < p; j++) {
    let d = A[j * p + j];
    for (let k = 0; k < j; k++) d -= L[j * p + k] * L[j * p + k];
    if (!(d > 1e-300)) return null;
    const ljj = Math.sqrt(d);
    L[j * p + j] = ljj;
    for (let i = j + 1; i < p; i++) {
      let s = A[i * p + j];
      for (let k = 0; k < j; k++) s -= L[i * p + k] * L[j * p + k];
      L[i * p + j] = s / ljj;
    }
  }
  return { n, p, X, L };
}

// Solve for beta given y, reusing the cached factorization. `out` lets hot
// loops recycle one buffer instead of allocating per field step.
export function olsSolve(m: OlsModel, y: ArrayLike<number>, out?: Float64Array): Float64Array {
  const { n, p, X, L } = m;
  const b = out ?? new Float64Array(p);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i * p + j] * y[i];
    b[j] = s;
  }
  for (let i = 0; i < p; i++) {           // forward:  L z = Xᵀy
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * p + k] * b[k];
    b[i] = s / L[i * p + i];
  }
  for (let i = p - 1; i >= 0; i--) {      // backward: Lᵀ β = z
    let s = b[i];
    for (let k = i + 1; k < p; k++) s -= L[k * p + i] * b[k];
    b[i] = s / L[i * p + i];
  }
  return b;
}

// R² of a fitted beta, evaluated straight off the cached design matrix.
export function olsR2(m: OlsModel, y: ArrayLike<number>, beta: Float64Array): { r2: number; rmse: number } {
  const { n, p, X } = m;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += y[i];
  mean /= n;
  let ssTot = 0, ssRes = 0;
  for (let i = 0; i < n; i++) {
    let yh = 0;
    for (let j = 0; j < p; j++) yh += X[i * p + j] * beta[j];
    ssTot += (y[i] - mean) * (y[i] - mean);
    ssRes += (y[i] - yh) * (y[i] - yh);
  }
  return { r2: ssTot > 0 ? 1 - ssRes / ssTot : 1, rmse: Math.sqrt(ssRes / n) };
}

// Design matrix for y = y0 + Σ [Cₙcos(nθ) + Sₙsin(nθ)], θ in degrees.
export function cosineDesign(angles: ArrayLike<number>, harmonics: number[]): { X: Float64Array; p: number } {
  const n = angles.length, p = 1 + 2 * harmonics.length;
  const X = new Float64Array(n * p);
  for (let i = 0; i < n; i++) {
    const ar = angles[i] * Math.PI / 180;
    X[i * p] = 1;
    for (let hi = 0; hi < harmonics.length; hi++) {
      const t = harmonics[hi] * ar;
      X[i * p + 1 + 2 * hi] = Math.cos(t);
      X[i * p + 2 + 2 * hi] = Math.sin(t);
    }
  }
  return { X, p };
}

// Design matrix for τ(α) = Σ [Sₙsin(nα) + Cₙcos(nα)], α in degrees.
export function sineFourierDesign(angles: ArrayLike<number>, nMax: number): { X: Float64Array; p: number } {
  const n = angles.length, p = 2 * nMax;
  const X = new Float64Array(n * p);
  for (let i = 0; i < n; i++) {
    const ar = angles[i] * Math.PI / 180;
    for (let k = 0; k < nMax; k++) {
      const t = (k + 1) * ar;
      X[i * p + 2 * k] = Math.sin(t);
      X[i * p + 2 * k + 1] = Math.cos(t);
    }
  }
  return { X, p };
}

// Linear-regression cosine fit: y = y0 + sum_n A_n*cos(n*pi*(x-x_n)/180)
// Converts to linear OLS via trig identity. harmonics controls which terms to include.
export const ALL_HARMONICS = [1, 2, 3, 4, 6, 12] as const;
export function fitCosineSeries(points: { x: number; y: number }[], harmonics: number[] = [...ALL_HARMONICS]): FitResult | null {
  const nParams = 1 + 2 * harmonics.length;
  if (points.length < nParams) return null;
  const n = points.length;
  const xs = new Float64Array(n), ys = new Float64Array(n);
  for (let i = 0; i < n; i++) { xs[i] = points[i].x; ys[i] = points[i].y; }
  const { X, p } = cosineDesign(xs, harmonics);
  const model = buildOlsModel(X, n, p);
  if (!model) return null;
  const beta = olsSolve(model, ys);
  return buildCosineFitResult(beta, harmonics, olsR2(model, ys, beta));
}

// Shared by fitCosineSeries and the cached-model field scans: turns a raw
// beta vector into the amplitude/phase FitResult shape.
export function buildCosineFitResult(
  beta: ArrayLike<number>,
  harmonics: number[],
  gof: { r2: number; rmse: number },
): FitResult {
  const res: any = { y0: beta[0], r2: gof.r2, rmse: gof.rmse };
  // Initialise all harmonics to zero, then overwrite fitted ones
  ALL_HARMONICS.forEach(h => { res[`A${h}`] = 0; res[`x${h}`] = 0; });
  harmonics.forEach((h, idx) => {
    const C = beta[1 + 2 * idx], S = beta[2 + 2 * idx];
    res[`A${h}`] = Math.sqrt(C * C + S * S);
    res[`x${h}`] = (180 / (h * Math.PI)) * Math.atan2(S, C);
  });
  res.predict = (x: number) => {
    let val = res.y0;
    for (const h of harmonics) val += res[`A${h}`] * Math.cos(h * Math.PI * (x - res[`x${h}`]) / 180);
    return val;
  };
  return res as FitResult;
}

// Fast interpolation using pre-parsed numeric arrays (no string parsing in hot path)
export function lerpFast(bVals: ArrayLike<number>, colVals: ArrayLike<number>, targetB: number): number {
  const n = bVals.length;
  if (n === 0) return NaN;
  if (targetB <= bVals[0]) return colVals[0];
  if (targetB >= bVals[n - 1]) return colVals[n - 1];
  let lo = 0, hi = n - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >>> 1;
    if (bVals[mid] <= targetB) lo = mid; else hi = mid;
  }
  const bLo = bVals[lo], bHi = bVals[hi];
  const vLo = colVals[lo], vHi = colVals[hi];
  if (!isFinite(vLo) || !isFinite(vHi) || bHi === bLo) return vLo;
  return vLo + (targetB - bLo) / (bHi - bLo) * (vHi - vLo);
}

// Polynomial fit: τ = a·H + b·H² + c·H³  (no constant — physically motivated)
export function fitPolynomialH(pts: { h: number; tau: number }[]): {
  a: number; b: number; c: number; r2: number; predict: (h: number) => number;
} | null {
  if (pts.length < 3) return null;
  const n = pts.length;
  const X = new Float64Array(n * 3), y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const h = pts[i].h;
    X[i * 3] = h; X[i * 3 + 1] = h * h; X[i * 3 + 2] = h * h * h;
    y[i] = pts[i].tau;
  }
  const model = buildOlsModel(X, n, 3);
  if (!model) return null;
  const beta = olsSolve(model, y);
  const [a, b, c] = [beta[0], beta[1], beta[2]];
  return {
    a, b, c,
    r2: olsR2(model, y, beta).r2,
    predict: (h: number) => a * h + b * h * h + c * h * h * h,
  };
}

// A sin(nα + φ) = S sin(nα) + C cos(nα)  →  A = √(S²+C²), φ = atan2(C,S)
export function buildFourierResult(beta: ArrayLike<number>, nMax: number): {
  coeffs: { n: number; A: number; phiDeg: number }[];
  r2: number; predict: (alpha: number) => number;
} {
  const coeffs = Array.from({ length: nMax }, (_, idx) => {
    const S = beta[2 * idx], C = beta[2 * idx + 1];
    return { n: idx + 1, A: Math.sqrt(S * S + C * C), phiDeg: Math.atan2(C, S) * 180 / Math.PI };
  });
  return {
    coeffs,
    r2: 1,
    predict: (alpha: number) => {
      const ar = alpha * Math.PI / 180;
      return coeffs.reduce((s, { n, A, phiDeg }) => s + A * Math.sin(n * ar + phiDeg * Math.PI / 180), 0);
    },
  };
}
