// ---------- Local user (no Firebase auth required) ----------
export interface LocalUser {
  uid: string;
  displayName: string | null;
  email: string | null;
  photoURL: string | null;
}
export type User = LocalUser;

// --- Core data types ---

export interface FFTResult {
  frequency: number;
  amplitude: number;
  threshold?: number;
}

export interface AnalysisResult {
  columnName: string;
  plotData: {
    b: number;
    r: number;
    bg: number;
    deltaR: number;
    invB: number;
  }[];
  fftResults: FFTResult[];
  peakFrequency: FFTResult | null;
  detectedPeaks: FFTResult[];
  suggestedMaxFreq: number;
  df: number;
  minB: number;
  maxB: number;
}

export interface FileEntry {
  id: string;
  name: string;
  data: any[];
  metadata: Record<string, string>;
  columns: string[];
  results: AnalysisResult[];
}

// One measurement type's files + the processing settings that go with them.
// Angle and temperature sweeps are usually different files with potentially
// different columns/ranges, so each bucket keeps its own settings rather than
// sharing one set across both.
export interface FileBucket {
  files: FileEntry[];
  settings: Record<string, any>;
}

// A temperature sweep is run at one fixed angle; a sample can have several
// such sweeps (one per angle of interest), each named by its filenames'
// temperature but grouped here under the angle it was taken at.
export interface TemperatureSweepGroup {
  id: string;
  label: string;
  files: FileEntry[];
}

// All temperature-sweep groups for one orientation share one set of
// processing settings — same physical setup/columns, just a different fixed
// angle per group.
export interface TemperatureBucket {
  groups: TemperatureSweepGroup[];
  settings: Record<string, any>;
}

// A sample can be measured at more than one physical orientation, and within
// one orientation you can run both kinds of sweep: an angle/BLM sweep at
// fixed temperature (feeds Torque/Landau Fan/Harmonics/Symmetry/Branch) and
// one or more temperature sweeps at fixed angle (feeds the L-K mass fit).
// Both live side by side under the same orientation as independent buckets.
export interface OrientationEntry {
  id: string;
  label: string;
  angle: FileBucket;
  temperature: TemperatureBucket;
}

// ── OLS with a cached design-matrix factorization ─────────────────────────
export type OlsModel = { n: number; p: number; X: Float64Array; L: Float64Array };

export type FitResult = {
  y0: number;
  A1: number; x1: number; A2: number; x2: number;
  A3: number; x3: number; A4: number; x4: number;
  A6: number; x6: number; A12: number; x12: number;
  r2: number; rmse: number;
  predict: (x: number) => number;
};

// ── Landau fan / Berry phase analysis ─────────────────────────────────────
// SdH oscillations are periodic in 1/B: ΔR ∝ cos[2π(F/B − ½ + β + δ)], so
// tagging the extrema (Landau levels), assigning indices n, and fitting n vs
// 1/B gives the frequency F (slope) and, from the intercept, the Berry phase.
export type LandauExtremum = { invB: number; b: number; val: number; type: 'max' | 'min' };
export type LandauPoint = LandauExtremum & { n: number };
export type FanResult = {
  extrema: LandauExtremum[];
  points: LandauPoint[];
  slope: number;     // F, oscillation frequency in Tesla
  intercept: number; // n extrapolated to 1/B → 0
  r2: number;
};

// Standard Landau-fan preprocessing result: FFT the oscillation in 1/B, keep
// only a chosen frequency band, inverse-transform to recover that single
// component, and index its extrema.
export type LandauFilterResult = {
  spectrum: { frequency: number; amplitude: number }[];
  recon: { invB: number; b: number; val: number }[];
  band: { lo: number; hi: number };       // bounding envelope of all bands, for single-band callers
  bands: { lo: number; hi: number }[];    // the individual band(s) actually kept — 2+ when combining frequencies
  peakF: number;
  filtered: boolean;
};
