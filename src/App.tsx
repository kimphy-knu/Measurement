/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  AreaChart,
  Area,
  Scatter,
  ScatterChart,
  ComposedChart,
  LabelList,
  ZAxis,
  ReferenceLine,
  ReferenceArea
} from 'recharts';
import Papa from 'papaparse';
import * as math from 'mathjs';
import _ from 'lodash';
import {
  Upload,
  FileText,
  Activity,
  Settings2,
  Download,
  Info,
  AlertCircle,
  RefreshCw,
  Copy,
  LogIn,
  LogOut,
  History,
  GitBranch,
  Layers,
  Check,
  Gauge,
  Trash2,
  Plus,
  Save,
  FolderOpen,
  X
} from 'lucide-react';

import type {
  User, LocalUser, FFTResult, AnalysisResult, FileEntry, OrientationEntry, FileBucket,
  TemperatureBucket, TemperatureSweepGroup, OlsModel,
} from './types';
import { lsGetPrefs, lsSavePrefs } from './lib/localPrefs';
import { dropQuadratureStrict } from './lib/quadrature';
import {
  getWindowCached, getFftInstance, applySpikeRemovalOpt, applySmoothingOpt,
  linearInterpUniform, decimateMinMax, downsampleFFT,
  buildOlsModel, olsSolve, olsR2, cosineDesign, sineFourierDesign,
  fitCosineSeries, lerpFast, fitPolynomialH, buildFourierResult,
} from './lib/dsp';
import {
  oscillationOnsetB, OSC_SCORE_MIN, OSC_WINDOW_FRAC, angleFromFileName,
  guessFieldColumn, splitSweepDirections, fileSignalColumns, fileOscillationScore,
  computeLandauFan, landauSpectrumAndFilter,
} from './lib/landau';
import { MiniFFT } from './components/MiniFFT';

// ---------- Local user (no Firebase auth required) ----------
const LOCAL_USER: LocalUser = {
  uid: 'local-user',
  displayName: 'Local User',
  email: 'local@localhost',
  photoURL: null,
};

// --- Components ---

// A "live preview" B-range input pair: the displayed draft updates on every
// keystroke, but the *committed* value (what the expensive re-FFT actually
// reacts to) only updates ~400ms after the user stops typing, and stays
// `null` ("not touched yet") until they do. Armed only from the onChange
// handler rather than a useEffect keyed on the draft — an effect fires once
// on mount too (twice under StrictMode), which would commit the initial
// draft value on its own moment after mount before the user ever touched
// the control (this exact bug already happened once in the FFT Range tab).
function useDebouncedRange(delayMs = 400) {
  const [minDraft, setMinDraft] = useState(0);
  const [maxDraft, setMaxDraft] = useState(1);
  const [minCommitted, setMinCommitted] = useState<number | null>(null);
  const [maxCommitted, setMaxCommitted] = useState<number | null>(null);
  const timers = useRef<{ min?: ReturnType<typeof setTimeout>; max?: ReturnType<typeof setTimeout> }>({});

  // `immediate` skips the debounce — used for wheel-scroll steps, which are
  // discrete deliberate actions (unlike keystrokes typing a multi-digit
  // number), so each tick should recompute right away.
  const changeMin = (v: number, immediate = false) => {
    setMinDraft(v);
    clearTimeout(timers.current.min);
    if (immediate) setMinCommitted(v);
    else timers.current.min = setTimeout(() => setMinCommitted(v), delayMs);
  };
  const changeMax = (v: number, immediate = false) => {
    setMaxDraft(v);
    clearTimeout(timers.current.max);
    if (immediate) setMaxCommitted(v);
    else timers.current.max = setTimeout(() => setMaxCommitted(v), delayMs);
  };
  // Reset to "not touched" — falls back to whatever the caller's own
  // fallback range is, and re-syncs the drafts so the inputs show it too.
  const reset = (fallbackMin: number, fallbackMax: number) => {
    clearTimeout(timers.current.min);
    clearTimeout(timers.current.max);
    setMinCommitted(null);
    setMaxCommitted(null);
    setMinDraft(fallbackMin);
    setMaxDraft(fallbackMax);
  };
  const active = minCommitted !== null || maxCommitted !== null;

  return { minDraft, maxDraft, minCommitted, maxCommitted, active, changeMin, changeMax, reset };
}

// Lets a live-range number input respond to the mouse wheel even when it
// isn't focused yet — the browser's native scroll-to-increment on
// <input type="number"> only kicks in after a click to focus it first,
// which read as "nothing happens until I click once" for these controls.
// Assigning onwheel as a DOM property (rather than a passive React
// onWheel prop) lets preventDefault actually stop the page from
// scrolling underneath the cursor while the user adjusts the range.
// Committed immediately (bypassing the debounce) since each wheel tick
// is a discrete, deliberate step rather than a keystroke mid-typing.
function wheelStepRef(current: number, onChange: (v: number, immediate?: boolean) => void, step = 0.1) {
  return (el: HTMLInputElement | null) => {
    if (!el) return;
    el.onwheel = (e: WheelEvent) => {
      e.preventDefault();
      const dir = e.deltaY < 0 ? 1 : -1;
      const next = Math.round((current + dir * step) * 100) / 100;
      onChange(next, true);
    };
  };
}

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [sampleIdDraft, setSampleIdDraft] = useState<string>('');
  const [orientations, setOrientations] = useState<OrientationEntry[]>([]);
  const [activeOrientationId, setActiveOrientationId] = useState<string>('');
  const [orientationLabelDraft, setOrientationLabelDraft] = useState<string>('');
  const [showAddOrientation, setShowAddOrientation] = useState(false);
  const [editingOrientationId, setEditingOrientationId] = useState<string | null>(null);
  const [editOrientationLabelDraft, setEditOrientationLabelDraft] = useState<string>('');
  // Files excluded from the FFT batch (by id). They stay loaded and keep feeding
  // the Raw and Torque tabs — only the FFT / peak / branch pipeline skips them.
  // Kept out of FileEntry on purpose: toggling one must not invalidate the
  // parsed-column caches keyed on `files`.
  const [fftSkip, setFftSkip] = useState<Set<string>>(new Set());
  const [oscScores, setOscScores] = useState<Record<string, number>>({});
  const [selectedBColumn, setSelectedBColumn] = useState<string>('');
  const [selectedRColumns, setSelectedRColumns] = useState<string[]>([]);
  const [polyDegree, setPolyDegree] = useState<number>(3);
  const [tempPolyDegree, setTempPolyDegree] = useState<number>(3);
  const [interpPoints, setInterpPoints] = useState<number>(1024);
  const [tempInterpPoints, setTempInterpPoints] = useState<number>(1024);
  const [windowType, setWindowType] = useState<string>('hanning');
  const [tempWindowType, setTempWindowType] = useState<string>('hanning');
  const [spikeRemoval, setSpikeRemoval] = useState<boolean>(false);
  const [tempSpikeRemoval, setTempSpikeRemoval] = useState<boolean>(false);
  const [spikeKernel, setSpikeKernel] = useState<number>(5);
  const [tempSpikeKernel, setTempSpikeKernel] = useState<number>(5);
  const [spikeThreshold, setSpikeThreshold] = useState<number>(2.0);
  const [tempSpikeThreshold, setTempSpikeThreshold] = useState<number>(2.0);
  const [smoothing, setSmoothing] = useState<boolean>(false);
  const [tempSmoothing, setTempSmoothing] = useState<boolean>(false);
  const [smoothWindow, setSmoothWindow] = useState<number>(5);
  const [tempSmoothWindow, setTempSmoothWindow] = useState<number>(5);
  const [minB, setMinB] = useState<number | ''>(30);
  const [tempMinB, setTempMinB] = useState<number | ''>(30);
  const [maxB, setMaxB] = useState<number | ''>(41.5);
  const [tempMaxB, setTempMaxB] = useState<number | ''>(41.5);
  const [complexMode, setComplexMode] = useState<boolean>(false);
  const [tempComplexMode, setTempComplexMode] = useState<boolean>(false);
  const [phase, setPhase] = useState<number>(0);
  const [tempPhase, setTempPhase] = useState<number>(0);
  const [autoFFTRange, setAutoFFTRange] = useState<boolean>(true);
  const [globalSuggestedMaxFreq, setGlobalSuggestedMaxFreq] = useState<number>(5000);
  const [manualMaxFreq, setManualMaxFreq] = useState<number>(5000);
  const [showPeaksInAssemble, setShowPeaksInAssemble] = useState<boolean>(true);
  const [targetFreqInterval, setTargetFreqInterval] = useState<number>(1.0);
  const [tempTargetFreqInterval, setTempTargetFreqInterval] = useState<number>(1.0);
  const [useTargetInterval, setUseTargetInterval] = useState<boolean>(true);
  const [tempUseTargetInterval, setTempUseTargetInterval] = useState<boolean>(true);
  const [minPeakWidth, setMinPeakWidth] = useState<number>(1);
  const [tempMinPeakWidth, setTempMinPeakWidth] = useState<number>(1);

  // Frequency-dependent thresholds (4 segments)
  // Auto peak-detection threshold from the per-file FFT noise floor (default on).
  // When off, the manual frequency-dependent %-of-max thresholds below are used.
  const [autoThreshold, setAutoThreshold] = useState<boolean>(true);
  const [freqThresholds, setFreqThresholds] = useState<{ limit: number; threshold: number }[]>([
    { limit: 1500, threshold: 40 },
    { limit: 3000, threshold: 30 },
    { limit: 5000, threshold: 20 },
    { limit: 10000, threshold: 20 }
  ]);
  const [tempFreqThresholds, setTempFreqThresholds] = useState<{ limit: number; threshold: number }[]>([
    { limit: 1500, threshold: 40 },
    { limit: 3000, threshold: 30 },
    { limit: 5000, threshold: 20 },
    { limit: 10000, threshold: 20 }
  ]);

  // Every column list in this component goes through here, which deliberately
  // shadows the module-level dropQuadratureStrict: the quadrature channel
  // (Output2 / Y) is never selected, fit, plotted or exported — except in complex
  // mode, whose whole job is to rotate in-phase and quadrature together.
  const dropQuadrature = useCallback(
    (cols: string[], pool: string[] = cols): string[] =>
      complexMode ? cols : dropQuadratureStrict(cols, pool),
    [complexMode],
  );

  const [selectedRawColumn, setSelectedRawColumn] = useState<string>('');
  const [activeBlms, setActiveBlms] = useState<Set<number>>(new Set());
  const [rawPlotData, setRawPlotData] = useState<{ x: number; y: number; fileName: string; columnName: string }[]>([]);
  const [maxDerivPlotData, setMaxDerivPlotData] = useState<{ x: number; y: number; fileName: string; columnName: string }[]>([]);
  const [maxDerivMinB, setMaxDerivMinB] = useState<number>(1);
  const [maxDerivMaxB, setMaxDerivMaxB] = useState<number>(16);
  const [fieldSliceValues, setFieldSliceValues] = useState<number[]>([1, 5, 10, 15, 20, 25, 30]);
  const [fieldSliceInput, setFieldSliceInput] = useState<string>('');
  const [rawBSliceTarget, setRawBSliceTarget] = useState<number | null>(null); // null = Max B
  const [rawBSliceInput, setRawBSliceInput] = useState<string>('');
  const [selectedHarmonics, setSelectedHarmonics] = useState<Set<number>>(new Set([2, 3, 4, 6]));
  const [scansArmed, setScansArmed] = useState(false);
  // Torque / Field-sweep analysis state
  const [tauHsfLimit, setTauHsfLimit] = useState<number>(10);
  const [tauResidThreshold, setTauResidThreshold] = useState<number>(0.05);
  const [sweepDensity, setSweepDensity] = useState<'fast' | 'full'>('fast');
  // Landau fan / Berry phase state
  const [landauResultKey, setLandauResultKey] = useState<string>('');
  const [landauIntegerAt, setLandauIntegerAt] = useState<'max' | 'min'>('max');
  const [landauN0, setLandauN0] = useState<number>(1);
  const [landauUseBoth, setLandauUseBoth] = useState<boolean>(true);
  const [landauSmoothWin, setLandauSmoothWin] = useState<number>(5);
  const [landauProm, setLandauProm] = useState<number>(0.05);
  const [landauDelta, setLandauDelta] = useState<number>(0); // dimensionality phase δ: 0 (2D), ±1/8 (3D)
  const [landauBandpass, setLandauBandpass] = useState<boolean>(true); // FFT→band-pass→IFFT before indexing
  const [landauBandLo, setLandauBandLo] = useState<number>(0);         // 0 ⇒ auto (from dominant peak)
  const [landauBandHi, setLandauBandHi] = useState<number>(0);
  const [landauEdge, setLandauEdge] = useState<number>(0.25);          // raised-cosine edge width fraction
  // IFFT verification state — "is this specific detected frequency a real
  // oscillation, or noise?" Reuses the same FFT→band-pass→IFFT machinery as
  // the Landau Fan tab, but overlays the reconstructed single-frequency
  // component directly on the raw ΔR(1/B) instead of indexing it.
  const [ifftResultKey, setIfftResultKey] = useState<string>('');
  const [ifftBandLo, setIfftBandLo] = useState<number>(0);   // 0,0 ⇒ auto (dominant peak)
  const [ifftBandHi, setIfftBandHi] = useState<number>(0);
  const [ifftEdge, setIfftEdge] = useState<number>(0.25);
  const [tauSampleVol, setTauSampleVol] = useState<number>(1.0);
  const [tauFourierNMax, setTauFourierNMax] = useState<number>(6);
  const [tauScaleFieldsStr, setTauScaleFieldsStr] = useState<string>('2,5,10,15,20');
  const [allRawSeries, setAllRawSeries] = useState<{ 
    fileName: string; 
    columnName: string; 
    blm: number; 
    peak: { b: number; val: number; type: 'max' | 'min' };
    data: { b: number; val: number }[];
    invBData: { invB: number; val: number }[];
  }[]>([]);

  const uniqueBlms = useMemo(() => {
    return Array.from(new Set(allRawSeries.map(s => s.blm))).sort((a: number, b: number) => a - b);
  }, [allRawSeries]);

  const [selectedPrefix, setSelectedPrefix] = useState<string>('');
  const [blmMode, setBlmMode] = useState<'metadata' | 'step'>('step');
  const [readAngleFromTitle, setReadAngleFromTitle] = useState<boolean>(true);
  const [blmPerDegree, setBlmPerDegree] = useState<number>(1.0);
  const [blmOffset, setBlmOffset] = useState<number>(0);
  const [assembleData, setAssembleData] = useState<{ x: number; fft: FFTResult[]; fileName: string; peaks?: FFTResult[] }[]>([]);
  const [exportMinFreq, setExportMinFreq] = useState<number>(0);
  const [exportMaxFreq, setExportMaxFreq] = useState<number>(5000);

  // ---- Export helpers ----

  const resolveBCol = useCallback((file: FileEntry): string => {
    return file.columns.find(c => c === selectedBColumn) ||
           file.columns.find(c => c.toLowerCase() === selectedBColumn.toLowerCase()) ||
           file.columns.find(k => {
             const low = k.toLowerCase();
             return (low === 'b' || low.startsWith('b ') || low.startsWith('b_') || low.includes('field') || low.includes('magnetic')) &&
                    !low.includes('blm') && !low.includes('angle') && !low.includes('position');
           }) ||
           file.columns.find(k => {
             const low = k.toLowerCase();
             return low.includes('b') && !low.includes('blm') && !low.includes('angle') && !low.includes('position');
           }) ||
           file.columns[0];
  }, [selectedBColumn]);

  const resolveBlm = useCallback((file: FileEntry, fIdx: number): number => {
    let blmValue = 0;
    if (readAngleFromTitle) {
      const fromName = angleFromFileName(file.name);
      blmValue = fromName != null ? fromName : fIdx + 1;
    } else if (blmMode === 'step') {
      blmValue = fIdx + 1;
    } else {
      const blmCol = file.columns.find(c =>
        c.toLowerCase().includes('blm') || c.toLowerCase().includes('position') ||
        c.toLowerCase().includes('angle') || c.toLowerCase().includes('step')
      );
      let blmStr = '0';
      if (blmCol && file.data.length > 0) {
        blmStr = String(file.data[0][blmCol]);
      } else {
        const entry = Object.entries(file.metadata).find(([k]) =>
          k.toLowerCase().includes('position') || k.toLowerCase().includes('blm') || k.toLowerCase().includes('angle')
        );
        blmStr = entry ? String(entry[1]) : '0';
      }
      const nm = blmStr.match(/[-+]?[0-9]*\.?[0-9]+/);
      blmValue = nm ? parseFloat(nm[0]) : parseFloat(blmStr);
      if (isNaN(blmValue)) blmValue = fIdx + 1;
    }
    return Number(((blmValue / blmPerDegree) + blmOffset).toFixed(2));
  }, [readAngleFromTitle, blmMode, blmPerDegree, blmOffset]);

  // Files ordered by angle, for the pickers that list them. The original upload
  // index travels with each entry because resolveBlm() falls back to it (step
  // mode, or a name with no number in it) — sorting must not renumber anything.
  // `label` disambiguates the rare case of two files landing on the same angle.
  const filesByAngle = useMemo(() => {
    const counts = new Map<number, number>();
    const entries = files.map((file, idx) => {
      const blm = resolveBlm(file, idx);
      counts.set(blm, (counts.get(blm) ?? 0) + 1);
      return { file, idx, blm };
    });
    return entries
      .map(e => ({ ...e, label: (counts.get(e.blm) ?? 0) > 1 ? `${e.blm}° · ${e.file.name}` : `${e.blm}°` }))
      .sort((a, b) => a.blm - b.blm || a.idx - b.idx);
  }, [files, resolveBlm]);

  const downloadCSV = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Export ALL columns from every file — wide format
  // Columns: File1_BLM_B(T), File1_BLM_ColA, File1_BLM_ColB, File2_BLM_B(T), ...
  const exportAllColumnsCSV = () => {
    if (files.length === 0) return;
    const fileData = files.map((file, fIdx) => {
      const bCol = resolveBCol(file);
      const blm = resolveBlm(file, fIdx);
      const rCols = dropQuadrature(file.columns.filter(c => c !== bCol), file.columns);
      return { file, bCol, rCols, blm };
    });

    const headerRow: string[] = [];
    fileData.forEach(fd => {
      fd.rCols.forEach(col => {
        headerRow.push(`"${fd.file.name}|BLM=${fd.blm}|B(T)"`);
        headerRow.push(`"${fd.file.name}|BLM=${fd.blm}|${col}"`);
      });
    });

    const maxRows = Math.max(...files.map(f => f.data.length));
    const rows: string[] = [headerRow.join(',')];
    for (let i = 0; i < maxRows; i++) {
      const rowData: string[] = [];
      fileData.forEach(fd => {
        const row = fd.file.data[i];
        fd.rCols.forEach(col => {
          if (row) {
            rowData.push(String(row[fd.bCol] ?? ''));
            rowData.push(String(row[col] ?? ''));
          } else {
            rowData.push('', '');
          }
        });
      });
      rows.push(rowData.join(','));
    }
    downloadCSV(rows.join('\n'), 'all_columns_all_files.csv');
  };

  // Export full-resolution processed ΔR — wide format, file-by-file side by
  // side (same layout as "All Raw Columns"): each file/column contributes a
  // B(T), DeltaR column pair, no interpolation onto a shared grid.
  const exportProcessedCSV = () => {
    const hasResults = files.some(f => f.results && f.results.length > 0);
    if (!hasResults) {
      setError('"Process All Files" 를 먼저 실행하세요.');
      return;
    }
    type SeriesEntry = { label: string; data: { b: number; deltaR: number }[] };
    const series: SeriesEntry[] = [];
    files.forEach((file, fIdx) => {
      const blm = resolveBlm(file, fIdx);
      (file.results || []).forEach(res => {
        series.push({ label: `${file.name}|BLM=${blm}|${res.columnName}`, data: res.plotData });
      });
    });
    if (series.length === 0) return;

    const headerRow: string[] = [];
    series.forEach(s => {
      headerRow.push(`"${s.label}|B(T)"`);
      headerRow.push(`"${s.label}|DeltaR"`);
    });

    const maxRows = Math.max(...series.map(s => s.data.length));
    const rows: string[] = [headerRow.join(',')];
    for (let i = 0; i < maxRows; i++) {
      const rowData: string[] = [];
      series.forEach(s => {
        const pt = s.data[i];
        if (pt) { rowData.push(String(pt.b), String(pt.deltaR)); }
        else { rowData.push('', ''); }
      });
      rows.push(rowData.join(','));
    }
    downloadCSV(rows.join('\n'), 'processed_data_deltaR_all_files.csv');
  };

  // Export oscillatory ΔR vs 1/B — same wide, file-by-file layout as above.
  const exportOscillatory1overBCSV = () => {
    if (allRawSeries.length === 0) return;
    const headerRow: string[] = [];
    allRawSeries.forEach(s => {
      headerRow.push(`"${s.fileName}|BLM=${s.blm}|${s.columnName}|1/B"`);
      headerRow.push(`"${s.fileName}|BLM=${s.blm}|${s.columnName}|DeltaR"`);
    });
    const maxRows = Math.max(...allRawSeries.map(s => s.invBData.length));
    const rows: string[] = [headerRow.join(',')];
    for (let i = 0; i < maxRows; i++) {
      const rowData: string[] = [];
      allRawSeries.forEach(s => {
        const pt = s.invBData[i];
        if (pt) { rowData.push(String(pt.invB), String(pt.val)); }
        else { rowData.push('', ''); }
      });
      rows.push(rowData.join(','));
    }
    downloadCSV(rows.join('\n'), 'oscillatory_1overB_all_files.csv');
  };

  // Raw R vs B, interpolated onto one shared B(T) grid (like FFT Assemble
  // does for spectra) — a single B column on the left, then one R column per
  // file/series, ordered by angle or temperature.
  const exportRawAlignedCSV = () => {
    if (fileNumericViews.length === 0) return;
    const dataLo = Math.min(...fileNumericViews.map(v => v.bMin));
    const dataHi = Math.max(...fileNumericViews.map(v => v.bMax));
    const lo = typeof minB === 'number' ? Math.max(minB, dataLo) : dataLo;
    const hi = typeof maxB === 'number' ? Math.min(maxB, dataHi) : dataHi;
    if (!(hi > lo) || !isFinite(lo) || !isFinite(hi)) return;

    const N = 500;
    const step = (hi - lo) / (N - 1);

    const entries = fileNumericViews
      .map((v, fIdx) => ({ v, cols: targetColsFor(v.file), ...alignedExportTag(v.file, fIdx) }))
      .filter(e => e.cols.length > 0 && e.v.bVals.length > 0)
      .sort((a, b) => a.sortVal - b.sortVal);
    if (entries.length === 0) return;

    const headerRow = ['B(T)'];
    entries.forEach(({ v, cols, label }) => cols.forEach(c => headerRow.push(`"${label}_${v.file.name}_${c}"`)));

    const rows = [headerRow.join(',')];
    for (let i = 0; i < N; i++) {
      const b = lo + i * step;
      const row = [b.toFixed(6)];
      entries.forEach(({ v, cols }) => {
        cols.forEach(c => {
          const val = lerpFast(v.bVals, v.colVals(c), b);
          row.push(isFinite(val) ? String(val) : '');
        });
      });
      rows.push(row.join(','));
    }
    downloadCSV(rows.join('\n'), 'raw_aligned_all_files.csv');
  };

  // ΔR (polynomial background removed) vs B — same shared-B(T)-grid layout as
  // exportRawAlignedCSV, but the background-subtracted value instead of the
  // raw one. Needs "Process All" to have run (uses res.plotData for ΔR).
  const exportBackgroundSubtractedAlignedCSV = () => {
    const hasResults = files.some(f => f.results && f.results.length > 0);
    if (!hasResults) {
      setError('"Process All Files" 를 먼저 실행하세요.');
      return;
    }
    let dataLo = Infinity, dataHi = -Infinity;
    files.forEach(f => (f.results || []).forEach(r => r.plotData.forEach(p => {
      if (p.b < dataLo) dataLo = p.b;
      if (p.b > dataHi) dataHi = p.b;
    })));
    if (!(dataHi > dataLo)) return;
    const lo = typeof minB === 'number' ? Math.max(minB, dataLo) : dataLo;
    const hi = typeof maxB === 'number' ? Math.min(maxB, dataHi) : dataHi;
    if (!(hi > lo)) return;

    const N = 500;
    const step = (hi - lo) / (N - 1);

    type SeriesEntry = { label: string; sortVal: number; bVals: Float64Array; deltaRVals: Float64Array };
    const series: SeriesEntry[] = [];
    files.forEach((file, fIdx) => {
      const { sortVal, label } = alignedExportTag(file, fIdx);
      (file.results || []).forEach(res => {
        const sorted = [...res.plotData].sort((a, b) => a.b - b.b);
        if (sorted.length < 2) return;
        series.push({
          label: `${label}_${file.name}_${res.columnName}`, sortVal,
          bVals: Float64Array.from(sorted.map(p => p.b)),
          deltaRVals: Float64Array.from(sorted.map(p => p.deltaR)),
        });
      });
    });
    if (series.length === 0) return;
    series.sort((a, b) => a.sortVal - b.sortVal);

    const headerRow = ['B(T)', ...series.map(s => `"${s.label}"`)];
    const rows = [headerRow.join(',')];
    for (let i = 0; i < N; i++) {
      const b = lo + i * step;
      const row = [b.toFixed(6)];
      series.forEach(s => {
        const val = lerpFast(s.bVals, s.deltaRVals, b);
        row.push(isFinite(val) ? String(val) : '');
      });
      rows.push(row.join(','));
    }
    downloadCSV(rows.join('\n'), 'background_subtracted_aligned_all_files.csv');
  };

  // Oscillatory ΔR vs 1/B, interpolated onto one shared 1/B grid — a single
  // 1/B column on the left, then one ΔR column per file/column, ordered by
  // angle or temperature. Needs "Process All" to have run (uses res.plotData).
  const exportOscillatoryAlignedCSV = () => {
    const hasResults = files.some(f => f.results && f.results.length > 0);
    if (!hasResults) {
      setError('"Process All Files" 를 먼저 실행하세요.');
      return;
    }
    let dataLo = Infinity, dataHi = -Infinity;
    files.forEach(f => (f.results || []).forEach(r => r.plotData.forEach(p => {
      if (p.invB < dataLo) dataLo = p.invB;
      if (p.invB > dataHi) dataHi = p.invB;
    })));
    if (!(dataHi > dataLo)) return;
    const lo = typeof maxB === 'number' && maxB > 0 ? Math.max(1 / maxB, dataLo) : dataLo;
    const hi = typeof minB === 'number' && minB > 0 ? Math.min(1 / minB, dataHi) : dataHi;
    if (!(hi > lo)) return;

    const N = interpPoints > 0 ? interpPoints : 1000;
    const step = (hi - lo) / (N - 1);

    type SeriesEntry = { label: string; sortVal: number; grid: Float64Array };
    const series: SeriesEntry[] = [];
    files.forEach((file, fIdx) => {
      const { sortVal, label } = alignedExportTag(file, fIdx);
      (file.results || []).forEach(res => {
        const sorted = [...res.plotData].sort((a, b) => a.invB - b.invB).map(p => ({ invB: p.invB, deltaR: p.deltaR }));
        if (sorted.length < 2) return;
        series.push({ label: `${label}_${file.name}_${res.columnName}`, sortVal, grid: linearInterpUniform(sorted, lo, step, N) });
      });
    });
    if (series.length === 0) return;
    series.sort((a, b) => a.sortVal - b.sortVal);

    const headerRow = ['1_over_B', ...series.map(s => `"${s.label}"`)];
    const rows = [headerRow.join(',')];
    for (let i = 0; i < N; i++) {
      const invB = lo + i * step;
      const row = [invB.toFixed(6)];
      series.forEach(s => row.push(String(s.grid[i])));
      rows.push(row.join(','));
    }
    downloadCSV(rows.join('\n'), 'oscillatory_1overB_aligned_all_files.csv');
  };

  // Export the L-K fit series + fitted parameters.
  // Data rows (one per file) plus the smooth fitted curve (temperature,
  // amplitude) side by side in the same wide CSV — the curve has many more
  // rows than there are files, so the two run to their own lengths and the
  // shorter side pads with blanks, same convention as the other "aligned"
  // exports in this app.
  const exportLKFitCSV = () => {
    if (!lkFitResult || lkFitResult.points.length === 0) return;
    const { points, mStar, A0, r2 } = lkFitResult;
    const curve = lkFitCurve;
    const rows: string[] = ['FileName,Temperature(K),Amplitude,B_eff(T),Predicted,m_star(m0),A0,R2,Fit_Temperature(K),Fit_Amplitude'];
    const maxRows = Math.max(points.length, curve.length);
    for (let i = 0; i < maxRows; i++) {
      const p = points[i];
      const c = curve[i];
      let dataCols: (string | number)[];
      if (p) {
        const X = mStar != null ? 14.69 * mStar * p.temp / p.bEff : NaN;
        const pred = mStar != null && A0 != null ? A0 * lkDampingFactor(X) : NaN;
        dataCols = [`"${p.fileName}"`, p.temp, p.amp, p.bEff, isFinite(pred) ? pred : '', mStar ?? '', A0 ?? '', r2 ?? ''];
      } else {
        dataCols = ['', '', '', '', '', '', '', ''];
      }
      const curveCols: (string | number)[] = c ? [c.temp, c.ampFit] : ['', ''];
      rows.push([...dataCols, ...curveCols].join(','));
    }
    downloadCSV(rows.join('\n'), `lk_fit_${lkFreq ?? 'freq'}T_${lkColumn || 'auto'}.csv`);
  };

  // Export full-resolution FFT spectra for all files — wide format
  const exportFFTSpectraCSV = () => {
    const hasResults = files.some(f => f.results && f.results.length > 0);
    if (!hasResults) {
      setError('"Process All Files" 를 먼저 실행하세요.');
      return;
    }
    type SeriesEntry = { label: string; fft: FFTResult[] };
    const series: SeriesEntry[] = [];
    files.forEach((file, fIdx) => {
      const blm = resolveBlm(file, fIdx);
      (file.results || []).forEach(res => {
        series.push({ label: `${file.name}|BLM=${blm}|${res.columnName}`, fft: res.fftResults });
      });
    });
    if (series.length === 0) return;

    const refLen = Math.max(...series.map(s => s.fft.length));
    const refFFT = series.find(s => s.fft.length === refLen)!.fft;

    const header = ['Frequency(T)', ...series.map(s => `"${s.label}"`)].join(',');
    const rows: string[] = [header];
    for (let i = 0; i < refLen; i++) {
      const row: string[] = [refFFT[i].frequency.toFixed(6)];
      series.forEach(s => {
        row.push(i < s.fft.length ? String(s.fft[i].amplitude) : '');
      });
      rows.push(row.join(','));
    }
    downloadCSV(rows.join('\n'), 'fft_spectra_all_files.csv');
  };

  // Export detected peaks per file in wide format (one column per file/BLM)
  const exportPeaksWideCSV = () => {
    const hasResults = files.some(f => f.results && f.results.length > 0);
    if (!hasResults) {
      setError('"Process All Files" 를 먼저 실행하세요.');
      return;
    }
    type ColEntry = { label: string; peaks: { freq: number; amp: number }[] };
    const cols: ColEntry[] = [];
    files.forEach((file, fIdx) => {
      const blm = resolveBlm(file, fIdx);
      (file.results || []).forEach(res => {
        cols.push({
          label: `${file.name}|BLM=${blm}|${res.columnName}`,
          peaks: res.detectedPeaks.map(p => ({ freq: p.frequency, amp: p.amplitude })),
        });
      });
    });
    if (cols.length === 0) return;

    const maxPeaks = Math.max(...cols.map(c => c.peaks.length));
    const headerFreq = cols.map(c => `"${c.label}|Freq(T)"`).join(',');
    const headerAmp  = cols.map(c => `"${c.label}|Amplitude"`).join(',');
    const rows: string[] = [headerFreq + ',' + headerAmp];
    for (let i = 0; i < maxPeaks; i++) {
      const freqPart = cols.map(c => i < c.peaks.length ? c.peaks[i].freq.toFixed(4) : '').join(',');
      const ampPart  = cols.map(c => i < c.peaks.length ? c.peaks[i].amp.toExponential(6) : '').join(',');
      rows.push(freqPart + ',' + ampPart);
    }
    downloadCSV(rows.join('\n'), 'peaks_wide_all_files.csv');
  };

  const exportRawDataCSV = () => {
    if (files.length === 0) return;

    // 1. Identify columns for each file
    const fileData = files.map((file, fIdx) => {
      const bCol = file.columns.find(c => c === selectedBColumn) || 
                   file.columns.find(c => c.toLowerCase() === selectedBColumn.toLowerCase()) ||
                   file.columns.find(k => {
                     const low = k.toLowerCase();
                     return (low === 'b' || low.startsWith('b ') || low.startsWith('b_') || low.includes('field') || low.includes('magnetic')) && 
                            !low.includes('blm') && !low.includes('angle') && !low.includes('position');
                   }) ||
                   file.columns.find(k => {
                     const low = k.toLowerCase();
                     return low.includes('b') && !low.includes('blm') && !low.includes('angle') && !low.includes('position');
                   }) ||
                   file.columns[0];

      let rCols: string[] = [];
      if (selectedPrefix) {
        rCols = file.columns.filter(c => c.startsWith(selectedPrefix + '_'));
      } else {
        rCols = selectedRColumns.map(target => {
            return file.columns.find(c => c === target) ||
                   file.columns.find(c => c.toLowerCase() === target.toLowerCase());
        }).filter((c): c is string => !!c);
      }
      rCols = dropQuadrature(rCols, file.columns);

      // Get BLM value
      let blmValue = 0;
      if (readAngleFromTitle) {
        // Extract the first number in the filename (e.g. "26down.csv" → 26)
        const fromName = angleFromFileName(file.name);
        blmValue = fromName != null ? fromName : fIdx + 1;
      } else if (blmMode === 'step') {
        blmValue = fIdx + 1;
      } else {
        const blmCol = file.columns.find(c => 
          c.toLowerCase().includes('blm') || 
          c.toLowerCase().includes('position') || 
          c.toLowerCase().includes('angle')
        );
        let blmStr = '0';
        if (blmCol && file.data.length > 0) {
          blmStr = String(file.data[0][blmCol]);
        } else {
          const blmEntry = Object.entries(file.metadata).find(([k]) => 
            k.toLowerCase().includes('position') || 
            k.toLowerCase().includes('blm') || 
            k.toLowerCase().includes('angle')
          );
          blmStr = blmEntry ? String(blmEntry[1]) : '0';
        }
        const numMatch = blmStr.match(/[-+]?[0-9]*\.?[0-9]+/);
        blmValue = numMatch ? parseFloat(numMatch[0]) : parseFloat(blmStr);
        if (isNaN(blmValue)) blmValue = fIdx + 1;
      }
      const finalBlm = Number(((blmValue / blmPerDegree) + blmOffset).toFixed(2));

      return { file, bCol, rCols, finalBlm };
    });

    // 2. Build CSV Header
    let csv = "";
    const headerRow: string[] = [];
    fileData.forEach(fd => {
      fd.rCols.forEach(rCol => {
        headerRow.push(`BLM ${fd.finalBlm} (Field)`);
        headerRow.push(`BLM ${fd.finalBlm} (Raw: ${rCol})`);
      });
    });
    csv += headerRow.join(",") + "\n";

    // 3. Find max rows
    const maxRows = Math.max(...files.map(f => f.data.length));

    // 4. Build CSV Body
    for (let i = 0; i < maxRows; i++) {
      const rowData: string[] = [];
      fileData.forEach(fd => {
        fd.rCols.forEach(rCol => {
          const row = fd.file.data[i];
          if (row) {
            rowData.push(String(row[fd.bCol] || ""));
            rowData.push(String(row[rCol] || ""));
          } else {
            rowData.push("");
            rowData.push("");
          }
        });
      });
      csv += rowData.join(",") + "\n";
    }

    // 5. Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `raw_data_all_files.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Branch Analysis State
  const [branchTolerance, setBranchTolerance] = useState<number>(50); // Frequency tolerance in Tesla
  const [tempBranchTolerance, setTempBranchTolerance] = useState<number>(50);
  const [minPeaksPerBranch, setMinPeaksPerBranch] = useState<number>(3);
  const [tempMinPeaksPerBranch, setTempMinPeaksPerBranch] = useState<number>(3);
  const [selectedBranchIds, setSelectedBranchIds] = useState<Set<string>>(new Set());
  const [harmonicTolerance, setHarmonicTolerance] = useState<number>(0.02); // 2% frequency tolerance
  const [selectedFileForHarmonics, setSelectedFileForHarmonics] = useState<string>('');
  const [fundamentalFreq, setFundamentalFreq] = useState<number | null>(null);
  const [lkColumn, setLkColumn] = useState<string>('');
  const [lkFreq, setLkFreq] = useState<number | null>(null);
  const [rangeScanMinB, setRangeScanMinB] = useState<number | null>(null);
  const [rangeScanMaxB, setRangeScanMaxB] = useState<number | null>(null);
  // Drafts the inputs actually bind to, so typing stays responsive — the
  // committed rangeScanMinB/MaxB above (which drive the full FFT re-batch)
  // only update ~400ms after the user stops typing, instead of on every
  // keystroke.
  const [rangeScanMinBDraft, setRangeScanMinBDraft] = useState<number>(0);
  const [rangeScanMaxBDraft, setRangeScanMaxBDraft] = useState<number>(1);
  // Same live-preview pattern as above, for two more spots that want an
  // instant "change the B-range, see the FFT move" control without a trip
  // to the FFT Range tab: the Dashboard's "All Detected Frequencies vs BLM"
  // (all files) and the per-file "3. FFT Power Spectrum" (just that file).
  const dashLiveRange = useDebouncedRange();
  const fileLiveRange = useDebouncedRange();
  // Seed both live-range controls with the currently committed B-Field ROI
  // instead of leaving them at the hook's hardcoded 0–1 default, so they
  // already show a sensible range before the user has touched them. Only
  // runs while a control is untouched ("not active") — an in-progress live
  // override is never clobbered by this.
  useEffect(() => {
    const fallbackMin = typeof minB === 'number' ? minB : 0;
    const fallbackMax = typeof maxB === 'number' ? maxB : 1;
    if (!dashLiveRange.active) dashLiveRange.reset(fallbackMin, fallbackMax);
    if (!fileLiveRange.active) fileLiveRange.reset(fallbackMin, fallbackMax);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [minB, maxB]);
  // Once locked, new uploads keep the chosen R-column prefix instead of
  // re-running the auto-detect heuristic — otherwise every added file (e.g.
  // building up a temperature series one file at a time) can silently drift
  // the prefix to whatever that file's own columns suggest.
  const [prefixLocked, setPrefixLocked] = useState(false);
  // Which bucket of the active orientation is currently loaded into `files` —
  // an orientation holds one angle-sweep bucket and one temperature-sweep
  // bucket side by side, and this picks which one is "live" right now.
  const [activeBucket, setActiveBucket] = useState<'angle' | 'temperature'>('angle');
  // Temperature sweeps run at different fixed angles are kept as separate
  // named groups inside the temperature bucket; this is which one is loaded.
  const [activeTempGroupId, setActiveTempGroupId] = useState<string>('');
  const [tempGroupLabelDraft, setTempGroupLabelDraft] = useState<string>('');
  const [showAddTempGroup, setShowAddTempGroup] = useState(false);
  const [editingTempGroupId, setEditingTempGroupId] = useState<string | null>(null);
  const [editTempGroupLabelDraft, setEditTempGroupLabelDraft] = useState<string>('');
  const [branches, setBranches] = useState<{
    id: string;
    name: string;
    peaks: { blm: number; freq: number; amp: number; fileName: string; id: string }[];
    fit?: { f0: number; r2: number; cosOffset: number; type: '1/cos' };
  }[]>([]);
  const [others, setOthers] = useState<{ blm: number; freq: number; amp: number; fileName: string; id: string }[]>([]);

  const prefixes = useMemo(() => {
    const allCols = files.flatMap(f => f.columns);
    const pSet = new Set<string>();
    // Output2 never drives prefix detection when Output1 is present
    dropQuadrature(allCols).forEach(col => {
      if (col.includes('_')) {
        const p = col.substring(0, col.lastIndexOf('_'));
        if (p) pSet.add(p);
      }
    });
    return Array.from(pSet).sort();
  }, [files, dropQuadrature]);

  // Which data columns the current selection maps to, for one file.
  const targetColsFor = useCallback((file: FileEntry): string[] => {
    if (selectedRawColumn) {
      const isPrefix = prefixes.includes(selectedRawColumn);
      return isPrefix
        ? dropQuadrature(file.columns.filter((c: string) => c.startsWith(selectedRawColumn + '_')))
        : file.columns.includes(selectedRawColumn) ? [selectedRawColumn] : [];
    }
    if (selectedPrefix) return dropQuadrature(file.columns.filter((c: string) => c.startsWith(selectedPrefix + '_')));
    // No prefix: fall back to the individually selected columns, so the Raw and
    // Torque tabs still have data for datasets that can't share one prefix.
    return dropQuadrature(selectedRColumns.filter((c: string) => file.columns.includes(c)), file.columns);
  }, [selectedRawColumn, selectedPrefix, prefixes, selectedRColumns, dropQuadrature]);

  // One parsed, B-sorted numeric view per file — built once and shared by every
  // Raw-tab memo below. fieldSliceData, rawPlotDisplayData and cosScanPrepared
  // each used to re-sort every file with a
  //   parseFloat(String(a[bCol])) - parseFloat(String(b[bCol]))
  // comparator (two string parses per comparison), and re-parse every column,
  // on every column or B-slice change. Here B is parsed once, an index
  // permutation is sorted numerically, and column arrays are cached on demand.
  type FileNumericView = {
    file: FileEntry; bCol: string; blm: number;
    sortedRows: any[]; bVals: Float64Array; bMin: number; bMax: number;
    colVals: (name: string) => Float64Array;
  };
  const fileNumericViews = useMemo<FileNumericView[]>(() => {
    return files.map((file: FileEntry, fIdx: number) => {
      const bCol = resolveBCol(file);
      const blm = resolveBlm(file, fIdx);
      const n = file.data.length;
      const rawB = new Float64Array(n);
      for (let i = 0; i < n; i++) rawB[i] = parseFloat(String(file.data[i][bCol]));
      const order = new Uint32Array(n);
      for (let i = 0; i < n; i++) order[i] = i;
      order.sort((x, y) => rawB[x] - rawB[y]);
      const sortedRows: any[] = new Array(n);
      const bVals = new Float64Array(n);
      for (let i = 0; i < n; i++) { const src = order[i]; sortedRows[i] = file.data[src]; bVals[i] = rawB[src]; }
      const cache = new Map<string, Float64Array>();
      const colVals = (name: string): Float64Array => {
        let arr = cache.get(name);
        if (!arr) {
          arr = new Float64Array(n);
          for (let i = 0; i < n; i++) arr[i] = parseFloat(String(sortedRows[i][name]));
          cache.set(name, arr);
        }
        return arr;
      };
      return {
        file, bCol, blm, sortedRows, bVals, colVals,
        bMin: n > 0 ? bVals[0] : Infinity,
        bMax: n > 0 ? bVals[n - 1] : -Infinity,
      };
    });
  }, [files, resolveBCol, resolveBlm]);

  // For each target B value, interpolate R in each file and return vs BLM
  const fieldSliceData = useMemo<Record<number, { blm: number; val: number; fileName: string }[]>>(() => {
    if (fileNumericViews.length === 0) return {};
    const prepared = fileNumericViews.map(v => ({
      v, cols: targetColsFor(v.file).map(c => v.colVals(c)),
    }));
    const result: Record<number, { blm: number; val: number; fileName: string }[]> = {};
    fieldSliceValues.forEach((targetB: number) => {
      const points: { blm: number; val: number; fileName: string }[] = [];
      prepared.forEach(({ v, cols }) => {
        if (cols.length === 0 || v.bVals.length === 0) return;
        let sum = 0, cnt = 0;
        for (const cv of cols) {
          const val = lerpFast(v.bVals, cv, targetB);
          if (isFinite(val)) { sum += val; cnt++; }
        }
        if (cnt === 0) return;
        points.push({ blm: v.blm, val: sum / cnt, fileName: v.file.name });
      });
      points.sort((a, b) => a.blm - b.blm);
      result[targetB] = points;
    });
    return result;
  }, [fileNumericViews, fieldSliceValues, targetColsFor]);

  // Reactive "R vs BLM" — interpolates at rawBSliceTarget, or uses max B when null.
  // Falls back to pre-computed rawPlotData only when no files are loaded.
  const rawPlotDisplayData = useMemo<{ x: number; y: number; fileName: string; columnName: string }[]>(() => {
    if (fileNumericViews.length === 0) return rawPlotData;
    const points: { x: number; y: number; fileName: string; columnName: string }[] = [];
    fileNumericViews.forEach(v => {
      const n = v.bVals.length;
      if (n === 0) return;
      targetColsFor(v.file).forEach((col: string) => {
        const cv = v.colVals(col);
        const val = rawBSliceTarget === null
          ? cv[n - 1]                                   // max B
          : lerpFast(v.bVals, cv, rawBSliceTarget);
        if (isFinite(val)) points.push({ x: v.blm, y: val, fileName: v.file.name, columnName: col });
      });
    });
    return points;
  }, [fileNumericViews, targetColsFor, rawBSliceTarget, rawPlotData]);

  // Only the BLM-filtered subset — must match what the scatter actually renders
  const rawPlotFitData = useMemo(
    () => rawPlotDisplayData.filter(d => activeBlms.size === 0 || activeBlms.has(d.x)),
    [rawPlotDisplayData, activeBlms],
  );

  // Cosine fit at the current B slice
  const cosFitAtSlice = useMemo(() => {
    const hArr = [...selectedHarmonics].sort((a, b) => a - b);
    if (hArr.length === 0 || rawPlotFitData.length < 1 + 2 * hArr.length) return null;
    return fitCosineSeries(rawPlotFitData.map(d => ({ x: d.x, y: d.y })), hArr);
  }, [rawPlotFitData, selectedHarmonics]);

  const cosFitCurve = useMemo(() => {
    if (!cosFitAtSlice || rawPlotFitData.length === 0) return [];
    const xs = rawPlotFitData.map(d => d.x);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    return Array.from({ length: 301 }, (_, i) => {
      const x = xMin + (i / 300) * (xMax - xMin);
      return { x, y: cosFitAtSlice.predict(x) };
    });
  }, [cosFitAtSlice, rawPlotFitData]);

  // Column selection projected onto the shared per-file views — the sort and the
  // B parse already happened once in fileNumericViews, and each column's numeric
  // array is cached there, so changing the column no longer re-sorts anything.
  type ScanPrep = { bCol: string; blm: number; targetCols: string[]; sortedRows: any[]; bVals: Float64Array; colNumVals: Float64Array[]; bMin: number; bMax: number };
  const cosScanPrepared = useMemo<ScanPrep[]>(() => {
    return fileNumericViews.map(v => {
      const targetCols = targetColsFor(v.file);
      return {
        bCol: v.bCol, blm: v.blm, targetCols,
        sortedRows: v.sortedRows,
        bVals: v.bVals,
        colNumVals: targetCols.map(c => v.colVals(c)),
        bMin: v.bMin, bMax: v.bMax,
      };
    });
  }, [fileNumericViews, targetColsFor]);

  // Shared field × angle lattice on 0.2 T steps. Built once with a single
  // monotone forward-pointer sweep per column and reused by every angular
  // scan below — each of those previously re-interpolated the whole dataset
  // with a per-point binary search, tripling identical work.
  type FieldAngleGrid = { bSteps: Float64Array; angles: Float64Array; values: Float64Array; S: number; A: number };
  const fieldAngleGrid = useMemo<FieldAngleGrid | null>(() => {
    if (!scansArmed || cosScanPrepared.length === 0) return null;
    let bMin = Infinity, bMax = -Infinity;
    cosScanPrepared.forEach(p => { if (p.bMin < bMin) bMin = p.bMin; if (p.bMax > bMax) bMax = p.bMax; });
    if (!isFinite(bMin) || !isFinite(bMax) || bMax <= bMin) return null;
    const scale = 5; // 0.2 T steps
    const startIdx = Math.ceil(bMin * scale), endIdx = Math.floor(bMax * scale);
    const S = endIdx - startIdx + 1;
    if (S <= 0) return null;
    const A = cosScanPrepared.length;
    const bSteps = new Float64Array(S);
    for (let s = 0; s < S; s++) bSteps[s] = (startIdx + s) / scale;
    const angles = new Float64Array(A);
    const values = new Float64Array(S * A).fill(NaN);

    cosScanPrepared.forEach(({ blm, bVals, colNumVals }, a) => {
      angles[a] = blm;
      const nRows = bVals.length, nCols = colNumVals.length;
      if (nRows === 0 || nCols === 0) return;
      const first = bVals[0], last = bVals[nRows - 1];
      let ptr = 0; // bSteps is ascending, so the bracket index only moves forward
      for (let s = 0; s < S; s++) {
        const b = bSteps[s];
        let lo: number, t: number;
        // Clamp outside this angle's own field range, matching lerpFast
        if (b <= first) { lo = 0; t = 0; }
        else if (b >= last) { lo = nRows - 1; t = 0; }
        else {
          while (ptr < nRows - 2 && bVals[ptr + 1] <= b) ptr++;
          lo = ptr;
          const bLo = bVals[lo], bHi = bVals[lo + 1];
          t = bHi === bLo ? 0 : (b - bLo) / (bHi - bLo);
        }
        let sum = 0, cnt = 0;
        for (let c = 0; c < nCols; c++) {
          const cv = colNumVals[c];
          const vLo = cv[lo];
          let v = vLo;
          if (t !== 0) {
            const vHi = cv[lo + 1];
            if (isFinite(vLo) && isFinite(vHi)) v = vLo + t * (vHi - vLo);
          }
          if (isFinite(v)) { sum += v; cnt++; }
        }
        if (cnt > 0) values[s * A + a] = sum / cnt;
      }
    });
    return { bSteps, angles, values, S, A };
  }, [scansArmed, cosScanPrepared]);

  // Runs an OLS fit at every field step, reusing one factorized design matrix
  // per distinct set of contributing angles (in practice: exactly one).
  const scanGrid = useCallback(<T,>(
    grid: FieldAngleGrid,
    minPts: number,
    design: (angles: Float64Array) => { X: Float64Array; p: number },
    emit: (b: number, beta: Float64Array, model: OlsModel, y: Float64Array) => T | null,
  ): T[] => {
    const { bSteps, angles, values, S, A } = grid;
    const models = new Map<string, OlsModel | null>();
    const out: T[] = [];
    const yBuf = new Float64Array(A);
    const idxBuf = new Int32Array(A);
    let betaBuf: Float64Array | null = null;
    for (let s = 0; s < S; s++) {
      let cnt = 0, key = '';
      for (let a = 0; a < A; a++) {
        const v = values[s * A + a];
        if (isFinite(v)) { yBuf[cnt] = v; idxBuf[cnt] = a; cnt++; key += a + ','; }
      }
      if (cnt < minPts) continue;
      let model = models.get(key);
      if (model === undefined) {
        const sub = new Float64Array(cnt);
        for (let i = 0; i < cnt; i++) sub[i] = angles[idxBuf[i]];
        const { X, p } = design(sub);
        model = buildOlsModel(X, cnt, p);
        models.set(key, model);
      }
      if (!model) continue;
      if (!betaBuf || betaBuf.length !== model.p) betaBuf = new Float64Array(model.p);
      const y = yBuf.subarray(0, cnt);
      const beta = olsSolve(model, y, betaBuf);
      const rec = emit(bSteps[s], beta, model, y as Float64Array);
      if (rec !== null) out.push(rec);
    }
    return out;
  }, []);

  // Harmonic amplitudes scanned at 0.2 T intervals — gated until Raw tab is first opened
  const cosHarmonicScanData = useMemo<{ b: number; A1: number; A2: number; A3: number; A4: number; A6: number }[]>(() => {
    if (!fieldAngleGrid) return [];
    const hArr = [...selectedHarmonics].sort((x, y) => x - y);
    if (hArr.length === 0) return [];
    return scanGrid(
      fieldAngleGrid,
      1 + 2 * hArr.length,
      (angles) => cosineDesign(angles, hArr),
      (b, beta) => {
        const rec = { b, A1: 0, A2: 0, A3: 0, A4: 0, A6: 0 };
        hArr.forEach((h, i) => {
          const C = beta[1 + 2 * i], S = beta[2 + 2 * i];
          if (h !== 12) (rec as any)[`A${h}`] = Math.sqrt(C * C + S * S);
        });
        return rec;
      },
    );
  }, [fieldAngleGrid, selectedHarmonics, scanGrid]);

  const cosHarmonicScanDataB2 = useMemo(
    () => cosHarmonicScanData.map(d => ({ ...d, b2: d.b * d.b })),
    [cosHarmonicScanData]
  );

  // Explains an empty Harmonic-Amplitudes chart instead of showing a bare
  // "Computing…" — reports the first stage of the pipeline that produced nothing.
  const harmonicScanDiag = useMemo<string | null>(() => {
    if (cosHarmonicScanDataB2.length > 0) return null;
    if (files.length === 0) return 'No files loaded.';
    if (!scansArmed) return 'Initialising…';
    const hArr = [...selectedHarmonics].sort((x, y) => x - y);
    if (hArr.length === 0) return 'No harmonics selected — pick at least n=1 above.';
    const needed = 1 + 2 * hArr.length;
    const withCols = cosScanPrepared.filter(p => p.colNumVals.length > 0 && p.bVals.length > 0);
    if (withCols.length === 0) {
      return `No numeric column selected. Pick a column or prefix in the header dropdown (${files.length} file${files.length !== 1 ? 's' : ''} loaded).`;
    }
    if (!fieldAngleGrid) return 'Field range is empty or degenerate across the loaded files.';
    // How many angles actually contribute at the best field step?
    const { values, S, A } = fieldAngleGrid;
    let best = 0;
    for (let s = 0; s < S; s++) {
      let cnt = 0;
      for (let a = 0; a < A; a++) if (isFinite(values[s * A + a])) cnt++;
      if (cnt > best) best = cnt;
    }
    if (best < needed) {
      return `Need ≥${needed} BLM angles for harmonics [${hArr.join(', ')}], but only ${best} contribute. Load more angles, or deselect high harmonics (each n costs 2 angles).`;
    }
    return `Fit is singular at every field step with ${best} angles and harmonics [${hArr.join(', ')}] — the angles are likely evenly spaced such that a harmonic aliases. Try deselecting the highest n.`;
  }, [cosHarmonicScanDataB2, files, scansArmed, selectedHarmonics, cosScanPrepared, fieldAngleGrid]);

  // ── Torque Analysis memos ──────────────────────────────────────

  const tauScaleFields = useMemo(
    () => tauScaleFieldsStr.split(',').map(s => parseFloat(s.trim())).filter(isFinite),
    [tauScaleFieldsStr],
  );

  interface FieldSweepEntry {
    blm: number; color: string;
    hTau: { h: number; tau: number }[];
    tauOverH: { h: number; v: number }[];
    lowFit: { a: number; b: number; c: number; r2: number; predict: (h: number) => number } | null;
    hSf: number; tauSat: number | null;
  }
  // Split in two: the per-angle τ(H) curves depend only on the loaded data, so
  // editing H_sf limit / threshold re-runs just the cheap fit stage instead of
  // rebuilding (and re-averaging) every raw sweep on each keystroke.
  type SweepBase = { blm: number; color: string; hTau: { h: number; tau: number }[]; tauOverH: { h: number; v: number }[] };
  const fieldSweepBase = useMemo<SweepBase[]>(() => {
    if (!scansArmed) return [];
    return cosScanPrepared.map(({ blm, bVals, colNumVals }, fIdx) => {
      const nCols = colNumVals.length;
      if (nCols === 0 || bVals.length < 4) return null;
      const hTau: { h: number; tau: number }[] = [];
      const tauOverH: { h: number; v: number }[] = [];
      for (let i = 0; i < bVals.length; i++) {
        const h = bVals[i];
        if (!(h > 0) || !isFinite(h)) continue;
        let sum = 0, cnt = 0;
        for (let c = 0; c < nCols; c++) {
          const v = colNumVals[c][i];
          if (isFinite(v)) { sum += v; cnt++; }
        }
        if (cnt === 0) continue;
        const tau = sum / cnt;
        hTau.push({ h, tau });
        if (h > 0.1) tauOverH.push({ h, v: tau / h });
      }
      if (hTau.length < 3) return null;
      // bVals is pre-sorted, so hTau stays ascending in h — relied on below.
      return { blm, color: `hsl(${(fIdx * 137) % 360}, 55%, 55%)`, hTau, tauOverH };
    }).filter(Boolean) as SweepBase[];
  }, [scansArmed, cosScanPrepared]);

  const fieldSweepByAngle = useMemo<FieldSweepEntry[]>(() => {
    return fieldSweepBase.map(({ blm, color, hTau, tauOverH }) => {
      // hTau ascending in h → the low-field window is a prefix
      let nLow = 0;
      while (nLow < hTau.length && hTau[nLow].h <= tauHsfLimit) nLow++;
      const lowPts = hTau.slice(0, nLow);
      const lowFit = lowPts.length >= 3 ? fitPolynomialH(lowPts) : null;
      let hSf = tauHsfLimit;
      if (lowFit && nLow > 0) {
        let maxTau = 0;
        for (let i = 0; i < nLow; i++) {
          const a = Math.abs(hTau[i].tau);
          if (a > maxTau) maxTau = a;
        }
        const denom = maxTau > 0 ? maxTau : 1;
        for (let i = nLow - 1; i >= 0; i--) {   // walk down from the highest field
          const { h, tau } = hTau[i];
          if (Math.abs(tau - lowFit.predict(h)) / denom <= tauResidThreshold) { hSf = h; break; }
        }
      }
      // Everything above hSf is the ascending suffix
      let iHigh = nLow;
      while (iHigh > 0 && hTau[iHigh - 1].h > hSf) iHigh--;
      const nHigh = hTau.length - iHigh;
      let tauSat: number | null = null;
      if (nHigh >= 3) {
        const take = Math.max(3, Math.floor(nHigh * 0.2));
        let sum = 0;
        for (let i = hTau.length - take; i < hTau.length; i++) sum += hTau[i].tau;
        tauSat = sum / take;
      }
      return { blm, color, hTau, tauOverH, lowFit, hSf, tauSat };
    });
  }, [fieldSweepBase, tauHsfLimit, tauResidThreshold]);

  // τ/H is drawn per angle from the full raw sweep, so the SVG node count is
  // (angles × rows) — tens of thousands of nodes is enough to make scrolling and
  // hovering crawl. 'fast' decimates for display only; exports stay full.
  const SWEEP_PLOT_MAX = 1200;
  const fieldSweepPlot = useMemo(() => {
    const total = fieldSweepByAngle.reduce((s, e) => s + e.tauOverH.length, 0);
    const series = fieldSweepByAngle.map(e => ({
      blm: e.blm, color: e.color,
      pts: sweepDensity === 'full' ? e.tauOverH : decimateMinMax(e.tauOverH, SWEEP_PLOT_MAX, p => p.v),
    }));
    return { series, total, drawn: series.reduce((s, e) => s + e.pts.length, 0) };
  }, [fieldSweepByAngle, sweepDensity]);

  interface FourierScanEntry { h: number; coeffs: { n: number; A: number; phiDeg: number }[]; r2: number; }
  const fourierScanData = useMemo<FourierScanEntry[]>(() => {
    if (!fieldAngleGrid) return [];
    return scanGrid(
      fieldAngleGrid,
      2 * tauFourierNMax,
      (angles) => sineFourierDesign(angles, tauFourierNMax),
      (h, beta, model, y) => ({
        h,
        coeffs: buildFourierResult(beta, tauFourierNMax).coeffs,
        r2: olsR2(model, y, beta).r2,
      }),
    );
  }, [fieldAngleGrid, tauFourierNMax, scanGrid]);

  const scalingVerifData = useMemo(() => {
    if (!scansArmed || cosScanPrepared.length === 0) return [];
    return tauScaleFields.map(h => {
      const pts = cosScanPrepared.map(({ blm, bVals, colNumVals }: any) => {
        if (colNumVals.length === 0 || bVals.length === 0) return null;
        const vals = colNumVals.map((cv: number[]) => lerpFast(bVals, cv, h)).filter(isFinite);
        if (vals.length === 0) return null;
        const tau = vals.reduce((a: number, v: number) => a + v, 0) / vals.length;
        return { alpha: blm, tau, tauOverH2: h > 0 ? tau / (h * h) : 0 };
      }).filter(Boolean) as { alpha: number; tau: number; tauOverH2: number }[];
      return { h, pts: pts.sort((a, b) => a.alpha - b.alpha) };
    }).filter(d => d.pts.length > 0);
  }, [scansArmed, cosScanPrepared, tauScaleFields]);

  const [error, setError] = useState<string | null>(null);
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'assemble' | 'raw' | 'torque' | 'landau' | 'branch' | 'harmonics' | 'symmetry' | 'lkfit' | 'fftrange' | string>('dashboard');
  const [symmetryCenter, setSymmetryCenter] = useState<number>(0);
  const [symmetryMaxFreq, setSymmetryMaxFreq] = useState<number>(5000);
  const [symmetryMinFreq, setSymmetryMinFreq] = useState<number>(0);
  const [appMode, setAppMode] = useState<'sdh'>('sdh');
  const [isProcessing, setIsProcessing] = useState(false);
  const [batchSummary, setBatchSummary] = useState<{ blm: number; freq: number; amp: number; fileName: string; id: string; temp: number }[]>([]);

  const isInitialLoad = useRef(true);
  // Set for a parameter change that must NOT wipe the results it just produced
  // (the batch's own B-window auto-correction writes minB/maxB mid-run).
  const suppressResultClear = useRef(false);

  // The angular field scans are gated behind this flag so they never run while
  // the user is on a tab that doesn't show them. Arm it the first time either
  // consumer tab opens — Raw needs it for the harmonic scan, Torque for the
  // field-sweep, Fourier and scaling sections.
  useEffect(() => {
    if (activeTab === 'raw' || activeTab === 'torque') setScansArmed(true);
  }, [activeTab]);

  // --- Local Auth (always signed in as local user) ---
  useEffect(() => {
    setUser(LOCAL_USER);
    setIsAuthReady(true);
  }, []);

  const handleLogin = async () => { /* no-op in local mode */ };
  const handleLogout = async () => { /* no-op in local mode */ };

  // --- Persistence: Load Preferences (localStorage) ---
  useEffect(() => {
    if (!isAuthReady || !user) return;

    const data = lsGetPrefs();
    if (Object.keys(data).length > 0 && isInitialLoad.current) {
      // Every restored value is written to BOTH the effective state and the
      // staged (temp) copy the sidebar displays. Restoring only the effective
      // half used to leave the two out of step, so the first "Apply All
      // Settings" click silently overwrote the restored values with whatever
      // factory defaults the inputs happened to be showing.
      if (data.selectedBColumn !== undefined) setSelectedBColumn(data.selectedBColumn);
      if (data.selectedRColumns !== undefined) setSelectedRColumns(data.selectedRColumns);
      if (data.minB !== undefined) { setMinB(data.minB); setTempMinB(data.minB); }
      if (data.maxB !== undefined) { setMaxB(data.maxB); setTempMaxB(data.maxB); }
      if (data.polyDegree !== undefined) { setPolyDegree(data.polyDegree); setTempPolyDegree(data.polyDegree); }
      if (data.interpPoints !== undefined) { setInterpPoints(data.interpPoints); setTempInterpPoints(data.interpPoints); }
      if (data.windowType !== undefined) { setWindowType(data.windowType); setTempWindowType(data.windowType); }
      if (data.spikeRemoval !== undefined) { setSpikeRemoval(data.spikeRemoval); setTempSpikeRemoval(data.spikeRemoval); }
      if (data.spikeKernel !== undefined) { setSpikeKernel(data.spikeKernel); setTempSpikeKernel(data.spikeKernel); }
      if (data.spikeThreshold !== undefined) { setSpikeThreshold(data.spikeThreshold); setTempSpikeThreshold(data.spikeThreshold); }
      if (data.smoothing !== undefined) { setSmoothing(data.smoothing); setTempSmoothing(data.smoothing); }
      if (data.smoothWindow !== undefined) { setSmoothWindow(data.smoothWindow); setTempSmoothWindow(data.smoothWindow); }
      if (data.freqThresholds !== undefined) { setFreqThresholds(data.freqThresholds); setTempFreqThresholds(data.freqThresholds); }
      if (data.complexMode !== undefined) { setComplexMode(data.complexMode); setTempComplexMode(data.complexMode); }
      if (data.phase !== undefined) { setPhase(data.phase); setTempPhase(data.phase); }
      if (data.useTargetInterval !== undefined) { setUseTargetInterval(data.useTargetInterval); setTempUseTargetInterval(data.useTargetInterval); }
      if (data.targetFreqInterval !== undefined) { setTargetFreqInterval(data.targetFreqInterval); setTempTargetFreqInterval(data.targetFreqInterval); }
      if (data.minPeakWidth !== undefined) { setMinPeakWidth(data.minPeakWidth); setTempMinPeakWidth(data.minPeakWidth); }
      if (data.autoThreshold !== undefined) setAutoThreshold(data.autoThreshold);
      if (data.blmMode !== undefined) setBlmMode(data.blmMode);
      if (data.blmPerDegree !== undefined) setBlmPerDegree(data.blmPerDegree);
      if (data.blmOffset !== undefined) setBlmOffset(data.blmOffset);
      if (data.selectedPrefix !== undefined) setSelectedPrefix(data.selectedPrefix);
      if (data.branchTolerance !== undefined) {
        setBranchTolerance(data.branchTolerance);
        setTempBranchTolerance(data.branchTolerance);
      }
      if (data.minPeaksPerBranch !== undefined) {
        setMinPeaksPerBranch(data.minPeaksPerBranch);
        setTempMinPeaksPerBranch(data.minPeaksPerBranch);
      }
      if (data.symmetryCenter !== undefined) setSymmetryCenter(data.symmetryCenter);
      if (data.symmetryMaxFreq !== undefined) setSymmetryMaxFreq(data.symmetryMaxFreq);
      if (data.symmetryMinFreq !== undefined) setSymmetryMinFreq(data.symmetryMinFreq);
      if (data.maxDerivMinB !== undefined) setMaxDerivMinB(data.maxDerivMinB);
      if (data.maxDerivMaxB !== undefined) setMaxDerivMaxB(data.maxDerivMaxB);
    }
    isInitialLoad.current = false;
  }, [isAuthReady, user]);

  // --- Clear results when parameters change to prevent stale data ---
  useEffect(() => {
    if (isInitialLoad.current) return;
    if (suppressResultClear.current) { suppressResultClear.current = false; return; }
    setFiles(prev => prev.map(f => ({ ...f, results: [] })));
  }, [
    selectedBColumn, selectedRColumns, minB, maxB, polyDegree,
    interpPoints, windowType, spikeRemoval, spikeKernel, spikeThreshold,
    smoothing, smoothWindow, freqThresholds, autoThreshold, complexMode, phase,
    useTargetInterval, targetFreqInterval, blmMode, blmPerDegree, blmOffset,
    selectedPrefix
  ]);

  // --- Persistence: Save Preferences (localStorage) ---
  useEffect(() => {
    if (!isAuthReady || !user || isInitialLoad.current) return;

    const timeoutId = setTimeout(() => {
      lsSavePrefs({
        selectedBColumn, selectedRColumns, minB, maxB, polyDegree,
        interpPoints, windowType, spikeRemoval, spikeKernel, spikeThreshold,
        smoothing, smoothWindow, freqThresholds, complexMode, phase,
        useTargetInterval, targetFreqInterval, blmMode, blmPerDegree, blmOffset,
        selectedPrefix, branchTolerance, minPeaksPerBranch, symmetryCenter,
        symmetryMaxFreq, symmetryMinFreq, maxDerivMinB, maxDerivMaxB,
        minPeakWidth, autoThreshold,
      });
    }, 1000);
    return () => clearTimeout(timeoutId);
  }, [
    isAuthReady, user, selectedBColumn, selectedRColumns, minB, maxB, polyDegree,
    interpPoints, windowType, spikeRemoval, spikeKernel, spikeThreshold,
    smoothing, smoothWindow, freqThresholds, complexMode, phase,
    useTargetInterval, targetFreqInterval, blmMode, blmPerDegree, blmOffset,
    selectedPrefix, branchTolerance, minPeaksPerBranch, symmetryCenter,
    symmetryMaxFreq, symmetryMinFreq, maxDerivMinB, maxDerivMaxB,
    minPeakWidth, autoThreshold,
  ]);
  const polyfit = useCallback((x: number[], y: number[], degree: number) => {
    const X = x.map(xi => {
      const row = [];
      for (let j = 0; j <= degree; j++) {
        row.push(Math.pow(xi, j));
      }
      return row;
    });
    const X_mat = math.matrix(X);
    const XT = math.transpose(X_mat);
    const XTX = math.multiply(XT, X_mat);
    const XTy = math.multiply(XT, math.matrix(y));
    const coeffs = math.lusolve(XTX, XTy) as any;
    return coeffs.toArray().map((c: any) => c[0]);
  }, []);

  const processFileData = useCallback((data: any[], bCol: string, rCol: string, rInCol?: string, rOutCol?: string, rangeOverride?: { min: number; max: number }): AnalysisResult | null => {
    if (!data || data.length < 10 || !bCol || (!rCol && (!rInCol || !rOutCol))) return null;

    const effectiveMinB = rangeOverride ? rangeOverride.min
      : (minB === '' || (minB === 0 && maxB === 0)) ? -Infinity : Number(minB);
    const effectiveMaxB = rangeOverride ? rangeOverride.max
      : (maxB === '' || (minB === 0 && maxB === 0)) ? Infinity : Number(maxB);

    let rawDataRaw = data.map(row => {
      let rVal = 0;
      if (complexMode && rInCol && rOutCol) {
        const rin = parseFloat(String(row[rInCol]));
        const rout = parseFloat(String(row[rOutCol]));
        const rad = (phase * Math.PI) / 180;
        rVal = rin * Math.cos(rad) + rout * Math.sin(rad);
      } else {
        rVal = parseFloat(String(row[rCol]));
      }
      return {
        b: parseFloat(String(row[bCol])),
        r: rVal
      };
    }).filter(d => !isNaN(d.b) && !isNaN(d.r))
      .filter(d => d.b >= effectiveMinB && d.b <= effectiveMaxB)
      .sort((a, b) => a.b - b.b);

    if (rawDataRaw.length < 10) return null;

    const bData = rawDataRaw.map(d => d.b);
    const rData = rawDataRaw.map(d => d.r);

    // 1. Background Removal (Polynomial Fit)
    const coeffs = polyfit(bData, rData, polyDegree);
    const bgData = bData.map(b => {
      let val = 0;
      for (let i = 0; i <= polyDegree; i++) {
        val += coeffs[i] * Math.pow(b, i);
      }
      return val;
    });

    let deltaR = rData.map((r, i) => r - bgData[i]);

    // 2. Pre-processing
    if (spikeRemoval) {
      deltaR = applySpikeRemovalOpt(deltaR, spikeKernel, spikeThreshold);
    }
    if (smoothing) {
      deltaR = applySmoothingOpt(deltaR, smoothWindow);
    }

    // 3. Interpolation in 1/B — single-pass O(N+M) instead of O(N·logM)
    let minInvB = Infinity, maxInvB = -Infinity;
    for (let i = 0; i < bData.length; i++) {
      const inv = 1 / bData[i];
      if (inv < minInvB) minInvB = inv;
      if (inv > maxInvB) maxInvB = inv;
    }
    const step = (maxInvB - minInvB) / (interpPoints - 1);
    const sorted = rawDataRaw.map((d, i) => ({ invB: 1 / d.b, deltaR: deltaR[i] })).sort((a, b) => a.invB - b.invB);
    const interpDeltaR = linearInterpUniform(sorted, minInvB, step, interpPoints);

    // 4. FFT
    const df_base = 1 / (maxInvB - minInvB);
    let fftSize = interpPoints;
    let paddingFactor = 1;

    if (useTargetInterval && targetFreqInterval > 0) {
      paddingFactor = Math.ceil(df_base / targetFreqInterval);
      // Ensure paddingFactor is at least 1 and fftSize is a power of 2
      // Since interpPoints is already a power of 2, we just need paddingFactor to be a power of 2
      paddingFactor = Math.pow(2, Math.ceil(Math.log2(Math.max(1, paddingFactor))));
      // A narrow field window (large df_base) combined with a small target
      // interval can drive this into the millions — the interval input has
      // no real floor (the "min" attribute is a UI hint only), so a typo
      // like 0.001 instead of 0.1 silently asks for an FFT with tens of
      // millions of points. Process All runs this per file per column with
      // no yielding in between, so an unbounded size here reads as the
      // whole tab freezing rather than a slow-but-alive computation.
      const MAX_FFT_SIZE = 65536;
      paddingFactor = Math.min(paddingFactor, Math.max(1, MAX_FFT_SIZE / interpPoints));
      fftSize = interpPoints * paddingFactor;
    }

    const win = getWindowCached(windowType, interpPoints);
    const fft = getFftInstance(fftSize);
    const out = fft.createComplexArray();
    const input = fft.createComplexArray();

    // Apply window in-place and fill padded input
    for (let i = 0; i < fftSize; i++) {
      input[2 * i] = i < interpPoints ? interpDeltaR[i] * win[i] : 0;
      input[2 * i + 1] = 0;
    }

    fft.transform(out, input);

    const halfSize = fftSize >> 1;
    const df = df_base / paddingFactor;
    const fftResults: FFTResult[] = new Array(halfSize);

    for (let i = 0; i < halfSize; i++) {
      const re = out[2 * i];
      const im = out[2 * i + 1];
      fftResults[i] = { frequency: i * df, amplitude: Math.sqrt(re * re + im * im) };
    }

    // 5. Peak Detection — loop instead of spread+map to avoid extra array allocation
    let maxAmp = 0;
    for (let i = 1; i < fftResults.length; i++) {
      if (fftResults[i].amplitude > maxAmp) maxAmp = fftResults[i].amplitude;
    }
    const detectedPeaks: FFTResult[] = [];

    if (autoThreshold) {
      // Rough, branch-friendly threshold built from a *local* noise floor.
      //
      // One global median+Kσ over the whole spectrum does not work on this data:
      // the low-frequency skirt left behind by an imperfect polynomial
      // background is often an order of magnitude taller than the branches, so
      // a single cut either sits above every real orbit (nothing detected) or,
      // once lowered, fills the peak list with skirt ripple. Both leave the
      // branch analysis with nothing to group.
      //
      // Instead the spectrum is split into blocks, each block gets a robust
      // floor (median + K·σ), and the per-bin threshold is interpolated between
      // block centres. σ comes from second differences, which cancel a locally
      // linear trend, so a steeply sloping skirt doesn't inflate the estimate
      // the way a plain MAD does. Deliberately loose (K = 3.5, plus a 1.5%
      // absolute floor): a peak that is noise at one angle won't recur across
      // angles and gets dropped by the branch grouping anyway.
      const K = 3.5;
      const lo = 1;                    // bin 0 is DC
      const med = (arr: number[]) => {
        arr.sort((a, b) => a - b);
        const m = arr.length >> 1;
        return arr.length % 2 ? arr[m] : (arr[m - 1] + arr[m]) / 2;
      };

      // Reference amplitude for the absolute floor ignores the lowest few % of
      // the frequency axis, so the skirt can't dictate the floor everywhere.
      const refStart = Math.min(halfSize - 1, lo + Math.max(2, Math.floor(halfSize * 0.02)));
      let refAmp = 0;
      for (let i = refStart; i < halfSize; i++) {
        if (fftResults[i].amplitude > refAmp) refAmp = fftResults[i].amplitude;
      }
      if (!(refAmp > 0)) refAmp = maxAmp;
      const absFloor = 0.015 * refAmp;

      const nBins = halfSize - lo;
      const nBlocks = Math.max(3, Math.min(24, Math.round(nBins / 96)));
      const blockLen = Math.max(16, Math.ceil(nBins / nBlocks));
      const bounds: [number, number][] = [];
      for (let s = lo; s < halfSize; s += blockLen) bounds.push([s, Math.min(halfSize, s + blockLen)]);
      // A stub last block has too few bins for a stable median — merge it back.
      if (bounds.length > 1 && bounds[bounds.length - 1][1] - bounds[bounds.length - 1][0] < 16) {
        const stub = bounds.pop()!;
        bounds[bounds.length - 1][1] = stub[1];
      }

      const centres: number[] = [], levels: number[] = [];
      bounds.forEach(([s, e]) => {
        const block: number[] = new Array(e - s);
        for (let i = s; i < e; i++) block[i - s] = fftResults[i].amplitude;
        const base = med([...block]);
        const d2: number[] = [];
        for (let i = s + 1; i < e - 1; i++) {
          d2.push(Math.abs(fftResults[i + 1].amplitude - 2 * fftResults[i].amplitude + fftResults[i - 1].amplitude));
        }
        // 1.4826·MAD → σ; /√6 undoes the second difference's variance inflation
        const sigma = d2.length > 0 ? (1.4826 * med(d2)) / Math.sqrt(6) : 0;
        centres.push((s + e - 1) / 2);
        levels.push(Math.max(base + K * sigma, absFloor));
      });

      let k = 1;
      for (let i = 0; i < halfSize; i++) {
        let thr: number;
        if (levels.length === 1 || i <= centres[0]) {
          thr = levels[0];
        } else if (i >= centres[centres.length - 1]) {
          thr = levels[levels.length - 1];
        } else {
          while (k < centres.length - 1 && centres[k] < i) k++;
          const x0 = centres[k - 1], x1 = centres[k];
          const t = x1 > x0 ? (i - x0) / (x1 - x0) : 0;
          thr = levels[k - 1] + t * (levels[k] - levels[k - 1]);
        }
        fftResults[i].threshold = thr;
      }
    } else {
      // Manual frequency-dependent %-of-max thresholds
      const sortedFreqThresholds = [...freqThresholds].sort((a, b) => a.limit - b.limit);
      const getThreshold = (freq: number) => {
        for (const seg of sortedFreqThresholds) {
          if (freq <= seg.limit) return maxAmp * (seg.threshold / 100);
        }
        return maxAmp * (sortedFreqThresholds[sortedFreqThresholds.length - 1].threshold / 100);
      };
      fftResults.forEach(r => { r.threshold = getThreshold(r.frequency); });
    }

    for (let i = 1; i < fftResults.length - 1; i++) {
      const currentFreq = fftResults[i].frequency;
      const threshold = fftResults[i].threshold || 0;

      if (fftResults[i].amplitude > threshold && 
          fftResults[i].amplitude > fftResults[i-1].amplitude && 
          fftResults[i].amplitude > fftResults[i+1].amplitude) {
        
        // Check width
        let width = 1;
        let l = i - 1;
        while (l >= 0 && fftResults[l].amplitude > threshold) { width++; l--; }
        let r = i + 1;
        while (r < fftResults.length && fftResults[r].amplitude > threshold) { width++; r++; }

        if (width >= minPeakWidth) {
          detectedPeaks.push(fftResults[i]);
        }
      }
    }

    const peakFrequency = detectedPeaks.length > 0 ? _.maxBy(detectedPeaks, 'amplitude')! : null;

    return {
      columnName: rCol || 'Combined',
      plotData: bData.map((b, i) => ({
        b,
        r: rData[i],
        bg: bgData[i],
        deltaR: deltaR[i],
        invB: 1/b
      })),
      fftResults,
      peakFrequency,
      detectedPeaks: _.orderBy(detectedPeaks, 'amplitude', 'desc'),
      suggestedMaxFreq: (peakFrequency?.frequency || 1000) * 1.5,
      df,
      minB: bData[0],
      maxB: bData[bData.length - 1]
    };
  }, [polyDegree, interpPoints, windowType, spikeRemoval, spikeKernel, spikeThreshold, smoothing, smoothWindow, minB, maxB, freqThresholds, autoThreshold, complexMode, phase, useTargetInterval, targetFreqInterval]);

  const getTemperature = useCallback((metadata: Record<string, any>) => {
    const tempEntry = Object.entries(metadata).find(([k]) => 
      k.toLowerCase().includes('temp') || 
      k.toLowerCase().includes('t (k)') || 
      k.toLowerCase().includes('temperature')
    );
    if (!tempEntry) return 1.5; // Default fallback
    const val = parseFloat(String(tempEntry[1]));
    return isNaN(val) ? 1.5 : val;
  }, []);


  // Detect the temperature column by name (each file is a sweep at one T).
  const findTempColumn = useCallback((file: FileEntry): string | undefined => {
    return file.columns.find(c => {
      const l = c.toLowerCase();
      return l.includes('temperature') || l.includes('temp') || l.includes('t (k)') || l.includes('t(k)') || l === 't';
    });
  }, []);

  // The file's temperature = FIRST value of its temperature column (user's
  // convention), falling back to the metadata scan when no column is present.
  const getFileTemperature = useCallback((file: FileEntry): number => {
    const col = findTempColumn(file);
    if (col && file.data.length > 0) {
      const v = parseFloat(String(file.data[0][col]));
      if (isFinite(v)) return v;
    }
    return getTemperature(file.metadata);
  }, [findTempColumn, getTemperature]);

  // Temperature for the L-K fit: prefer a "...4.2K..." style pattern in the
  // filename (temperature-sweep datasets name the file after the setpoint),
  // falling back to the existing column/metadata-based resolver otherwise.
  const resolveTemperature = useCallback((file: FileEntry): number => {
    const m = file.name.match(/(\d+(?:\.\d+)?)\s*[kK](?![a-zA-Z0-9])/);
    if (m) {
      const v = parseFloat(m[1]);
      if (isFinite(v)) return v;
    }
    return getFileTemperature(file);
  }, [getFileTemperature]);

  // File sort key/label for the two "aligned" exports (exportRawAlignedCSV /
  // exportOscillatoryAlignedCSV, defined earlier): angle when the active
  // bucket is a BLM sweep, temperature when it's a T-sweep — whichever axis
  // actually varies across these files.
  const alignedExportTag = useCallback((file: FileEntry, fIdx: number): { sortVal: number; label: string } => {
    if (activeBucket === 'temperature') {
      const t = resolveTemperature(file);
      return { sortVal: t, label: `${t.toFixed(2)}K` };
    }
    const blm = resolveBlm(file, fIdx);
    return { sortVal: blm, label: `${blm}deg` };
  }, [activeBucket, resolveTemperature, resolveBlm]);

  // Every processed column name available across all files, for the L-K Fit
  // column picker.
  const lkColumnOptions = useMemo(() => {
    const set = new Set<string>();
    files.forEach(f => (f.results || []).forEach(r => set.add(r.columnName)));
    return Array.from(set).sort();
  }, [files]);

  // Detected peak frequencies clustered (within 2% relative tolerance) across
  // every file/column, ranked by how often they show up — gives the user a
  // one-click shortlist of "the" oscillation frequency to fit.
  const lkFreqClusters = useMemo(() => {
    const allFreqs: number[] = [];
    files.forEach(f => (f.results || []).forEach(r => r.detectedPeaks.forEach(p => allFreqs.push(p.frequency))));
    if (allFreqs.length === 0) return [] as { freq: number; count: number }[];
    const sorted = [...allFreqs].sort((a, b) => a - b);
    const clusters: { freq: number; count: number }[] = [];
    let sum = sorted[0], count = 1;
    for (let i = 1; i < sorted.length; i++) {
      const mean = sum / count;
      if (Math.abs(sorted[i] - mean) / mean < 0.02) {
        sum += sorted[i]; count++;
      } else {
        clusters.push({ freq: sum / count, count });
        sum = sorted[i]; count = 1;
      }
    }
    clusters.push({ freq: sum / count, count });
    return _.orderBy(clusters, ['count'], ['desc']).slice(0, 8);
  }, [files]);

  // Lifshitz-Kosevich thermal damping factor R_T(X) = X / sinh(X)
  const lkDampingFactor = (X: number): number => (Math.abs(X) < 1e-8 ? 1 : X / Math.sinh(X));

  // (temperature, amplitude, B_eff) series for the currently selected
  // frequency/column, then a 1-D scan + golden-section search over the
  // effective mass m* (in units of m0) — A0 is linear given m*, so it is
  // solved in closed form at each trial m*.
  const lkFitResult = useMemo(() => {
    if (lkFreq == null || files.length === 0) return null;

    const points: { fileName: string; temp: number; amp: number; bEff: number; minB: number; maxB: number }[] = [];
    files.forEach(file => {
      const res = lkColumn
        ? (file.results || []).find(r => r.columnName === lkColumn)
        : (file.results || [])[0];
      if (!res || res.fftResults.length === 0) return;
      const temp = resolveTemperature(file);
      if (!isFinite(temp) || temp <= 0) return;

      const fft = res.fftResults;
      let amp: number;
      if (lkFreq <= fft[0].frequency) amp = fft[0].amplitude;
      else if (lkFreq >= fft[fft.length - 1].frequency) amp = fft[fft.length - 1].amplitude;
      else {
        let amp_: number | null = null;
        for (let i = 0; i < fft.length - 1; i++) {
          if (fft[i].frequency <= lkFreq && fft[i + 1].frequency >= lkFreq) {
            const span = fft[i + 1].frequency - fft[i].frequency;
            const t = span > 0 ? (lkFreq - fft[i].frequency) / span : 0;
            amp_ = fft[i].amplitude + t * (fft[i + 1].amplitude - fft[i].amplitude);
            break;
          }
        }
        amp = amp_ ?? fft[fft.length - 1].amplitude;
      }

      const bEff = (2 * res.minB * res.maxB) / (res.minB + res.maxB);
      if (!isFinite(bEff) || bEff <= 0) return;
      points.push({ fileName: file.name, temp, amp, bEff, minB: res.minB, maxB: res.maxB });
    });

    if (points.length < 3) return { points, mStar: null, A0: null, r2: null };

    const fitForM = (m: number) => {
      let sumFA = 0, sumFF = 0;
      const fs = points.map(p => lkDampingFactor(14.69 * m * p.temp / p.bEff));
      fs.forEach((f, i) => { sumFA += f * points[i].amp; sumFF += f * f; });
      const A0 = sumFF > 0 ? sumFA / sumFF : 0;
      let ssr = 0;
      fs.forEach((f, i) => { const d = points[i].amp - A0 * f; ssr += d * d; });
      return { A0, ssr };
    };

    const mMin = 0.01, mMax = 20, steps = 400;
    let bestM = mMin, bestSsr = Infinity;
    for (let i = 0; i <= steps; i++) {
      const m = mMin + ((mMax - mMin) * i) / steps;
      const { ssr } = fitForM(m);
      if (ssr < bestSsr) { bestSsr = ssr; bestM = m; }
    }

    const gr = (Math.sqrt(5) - 1) / 2;
    const bracket = (mMax - mMin) / steps * 2;
    let a = Math.max(mMin, bestM - bracket), b = Math.min(mMax, bestM + bracket);
    let c = b - gr * (b - a), d = a + gr * (b - a);
    for (let iter = 0; iter < 60 && Math.abs(b - a) > 1e-7; iter++) {
      const fc = fitForM(c).ssr, fd = fitForM(d).ssr;
      if (fc < fd) b = d; else a = c;
      c = b - gr * (b - a); d = a + gr * (b - a);
    }
    const mStar = (a + b) / 2;
    const { A0, ssr } = fitForM(mStar);

    const meanAmp = points.reduce((s, p) => s + p.amp, 0) / points.length;
    const sst = points.reduce((s, p) => s + Math.pow(p.amp - meanAmp, 2), 0);
    const r2 = sst > 0 ? 1 - ssr / sst : 1;

    return { points: _.orderBy(points, ['temp'], ['asc']), mStar, A0, r2 };
    // Only recomputed while the L-K Fit tab is actually open — this runs a
    // 400-step scan + golden-section search over every file, no reason to pay
    // for it while the user is on a different tab.
  }, [activeTab, files, lkColumn, lkFreq, resolveTemperature]);

  // The smooth (temperature, fitted amplitude) curve — 100 points evaluated
  // at the mean B_eff across the series. Shared by the chart and the CSV
  // export so both draw from the exact same fitted function.
  const lkFitCurve = useMemo(() => {
    if (!lkFitResult || lkFitResult.mStar == null || lkFitResult.A0 == null || lkFitResult.points.length === 0) return [];
    const { points, mStar, A0 } = lkFitResult;
    const avgBeff = points.reduce((s, p) => s + p.bEff, 0) / points.length;
    const tMin = Math.min(...points.map(p => p.temp));
    const tMax = Math.max(...points.map(p => p.temp));
    const steps = 100;
    const curve: { temp: number; ampFit: number }[] = [];
    for (let i = 0; i <= steps; i++) {
      const t = tMin + ((tMax - tMin) * i) / steps;
      const X = 14.69 * mStar * t / avgBeff;
      curve.push({ temp: t, ampFit: A0 * lkDampingFactor(X) });
    }
    return curve;
  }, [lkFitResult]);

  // ---- FFT Range Scan: how do ALL detected peaks, across every file in the
  // batch, shift when the B-field window changes? Not a single-file drill
  // down — the point is comparing peaks-vs-angle under the currently
  // committed range against a candidate range, side by side, so a window
  // choice can be judged for stability across the whole sweep at once.
  // Reuses processFileData's rangeOverride so nothing else on the page
  // reprocesses while the candidate range is being tuned.
  //
  // Gated on activeTab: this reprocesses every included file twice (full
  // background-subtraction + FFT), and processFileData's identity changes on
  // almost any Dashboard setting — without the gate this was refiring on
  // every sidebar tweak even while looking at a completely different tab.
  const rangeScanFileBExtent = useMemo(() => {
    if (activeTab !== 'fftrange') return null;
    let lo = Infinity, hi = -Infinity;
    files.forEach(f => {
      const bCol = resolveBCol(f);
      f.data.forEach(row => {
        const v = parseFloat(String(row[bCol]));
        if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
      });
    });
    return isFinite(lo) && isFinite(hi) && hi > lo ? { lo, hi } : null;
  }, [activeTab, files, resolveBCol]);

  const rangeScanMinBEff = rangeScanMinB ?? (typeof minB === 'number' ? minB : rangeScanFileBExtent?.lo ?? 0);
  const rangeScanMaxBEff = rangeScanMaxB ?? (typeof maxB === 'number' ? maxB : rangeScanFileBExtent?.hi ?? 1);

  // Sync the input drafts whenever the tab is (re)opened, so they start from
  // whatever candidate/committed range was last in effect rather than
  // whatever was left over from before. Only ever touches the *draft* — the
  // committed rangeScanMinB/MaxB below deliberately stay null until the user
  // actually edits something, since null is what makes rangeScanMinBEff fall
  // back to the real committed range in the first place.
  useEffect(() => {
    if (activeTab === 'fftrange') {
      setRangeScanMinBDraft(rangeScanMinBEff);
      setRangeScanMaxBDraft(rangeScanMaxBEff);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  // Debounce draft -> committed candidate range, so the full re-batch below
  // (which reprocesses every included file) runs once after the user stops
  // typing instead of on every keystroke. Implemented as plain event-handler
  // timers rather than a useEffect keyed on the draft: an effect fires on
  // mount too (twice under StrictMode), which — since the draft starts at a
  // real number, not "unset" — would commit that value on its own a moment
  // after mount, before the user ever touched the tab. A timer armed only
  // from the onChange handler can't fire except in response to a real edit.
  const rangeScanDebounceRef = useRef<{ min?: ReturnType<typeof setTimeout>; max?: ReturnType<typeof setTimeout> }>({});
  const handleRangeScanMinBChange = (v: number) => {
    setRangeScanMinBDraft(v);
    clearTimeout(rangeScanDebounceRef.current.min);
    rangeScanDebounceRef.current.min = setTimeout(() => setRangeScanMinB(v), 400);
  };
  const handleRangeScanMaxBChange = (v: number) => {
    setRangeScanMaxBDraft(v);
    clearTimeout(rangeScanDebounceRef.current.max);
    rangeScanDebounceRef.current.max = setTimeout(() => setRangeScanMaxB(v), 400);
  };

  const committedMinB = typeof minB === 'number' ? minB : -Infinity;
  const committedMaxB = typeof maxB === 'number' ? maxB : Infinity;

  // Peaks under the currently *committed* (Process All) range — one row per
  // included file plus every detected peak for the chart. Split out from the
  // candidate-range computation below so that typing a new candidate value
  // doesn't also redundantly redo this half of the work: it only changes
  // when the committed range itself changes (i.e. after Apply/Process All).
  const rangeScanCommitted = useMemo(() => {
    if (activeTab !== 'fftrange') return { peakByFile: new Map<string, number | null>(), peaks: [] as { blm: number; frequency: number }[] };
    const peakByFile = new Map<string, number | null>();
    const peaks: { blm: number; frequency: number }[] = [];
    filesByAngle
      .filter(({ file }) => !fftSkip.has(file.id))
      .forEach(({ file, blm }) => {
        const bCol = resolveBCol(file);
        const rCol = targetColsFor(file)[0];
        if (!rCol) { peakByFile.set(file.id, null); return; }
        const res = processFileData(file.data, bCol, rCol, undefined, undefined, { min: committedMinB, max: committedMaxB });
        peakByFile.set(file.id, res?.peakFrequency?.frequency ?? null);
        (res?.detectedPeaks ?? []).forEach(p => peaks.push({ blm, frequency: p.frequency }));
      });
    return { peakByFile, peaks };
  }, [activeTab, filesByAngle, fftSkip, resolveBCol, targetColsFor, committedMinB, committedMaxB, processFileData]);

  // Peaks under the *candidate* range — recomputes on every debounced edit to
  // the Min/Max B inputs above (that's the live-preview point of this tab).
  const rangeScanCandidate = useMemo(() => {
    if (activeTab !== 'fftrange') return { peakByFile: new Map<string, number | null>(), peaks: [] as { blm: number; frequency: number }[] };
    const peakByFile = new Map<string, number | null>();
    const peaks: { blm: number; frequency: number }[] = [];
    filesByAngle
      .filter(({ file }) => !fftSkip.has(file.id))
      .forEach(({ file, blm }) => {
        const bCol = resolveBCol(file);
        const rCol = targetColsFor(file)[0];
        if (!rCol) { peakByFile.set(file.id, null); return; }
        const res = processFileData(file.data, bCol, rCol, undefined, undefined, { min: rangeScanMinBEff, max: rangeScanMaxBEff });
        peakByFile.set(file.id, res?.peakFrequency?.frequency ?? null);
        (res?.detectedPeaks ?? []).forEach(p => peaks.push({ blm, frequency: p.frequency }));
      });
    return { peakByFile, peaks };
  }, [activeTab, filesByAngle, fftSkip, resolveBCol, targetColsFor, rangeScanMinBEff, rangeScanMaxBEff, processFileData]);

  const rangeScanBatch = useMemo(() => {
    if (activeTab !== 'fftrange') return [];
    return filesByAngle
      .filter(({ file }) => !fftSkip.has(file.id))
      .map(({ file, blm }) => ({
        blm, fileName: file.name,
        committedPeak: rangeScanCommitted.peakByFile.get(file.id) ?? null,
        candidatePeak: rangeScanCandidate.peakByFile.get(file.id) ?? null,
      }))
      .filter(d => d.committedPeak !== null || d.candidatePeak !== null)
      .sort((a, b) => a.blm - b.blm);
  }, [activeTab, filesByAngle, fftSkip, rangeScanCommitted, rangeScanCandidate]);
  const rangeScanCurrentPeaks = rangeScanCommitted.peaks;
  const rangeScanCandidatePeaks = rangeScanCandidate.peaks;

  const rangeScanStability = useMemo(() => {
    const withBoth = rangeScanBatch.filter(d => d.committedPeak !== null && d.candidatePeak !== null);
    if (withBoth.length === 0) return null;
    const deltas = withBoth.map(d => Math.abs((d.candidatePeak as number) - (d.committedPeak as number)));
    const maxDelta = Math.max(...deltas);
    const meanDelta = deltas.reduce((s, d) => s + d, 0) / deltas.length;
    return { n: withBoth.length, maxDelta, meanDelta };
  }, [rangeScanBatch]);

  // ---- Live-preview B-range for "All Detected Frequencies vs BLM" ----
  // Recomputes every included file's detected peaks under a candidate range
  // with no Process All needed. null (untouched) means "just show the
  // normal batchSummary from the last Process All" — dashLivePeaks only
  // takes over once the user actually edits the range, and only while the
  // Dashboard is the visible tab.
  const dashLiveMinBEff = dashLiveRange.minCommitted ?? (typeof minB === 'number' ? minB : -Infinity);
  const dashLiveMaxBEff = dashLiveRange.maxCommitted ?? (typeof maxB === 'number' ? maxB : Infinity);
  const dashLivePeaks = useMemo(() => {
    if (activeTab !== 'dashboard' || !dashLiveRange.active) return null;
    const out: { blm: number; freq: number; amp: number; fileName: string; id: string; temp: number }[] = [];
    filesByAngle
      .filter(({ file }) => !fftSkip.has(file.id))
      .forEach(({ file, blm }) => {
        const bCol = resolveBCol(file);
        const cols = targetColsFor(file);
        const temp = getFileTemperature(file);
        const results = complexMode && cols.length >= 2
          ? [processFileData(file.data, bCol, 'Combined', cols[0], cols[1], { min: dashLiveMinBEff, max: dashLiveMaxBEff })]
          : cols.map(col => processFileData(file.data, bCol, col, undefined, undefined, { min: dashLiveMinBEff, max: dashLiveMaxBEff }));
        results.forEach((res, i) => {
          if (!res) return;
          const colName = complexMode && cols.length >= 2 ? 'Combined' : cols[i];
          res.detectedPeaks.forEach((peak, pIdx) => {
            out.push({ blm, freq: peak.frequency, amp: peak.amplitude, fileName: file.name, id: `${file.id}-${colName}-${pIdx}`, temp });
          });
        });
      });
    return out;
  }, [activeTab, dashLiveRange.active, filesByAngle, fftSkip, resolveBCol, targetColsFor, complexMode, dashLiveMinBEff, dashLiveMaxBEff, processFileData, getFileTemperature]);
  const displayedBatchSummary = dashLivePeaks ?? batchSummary;

  const findAutoPhase = () => {
    if (selectedRColumns.length < 2 || files.length === 0) return;
    
    setIsProcessing(true);
    setTimeout(() => {
      let bestPhase = 0;
      let maxAmp = -1;

      const fileData = files[0].data;
      const rIn = fileData.map(row => parseFloat(row[selectedRColumns[0]]));
      const rOut = fileData.map(row => parseFloat(row[selectedRColumns[1]]));
      const bValues = fileData.map(row => parseFloat(row[selectedBColumn]));

      // Pre-filter valid data once; only invB ordering changes per phase
      const effectiveMinB = minB === '' ? -Infinity : Number(minB);
      const effectiveMaxB = maxB === '' ? Infinity : Number(maxB);
      const validIdx: number[] = [];
      for (let i = 0; i < bValues.length; i++) {
        if (!isNaN(bValues[i]) && !isNaN(rIn[i]) && !isNaN(rOut[i]) &&
            bValues[i] >= effectiveMinB && bValues[i] <= effectiveMaxB) {
          validIdx.push(i);
        }
      }
      if (validIdx.length < 10) { setIsProcessing(false); return; }

      // Pre-compute sorted 1/B values and indices (same for all phases)
      const sortedByInvB = validIdx
        .map(i => ({ idx: i, invB: 1 / bValues[i] }))
        .sort((a, b) => a.invB - b.invB);
      const invBArr = sortedByInvB.map(s => s.invB);
      const minInvBp = invBArr[0];
      const maxInvBp = invBArr[invBArr.length - 1];
      const stepP = (maxInvBp - minInvBp) / (interpPoints - 1);

      // Reuse single FFT instance for all phase iterations
      const fftP = getFftInstance(interpPoints);
      const outP = fftP.createComplexArray();
      const inputP = fftP.createComplexArray();

      // Search every 5 degrees
      for (let p = 0; p < 360; p += 5) {
        const rad = (p * Math.PI) / 180;
        const cosRad = Math.cos(rad);
        const sinRad = Math.sin(rad);

        // Build sorted deltaR for this phase (background = mean)
        let sumR = 0;
        for (let k = 0; k < sortedByInvB.length; k++) {
          const i = sortedByInvB[k].idx;
          sumR += rIn[i] * cosRad + rOut[i] * sinRad;
        }
        const meanR = sumR / sortedByInvB.length;

        const sortedForInterp = sortedByInvB.map(s => ({
          invB: s.invB,
          deltaR: (rIn[s.idx] * cosRad + rOut[s.idx] * sinRad) - meanR,
        }));

        const interpDeltaR = linearInterpUniform(sortedForInterp, minInvBp, stepP, interpPoints);

        for (let i = 0; i < interpPoints; i++) {
          inputP[2 * i] = interpDeltaR[i];
          inputP[2 * i + 1] = 0;
        }
        fftP.transform(outP, inputP);

        for (let i = 5; i < interpPoints / 2; i++) {
          const amp = Math.sqrt(outP[2 * i] ** 2 + outP[2 * i + 1] ** 2);
          if (amp > maxAmp) { maxAmp = amp; bestPhase = p; }
        }
      }

      setPhase(bestPhase);
      setIsProcessing(false);
    }, 100);
  };

  // getWindow / applySpikeRemoval / applySmoothing moved to module scope as optimized versions

  // --- CSV Parsing ---
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = event.target.files;
    if (!fileList || fileList.length === 0) return;

    setError(null);
    setIsProcessing(true);

    const newFiles: FileEntry[] = [];
    let filesProcessed = 0;

    Array.from(fileList).forEach((file: File) => {
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        const text = e.target?.result as string;
        const lines = text.split(/\r?\n/);
        
        // Extract metadata
        const extractedMetadata: Record<string, string> = {};
        const blmLine = lines.find(l => {
          const low = l.toLowerCase();
          return low.includes('blm') || low.includes('angle') || low.includes('position') || low.includes('step');
        });
        if (blmLine) {
          // Improved regex to look for the number after "BLM" or similar keywords
          const match = blmLine.match(/(?:blm|angle|position|step)[^0-9]*?([-+]?[0-9]*\.?[0-9]+)/i);
          if (match) {
            extractedMetadata['BLM Position'] = match[1];
          } else {
            // Fallback to the first number on the line if the specific pattern isn't found
            const fallbackMatch = blmLine.match(/[-+]?[0-9]*\.?[0-9]+/);
            if (fallbackMatch) {
              extractedMetadata['BLM Position'] = fallbackMatch[0];
            }
          }
        }
        
        lines.slice(0, 100).forEach(line => {
          if (line.includes(':') && line.length < 100) {
            const [key, val] = line.split(':');
            if (key && val && val.trim().length > 0 && !key.toLowerCase().includes('date') && !key.toLowerCase().includes('time')) {
              extractedMetadata[key.trim()] = val.trim();
            }
          }
        });

        let startLine = 0;
        const endHistoryIdx = _.findLastIndex(lines, l => l.includes('End of Sequence History'));

        if (endHistoryIdx !== -1) {
          // If history block found, the data header is the first non-empty line after it
          startLine = endHistoryIdx + 1;
          while (startLine < lines.length && lines[startLine].trim().length === 0) {
            startLine++;
          }
        } else {
          // Anchor on the data itself: find the first row whose every field is a
          // number, then take the non-empty line above it as the header. The old
          // keyword blacklist ("Offsets:", "Magnet:", …) could not survive a new
          // instrument's preamble — a title line like
          // "20260824_1.006.txt Mon, Aug 24, 2026 …" has commas, no keyword, and
          // was accepted as the header, so the whole file parsed into nonsense.
          const fieldsOf = (l: string) => l.split(/[\t,]|\s{2,}/).map(s => s.trim()).filter(s => s.length > 0);
          // "nan" counts as numeric: instrument files park unused channels there,
          // and a whole data row of them must still read as data.
          const NUMERIC = /^([-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?|[-+]?(nan|inf(inity)?))$/i;
          const isDataRow = (l: string) => {
            const f = fieldsOf(l);
            return f.length >= 2 && f.every(s => NUMERIC.test(s));
          };
          const firstData = lines.findIndex(isDataRow);
          if (firstData > 0) {
            let hdr = firstData - 1;
            while (hdr > 0 && lines[hdr].trim().length === 0) hdr--;
            // Only treat it as a header if it isn't itself data and has names for
            // the columns; otherwise start at the data and let Papa name them.
            startLine = (!isDataRow(lines[hdr]) && fieldsOf(lines[hdr]).length >= 2) ? hdr : firstData;
          } else if (firstData === 0) {
            startLine = 0;
          } else {
            // No numeric row at all — fall back to the old keyword skip.
            const metadataKeywords = ['offsets:', 'scale factors:', 'magnet:', 'trigger', 'this data', '*****', 'date:', 'time:'];
            while (startLine < lines.length) {
              const line = lines[startLine].trim();
              const isMetadata = metadataKeywords.some(k => line.toLowerCase().includes(k));
              const cols = line.split(/[\t,]+/).filter(c => c.trim().length > 0).length;
              if (line.length > 0 && !isMetadata && cols > 1) break;
              startLine++;
            }
          }
        }

        const cleanText = lines.slice(startLine).join('\n');

        Papa.parse(cleanText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          comments: '#',
          complete: (results) => {
            const data = results.data as any[];
            if (data.length > 0) {
              const keys = Object.keys(data[0]);
              // An up-and-down sweep becomes two entries, one per branch.
              const branches = splitSweepDirections(data, guessFieldColumn(keys));
              if (branches.length >= 2) {
                branches.forEach(br => newFiles.push({
                  id: Math.random().toString(36).substr(2, 9),
                  name: `${file.name} (${br.dir})`,
                  data: br.rows,
                  metadata: extractedMetadata,
                  columns: keys,
                  results: [],
                }));
              } else {
                newFiles.push({
                  id: Math.random().toString(36).substr(2, 9),
                  name: file.name,
                  data,
                  metadata: extractedMetadata,
                  columns: keys,
                  results: []
                });
              }
            }
            filesProcessed++;
            if (filesProcessed === fileList.length) {
              if (newFiles.length > 0) {
                // Adding to an already-populated set: append the new files,
                // score them for the FFT-inclusion checklist, and leave every
                // other setting (columns, B-range, prefix) exactly as the user
                // already has it — only a from-empty upload gets to redetect
                // all of that below.
                const wasEmpty = files.length === 0;
                setFiles(prev => [...prev, ...newFiles]);
                if (!wasEmpty) {
                  const scores: Record<string, number> = {};
                  newFiles.forEach(f => {
                    const bk = resolveBCol(f);
                    scores[f.id] = fileOscillationScore(f, bk).score;
                  });
                  setOscScores(prev => ({ ...prev, ...scores }));
                  setFftSkip(prev => {
                    const next = new Set(prev);
                    newFiles.forEach(f => { if (scores[f.id] < OSC_SCORE_MIN) next.add(f.id); });
                    return next;
                  });
                  setIsProcessing(false);
                  return;
                }
                const firstKeys = newFiles[0].columns;
                const bKey = guessFieldColumn(firstKeys);
                setSelectedBColumn(bKey);

                // Auto B-field ROI focused on where oscillations (branches) are
                // visible: Max B = highest measured field; Min B = onset of
                // oscillations (median across files), not the absolute data min.
                // Rank each file's real signal columns. Everything downstream — the
                // score, the onset detector, the initial column selection — needs
                // the channel that actually holds the measurement; a name-based
                // guess lands on "Current (A)" or an all-NaN spare lock-in.
                const ranked = newFiles.map(f => ({ f, cols: fileSignalColumns(f, bKey) }));
                const signalCols = new Set<string>();
                ranked.forEach(({ cols }) => cols.forEach(c => signalCols.add(c.col)));

                // The file with the clearest oscillation decides which channel role
                // is the signal (the part after the last "_", e.g. "X"), and every
                // other file uses its own column of that same role. Without this the
                // angles that show nothing would each be analysed on whichever noisy
                // channel happened to rank first.
                const roleOf = (col: string) => col.includes('_') ? col.substring(col.lastIndexOf('_') + 1) : col;
                let champion: { col: string; score: number } | null = null;
                ranked.forEach(({ cols }) => {
                  if (cols[0] && (!champion || cols[0].score > champion.score)) champion = cols[0];
                });
                const wantedRole = champion ? roleOf(champion.col) : null;

                let bMax = -Infinity, bMinAll = Infinity;
                const onsets: number[] = [];
                const scores: Record<string, number> = {};
                const bestCols: string[] = [];             // one per file
                ranked.forEach(({ f, cols }) => {
                  const pick = (wantedRole && cols.find(c => roleOf(c.col) === wantedRole)) || cols[0] || null;
                  const col = pick?.col ?? null;
                  // The score always reports the file's best channel, so a file is
                  // never dropped just because the shared role is quiet in it.
                  scores[f.id] = cols[0]?.score ?? 0;
                  if (col) bestCols.push(col);

                  const bv: number[] = [], rv: number[] = [];
                  for (const row of f.data) {
                    const bVal = parseFloat(String(row[bKey]));
                    const rVal = col ? parseFloat(String(row[col])) : NaN;
                    if (isFinite(bVal)) {
                      if (bVal > bMax) bMax = bVal;
                      if (bVal > 0 && bVal < bMinAll) bMinAll = bVal;
                      if (isFinite(rVal)) { bv.push(bVal); rv.push(rVal); }
                    }
                  }
                  const onset = oscillationOnsetB(bv, rv);
                  if (onset != null && isFinite(onset)) onsets.push(onset);
                });

                // Pre-select the FFT batch: files with no visible oscillation are
                // unchecked so they don't pollute the peak list. The user can
                // re-check any of them in the sidebar.
                setOscScores(scores);
                setFftSkip(new Set(newFiles.filter(f => scores[f.id] < OSC_SCORE_MIN).map(f => f.id)));

                // Always land on a usable B window, so the 30–41.5 T factory
                // default never survives an upload from a magnet that only went to
                // 9 T. Use the oscillation onset when it was found; otherwise fall
                // back to the same high-field window the score uses (everything
                // above 0.6·B_max), which keeps a narrow high-field sweep whole.
                if (isFinite(bMax) && isFinite(bMinAll) && bMax > bMinAll) {
                  const cap = bMinAll + 0.5 * (bMax - bMinAll);
                  let onset: number;
                  if (onsets.length > 0) {
                    onsets.sort((a, b) => a - b);
                    onset = Math.min(onsets[onsets.length >> 1], cap);
                  } else {
                    onset = Math.max(bMinAll, OSC_WINDOW_FRAC * bMax);
                  }
                  const lo = Math.max(0, Math.floor(onset * 10) / 10);
                  const hi = Math.ceil(bMax * 10) / 10;
                  if (hi > lo) {
                    setMinB(lo); setMaxB(hi);
                    setTempMinB(lo); setTempMaxB(hi); // keep the staged inputs in sync
                  }
                }

                // Analyse the channel that carries the oscillation, one per file.
                // Taking the union of the per-file winners is what makes files that
                // name their channels after their own angle work
                // ("1.7K_-175deg_X" / "1.7K_95deg_X"): each file resolves only the
                // name it owns. Output2 is already dropped by the ranking.
                const rKeys = Array.from(new Set(bestCols));
                setSelectedRColumns(rKeys.length > 0 ? rKeys : [firstKeys[1] || firstKeys[0]]);

                // A prefix is only worth pre-selecting if it resolves to a real
                // signal column in *every* file — otherwise the batch silently
                // produces nothing for the files it doesn't cover. Prefixes are
                // taken from data-bearing columns only, so housekeeping ("T" from
                // T_sample) and unused lock-ins never become the default.
                const pSet = new Set<string>();
                signalCols.forEach(col => {
                  if (col.includes('_')) {
                    const p = col.substring(0, col.lastIndexOf('_'));
                    if (p) pSet.add(p);
                  }
                });
                if (!prefixLocked) {
                  const covering = Array.from(pSet).sort().filter(p =>
                    newFiles.every(f => f.columns.some(c => c.startsWith(p + '_') && signalCols.has(c)))
                  );
                  setSelectedPrefix(covering.length > 0 ? covering[0] : '');
                }
              } else {
                setError("No valid data found in selected files.");
              }
              setIsProcessing(false);
            }
          },
          error: () => {
            filesProcessed++;
            if (filesProcessed === fileList.length) setIsProcessing(false);
          }
        });
      };
      reader.readAsText(file);
    });
  };

  // Uploading while the temperature bucket has no active group would silently
  // lose the files at the next switch (nothing to capture them into) — create
  // a default group first so there's always somewhere for them to land. The
  // user can rename it (e.g. to the angle it was taken at) afterwards.
  const handleFileUploadIntoActiveBucket = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (activeBucket === 'temperature') {
      const current = orientations.find(o => o.id === activeOrientationId);
      const hasActiveGroup = current?.temperature.groups.some(g => g.id === activeTempGroupId);
      if (!hasActiveGroup) {
        const newId = `tg-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        setOrientations(prev => prev.map(o => o.id === activeOrientationId
          ? { ...o, temperature: { ...o.temperature, groups: [...o.temperature.groups, { id: newId, label: `Group ${o.temperature.groups.length + 1}`, files: [] }] } }
          : o));
        setActiveTempGroupId(newId);
      }
    }
    handleFileUpload(event);
  };

  // Remove a single uploaded file — index-shift selectedFileIndex, drop it
  // from the FFT-skip set, and bounce off its per-file detail tab if it was
  // the one showing.
  const removeFile = useCallback((id: string) => {
    const idx = files.findIndex(f => f.id === id);
    if (idx === -1) return;
    setFiles(prev => prev.filter(f => f.id !== id));
    setFftSkip(prev => {
      if (!prev.has(id)) return prev;
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
    setActiveTab(at => (at === id ? 'dashboard' : at));
    setSelectedFileIndex(prevIdx => {
      if (prevIdx > idx) return prevIdx - 1;
      if (prevIdx === idx) return Math.max(0, idx - 1);
      return prevIdx;
    });
  }, [files]);

  const currentFileResults = useMemo(() => {
    if (files.length === 0) return null;
    
    // Determine which file to analyze based on activeTab or selectedFileIndex
    let fileIdx = selectedFileIndex;
    if (activeTab !== 'dashboard') {
      const foundIdx = files.findIndex(f => f.id === activeTab);
      if (foundIdx !== -1) fileIdx = foundIdx;
    }
    
    const idx = Math.min(fileIdx, files.length - 1);
    const file = files[idx];
    
    // PERFORMANCE OPTIMIZATION: If results are already stored in the file entry (from batch process),
    // use them directly instead of recalculating on every render/tab change.
    if (file.results && file.results.length > 0) {
      return file;
    }
    
    // If no results yet (e.g. fresh upload), we can still do a quick calc for the current file
    // unless a batch process is already running — or the file is unchecked for FFT.
    if (isProcessing || fftSkip.has(file.id)) return file;

    // Smart matching for B column
    const bCol = file.columns.find(c => c === selectedBColumn) || 
                 file.columns.find(c => c.toLowerCase() === selectedBColumn.toLowerCase()) ||
                 file.columns.find(k => {
                   const low = k.toLowerCase();
                   return (low === 'b' || low.startsWith('b ') || low.startsWith('b_') || low.includes('field') || low.includes('magnetic')) && 
                          !low.includes('blm') && !low.includes('angle') && !low.includes('position');
                 }) ||
                 file.columns.find(k => {
                   const low = k.toLowerCase();
                   return low.includes('b') && !low.includes('blm') && !low.includes('angle') && !low.includes('position');
                 }) ||
                 file.columns[0];

    // Smart matching for R columns — Output2 never reaches the fit
    const rCols = dropQuadrature(selectedRColumns.map(target => {
        return file.columns.find(c => c === target) ||
               file.columns.find(c => c.toLowerCase() === target.toLowerCase());
    }).filter((c): c is string => !!c), file.columns);

    const results = complexMode && rCols.length >= 2
      ? [processFileData(file.data, bCol, 'Combined', rCols[0], rCols[1])]
      : rCols.map(rCol => processFileData(file.data, bCol, rCol));
    
    return { ...file, results: results.filter((r): r is AnalysisResult => r !== null) };
  }, [files, selectedFileIndex, activeTab, selectedBColumn, selectedRColumns, complexMode, processFileData, isProcessing, fftSkip]);

  // ---- Live-preview B-range for the per-file "3. FFT Power Spectrum" ----
  // Same idea, scoped to just the file currently open in the per-file detail
  // view — keyed by column name so each processed column gets its own live
  // spectrum. The rare complex-mode "Combined" pseudo-column has no recorded
  // In/Out pair to redo the mix with, so it's left out of the live preview.
  const fileLiveMinBEff = fileLiveRange.minCommitted ?? (typeof minB === 'number' ? minB : -Infinity);
  const fileLiveMaxBEff = fileLiveRange.maxCommitted ?? (typeof maxB === 'number' ? maxB : Infinity);
  const fileLiveResults = useMemo(() => {
    if (!fileLiveRange.active || !currentFileResults || !files.some(f => f.id === activeTab)) return null;
    const bCol = resolveBCol(currentFileResults);
    const map = new Map<string, AnalysisResult | null>();
    (currentFileResults.results || []).forEach(res => {
      if (res.columnName === 'Combined') { map.set(res.columnName, null); return; }
      map.set(res.columnName, processFileData(currentFileResults.data, bCol, res.columnName, undefined, undefined, { min: fileLiveMinBEff, max: fileLiveMaxBEff }));
    });
    return map;
  }, [fileLiveRange.active, currentFileResults, files, activeTab, resolveBCol, fileLiveMinBEff, fileLiveMaxBEff, processFileData]);

  // Commit the staged (sidebar) values to the effective settings. Shared by the
  // "Apply All Settings" button and by Process All, so the numbers on screen are
  // always the numbers that get used — no separate Apply click required.
  const commitStagedSettings = () => {
    setMinB(tempMinB);
    setMaxB(tempMaxB);
    setPolyDegree(tempPolyDegree);
    setSpikeRemoval(tempSpikeRemoval);
    setSpikeKernel(tempSpikeKernel);
    setSpikeThreshold(tempSpikeThreshold);
    setSmoothing(tempSmoothing);
    setSmoothWindow(tempSmoothWindow);
    setInterpPoints(tempInterpPoints);
    setWindowType(tempWindowType);
    setUseTargetInterval(tempUseTargetInterval);
    setTargetFreqInterval(tempTargetFreqInterval);
    setMinPeakWidth(tempMinPeakWidth);
    setComplexMode(tempComplexMode);
    setPhase(tempPhase);
    setBranchTolerance(tempBranchTolerance);
    setMinPeaksPerBranch(tempMinPeaksPerBranch);
    // Keep the array identity when nothing changed, so this doesn't look like a
    // parameter change and needlessly clear the results.
    setFreqThresholds(prev => {
      const same = prev.length === tempFreqThresholds.length &&
        prev.every((p, i) => p.limit === tempFreqThresholds[i].limit && p.threshold === tempFreqThresholds[i].threshold);
      return same ? prev : tempFreqThresholds.map(t => ({ ...t }));
    });
  };

  // ---- Sample / Orientation management ----
  // Each orientation owns its own files[] plus a full processing-settings
  // snapshot. Only one orientation is "live" in the ordinary `files`/`minB`/…
  // state at a time; switching orientations saves the outgoing one and
  // restores the incoming one, so the rest of the app (every tab) keeps
  // reading the same top-level state it always has.
  const getSettingsSnapshot = useCallback((): Record<string, any> => ({
    selectedBColumn, selectedRColumns, minB, maxB, polyDegree,
    interpPoints, windowType, spikeRemoval, spikeKernel, spikeThreshold,
    smoothing, smoothWindow, freqThresholds, complexMode, phase,
    useTargetInterval, targetFreqInterval, blmMode, blmPerDegree, blmOffset,
    selectedPrefix, branchTolerance, minPeaksPerBranch, symmetryCenter,
    symmetryMaxFreq, symmetryMinFreq, maxDerivMinB, maxDerivMaxB,
    minPeakWidth, autoThreshold, readAngleFromTitle, prefixLocked,
  }), [
    selectedBColumn, selectedRColumns, minB, maxB, polyDegree,
    interpPoints, windowType, spikeRemoval, spikeKernel, spikeThreshold,
    smoothing, smoothWindow, freqThresholds, complexMode, phase,
    useTargetInterval, targetFreqInterval, blmMode, blmPerDegree, blmOffset,
    selectedPrefix, branchTolerance, minPeaksPerBranch, symmetryCenter,
    symmetryMaxFreq, symmetryMinFreq, maxDerivMinB, maxDerivMaxB,
    minPeakWidth, autoThreshold, readAngleFromTitle, prefixLocked,
  ]);

  const applySettingsSnapshot = useCallback((data: Record<string, any>) => {
    if (!data) return;
    if (data.selectedBColumn !== undefined) setSelectedBColumn(data.selectedBColumn);
    if (data.selectedRColumns !== undefined) setSelectedRColumns(data.selectedRColumns);
    if (data.minB !== undefined) { setMinB(data.minB); setTempMinB(data.minB); }
    if (data.maxB !== undefined) { setMaxB(data.maxB); setTempMaxB(data.maxB); }
    if (data.polyDegree !== undefined) { setPolyDegree(data.polyDegree); setTempPolyDegree(data.polyDegree); }
    if (data.interpPoints !== undefined) { setInterpPoints(data.interpPoints); setTempInterpPoints(data.interpPoints); }
    if (data.windowType !== undefined) { setWindowType(data.windowType); setTempWindowType(data.windowType); }
    if (data.spikeRemoval !== undefined) { setSpikeRemoval(data.spikeRemoval); setTempSpikeRemoval(data.spikeRemoval); }
    if (data.spikeKernel !== undefined) { setSpikeKernel(data.spikeKernel); setTempSpikeKernel(data.spikeKernel); }
    if (data.spikeThreshold !== undefined) { setSpikeThreshold(data.spikeThreshold); setTempSpikeThreshold(data.spikeThreshold); }
    if (data.smoothing !== undefined) { setSmoothing(data.smoothing); setTempSmoothing(data.smoothing); }
    if (data.smoothWindow !== undefined) { setSmoothWindow(data.smoothWindow); setTempSmoothWindow(data.smoothWindow); }
    if (data.freqThresholds !== undefined) { setFreqThresholds(data.freqThresholds); setTempFreqThresholds(data.freqThresholds); }
    if (data.complexMode !== undefined) { setComplexMode(data.complexMode); setTempComplexMode(data.complexMode); }
    if (data.phase !== undefined) { setPhase(data.phase); setTempPhase(data.phase); }
    if (data.useTargetInterval !== undefined) { setUseTargetInterval(data.useTargetInterval); setTempUseTargetInterval(data.useTargetInterval); }
    if (data.targetFreqInterval !== undefined) { setTargetFreqInterval(data.targetFreqInterval); setTempTargetFreqInterval(data.targetFreqInterval); }
    if (data.minPeakWidth !== undefined) { setMinPeakWidth(data.minPeakWidth); setTempMinPeakWidth(data.minPeakWidth); }
    if (data.autoThreshold !== undefined) setAutoThreshold(data.autoThreshold);
    if (data.blmMode !== undefined) setBlmMode(data.blmMode);
    if (data.blmPerDegree !== undefined) setBlmPerDegree(data.blmPerDegree);
    if (data.blmOffset !== undefined) setBlmOffset(data.blmOffset);
    if (data.selectedPrefix !== undefined) setSelectedPrefix(data.selectedPrefix);
    if (data.branchTolerance !== undefined) { setBranchTolerance(data.branchTolerance); setTempBranchTolerance(data.branchTolerance); }
    if (data.minPeaksPerBranch !== undefined) { setMinPeaksPerBranch(data.minPeaksPerBranch); setTempMinPeaksPerBranch(data.minPeaksPerBranch); }
    if (data.symmetryCenter !== undefined) setSymmetryCenter(data.symmetryCenter);
    if (data.symmetryMaxFreq !== undefined) setSymmetryMaxFreq(data.symmetryMaxFreq);
    if (data.symmetryMinFreq !== undefined) setSymmetryMinFreq(data.symmetryMinFreq);
    if (data.maxDerivMinB !== undefined) setMaxDerivMinB(data.maxDerivMinB);
    if (data.maxDerivMaxB !== undefined) setMaxDerivMaxB(data.maxDerivMaxB);
    if (data.readAngleFromTitle !== undefined) setReadAngleFromTitle(data.readAngleFromTitle);
    if (data.prefixLocked !== undefined) setPrefixLocked(data.prefixLocked);
  }, []);

  const emptyTemperatureBucket = (settings: Record<string, any> = {}): TemperatureBucket => ({ groups: [], settings });

  // Snapshot (files, settings) into whichever bucket/group is currently live,
  // for one orientation's entry — used before switching away from it (to
  // another orientation, another bucket, another temperature group, or out to
  // disk). No-ops safely if there's no active temperature group yet.
  const captureActive = useCallback((o: OrientationEntry, outgoingSettings: Record<string, any>): OrientationEntry => {
    if (activeBucket === 'angle') {
      return { ...o, angle: { files, settings: outgoingSettings } };
    }
    return {
      ...o,
      temperature: {
        settings: outgoingSettings,
        groups: o.temperature.groups.map(g => g.id === activeTempGroupId ? { ...g, files } : g),
      },
    };
  }, [activeBucket, activeTempGroupId, files]);

  const loadAngleBucket = useCallback((bucket: FileBucket | undefined) => {
    setFiles(bucket?.files ?? []);
    if (bucket?.settings) applySettingsSnapshot(bucket.settings);
  }, [applySettingsSnapshot]);

  // Loads a temperature bucket, preferring `preferredGroupId` if it still
  // exists in it, else its first group, else empty (no groups yet).
  const loadTemperatureBucket = useCallback((bucket: TemperatureBucket | undefined, preferredGroupId?: string) => {
    if (bucket?.settings) applySettingsSnapshot(bucket.settings);
    const group = bucket?.groups.find(g => g.id === preferredGroupId) ?? bucket?.groups[0];
    setActiveTempGroupId(group?.id ?? '');
    setFiles(group?.files ?? []);
  }, [applySettingsSnapshot]);

  const switchOrientation = useCallback((targetId: string) => {
    if (targetId === activeOrientationId) return;
    const outgoingSettings = getSettingsSnapshot();
    const target = orientations.find(o => o.id === targetId);
    setOrientations(prev => prev.map(o => o.id === activeOrientationId ? captureActive(o, outgoingSettings) : o));
    if (target) {
      if (activeBucket === 'angle') loadAngleBucket(target.angle);
      else loadTemperatureBucket(target.temperature);
    }
    setActiveOrientationId(targetId);
    setActiveTab('dashboard');
  }, [activeOrientationId, orientations, activeBucket, getSettingsSnapshot, captureActive, loadAngleBucket, loadTemperatureBucket]);

  // Switch which sweep type (angle vs temperature) is loaded for the SAME
  // orientation — the two buckets are independent file sets + settings.
  const switchBucket = useCallback((targetBucket: 'angle' | 'temperature') => {
    if (targetBucket === activeBucket) return;
    const outgoingSettings = getSettingsSnapshot();
    const current = orientations.find(o => o.id === activeOrientationId);
    setOrientations(prev => prev.map(o => o.id === activeOrientationId ? captureActive(o, outgoingSettings) : o));
    if (current) {
      if (targetBucket === 'angle') loadAngleBucket(current.angle);
      else loadTemperatureBucket(current.temperature, activeTempGroupId);
    }
    setActiveBucket(targetBucket);
    const angleOnlyTabs = new Set(['torque', 'landau', 'harmonics', 'symmetry', 'branch']);
    if (targetBucket === 'temperature' && angleOnlyTabs.has(activeTab)) setActiveTab('dashboard');
  }, [activeBucket, activeOrientationId, orientations, activeTempGroupId, getSettingsSnapshot, captureActive, loadAngleBucket, loadTemperatureBucket, activeTab]);

  // ---- Temperature-sweep groups (multiple fixed-angle T-sweeps per orientation) ----
  const addTemperatureGroup = useCallback((label: string) => {
    const newId = `tg-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const outgoingSettings = getSettingsSnapshot();
    setOrientations(prev => prev.map(o => {
      if (o.id !== activeOrientationId) return o;
      const groups = o.temperature.groups.map(g => g.id === activeTempGroupId ? { ...g, files } : g);
      return { ...o, temperature: { settings: outgoingSettings, groups: [...groups, { id: newId, label: label || `Group ${groups.length + 1}`, files: [] }] } };
    }));
    setFiles([]);
    setActiveTempGroupId(newId);
  }, [activeOrientationId, activeTempGroupId, files, getSettingsSnapshot]);

  const switchTemperatureGroup = useCallback((targetId: string) => {
    if (targetId === activeTempGroupId) return;
    const outgoingSettings = getSettingsSnapshot();
    const current = orientations.find(o => o.id === activeOrientationId);
    const targetGroup = current?.temperature.groups.find(g => g.id === targetId);
    setOrientations(prev => prev.map(o => o.id === activeOrientationId
      ? { ...o, temperature: { settings: outgoingSettings, groups: o.temperature.groups.map(g => g.id === activeTempGroupId ? { ...g, files } : g) } }
      : o));
    setFiles(targetGroup?.files ?? []);
    setActiveTempGroupId(targetId);
  }, [activeOrientationId, activeTempGroupId, orientations, files, getSettingsSnapshot]);

  const renameTemperatureGroup = useCallback((id: string, label: string) => {
    setOrientations(prev => prev.map(o => o.id === activeOrientationId
      ? { ...o, temperature: { ...o.temperature, groups: o.temperature.groups.map(g => g.id === id ? { ...g, label } : g) } }
      : o));
  }, [activeOrientationId]);

  const removeTemperatureGroup = useCallback((id: string) => {
    const current = orientations.find(o => o.id === activeOrientationId);
    const remaining = (current?.temperature.groups ?? []).filter(g => g.id !== id);
    setOrientations(prev => prev.map(o => o.id === activeOrientationId ? { ...o, temperature: { ...o.temperature, groups: remaining } } : o));
    if (id === activeTempGroupId) {
      setFiles(remaining[0]?.files ?? []);
      setActiveTempGroupId(remaining[0]?.id ?? '');
    }
  }, [orientations, activeOrientationId, activeTempGroupId]);

  const addOrientation = useCallback((label: string) => {
    const newId = `ori-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const outgoingSettings = getSettingsSnapshot();
    setOrientations(prev => [
      ...prev.map(o => o.id === activeOrientationId ? captureActive(o, outgoingSettings) : o),
      {
        id: newId, label: label || `Orientation ${prev.length + 1}`,
        angle: { files: [], settings: outgoingSettings },
        temperature: emptyTemperatureBucket(outgoingSettings),
      },
    ]);
    setFiles([]);
    setActiveBucket('angle');
    setActiveTempGroupId('');
    setActiveOrientationId(newId);
    setActiveTab('dashboard');
  }, [activeOrientationId, getSettingsSnapshot, captureActive]);

  const renameOrientation = useCallback((id: string, label: string) => {
    setOrientations(prev => prev.map(o => o.id === id ? { ...o, label } : o));
  }, []);

  const removeOrientation = useCallback((id: string) => {
    const remaining = orientations.filter(o => o.id !== id);
    if (id !== activeOrientationId) { setOrientations(remaining); return; }
    if (remaining.length > 0) {
      if (activeBucket === 'angle') loadAngleBucket(remaining[0].angle);
      else loadTemperatureBucket(remaining[0].temperature);
      setActiveOrientationId(remaining[0].id);
      setOrientations(remaining);
    } else {
      const newId = `ori-${Date.now()}`;
      const settings = getSettingsSnapshot();
      setFiles([]);
      setActiveBucket('angle');
      setActiveTempGroupId('');
      setActiveOrientationId(newId);
      setOrientations([{ id: newId, label: 'Orientation 1', angle: { files: [], settings }, temperature: emptyTemperatureBucket(settings) }]);
    }
  }, [orientations, activeOrientationId, activeBucket, loadAngleBucket, loadTemperatureBucket, getSettingsSnapshot]);

  // Full-session export: every orientation's raw files + processed results +
  // settings, as one local JSON file named after the sample ID.
  // FFT/plotData/detectedPeaks are all cheaply regenerable from raw data +
  // settings via "Process All" — they're also the bulk of a processed
  // dataset's size (a full spectrum per column per file). Stripping them
  // before serializing keeps the saved file small and JSON.stringify fast;
  // the live `orientations` state (used by the charts on screen) is
  // untouched. Reloading a session just needs one "Process All" click to
  // regenerate everything.
  const stripResultsForSave = (files: FileEntry[]): FileEntry[] => files.map(f => ({ ...f, results: [] }));

  const buildSessionPayload = useCallback(() => {
    const outgoingSettings = getSettingsSnapshot();
    const finalOrientations = orientations.map(o => o.id === activeOrientationId ? captureActive(o, outgoingSettings) : o);
    const serializable = finalOrientations.map(o => ({
      ...o,
      angle: { ...o.angle, files: stripResultsForSave(o.angle.files) },
      temperature: { ...o.temperature, groups: o.temperature.groups.map(g => ({ ...g, files: stripResultsForSave(g.files) })) },
    }));
    const payload = {
      version: 3,
      sampleId,
      activeOrientationId,
      activeBucket,
      activeTempGroupId,
      orientations: serializable,
      savedAt: new Date().toISOString(),
    };
    return { payload, finalOrientations };
  }, [sampleId, orientations, activeOrientationId, activeBucket, activeTempGroupId, getSettingsSnapshot, captureActive]);

  // Once the user has saved or loaded via the native file picker, this holds
  // a writable handle to that exact file, so later saves overwrite it in
  // place instead of downloading a new copy each time. Chromium-only
  // (File System Access API); falls back to a plain download elsewhere. Not
  // persisted across reloads — a fresh page load needs one manual Save/Load
  // to (re)establish the link.
  const sessionFileHandleRef = useRef<any>(null);
  const [linkedSessionFileName, setLinkedSessionFileName] = useState<string | null>(null);
  const supportsFileSystemAccess = typeof window !== 'undefined' && typeof (window as any).showSaveFilePicker === 'function';
  // Visible confirmation that each click actually did something — a silent
  // handle-overwrite (no dialog, unlike the first save) gave no feedback at
  // all, which read as "only the first save works" even when later saves
  // were succeeding.
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [lastSavedAt, setLastSavedAt] = useState<Date | null>(null);

  const downloadSessionBlob = (payload: any) => {
    const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(sampleId || 'sample').replace(/[^a-zA-Z0-9_-]+/g, '_')}_session.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Chrome can silently drop a handle's write permission (tab backgrounded a
  // while, browser restart, etc.) — re-request before trusting it instead of
  // finding out via a thrown error from createWritable().
  const ensureWritePermission = async (handle: any): Promise<boolean> => {
    if (typeof handle.queryPermission !== 'function') return true; // older/partial impl — just try the write
    const opts = { mode: 'readwrite' as const };
    if ((await handle.queryPermission(opts)) === 'granted') return true;
    if (typeof handle.requestPermission !== 'function') return false;
    return (await handle.requestPermission(opts)) === 'granted';
  };

  // Manual save (button click only — never called automatically). Writes to
  // the linked file if one exists; otherwise offers the native "Save As"
  // picker to establish one (Chromium), or falls back to a download.
  const saveSessionToFile = useCallback(async () => {
    if (!sampleId) return;
    setSaveStatus('saving');
    const { payload, finalOrientations } = buildSessionPayload();
    setOrientations(finalOrientations);
    const json = JSON.stringify(payload);

    if (sessionFileHandleRef.current) {
      try {
        if (!(await ensureWritePermission(sessionFileHandleRef.current))) {
          throw new Error('write permission not granted');
        }
        const writable = await sessionFileHandleRef.current.createWritable();
        await writable.write(json);
        await writable.close();
        setSaveStatus('saved');
        setLastSavedAt(new Date());
        return;
      } catch {
        // Handle went stale (file moved/permission revoked) — fall through
        // to re-establishing one below.
        sessionFileHandleRef.current = null;
        setLinkedSessionFileName(null);
      }
    }

    if (supportsFileSystemAccess) {
      try {
        const handle = await (window as any).showSaveFilePicker({
          suggestedName: `${sampleId.replace(/[^a-zA-Z0-9_-]+/g, '_')}_session.json`,
          types: [{ description: 'Session JSON', accept: { 'application/json': ['.json'] } }],
        });
        const writable = await handle.createWritable();
        await writable.write(json);
        await writable.close();
        sessionFileHandleRef.current = handle;
        setLinkedSessionFileName(handle.name);
        setSaveStatus('saved');
        setLastSavedAt(new Date());
        return;
      } catch (err: any) {
        if (err?.name === 'AbortError') { setSaveStatus('idle'); return; } // user cancelled the picker
        // Fall through to the download fallback below.
      }
    }
    try {
      downloadSessionBlob(payload);
      setSaveStatus('saved');
      setLastSavedAt(new Date());
    } catch {
      setSaveStatus('error');
      setError('세션을 저장하지 못했습니다.');
    }
  }, [sampleId, supportsFileSystemAccess, buildSessionPayload]);

  const applyLoadedSession = useCallback((payload: any) => {
    if (!payload || !Array.isArray(payload.orientations) || payload.orientations.length === 0) {
      throw new Error('empty session');
    }
    // v1: one flat {files,settings,mode} per orientation.
    // v2: {angle: FileBucket, temperature: FileBucket} — one flat temperature file list.
    // v3 (current): temperature is {groups: TemperatureSweepGroup[], settings}.
    const normalized: OrientationEntry[] = payload.orientations.map((o: any) => {
      const angle: FileBucket = o.angle ?? { files: o.mode === 'temperature' ? [] : (o.files ?? []), settings: o.mode === 'temperature' ? {} : (o.settings ?? {}) };
      let temperature: TemperatureBucket;
      if (o.temperature?.groups) {
        temperature = o.temperature;
      } else if (o.temperature) {
        // v2 flat bucket -> a single unnamed group
        const t = o.temperature as { files: FileEntry[]; settings: Record<string, any> };
        temperature = { settings: t.settings ?? {}, groups: t.files?.length ? [{ id: `tg-${o.id}`, label: 'Group 1', files: t.files }] : [] };
      } else if (o.mode === 'temperature' && o.files) {
        temperature = { settings: o.settings ?? {}, groups: [{ id: `tg-${o.id}`, label: 'Group 1', files: o.files }] };
      } else {
        temperature = { settings: {}, groups: [] };
      }
      return { id: o.id, label: o.label, angle, temperature };
    });
    const activeId = payload.activeOrientationId && normalized.some((o) => o.id === payload.activeOrientationId)
      ? payload.activeOrientationId
      : normalized[0].id;
    const active = normalized.find((o) => o.id === activeId);
    const bucket: 'angle' | 'temperature' = payload.activeBucket === 'temperature' ? 'temperature' : 'angle';
    setOrientations(normalized);
    setActiveOrientationId(activeId);
    setActiveBucket(bucket);
    if (bucket === 'angle') loadAngleBucket(active?.angle);
    else loadTemperatureBucket(active?.temperature, payload.activeTempGroupId);
    setSampleId(payload.sampleId || 'Untitled');
    setActiveTab('dashboard');
    setError(null);
  }, [loadAngleBucket, loadTemperatureBucket]);

  // Fallback loader (any browser): plain <input type="file">, no write-back handle.
  const loadSessionFromFile = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const f = event.target.files?.[0];
    event.target.value = '';
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        applyLoadedSession(JSON.parse(String(ev.target?.result)));
      } catch {
        setError('세션 파일을 불러오지 못했습니다 — 형식이 올바르지 않습니다.');
      }
    };
    reader.readAsText(f);
  }, [applyLoadedSession]);

  // Chromium loader: native "Open" picker, keeps the handle so later Save
  // Session calls overwrite this exact file instead of downloading a copy.
  const loadSessionViaPicker = useCallback(async () => {
    if (!supportsFileSystemAccess) return;
    try {
      const [handle] = await (window as any).showOpenFilePicker({
        types: [{ description: 'Session JSON', accept: { 'application/json': ['.json'] } }],
      });
      const file = await handle.getFile();
      const text = await file.text();
      applyLoadedSession(JSON.parse(text));
      sessionFileHandleRef.current = handle;
      setLinkedSessionFileName(handle.name);
    } catch (err: any) {
      if (err?.name === 'AbortError') return;
      setError('세션 파일을 불러오지 못했습니다 — 형식이 올바르지 않습니다.');
    }
  }, [supportsFileSystemAccess, applyLoadedSession]);

  // Process All runs with the settings as they were last applied — staged
  // sidebar edits need "Apply All Settings" first.
  const handleProcessAll = () => {
    if (files.length === 0) return;

    setIsProcessing(true);
    setTimeout(() => {
      try {
        const summary: { blm: number; freq: number; amp: number; fileName: string; id: string; temp: number }[] = [];
        const assemble: { x: number; fft: FFTResult[]; fileName: string; peaks?: FFTResult[] }[] = [];
        const rawPlot: { x: number; y: number; fileName: string; columnName: string }[] = [];
        const maxDerivPlot: { x: number; y: number; fileName: string; columnName: string }[] = [];
        const rawSeries: { 
          fileName: string; 
          columnName: string; 
          blm: number; 
          peak: { b: number; val: number; type: 'max' | 'min' };
          data: { b: number; val: number }[];
          invBData: { invB: number; val: number }[];
        }[] = [];
        
        // Calculate total series to be plotted for adaptive downsampling
        let totalSeriesCount = 0;
        files.forEach(file => {
          totalSeriesCount += selectedPrefix
            ? dropQuadrature(file.columns.filter(c => c.startsWith(selectedPrefix + '_'))).length
            : dropQuadrature(selectedRColumns.filter(c => file.columns.includes(c)), file.columns).length;
        });

        // Target total points for the chart to maintain performance
        const totalPointsTarget = 1000;
        const pointsPerSeries = totalSeriesCount > 0 
          ? Math.max(30, Math.floor(totalPointsTarget / totalSeriesCount))
          : 300;

        const updatedFiles = [...files];

        // Make "Process All" self-sufficient: if the current B window holds too
        // few points to analyse in every file (a stale window from a different
        // magnet, or a hand-typed range), fall back to the measured field range
        // for this run and move the sidebar inputs with it, instead of silently
        // producing an empty dashboard.
        const runRange = (() => {
          let dataMin = Infinity, dataMax = -Infinity, bestInWindow = 0;
          const winLo = minB === '' ? -Infinity : Number(minB);
          const winHi = maxB === '' ? Infinity : Number(maxB);
          files.forEach(file => {
            const bc = resolveBCol(file);
            let inWindow = 0;
            for (const row of file.data) {
              const b = parseFloat(String(row[bc]));
              if (!isFinite(b)) continue;
              if (b < dataMin) dataMin = b;
              if (b > dataMax) dataMax = b;
              if (b >= winLo && b <= winHi) inWindow++;
            }
            if (inWindow > bestInWindow) bestInWindow = inWindow;
          });
          if (bestInWindow >= 10 || !isFinite(dataMin) || !(dataMax > dataMin)) return undefined;
          const lo = Math.max(0, Math.floor(dataMin * 10) / 10);
          const hi = Math.ceil(dataMax * 10) / 10;
          suppressResultClear.current = true;   // don't wipe the results of this run
          setMinB(lo); setMaxB(hi);
          setTempMinB(lo); setTempMaxB(hi);
          setError(`B range ${winLo}–${winHi} T 안에 데이터가 없어 측정 범위 ${lo}–${hi} T 로 자동 조정했습니다.`);
          return { min: lo, max: hi };
        })();

        files.forEach((file, fIdx) => {
          // Use smart matching logic here too
          const bCol = file.columns.find(c => c === selectedBColumn) || 
                       file.columns.find(c => c.toLowerCase() === selectedBColumn.toLowerCase()) ||
                       file.columns.find(k => {
                         const low = k.toLowerCase();
                         return (low === 'b' || low.startsWith('b ') || low.startsWith('b_') || low.includes('field') || low.includes('magnetic')) && 
                                !low.includes('blm') && !low.includes('angle') && !low.includes('position');
                       }) ||
                       file.columns.find(k => {
                         const low = k.toLowerCase();
                         return low.includes('b') && !low.includes('blm') && !low.includes('angle') && !low.includes('position');
                       }) ||
                       file.columns[0];

          let rCols: string[] = [];
          if (selectedPrefix) {
            rCols = file.columns.filter(c => c.startsWith(selectedPrefix + '_'));
          } else {
            rCols = selectedRColumns.map(target => {
                return file.columns.find(c => c === target) || 
                       file.columns.find(c => c.toLowerCase() === target.toLowerCase());
            }).filter((c): c is string => !!c);
          }
          
          // Strictly exclude Output2 from calculation whenever Output1 exists
          rCols = dropQuadrature(rCols, file.columns);

          // Unchecked files skip the whole FFT pipeline (background removal, FFT,
          // peak detection, branch input). Their raw traces below are still
          // collected, so the Raw and Torque tabs keep showing them.
          const skipFFT = fftSkip.has(file.id);

          const resultsList = skipFFT
            ? []
            : complexMode && rCols.length >= 2
              ? [processFileData(file.data, bCol, 'Combined', rCols[0], rCols[1], runRange)]
              : rCols.map(rCol => processFileData(file.data, bCol, rCol, undefined, undefined, runRange));

          const validResults = resultsList.filter((r): r is AnalysisResult => r !== null);

          // Store results in the file entry
          updatedFiles[fIdx] = { ...file, results: validResults };

          let blmValue = 0;
          if (readAngleFromTitle) {
            // Extract the first number in the filename (e.g. "26down.csv" → 26)
            const fromName = angleFromFileName(file.name);
            blmValue = fromName != null ? fromName : fIdx + 1;
          } else if (blmMode === 'step') {
            blmValue = fIdx + 1;
          } else {
            // Check columns first (user request: "BLM, BLM position 이런 단어가 들어간 column의 값")
            const blmCol = file.columns.find(c => 
              c.toLowerCase().includes('blm') || 
              c.toLowerCase().includes('position') || 
              c.toLowerCase().includes('angle') ||
              c.toLowerCase().includes('step')
            );
            
            let blmStr = '0';
            if (blmCol && file.data.length > 0) {
              blmStr = String(file.data[0][blmCol]);
            } else {
              // Fallback to metadata
              const blmEntry = Object.entries(file.metadata).find(([k]) => 
                k.toLowerCase().includes('position') || 
                k.toLowerCase().includes('blm') || 
                k.toLowerCase().includes('angle')
              );
              blmStr = blmEntry ? String(blmEntry[1]) : '0';
            }
            
            // Extract the first number from the string to handle cases like "93 (deg)" or "Pos: 93"
            const numMatch = blmStr.match(/[-+]?[0-9]*\.?[0-9]+/);
            blmValue = numMatch ? parseFloat(numMatch[0]) : parseFloat(blmStr);
            if (isNaN(blmValue)) blmValue = fIdx + 1; // Fallback to step if metadata/column is NaN
          }
          
          const finalBlm = Number(((blmValue / blmPerDegree) + blmOffset).toFixed(2));

          // Extract raw data for plotting. Without a prefix this used to collect
          // nothing at all, which left the Raw tab empty for datasets whose
          // channels are named per file and can't share one prefix — fall back to
          // the same columns the FFT just used.
          const rawTargetCols = selectedPrefix
            ? dropQuadrature(file.columns.filter(c => c.startsWith(selectedPrefix + '_')))
            : rCols;
          if (rawTargetCols.length > 0 && file.data.length > 0) {
            const targetCols = rawTargetCols;

            // Parse the B column once for the whole file — it's identical across
            // every R column, so re-parsing it per column was pure waste.
            const parsedB = new Float64Array(file.data.length);
            for (let i = 0; i < file.data.length; i++) parsedB[i] = parseFloat(String(file.data[i][bCol]));

            targetCols.forEach(col => {
              // 2. Full series for Raw vs B-Field (all files)
              const seriesData: { b: number; val: number }[] = [];
              for (let i = 0; i < file.data.length; i++) {
                const bVal = parsedB[i];
                const rVal = parseFloat(String(file.data[i][col]));
                if (!isNaN(bVal) && !isNaN(rVal)) {
                  seriesData.push({ b: bVal, val: rVal });
                }
              }

              // Sort by B once and reuse — the derivative, downsampling and
              // max-B lookup all previously re-sorted or re-scanned this array.
              const sortedByB = [...seriesData].sort((a, b) => a.b - b.b);

              // 1. Single point for Raw vs BLM (Value at Max B)
              if (sortedByB.length > 0) {
                const maxBPoint = sortedByB[sortedByB.length - 1];
                rawPlot.push({ x: finalBlm, y: maxBPoint.val, fileName: file.name, columnName: col });
              }

              // Calculate derivative and find field at max derivative (specifically in range, with smoothing)
              if (seriesData.length > 1) {
                const filteredForDeriv = sortedByB.filter(d => d.b >= maxDerivMinB && d.b <= maxDerivMaxB);

                if (filteredForDeriv.length > 5) {
                  // 1. Calculate raw derivatives
                  const rawDerivs: { b: number; deriv: number }[] = [];
                  for (let i = 1; i < filteredForDeriv.length; i++) {
                    const dB = filteredForDeriv[i].b - filteredForDeriv[i - 1].b;
                    const dR = filteredForDeriv[i].val - filteredForDeriv[i - 1].val;
                    if (dB !== 0) {
                      rawDerivs.push({
                        b: (filteredForDeriv[i].b + filteredForDeriv[i - 1].b) / 2,
                        deriv: Math.abs(dR / dB)
                      });
                    }
                  }

                  // 2. Apply moving average smoothing to derivatives (window size 7)
                  const windowSize = 7;
                  const smoothedDerivs: { b: number; deriv: number }[] = [];
                  for (let i = 0; i < rawDerivs.length; i++) {
                    const start = Math.max(0, i - Math.floor(windowSize / 2));
                    const end = Math.min(rawDerivs.length - 1, i + Math.floor(windowSize / 2));
                    let sum = 0;
                    for (let j = start; j <= end; j++) {
                      sum += rawDerivs[j].deriv;
                    }
                    smoothedDerivs.push({
                      b: rawDerivs[i].b,
                      deriv: sum / (end - start + 1)
                    });
                  }

                  // 3. Find peak of smoothed derivatives
                  if (smoothedDerivs.length > 0) {
                    let peak = smoothedDerivs[0];
                    for (let i = 1; i < smoothedDerivs.length; i++) {
                      if (smoothedDerivs[i].deriv > peak.deriv) peak = smoothedDerivs[i];
                    }
                    maxDerivPlot.push({ x: finalBlm, y: peak.b, fileName: file.name, columnName: col });
                  }
                }
              }

              // Adaptive Downsampling with Linear Interpolation for B-Field (Full Range)
              let processedData: { b: number; val: number }[] = [];
              if (seriesData.length > pointsPerSeries) {
                const minBData = sortedByB.length > 0 ? sortedByB[0].b : 0;
                const maxBData = sortedByB.length > 0 ? sortedByB[sortedByB.length - 1].b : 1;
                const step = (maxBData - minBData) / (pointsPerSeries - 1);
                const sorted = sortedByB;
                let ptr = 0;
                for (let i = 0; i < pointsPerSeries; i++) {
                  const targetB = minBData + i * step;
                  while (ptr < sorted.length && sorted[ptr].b < targetB) ptr++;
                  if (ptr === 0) processedData.push({ b: targetB, val: sorted[0].val });
                  else if (ptr >= sorted.length) processedData.push({ b: targetB, val: sorted[sorted.length - 1].val });
                  else {
                    const s0 = sorted[ptr - 1];
                    const s1 = sorted[ptr];
                    const t = (targetB - s0.b) / (s1.b - s0.b);
                    processedData.push({ b: targetB, val: s0.val + t * (s1.val - s0.val) });
                  }
                }
              } else {
                processedData = seriesData.map(s => ({ b: s.b, val: s.val }));
              }

              // Reuse already-computed result — avoids a full redundant FFT per column
              const analysisRes = validResults.find(r => r.columnName === col) || null;
              let processedInvBData: { invB: number; val: number }[] = [];
              
              if (analysisRes && analysisRes.plotData.length > 0) {
                // The plotData in analysisRes is already filtered by minB/maxB
                // We want to plot deltaR vs invB
                const oscData = analysisRes.plotData.map(d => ({ invB: d.invB, val: d.deltaR }));
                
                // Sort by invB for plotting
                const sortedOsc = _.sortBy(oscData, 'invB');
                
                // Downsample if needed
                if (sortedOsc.length > pointsPerSeries) {
                  const minInv = sortedOsc[0].invB;
                  const maxInv = sortedOsc[sortedOsc.length - 1].invB;
                  const step = (maxInv - minInv) / (pointsPerSeries - 1);
                  let ptr = 0;
                  for (let i = 0; i < pointsPerSeries; i++) {
                    const targetInv = minInv + i * step;
                    while (ptr < sortedOsc.length && sortedOsc[ptr].invB < targetInv) ptr++;
                    if (ptr === 0) processedInvBData.push({ invB: targetInv, val: sortedOsc[0].val });
                    else if (ptr >= sortedOsc.length) processedInvBData.push({ invB: targetInv, val: sortedOsc[sortedOsc.length - 1].val });
                    else {
                      const p0 = sortedOsc[ptr - 1];
                      const p1 = sortedOsc[ptr];
                      const t = (targetInv - p0.invB) / (p1.invB - p0.invB);
                      processedInvBData.push({ invB: targetInv, val: p0.val + t * (p1.val - p0.val) });
                    }
                  }
                } else {
                  processedInvBData = sortedOsc;
                }
              }
                
              // Single pass for both extremes instead of four full-array scans
              let maxPt = seriesData[0], minPt = seriesData[0];
              for (let i = 1; i < seriesData.length; i++) {
                if (seriesData[i].val > maxPt.val) maxPt = seriesData[i];
                if (seriesData[i].val < minPt.val) minPt = seriesData[i];
              }
              const maxVal = maxPt?.val || 0;
              const minVal = minPt?.val || 0;
              const peakType = Math.abs(maxVal) >= Math.abs(minVal) ? 'max' : 'min';
              const peakPoint = peakType === 'max' ? maxPt : minPt;

              rawSeries.push({ 
                fileName: file.name, 
                columnName: col, 
                blm: finalBlm, 
                peak: { b: peakPoint?.b || 0, val: peakPoint?.val || 0, type: peakType as 'max' | 'min' },
                data: processedData,
                invBData: processedInvBData
              });
            });
          }
          
          validResults.forEach(res => {
            // Store full FFT for assemble
            assemble.push({
              x: finalBlm,
              fft: res.fftResults,
              fileName: file.name,
              peaks: res.detectedPeaks
            });

            const temp = getFileTemperature(file);

            res.detectedPeaks.forEach((peak, pIdx) => {
              summary.push({
                blm: finalBlm,
                freq: peak.frequency,
                amp: peak.amplitude,
                fileName: file.name,
                id: `${file.id}-${res.columnName}-${pIdx}`,
                temp
              });
            });
          });
        });
        
        setBatchSummary(_.orderBy(summary, ['blm', 'amp'], ['asc', 'desc']));
        
        // Calculate global suggested max frequency (1.5x the highest peak found across all files)
        const allPeaks = summary.map(s => s.freq);
        if (allPeaks.length > 0) {
          let maxPeak = 0;
          for (const f of allPeaks) if (f > maxPeak) maxPeak = f;
          // Round the auto display max up to a clean value (nearest 100) so axis
          // ticks read as e.g. 3000 rather than 2989.0630640987928.
          setGlobalSuggestedMaxFreq(Math.max(100, Math.ceil(maxPeak * 1.5 / 100) * 100));
        } else {
          setGlobalSuggestedMaxFreq(1000);
        }

        setAssembleData(_.orderBy(assemble, ['x'], ['asc']));
        setRawPlotData(rawPlot);
        setMaxDerivPlotData(maxDerivPlot);
        setAllRawSeries(rawSeries);
        setFiles(updatedFiles);
        const uniqueBlms = Array.from(new Set(rawSeries.map(s => s.blm)));
        setActiveBlms(new Set(uniqueBlms));
        
        // Auto-run branch analysis after processing all
        runBranchAnalysis(summary);
      } catch (err) {
        setError("Error during batch processing: " + (err instanceof Error ? err.message : String(err)));
      } finally {
        setIsProcessing(false);
      }
    }, 100);
  };

  const runBranchAnalysis = (summaryData: { blm: number; freq: number; amp: number; fileName: string; id: string }[]) => {
    if (summaryData.length === 0) return;

    // Group peaks by BLM angle and sort angles
    const peaksByBlm = _.groupBy(summaryData, 'blm');
    const sortedBlms = Object.keys(peaksByBlm)
      .map(Number)
      .sort((a, b) => a - b);

    let extractedBranches: {
      id: string;
      name: string;
      peaks: { blm: number; freq: number; amp: number; fileName: string; id: string }[];
    }[] = [];

    sortedBlms.forEach((blm, blmIdx) => {
      const currentPeaks = peaksByBlm[blm];

      currentPeaks.forEach(peak => {
        let bestBranchIdx = -1;
        let minDiff = Infinity;

        if (blmIdx > 0) {
          const prevBlm = sortedBlms[blmIdx - 1];
          extractedBranches.forEach((branch, bIdx) => {
            const lastPeak = branch.peaks[branch.peaks.length - 1];
            if (lastPeak.blm === prevBlm) {
              const diff = Math.abs(peak.freq - lastPeak.freq);
              if (diff <= branchTolerance && diff < minDiff) {
                minDiff = diff;
                bestBranchIdx = bIdx;
              }
            }
          });
        }

        if (bestBranchIdx !== -1) {
          extractedBranches[bestBranchIdx].peaks.push(peak);
        } else {
          extractedBranches.push({
            id: `branch-${extractedBranches.length + 1}`,
            name: `Branch ${extractedBranches.length + 1}`,
            peaks: [peak]
          });
        }
      });
    });

    // --- Branch Merging Step ---
    // Try to merge branches that are fragmented but likely the same
    let merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < extractedBranches.length; i++) {
        for (let j = 0; j < extractedBranches.length; j++) {
          if (i === j) continue;
          const b1 = extractedBranches[i];
          const b2 = extractedBranches[j];
          if (!b1 || !b2) continue;

          const b1Last = b1.peaks[b1.peaks.length - 1];
          const b2First = b2.peaks[0];

          // If b1 ends just before b2 starts (or at the same BLM if no overlap)
          // and frequency is close
          const blmGap = b2First.blm - b1Last.blm;
          const freqDiff = Math.abs(b2First.freq - b1Last.freq);

          // Allow a gap of up to 2 BLM steps for merging
          const blmStep = sortedBlms.length > 1 ? sortedBlms[1] - sortedBlms[0] : 1;
          
          if (blmGap >= 0 && blmGap <= blmStep * 2.1 && freqDiff <= branchTolerance) {
            // Check if they have overlapping BLM angles
            const b1Blms = new Set(b1.peaks.map(p => p.blm));
            const b2Blms = new Set(b2.peaks.map(p => p.blm));
            const hasOverlap = [...b2Blms].some(b => b1Blms.has(b));

            if (!hasOverlap) {
              b1.peaks.push(...b2.peaks);
              extractedBranches.splice(j, 1);
              merged = true;
              break;
            }
          }
        }
        if (merged) break;
      }
    }

    // Filter branches by minimum peaks
    const finalBranches = extractedBranches
      .filter(b => b.peaks.length >= minPeaksPerBranch)
      .map((b, idx) => ({ ...b, id: `branch-${idx + 1}`, name: `Branch ${idx + 1}` }));

    const branchPeakIds = new Set(finalBranches.flatMap(b => b.peaks.map(p => p.id)));
    const finalOthers = summaryData.filter(p => !branchPeakIds.has(p.id));

    // Apply fitting if selected (we'll handle this in a separate effect or here)
    // For now, let's just set the branches. The fitting will be triggered by the UI selection.
    setBranches(_.orderBy(finalBranches, [b => _.meanBy(b.peaks, 'freq')], ['asc']));
    setOthers(finalOthers);
  };

  const branchesWithFits = useMemo(() => {
    return branches.map(branch => {
      if (!selectedBranchIds.has(branch.id)) return branch;

      // Filter peaks by symmetry frequency range for fitting
      const filteredPeaks = branch.peaks.filter(p => p.freq >= (symmetryMinFreq || 0) && p.freq <= (symmetryMaxFreq || 100000));
      if (filteredPeaks.length < 2) return { ...branch, fit: null };

      // Optimize cosOffset for this specific branch
      let bestOffset = symmetryCenter;
      let bestF0 = 0;
      let maxR2 = -Infinity;

      // 1/cos(θ-φ) has period 180° in φ, so scanning any 180°-wide window is
      // guaranteed to contain the true global optimum wherever it actually
      // sits — the previous ±30° window only found it if the in-plane
      // reference (symmetryCenter, default 0°) already happened to be within
      // 30° of the true out-of-plane angle. A sample mounted so 90° (not 0°)
      // is out-of-plane would never be found by that narrow window; ±90°
      // covers the full period and finds it automatically regardless of
      // which angle turns out to be the physical zero.
      const searchRange = 90;
      const start = symmetryCenter - searchRange;
      const end = symmetryCenter + searchRange;
      const step = 0.02;
      
      for (let offset = start; offset <= end; offset += step) {
        const data = filteredPeaks.map(p => {
          const theta = ((p.blm - offset) * Math.PI) / 180;
          return {
            x: theta,
            y: p.freq,
            invCos: 1 / Math.cos(theta)
          };
        });

        // Linear regression for F0: y = F0 * (1/cos(theta))
        // F0 = sum(y_i * invCos_i) / sum(invCos_i^2)
        let sumYInvCos = 0;
        let sumInvCosSq = 0;
        let validPoints = 0;
        
        data.forEach(d => {
          if (Math.abs(Math.cos(d.x)) > 0.1) { // Avoid singularities
            sumYInvCos += d.y * d.invCos;
            sumInvCosSq += d.invCos * d.invCos;
            validPoints++;
          }
        });

        if (validPoints < 2) continue;

        const f0 = sumInvCosSq !== 0 ? sumYInvCos / sumInvCosSq : 0;
        if (f0 <= 0) continue;

        const yMean = _.meanBy(data, 'y');
        let ssRes = 0;
        let ssTot = 0;
        data.forEach(d => {
          const yPred = f0 * d.invCos;
          ssRes += Math.pow(d.y - yPred, 2);
          ssTot += Math.pow(d.y - yMean, 2);
        });
        const r2 = ssTot !== 0 ? 1 - (ssRes / ssTot) : 0;

        if (r2 > maxR2) {
          maxR2 = r2;
          bestOffset = offset;
          bestF0 = f0;
        }
      }

      return {
        ...branch,
        fit: { f0: bestF0, r2: maxR2, cosOffset: Number(bestOffset.toFixed(2)), type: '1/cos' as const }
      };
    });
  }, [branches, selectedBranchIds, symmetryCenter, symmetryMinFreq, symmetryMaxFreq]);

  const branchComparisonTable = useMemo(() => {
    if (branchesWithFits.length === 0 && others.length === 0) return [];

    const allBlms = Array.from(new Set([
      ...branchesWithFits.flatMap(b => b.peaks.map(p => p.blm)),
      ...others.map(p => p.blm)
    ])).sort((a, b) => a - b);

    return allBlms.map(blm => {
      const row: any = { blm };
      branchesWithFits.forEach(branch => {
        const peak = branch.peaks.find(p => p.blm === blm);
        row[branch.id] = peak ? { freq: peak.freq, amp: peak.amp } : null;
      });
      const otherPeak = others.find(p => p.blm === blm);
      row['others'] = otherPeak ? { freq: otherPeak.freq, amp: otherPeak.amp } : null;
      return row;
    });
  }, [branchesWithFits, others]);

  const analysisResults = useMemo(() => {
    return currentFileResults?.results || [];
  }, [currentFileResults]);

  // Downsampled FFT (1000 pts) for the big waterfall chart. The per-file
  // sparklines get the full fft and clip+decimate their own visible window
  // inside MiniFFT, so no separate mini series is precomputed here.
  const assembleChartData = useMemo(() =>
    assembleData.map(d => ({ ...d, fftDisplay: downsampleFFT(d.fft, 1000) })),
    [assembleData]
  );

  // The Assemble header recomputed Math.max(...allPeaks) on every render via an
  // inline IIFE — a spread over every peak of every file. Memoize it, and use a
  // loop so a large peak set can't blow the call stack.
  const assembleFreqMax = useMemo(() => {
    let max = 0;
    for (const d of assembleData) {
      for (const p of d.peaks || []) if (p.frequency > max) max = p.frequency;
    }
    return max;
  }, [assembleData]);

  // ── Landau fan / Berry phase memos ──────────────────────────────
  // Every processed (file, column) result is a candidate fan series.
  const landauSeriesList = useMemo(() => {
    const list: { key: string; fileIdx: number; blm: number; col: string; result: AnalysisResult }[] = [];
    files.forEach((f, fi) => {
      const blm = resolveBlm(f, fi);
      f.results.forEach(r => {
        if (r.plotData && r.plotData.length > 4) {
          list.push({ key: `${fi}::${r.columnName}`, fileIdx: fi, blm, col: r.columnName, result: r });
        }
      });
    });
    return list;
  }, [files, resolveBlm]);

  const landauOpts = useMemo(() => ({
    smoothWin: landauSmoothWin, prominenceFrac: landauProm,
    integerAt: landauIntegerAt, n0: landauN0, useBoth: landauUseBoth,
  }), [landauSmoothWin, landauProm, landauIntegerAt, landauN0, landauUseBoth]);

  const landauSelected = useMemo(() => {
    if (landauSeriesList.length === 0) return null;
    const found = landauSeriesList.find(s => s.key === landauResultKey) || landauSeriesList[0];
    const series = found.result.plotData
      .map(d => ({ invB: d.invB, b: d.b, val: d.deltaR }))
      .sort((a, b) => a.invB - b.invB);
    return { ...found, series, freqFFT: found.result.peakFrequency?.frequency ?? null };
  }, [landauSeriesList, landauResultKey]);

  // FFT → band-pass → IFFT of the selected series. `null` band ⇒ auto-band from
  // the dominant peak. When band-pass is off, recon is the raw series.
  const landauBand = useMemo(
    () => (landauBandHi > landauBandLo && landauBandHi > 0 ? { lo: landauBandLo, hi: landauBandHi } : null),
    [landauBandLo, landauBandHi],
  );
  const landauFiltered = useMemo(
    () => landauSelected ? landauSpectrumAndFilter(landauSelected.series, landauBand, landauEdge, landauBandpass) : null,
    [landauSelected, landauBand, landauEdge, landauBandpass],
  );

  // Index the reconstructed (single-frequency) signal, not the raw multi-band ΔR.
  const landauFan = useMemo(
    () => landauFiltered ? computeLandauFan(landauFiltered.recon, landauOpts) : null,
    [landauFiltered, landauOpts],
  );

  // Signal shown in the picking chart — the band-passed reconstruction when on
  const landauLine = useMemo(
    () => landauFiltered ? decimateMinMax(landauFiltered.recon, 1500, p => p.val) : [],
    [landauFiltered],
  );

  // Auto-fan every angle: per-angle auto band-pass (frequency shifts with angle),
  // then index → Berry phase & F vs angle.
  const landauByAngle = useMemo(() => {
    return landauSeriesList.map(s => {
      const series = s.result.plotData
        .map(d => ({ invB: d.invB, b: d.b, val: d.deltaR }))
        .sort((a, b) => a.invB - b.invB);
      const filt = landauSpectrumAndFilter(series, null, landauEdge, landauBandpass);
      const fan = computeLandauFan(filt ? filt.recon : series, landauOpts);
      const beta = fan.intercept + 0.5 - landauDelta; // n = F/B − ½ + β + δ
      return {
        blm: s.blm, col: s.col, slope: fan.slope, intercept: fan.intercept,
        r2: fan.r2, nPts: fan.points.length, berryOverPi: 2 * beta,
      };
    }).filter(d => isFinite(d.slope) && d.nPts >= 3).sort((a, b) => a.blm - b.blm);
  }, [landauSeriesList, landauOpts, landauDelta, landauEdge, landauBandpass]);

  // ── IFFT verification memos ─────────────────────────────────────────────
  // Shares landauSeriesList (every processed file+column) as the picker —
  // it's not Landau-specific, just "every result available to inspect".
  const ifftSelected = useMemo(() => {
    if (landauSeriesList.length === 0) return null;
    const found = landauSeriesList.find(s => s.key === ifftResultKey) || landauSeriesList[0];
    const series = found.result.plotData
      .map(d => ({ invB: d.invB, b: d.b, val: d.deltaR }))
      .sort((a, b) => a.invB - b.invB);
    return { ...found, series };
  }, [landauSeriesList, ifftResultKey]);

  const ifftBand = useMemo(
    () => (ifftBandHi > ifftBandLo && ifftBandHi > 0 ? { lo: ifftBandLo, hi: ifftBandHi } : null),
    [ifftBandLo, ifftBandHi],
  );
  const ifftFiltered = useMemo(
    () => ifftSelected ? landauSpectrumAndFilter(ifftSelected.series, ifftBand, ifftEdge, true) : null,
    [ifftSelected, ifftBand, ifftEdge],
  );

  // Both decimated the same way so an honest visual comparison (same point
  // density either side) is what actually reaches the chart.
  const ifftOriginalLine = useMemo(
    () => ifftSelected ? decimateMinMax(ifftSelected.series, 1500, p => p.val) : [],
    [ifftSelected],
  );
  const ifftReconLine = useMemo(
    () => ifftFiltered ? decimateMinMax(ifftFiltered.recon, 1500, p => p.val) : [],
    [ifftFiltered],
  );

  const toggleRColumn = (col: string) => {
    setSelectedRColumns(prev =>
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    );
  };

  const columns = useMemo(() => {
    if (files.length === 0) return [];
    let fileIdx = selectedFileIndex;
    if (activeTab !== 'dashboard') {
      const foundIdx = files.findIndex(f => f.id === activeTab);
      if (foundIdx !== -1) fileIdx = foundIdx;
    }
    const idx = Math.min(fileIdx, files.length - 1);
    const allCols = files[idx].columns;
    
    // If prefix is selected, filter columns to only show matching ones (+ B column)
    if (selectedPrefix) {
      return dropQuadrature(
        allCols.filter(c => c === selectedBColumn || c.startsWith(selectedPrefix + '_')),
        allCols
      );
    }

    return dropQuadrature(allCols);
  }, [files, selectedFileIndex, activeTab, selectedPrefix, selectedBColumn, dropQuadrature]);

  const fileName = useMemo(() => {
    if (files.length === 0) return '';
    if (files.length === 1) return files[0].name;
    return `${files.length} files selected`;
  }, [files]);

  const copyPeaksToClipboard = (peaks: FFTResult[]) => {
    const text = "Frequency (T)\tAmplitude\n" + 
      peaks.map(p => `${p.frequency.toFixed(4)}\t${p.amplitude.toExponential(4)}`).join('\n');
    
    navigator.clipboard.writeText(text).then(() => {
      alert("Peaks copied to clipboard!");
    }).catch(err => {
      console.error('Failed to copy: ', err);
    });
  };

  // The processing control sidebar is only relevant to the data-processing views
  // (Dashboard, Raw) and the per-file detail views; the self-contained analysis
  // tabs (Torque, Landau, Harmonics, Symmetry, Branch, FFT Assemble) render
  // full-width with their own inline controls.
  const showControlSidebar = activeTab === 'dashboard' || activeTab === 'raw' || files.some(f => f.id === activeTab);

  // Gate: every session starts by naming (or reloading) a sample, before any
  // dashboard is shown. This is what saveSessionToFile/loadSessionFromFile
  // key their output to.
  if (!sampleId) {
    const startNewSample = () => {
      const id = sampleIdDraft.trim();
      if (!id) return;
      const firstId = `ori-${Date.now()}`;
      const blankSettings = getSettingsSnapshot();
      setOrientations([{
        id: firstId, label: 'Orientation 1',
        angle: { files: [], settings: blankSettings },
        temperature: { groups: [], settings: blankSettings },
      }]);
      setActiveOrientationId(firstId);
      setSampleId(id);
    };
    return (
      <div className="min-h-screen bg-[#E4E3E0] text-[#141414] font-sans flex items-center justify-center p-4">
        <div className="w-full max-w-md bg-[#141414] text-[#E4E3E0] border border-[#E4E3E0]/10 p-10">
          <h1 className="text-3xl font-bold tracking-tighter uppercase italic font-serif mb-1">Quantum Analysis</h1>
          <p className="text-xs opacity-50 font-mono uppercase tracking-widest mb-8">SdH Analysis Suite</p>

          <label className="block text-[10px] font-mono uppercase opacity-60 mb-2">Sample ID</label>
          <input
            type="text"
            autoFocus
            value={sampleIdDraft}
            onChange={(e) => setSampleIdDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') startNewSample(); }}
            placeholder="e.g. PrAlGe-01"
            className="w-full bg-[#0a0a0a] border border-[#E4E3E0]/30 p-3 font-mono text-sm text-[#E4E3E0] focus:outline-none focus:border-[#F27D26] mb-4"
          />
          <button
            onClick={startNewSample}
            disabled={!sampleIdDraft.trim()}
            className="w-full bg-[#F27D26] text-[#141414] font-mono text-xs uppercase tracking-widest font-bold py-3 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-opacity-90 transition-all"
          >
            Start
          </button>

          <div className="flex items-center gap-3 my-6">
            <div className="h-px bg-white/10 flex-1" />
            <span className="text-[9px] font-mono uppercase opacity-30">or</span>
            <div className="h-px bg-white/10 flex-1" />
          </div>

          {supportsFileSystemAccess ? (
            <button
              onClick={loadSessionViaPicker}
              className="w-full flex items-center justify-center gap-2 border border-white/20 py-3 font-mono text-xs uppercase tracking-widest hover:bg-white/5 transition-all"
            >
              <FolderOpen size={14} />
              Load Session File
            </button>
          ) : (
            <label className="w-full cursor-pointer flex items-center justify-center gap-2 border border-white/20 py-3 font-mono text-xs uppercase tracking-widest hover:bg-white/5 transition-all">
              <FolderOpen size={14} />
              Load Session File
              <input type="file" accept=".json" onChange={loadSessionFromFile} className="hidden" />
            </label>
          )}
          {error && <p className="mt-4 text-[11px] font-mono text-red-400">{error}</p>}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#E4E3E0] text-[#141414] font-sans p-4 md:p-8">
      {/* Header */}
      <header className="max-w-7xl mx-auto mb-4 border-b border-[#141414] pb-4 flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
        <div className="flex flex-col md:flex-row items-start md:items-center gap-8">
          <div>
            <h1 className="text-4xl font-bold tracking-tighter uppercase italic font-serif">
              Quantum Analysis
            </h1>
            <p className="text-sm opacity-60 mt-1 font-mono uppercase tracking-widest">
              SdH Analysis Suite
            </p>
          </div>
          
          <nav className="flex bg-[#141414] p-1 rounded-none">
            <button 
              onClick={() => {
                setAppMode('sdh');
                setActiveTab('dashboard');
              }}
              className={`px-6 py-2 font-mono text-xs uppercase tracking-widest transition-all ${
                appMode === 'sdh' 
                  ? 'bg-[#F27D26] text-[#141414] font-bold' 
                  : 'text-[#E4E3E0] hover:bg-white/10'
              }`}
            >
              SdH Analysis
            </button>
          </nav>
        </div>
        <div className="flex items-center gap-4">
          {isAuthReady && (
            <div className="flex items-center gap-3 border-l pl-4 border-[#141414]/20">
              {user ? (
                <div className="flex items-center gap-3">
                  <div className="text-right hidden sm:block">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-[#141414]">{user.displayName}</p>
                    <p className="text-[9px] font-mono opacity-50">{user.email}</p>
                  </div>
                  {user.photoURL && (
                    <img src={user.photoURL} alt="" className="w-8 h-8 rounded-none border border-[#141414]" referrerPolicy="no-referrer" />
                  )}
                  <button
                    onClick={handleLogout}
                    className="p-2 text-[#141414] hover:text-red-600 transition-colors"
                    title="Sign Out"
                  >
                    <LogOut className="w-5 h-5" />
                  </button>
                </div>
              ) : (
                <button
                  onClick={handleLogin}
                  className="flex items-center gap-2 px-4 py-2 bg-[#141414] text-[#E4E3E0] hover:bg-opacity-80 transition-all font-mono text-[10px] uppercase tracking-widest"
                >
                  <LogIn className="w-4 h-4" />
                  Sign In
                </button>
              )}
            </div>
          )}
          {activeTab !== 'dashboard' && currentFileResults && (
            (() => {
              const blmCol = currentFileResults.columns.find(c => 
                c.toLowerCase().includes('blm') || 
                c.toLowerCase().includes('position') || 
                c.toLowerCase().includes('angle')
              );
              
              let displayVal = '';
              if (blmCol && currentFileResults.data.length > 0) {
                displayVal = String(currentFileResults.data[0][blmCol]);
              } else {
                const blmEntry = Object.entries(currentFileResults.metadata).find(([k]) => 
                  k.toLowerCase().includes('position') || 
                  k.toLowerCase().includes('blm') || 
                  k.toLowerCase().includes('angle')
                );
                displayVal = blmEntry ? String(blmEntry[1]) : '';
              }
              
              if (!displayVal) return null;
              
              return (
                <div className="bg-[#141414] text-[#E4E3E0] px-4 py-2 flex flex-col items-end border-l-4 border-[#F27D26]">
                  <span className="text-[9px] font-mono uppercase opacity-50">BLM Position (Angle)</span>
                  <span className="text-xl font-mono font-bold">
                    {displayVal}{!displayVal.includes('°') && '°'}
                  </span>
                </div>
              );
            })()
          )}
          <label className="cursor-pointer bg-[#141414] text-[#E4E3E0] px-6 py-3 rounded-none hover:bg-opacity-90 transition-all flex items-center gap-2 font-mono text-sm uppercase tracking-wider">
            <Upload size={16} />
            {files.length > 0 ? 'Add Files' : 'Upload Data'}
            <input type="file" accept=".csv,.dat,.txt" onChange={handleFileUploadIntoActiveBucket} className="hidden" multiple />
          </label>
        </div>
      </header>

      {/* Sample / Orientation bar */}
      <div className="max-w-7xl mx-auto mb-4 flex items-center gap-3 flex-wrap bg-[#141414] text-[#E4E3E0] px-4 py-2 border border-[#E4E3E0]/10">
        <div className="flex items-center gap-2 pr-3 border-r border-white/10">
          <span className="text-[9px] font-mono uppercase opacity-40">Sample</span>
          <span className="font-mono text-sm font-bold text-[#F27D26]">{sampleId}</span>
          <button
            title="다른 샘플로 전환 (저장 안 된 변경사항은 사라집니다)"
            onClick={() => {
              if (window.confirm('다른 샘플로 전환하시겠습니까? 저장하지 않은 변경사항은 사라집니다.')) {
                setSampleId(null);
                setSampleIdDraft('');
                setOrientations([]);
                setActiveOrientationId('');
                setFiles([]);
                sessionFileHandleRef.current = null;
                setLinkedSessionFileName(null);
                setSaveStatus('idle');
                setLastSavedAt(null);
              }
            }}
            className="opacity-40 hover:opacity-100 p-0.5"
          >
            <X size={12} />
          </button>
        </div>

        <div className="flex items-center gap-1 flex-wrap flex-1">
          {orientations.map(o => (
            <div key={o.id} className="group relative">
              {editingOrientationId === o.id ? (
                <input
                  type="text"
                  autoFocus
                  value={editOrientationLabelDraft}
                  onChange={(e) => setEditOrientationLabelDraft(e.target.value)}
                  onBlur={() => { const v = editOrientationLabelDraft.trim(); if (v) renameOrientation(o.id, v); setEditingOrientationId(null); }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') { const v = editOrientationLabelDraft.trim(); if (v) renameOrientation(o.id, v); setEditingOrientationId(null); }
                    if (e.key === 'Escape') setEditingOrientationId(null);
                  }}
                  className="w-24 bg-[#0a0a0a] border border-[#F27D26] px-1.5 py-1 font-mono text-[10px] text-[#E4E3E0] focus:outline-none"
                />
              ) : (
                <button
                  onDoubleClick={() => { setEditingOrientationId(o.id); setEditOrientationLabelDraft(o.label); }}
                  onClick={() => switchOrientation(o.id)}
                  title="더블클릭으로 이름 변경"
                  className={`px-2.5 py-1 font-mono text-[10px] uppercase border transition-colors ${
                    o.id === activeOrientationId
                      ? 'border-[#F27D26] text-[#F27D26] bg-[#F27D26]/10'
                      : 'border-white/15 opacity-60 hover:opacity-100'
                  }`}
                >
                  {o.label}
                </button>
              )}
              {orientations.length > 1 && editingOrientationId !== o.id && (
                <button
                  onClick={(e) => { e.stopPropagation(); if (window.confirm(`"${o.label}" orientation을 제거할까요?`)) removeOrientation(o.id); }}
                  title="Orientation 제거"
                  className="absolute -top-1.5 -right-1.5 hidden group-hover:flex items-center justify-center w-3.5 h-3.5 bg-red-600 text-white rounded-full"
                >
                  <X size={9} />
                </button>
              )}
            </div>
          ))}

          {showAddOrientation ? (
            <div className="flex items-center gap-1">
              <input
                type="text"
                autoFocus
                value={orientationLabelDraft}
                onChange={(e) => setOrientationLabelDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') { addOrientation(orientationLabelDraft.trim()); setOrientationLabelDraft(''); setShowAddOrientation(false); }
                  if (e.key === 'Escape') { setOrientationLabelDraft(''); setShowAddOrientation(false); }
                }}
                placeholder="e.g. [001]"
                className="w-24 bg-[#0a0a0a] border border-[#E4E3E0]/30 px-1.5 py-1 font-mono text-[10px] text-[#E4E3E0] focus:outline-none focus:border-[#F27D26]"
              />
              <button
                onClick={() => { addOrientation(orientationLabelDraft.trim()); setOrientationLabelDraft(''); setShowAddOrientation(false); }}
                className="px-1.5 py-1 font-mono text-[9px] uppercase border border-[#F27D26] text-[#F27D26]"
              >
                <Check size={11} />
              </button>
            </div>
          ) : (
            <button
              onClick={() => { setOrientationLabelDraft(`Orientation ${orientations.length + 1}`); setShowAddOrientation(true); }}
              title="새 Orientation 추가"
              className="px-1.5 py-1 border border-dashed border-white/20 opacity-50 hover:opacity-100 hover:border-white/40"
            >
              <Plus size={12} />
            </button>
          )}
        </div>

        <div className="flex items-center gap-1 pl-3 border-l border-white/10" title="이 orientation 안의 두 파일 세트 — 각도 스윕 / 온도 스윕은 서로 다른 파일·설정을 가짐">
          <span className="text-[9px] font-mono uppercase opacity-40 mr-1">Files</span>
          {(['angle', 'temperature'] as const).map(m => (
            <button
              key={m}
              onClick={() => switchBucket(m)}
              className={`px-2 py-1 font-mono text-[9px] uppercase border transition-colors ${
                activeBucket === m
                  ? 'border-[#F27D26] text-[#F27D26] bg-[#F27D26]/10'
                  : 'border-white/15 opacity-50 hover:opacity-100'
              }`}
            >
              {m === 'angle' ? 'Angle Sweep' : 'Temp Sweep'}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 pl-3 border-l border-white/10">
          {linkedSessionFileName && (
            <span className="text-[8px] font-mono opacity-40 max-w-[140px] truncate" title={`저장/불러오기 시 이 파일에 덮어씁니다: ${linkedSessionFileName}`}>
              🔗 {linkedSessionFileName}
            </span>
          )}
          <button onClick={saveSessionToFile} title={linkedSessionFileName ? `"${linkedSessionFileName}" 파일에 덮어씁니다` : '샘플 전체(raw+설정)를 로컬 JSON 파일로 저장'}
            className="px-2 py-1 font-mono text-[9px] uppercase border border-white/15 hover:bg-white/5 flex items-center gap-1">
            <Save size={11} />{saveStatus === 'saving' ? 'Saving…' : 'Save Session'}
          </button>
          {saveStatus === 'saved' && lastSavedAt && (
            <span className="text-[8px] font-mono text-[#4ADE80]">✓ Saved {lastSavedAt.toLocaleTimeString()}</span>
          )}
          {saveStatus === 'error' && (
            <span className="text-[8px] font-mono text-red-400">✕ Save failed</span>
          )}
          {supportsFileSystemAccess ? (
            <button
              onClick={() => { if (window.confirm('현재 작업을 덮어씁니다. 계속할까요?')) loadSessionViaPicker(); }}
              title="세션 파일 불러오기 (현재 작업을 덮어씁니다) — 이후 Save는 이 파일에 바로 덮어씁니다"
              className="px-2 py-1 font-mono text-[9px] uppercase border border-white/15 hover:bg-white/5 flex items-center gap-1"
            >
              <FolderOpen size={11} />Load Session
            </button>
          ) : (
            <label title="저장된 세션 파일 불러오기 (현재 작업을 덮어씁니다)"
              className="cursor-pointer px-2 py-1 font-mono text-[9px] uppercase border border-white/15 hover:bg-white/5 flex items-center gap-1">
              <FolderOpen size={11} />Load Session
              <input type="file" accept=".json" onChange={(e) => { if (window.confirm('현재 작업을 덮어씁니다. 계속할까요?')) loadSessionFromFile(e); else e.target.value = ''; }} className="hidden" />
            </label>
          )}
        </div>
      </div>

      {/* Temperature-sweep groups: several fixed-angle T-sweeps can live side
          by side inside the same orientation's temperature bucket. */}
      {activeBucket === 'temperature' && (() => {
        const currentOrientation = orientations.find(o => o.id === activeOrientationId);
        const groups = currentOrientation?.temperature.groups ?? [];
        return (
          <div className="max-w-7xl mx-auto mb-4 flex items-center gap-2 flex-wrap bg-white/50 border border-[#141414]/10 px-4 py-2">
            <span className="text-[9px] font-mono uppercase opacity-40">T-Sweep Set (angle)</span>
            {groups.map(g => (
              <div key={g.id} className="group relative">
                {editingTempGroupId === g.id ? (
                  <input
                    type="text"
                    autoFocus
                    value={editTempGroupLabelDraft}
                    onChange={(e) => setEditTempGroupLabelDraft(e.target.value)}
                    onBlur={() => { const v = editTempGroupLabelDraft.trim(); if (v) renameTemperatureGroup(g.id, v); setEditingTempGroupId(null); }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') { const v = editTempGroupLabelDraft.trim(); if (v) renameTemperatureGroup(g.id, v); setEditingTempGroupId(null); }
                      if (e.key === 'Escape') setEditingTempGroupId(null);
                    }}
                    className="w-20 bg-white border border-[#F27D26] px-1.5 py-1 font-mono text-[10px] text-[#141414] focus:outline-none"
                  />
                ) : (
                  <button
                    onDoubleClick={() => { setEditingTempGroupId(g.id); setEditTempGroupLabelDraft(g.label); }}
                    onClick={() => switchTemperatureGroup(g.id)}
                    title="더블클릭으로 이름 변경 (예: 각도)"
                    className={`px-2.5 py-1 font-mono text-[10px] uppercase border transition-colors ${
                      g.id === activeTempGroupId
                        ? 'border-[#F27D26] text-[#F27D26] bg-[#F27D26]/10'
                        : 'border-[#141414]/20 opacity-60 hover:opacity-100'
                    }`}
                  >
                    {g.label}
                  </button>
                )}
                {editingTempGroupId !== g.id && (
                  <button
                    onClick={(e) => { e.stopPropagation(); if (window.confirm(`"${g.label}" T-sweep set을 제거할까요?`)) removeTemperatureGroup(g.id); }}
                    title="제거"
                    className="absolute -top-1.5 -right-1.5 hidden group-hover:flex items-center justify-center w-3.5 h-3.5 bg-red-600 text-white rounded-full"
                  >
                    <X size={9} />
                  </button>
                )}
              </div>
            ))}

            {showAddTempGroup ? (
              <div className="flex items-center gap-1">
                <input
                  type="text"
                  autoFocus
                  value={tempGroupLabelDraft}
                  onChange={(e) => setTempGroupLabelDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') { addTemperatureGroup(tempGroupLabelDraft.trim()); setTempGroupLabelDraft(''); setShowAddTempGroup(false); }
                    if (e.key === 'Escape') { setTempGroupLabelDraft(''); setShowAddTempGroup(false); }
                  }}
                  placeholder="e.g. 15deg"
                  className="w-20 bg-white border border-[#141414]/30 px-1.5 py-1 font-mono text-[10px] text-[#141414] focus:outline-none focus:border-[#F27D26]"
                />
                <button
                  onClick={() => { addTemperatureGroup(tempGroupLabelDraft.trim()); setTempGroupLabelDraft(''); setShowAddTempGroup(false); }}
                  className="px-1.5 py-1 font-mono text-[9px] uppercase border border-[#F27D26] text-[#F27D26]"
                >
                  <Check size={11} />
                </button>
              </div>
            ) : (
              <button
                onClick={() => { setTempGroupLabelDraft(`Group ${groups.length + 1}`); setShowAddTempGroup(true); }}
                title="새 T-sweep set 추가 (다른 각도)"
                className="px-1.5 py-1 border border-dashed border-[#141414]/20 opacity-50 hover:opacity-100 hover:border-[#141414]/40"
              >
                <Plus size={12} />
              </button>
            )}
            {groups.length === 0 && (
              <span className="text-[9px] font-mono opacity-30 italic">+ 를 눌러 첫 T-sweep set을 추가하세요 (각도별로 여러 개 가능)</span>
            )}
          </div>
        );
      })()}

      {/* Tabs Navigation */}
      {files.length > 0 && (
        <nav className="max-w-7xl mx-auto mb-8 flex items-center gap-0.5 overflow-x-auto no-scrollbar bg-white/50 border border-[#141414]/10 px-2">
          {(() => {
            const angleOnlyTabs = new Set(['torque', 'landau', 'harmonics', 'symmetry', 'branch']);
            const allTabs: [string, string][] = [
              ['dashboard', 'Dashboard'],
              ['raw', 'Raw Data'], ['assemble', 'FFT Assemble'],
              ['torque', 'Torque'], ['landau', 'Landau Fan'], ['harmonics', 'Harmonics'], ['symmetry', 'Symmetry'], ['branch', 'Branch'],
              ['lkfit', 'L-K Fit'], ['fftrange', 'FFT Range'], ['ifft', 'IFFT'],
            ];
            // Angle-dependent tools (Torque/Landau Fan/Harmonics/Symmetry/Branch)
            // assume a BLM sweep at fixed T; the temperature-sweep bucket has no
            // angle axis, so they'd just show noise — hide them while it's active.
            return activeBucket === 'temperature' ? allTabs.filter(([id]) => !angleOnlyTabs.has(id)) : allTabs;
          })().map(([id, label]) => (
            <React.Fragment key={id}>
              {(id === 'raw' || id === 'torque') && <span className="w-px h-4 bg-[#141414]/15 mx-2 shrink-0" />}
              <button
                onClick={() => setActiveTab(id)}
                className={`px-4 py-3.5 font-mono text-xs uppercase tracking-wide whitespace-nowrap transition-colors border-b-2 -mb-px ${
                  activeTab === id
                    ? 'border-[#F27D26] text-[#141414] font-bold'
                    : 'border-transparent text-[#141414]/50 hover:text-[#141414]'
                }`}
              >
                {label}
              </button>
            </React.Fragment>
          ))}
          {/* Per-file detail collapsed into a right-aligned dropdown */}
          <div className="ml-auto flex items-center gap-2 pl-4 shrink-0">
            <span className="text-[9px] font-mono uppercase tracking-widest opacity-30">File</span>
            <select
              value={files.some(f => f.id === activeTab) ? activeTab : ''}
              onChange={(e) => { const id = e.target.value; if (id) { setActiveTab(id); setSelectedFileIndex(files.findIndex(f => f.id === id)); } }}
              className={`bg-white border py-1.5 px-2 font-mono text-[11px] focus:outline-none focus:border-[#F27D26] cursor-pointer max-w-[180px] ${
                files.some(f => f.id === activeTab)
                  ? 'border-[#F27D26] text-[#141414] font-bold'
                  : 'border-[#141414]/20 text-[#141414]/60'
              }`}
            >
              <option value="">Detail…</option>
              {filesByAngle.map(({ file, label }) => (
                <option key={file.id} value={file.id} title={file.name}>{label}</option>
              ))}
            </select>
          </div>
        </nav>
      )}

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Sidebar / Controls — only on processing & per-file views */}
        {showControlSidebar && (
        <aside className="lg:col-span-3 space-y-6">
          {files.length > 0 && (
            <button 
              onClick={handleProcessAll}
              disabled={isProcessing}
              className="w-full bg-[#F27D26] text-[#141414] font-bold py-4 px-6 rounded-none hover:bg-opacity-90 transition-all flex items-center justify-center gap-2 font-mono text-sm uppercase tracking-wider mb-6 disabled:opacity-50 disabled:cursor-not-allowed shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none"
            >
              <RefreshCw size={18} className={isProcessing ? 'animate-spin' : ''} />
              Process All Files
            </button>
          )}

          {/* Which files go into the FFT batch */}
          {files.length > 0 && (
            <section className="bg-white/50 p-4 border border-[#141414]/10">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2 text-[#141414]">
                  <Activity size={14} className="text-[#F27D26]" />
                  <h2 className="font-serif italic font-bold text-sm">FFT Files</h2>
                  <span className="text-[9px] font-mono opacity-40">{files.length - fftSkip.size}/{files.length}</span>
                </div>
                <div className="flex items-center gap-1">
                  <button onClick={() => setFftSkip(new Set())}
                    className="px-1.5 py-0.5 font-mono text-[8px] uppercase border border-[#141414]/20 hover:bg-[#141414]/5">All</button>
                  <button onClick={() => setFftSkip(new Set(files.map(f => f.id)))}
                    className="px-1.5 py-0.5 font-mono text-[8px] uppercase border border-[#141414]/20 hover:bg-[#141414]/5">None</button>
                  <button onClick={() => setFftSkip(new Set(files.filter(f => (oscScores[f.id] ?? 0) < OSC_SCORE_MIN).map(f => f.id)))}
                    title={`oscillation 점수 ${OSC_SCORE_MIN} 미만 자동 해제`}
                    className="px-1.5 py-0.5 font-mono text-[8px] uppercase border border-[#F27D26] text-[#F27D26] hover:bg-[#F27D26]/10">Auto</button>
                </div>
              </div>
              <div className="max-h-56 overflow-y-auto border border-[#141414]/10 bg-white">
                {filesByAngle.map(({ file, label }) => {
                  const included = !fftSkip.has(file.id);
                  const score = oscScores[file.id];
                  return (
                    <label key={file.id} title={file.name}
                      className="flex items-center gap-2 px-2 py-1.5 cursor-pointer hover:bg-[#141414]/5 border-b border-[#141414]/5 last:border-b-0">
                      <input
                        type="checkbox"
                        checked={included}
                        onChange={() => setFftSkip(prev => {
                          const next = new Set(prev);
                          next.has(file.id) ? next.delete(file.id) : next.add(file.id);
                          return next;
                        })}
                        className="accent-[#F27D26] shrink-0"
                      />
                      <span className={`flex-1 truncate font-mono text-[11px] tabular-nums ${included ? '' : 'opacity-35 line-through'}`}>
                        {label}
                      </span>
                      {score !== undefined && (
                        <span className={`font-mono text-[9px] shrink-0 ${score >= OSC_SCORE_MIN ? 'text-[#F27D26] font-bold' : 'opacity-35'}`}>
                          {score >= 99.5 ? '99+' : score.toFixed(1)}
                        </span>
                      )}
                      <button
                        type="button"
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); if (window.confirm(`"${file.name}" 파일을 제거할까요?`)) removeFile(file.id); }}
                        title="파일 제거"
                        className="shrink-0 opacity-30 hover:opacity-100 hover:text-red-600 p-0.5"
                      >
                        <Trash2 size={11} />
                      </button>
                    </label>
                  );
                })}
              </div>
              <p className="text-[9px] font-mono opacity-40 leading-relaxed mt-2">
                각도 옆 숫자는 oscillation 점수(1/B 스펙트럼의 최대 라인 ÷ 그 옆 노이즈).
                업로드 시 {OSC_SCORE_MIN} 미만은 자동으로 해제됩니다.
                해제된 파일은 FFT·피크·Branch에서 빠지고 Raw·Torque 탭에는 그대로 남습니다. 변경 후 Process All 을 다시 누르세요.
                휴지통 아이콘으로 파일 자체를 목록에서 제거할 수 있습니다.
              </p>
            </section>
          )}

          {appMode === 'sdh' && (
            <section className="bg-white/50 p-6 border border-[#141414]/10">
              <div className="flex items-center gap-2 mb-4 text-[#141414]">
                <FileText size={18} />
                <h2 className="font-serif italic font-bold text-lg">Columns</h2>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-mono uppercase opacity-60 mb-1">B (Field) Column</label>
                  <select 
                    value={selectedBColumn} 
                    onChange={(e) => setSelectedBColumn(e.target.value)}
                    className="w-full bg-white border border-[#141414] p-2 font-mono text-sm focus:outline-none focus:ring-1 focus:ring-[#141414]"
                  >
                    {columns.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="block text-xs font-mono uppercase opacity-60">R (Resistance) Column Prefix</label>
                    {!prefixLocked && (
                      <button
                        type="button"
                        onClick={() => setPrefixLocked(true)}
                        disabled={!selectedPrefix}
                        title="이 prefix로 고정 — 앞으로 받는 파일도 이 컬럼만 사용, 선택 UI는 숨겨집니다"
                        className="px-1.5 py-0.5 font-mono text-[9px] uppercase border border-[#141414]/20 hover:bg-[#141414]/5 flex items-center gap-1 disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        <div className="w-2.5 h-2.5" />Lock
                      </button>
                    )}
                  </div>

                  {prefixLocked ? (
                    <div className="flex items-center justify-between p-2 border border-[#F27D26] bg-[#F27D26]/5">
                      <span className="font-mono text-sm font-bold text-[#F27D26]">{selectedPrefix}_*</span>
                      <button
                        type="button"
                        onClick={() => setPrefixLocked(false)}
                        title="잠금 해제 — 새 파일 업로드 시 다시 자동 감지됩니다"
                        className="px-1.5 py-0.5 font-mono text-[9px] uppercase border border-[#F27D26] text-[#F27D26] hover:bg-[#F27D26]/10 flex items-center gap-1"
                      >
                        <Check size={10} />Locked
                      </button>
                    </div>
                  ) : (
                    <>
                      {prefixes.length > 0 ? (
                        <select
                          value={selectedPrefix}
                          onChange={(e) => setSelectedPrefix(e.target.value)}
                          className="w-full bg-white border border-[#141414] p-2 font-mono text-sm focus:outline-none focus:ring-1 focus:ring-[#141414]"
                        >
                          <option value="">Select Prefix...</option>
                          {prefixes.map(p => (
                            <option key={p} value={p}>{p}_*</option>
                          ))}
                        </select>
                      ) : (
                        <div className="text-[10px] font-mono opacity-40 italic p-2 border border-[#141414]/10 bg-white">
                          No prefixes detected (XX_# format)
                        </div>
                      )}

                      {/* Optional: Show individual selection if no prefix is selected or available */}
                      {!selectedPrefix && (
                        <div className="mt-4">
                          <label className="block text-xs font-mono uppercase opacity-60 mb-1">Individually Select R Columns</label>
                          <div className="max-h-48 overflow-y-auto border border-[#141414] p-2 bg-white space-y-1">
                            {columns.map(col => (
                              <label key={col} className="flex items-center gap-2 text-xs font-mono cursor-pointer hover:bg-[#141414]/5 p-1">
                                <input
                                  type="checkbox"
                                  checked={selectedRColumns.includes(col)}
                                  onChange={() => toggleRColumn(col)}
                                  disabled={col === selectedBColumn}
                                  className="accent-[#141414]"
                                />
                                <span className={col === selectedBColumn ? 'opacity-30' : ''}>{col}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </section>
          )}

          <section className="bg-white/50 p-6 border border-[#141414]/10">
            <div className="flex items-center gap-2 mb-4 text-[#141414]">
              <Settings2 size={18} />
              <h2 className="font-serif italic font-bold text-lg">Parameters</h2>
            </div>
            
            <div className="space-y-6">
              {/* ROI Selection */}
              <div>
                <label className="block text-[10px] font-mono uppercase opacity-60 mb-2">B-Field Range (ROI)</label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-[10px] font-mono block mb-1">Min B (T):</span>
                    <input 
                      type="number" 
                      step="0.1"
                      placeholder="Auto"
                      value={tempMinB} 
                      onChange={(e) => setTempMinB(e.target.value === '' ? '' : parseFloat(e.target.value))}
                      className="w-full bg-white border border-[#141414] p-1 font-mono text-xs focus:outline-none"
                    />
                  </div>
                  <div>
                    <span className="text-[10px] font-mono block mb-1">Max B (T):</span>
                    <input 
                      type="number" 
                      step="0.1"
                      placeholder="Auto"
                      value={tempMaxB} 
                      onChange={(e) => setTempMaxB(e.target.value === '' ? '' : parseFloat(e.target.value))}
                      className="w-full bg-white border border-[#141414] p-1 font-mono text-xs focus:outline-none"
                    />
                  </div>
                </div>
              </div>

              {/* Background Subtraction */}
              <div>
                <label className="block text-[10px] font-mono uppercase opacity-60 mb-2">Background Subtraction</label>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono">Poly Degree:</span>
                  <input 
                    type="number" 
                    min="1" 
                    max="10" 
                    value={tempPolyDegree} 
                    onChange={(e) => setTempPolyDegree(parseInt(e.target.value) || 1)}
                    className="w-16 bg-white border border-[#141414] p-1 font-mono text-xs focus:outline-none"
                  />
                </div>
              </div>

              {/* Spike Removal */}
              <div className="pt-4 border-t border-[#141414]/5">
                <label className="flex items-center gap-2 cursor-pointer mb-2">
                  <input 
                    type="checkbox" 
                    checked={tempSpikeRemoval} 
                    onChange={(e) => setTempSpikeRemoval(e.target.checked)}
                    className="accent-[#141414]"
                  />
                  <span className="text-[10px] font-mono uppercase opacity-60">Spike Removal</span>
                </label>
                {tempSpikeRemoval && (
                  <div className="space-y-2 pl-5">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-mono">Kernel:</span>
                      <input 
                        type="number" 
                        min="3" 
                        step="2"
                        value={tempSpikeKernel} 
                        onChange={(e) => setTempSpikeKernel(parseInt(e.target.value) || 3)}
                        className="w-12 bg-white border border-[#141414] p-1 font-mono text-[10px]"
                      />
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-mono">Threshold:</span>
                      <input 
                        type="number" 
                        step="0.1"
                        value={tempSpikeThreshold} 
                        onChange={(e) => setTempSpikeThreshold(parseFloat(e.target.value) || 1)}
                        className="w-12 bg-white border border-[#141414] p-1 font-mono text-[10px]"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Smoothing */}
              <div className="pt-4 border-t border-[#141414]/5">
                <label className="flex items-center gap-2 cursor-pointer mb-2">
                  <input 
                    type="checkbox" 
                    checked={tempSmoothing} 
                    onChange={(e) => setTempSmoothing(e.target.checked)}
                    className="accent-[#141414]"
                  />
                  <span className="text-[10px] font-mono uppercase opacity-60">Smoothing</span>
                </label>
                {tempSmoothing && (
                  <div className="flex justify-between items-center pl-5">
                    <span className="text-[10px] font-mono">Window:</span>
                    <input 
                      type="number" 
                      min="3" 
                      value={tempSmoothWindow} 
                      onChange={(e) => setTempSmoothWindow(parseInt(e.target.value) || 3)}
                      className="w-12 bg-white border border-[#141414] p-1 font-mono text-[10px]"
                    />
                  </div>
                )}
              </div>

              {/* FFT Settings */}
              <div className="pt-4 border-t border-[#141414]/5 space-y-3">
                <label className="block text-[10px] font-mono uppercase opacity-60">FFT & Windowing</label>
                <div>
                  <label className="block text-[10px] font-mono mb-1">Points:</label>
                  <select 
                    value={tempInterpPoints} 
                    onChange={(e) => setTempInterpPoints(parseInt(e.target.value))}
                    className="w-full bg-white border border-[#141414] p-1 font-mono text-xs"
                  >
                    {[256, 512, 1024, 2048, 4096, 8192].map(p => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] font-mono mb-1">Window:</label>
                  <select 
                    value={tempWindowType} 
                    onChange={(e) => setTempWindowType(e.target.value)}
                    className="w-full bg-white border border-[#141414] p-1 font-mono text-xs"
                  >
                    {['hanning', 'hamming', 'bartlett', 'blackman', 'kaiser', 'flat'].map(w => (
                      <option key={w} value={w}>{w.charAt(0).toUpperCase() + w.slice(1)}</option>
                    ))}
                  </select>
                </div>

                <div className="pt-2 border-t border-[#141414]/5">
                  <label className="flex items-center gap-2 cursor-pointer mb-1">
                    <input 
                      type="checkbox" 
                      checked={tempUseTargetInterval} 
                      onChange={(e) => setTempUseTargetInterval(e.target.checked)}
                      className="accent-[#141414]"
                    />
                    <span className="text-[10px] font-mono uppercase opacity-60">Target Freq Interval</span>
                  </label>
                  {tempUseTargetInterval && (
                    <div className="flex items-center gap-2 pl-5">
                      <input 
                        type="number" 
                        step="0.1"
                        min="0.01"
                        value={tempTargetFreqInterval} 
                        onChange={(e) => setTempTargetFreqInterval(parseFloat(e.target.value) || 0.1)}
                        className="w-full bg-white border border-[#141414] p-1 font-mono text-xs"
                      />
                      <span className="text-[10px] font-mono opacity-50">T</span>
                    </div>
                  )}
                </div>

                <div className="space-y-2 pt-2 border-t border-[#141414]/5">
                  <div className="flex items-center justify-between">
                    <label className="block text-[10px] font-mono uppercase opacity-60">Peak Threshold</label>
                    <button
                      onClick={() => setAutoThreshold(v => !v)}
                      className={`px-2 py-0.5 font-mono text-[8px] uppercase border transition-all ${autoThreshold ? 'border-[#F27D26] text-[#F27D26] bg-[#F27D26]/10 font-bold' : 'border-[#141414]/20 opacity-50 hover:opacity-100'}`}
                    >{autoThreshold ? 'Auto ✓' : 'Auto'}</button>
                  </div>
                  {autoThreshold ? (
                    <p className="text-[9px] font-mono opacity-40 leading-relaxed pt-1">
                      Auto — local noise floor per file: the spectrum is split into blocks and each gets median + 3.5σ (σ from second differences), so the low-frequency background skirt no longer sets the bar for the whole spectrum. Deliberately loose so weak branches survive at every angle. Toggle off for manual frequency-dependent thresholds.
                    </p>
                  ) : (
                    <div className="space-y-2 pt-1">
                      <div className="grid grid-cols-2 gap-2 text-[9px] font-mono opacity-50 uppercase mb-1">
                        <span>Max Freq (T)</span>
                        <span>Threshold (%)</span>
                      </div>
                      {tempFreqThresholds.map((seg, idx) => (
                        <div key={idx} className="flex gap-2 items-center">
                          <div className="flex-1">
                            <input
                              type="number"
                              value={seg.limit === 10000 ? '' : seg.limit}
                              placeholder={idx === 3 ? '∞' : ''}
                              disabled={idx === 3}
                              onChange={(e) => {
                                const newSegs = [...tempFreqThresholds];
                                newSegs[idx].limit = parseFloat(e.target.value) || 0;
                                setTempFreqThresholds(newSegs);
                              }}
                              className={`w-full bg-white border border-[#141414] p-1 font-mono text-[10px] ${idx === 3 ? 'opacity-50' : ''}`}
                            />
                          </div>
                          <div className="flex-1">
                            <input
                              type="number"
                              value={seg.threshold}
                              onChange={(e) => {
                                const newSegs = [...tempFreqThresholds];
                                newSegs[idx].threshold = parseFloat(e.target.value) || 0;
                                setTempFreqThresholds(newSegs);
                              }}
                              className="w-full bg-white border border-[#141414] p-1 font-mono text-[10px]"
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="space-y-2 pt-2 border-t border-[#141414]/5">
                  <label className="block text-[10px] font-mono uppercase opacity-60">Branch Analysis</label>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-mono">Tolerance (T):</span>
                      <input 
                        type="number" 
                        value={tempBranchTolerance} 
                        onChange={(e) => setTempBranchTolerance(parseFloat(e.target.value) || 0)}
                        className="w-16 bg-white border border-[#141414] p-1 font-mono text-[10px]"
                      />
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-mono">Min Peaks:</span>
                      <input 
                        type="number" 
                        min="2"
                        value={tempMinPeaksPerBranch} 
                        onChange={(e) => setTempMinPeaksPerBranch(parseInt(e.target.value) || 2)}
                        className="w-16 bg-white border border-[#141414] p-1 font-mono text-[10px]"
                      />
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-mono">Min Peak Width:</span>
                      <input 
                        type="number" 
                        min="1"
                        value={tempMinPeakWidth} 
                        onChange={(e) => setTempMinPeakWidth(parseInt(e.target.value) || 1)}
                        className="w-16 bg-white border border-[#141414] p-1 font-mono text-[10px]"
                      />
                    </div>
                  </div>
                </div>

                <button
                  onClick={commitStagedSettings}
                  className="w-full bg-[#141414] text-[#E4E3E0] py-2 font-mono text-[10px] uppercase tracking-wider hover:bg-opacity-80 transition-all"
                >
                  Apply All Settings
                </button>
                <p className="text-[9px] font-mono opacity-40 mt-1.5 leading-relaxed">
                  위 설정을 바꾼 뒤에는 Apply 를 먼저 누르고 Process All 을 실행하세요.
                </p>
              </div>

              {/* Complex Analysis */}
              <div className="pt-4 border-t border-[#141414]/5">
                <label className="flex items-center gap-2 cursor-pointer mb-2">
                  <input 
                    type="checkbox" 
                    checked={tempComplexMode} 
                    onChange={(e) => setTempComplexMode(e.target.checked)}
                    className="accent-[#141414]"
                  />
                  <span className="text-[10px] font-mono uppercase opacity-60">Complex Mode (In+Out)</span>
                </label>
                {tempComplexMode && (
                  <div className="space-y-3 pl-5">
                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] font-mono">Phase: {tempPhase}°</span>
                        <button 
                          onClick={() => {
                            // findAutoPhase logic needs to be adapted for temp states if needed, 
                            // but for now we'll just let it set the tempPhase
                            findAutoPhase(); 
                          }}
                          className="text-[10px] font-mono bg-[#141414] text-[#E4E3E0] px-2 py-0.5 hover:bg-opacity-80"
                        >
                          Auto
                        </button>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="360" 
                        step="5"
                        value={tempPhase} 
                        onChange={(e) => setTempPhase(parseInt(e.target.value))}
                        className="w-full accent-[#141414]"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>

          <section className="bg-white/50 p-6 border border-[#141414]/10">
            <div className="flex items-center gap-2 mb-4 text-[#141414]">
              <Activity size={18} />
              <h2 className="font-serif italic font-bold text-lg">FFT Display Settings</h2>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-[10px] font-mono uppercase opacity-60">Auto Frequency Range</label>
                <input 
                  type="checkbox" 
                  checked={autoFFTRange} 
                  onChange={(e) => setAutoFFTRange(e.target.checked)}
                  className="accent-[#F27D26]"
                />
              </div>
              {!autoFFTRange && (
                <div>
                  <label className="block text-[10px] font-mono uppercase opacity-60 mb-1">Max Frequency (T)</label>
                  <input 
                    type="number" 
                    value={manualMaxFreq} 
                    onChange={(e) => setManualMaxFreq(Number(e.target.value))}
                    className="w-full bg-white border border-[#141414] p-1 font-mono text-xs"
                  />
                </div>
              )}
            </div>
          </section>

          <section className="bg-white/50 p-6 border border-[#141414]/10">
            <div className="flex items-center gap-2 mb-4 text-[#141414]">
              <Settings2 size={18} />
              <h2 className="font-serif italic font-bold text-lg">BLM / Steps Settings</h2>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-[10px] font-mono uppercase opacity-60 mb-1">X-Axis Mode</label>
                <select 
                  value={blmMode} 
                  onChange={(e) => setBlmMode(e.target.value as 'metadata' | 'step')}
                  className="w-full bg-white border border-[#141414] p-1 font-mono text-xs"
                >
                  <option value="metadata">BLM (Metadata)</option>
                  <option value="step">BLM (Steps)</option>
                </select>
              </div>
              <div className="flex items-center gap-2 py-2">
                <button 
                  onClick={() => setReadAngleFromTitle(!readAngleFromTitle)}
                  className={`flex-1 flex items-center justify-between p-2 border border-[#141414] font-mono text-xs transition-colors ${readAngleFromTitle ? 'bg-[#141414] text-white' : 'bg-white text-[#141414]'}`}
                >
                  <span>Read Angle from Title</span>
                  {readAngleFromTitle ? <Check size={14} /> : <div className="w-3.5 h-3.5" />}
                </button>
              </div>
              <div>
                <label className="block text-[10px] font-mono uppercase opacity-60 mb-1">1° = [ ] BLM units</label>
                <input 
                  type="number" 
                  step="0.01"
                  value={blmPerDegree} 
                  onChange={(e) => setBlmPerDegree(parseFloat(e.target.value) || 1.0)}
                  className="w-full bg-white border border-[#141414] p-1 font-mono text-xs"
                />
              </div>
              <div>
                <label className="block text-[10px] font-mono uppercase opacity-60 mb-1">BLM Offset</label>
                <input 
                  type="number" 
                  step="0.1"
                  value={blmOffset} 
                  onChange={(e) => setBlmOffset(parseFloat(e.target.value) || 0)}
                  className="w-full bg-white border border-[#141414] p-1 font-mono text-xs"
                />
              </div>
            </div>
          </section>

          {error && (
            <div className="bg-red-50 border border-red-200 p-4 text-red-700 text-sm flex gap-2">
              <AlertCircle size={16} className="shrink-0" />
              <p>{error}</p>
            </div>
          )}
        </aside>
        )}

        {/* Main Content / Charts */}
        <div className={`space-y-12 ${showControlSidebar ? 'lg:col-span-9' : 'lg:col-span-12'}`}>
          {activeTab === 'dashboard' ? (
            <div className="space-y-12">
              {files.length > 0 && (
                <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-3">
                      <Activity size={24} className="text-[#F27D26]" />
                      <h3 className="text-2xl font-serif italic font-bold">Multi-File Analysis Summary</h3>
                    </div>
                    <div className="flex items-center gap-6">
                      {prefixes.length > 0 && (
                        <div className="flex flex-col items-end">
                          <span className="text-[9px] font-mono uppercase opacity-40 mb-1">Column Prefix (XX_#)</span>
                          <select 
                            value={selectedPrefix}
                            onChange={(e) => setSelectedPrefix(e.target.value)}
                            className="bg-transparent border-b border-[#E4E3E0]/20 text-[#E4E3E0] font-mono text-xs focus:outline-none focus:border-[#F27D26] transition-colors cursor-pointer"
                          >
                            <option value="" className="bg-[#141414]">All Files</option>
                            {prefixes.map(p => (
                              <option key={p} value={p} className="bg-[#141414]">{p}</option>
                            ))}
                          </select>
                        </div>
                      )}
                      <button 
                        onClick={handleProcessAll}
                        disabled={isProcessing}
                        className="flex items-center gap-2 bg-[#F27D26] hover:bg-[#F27D26]/80 text-white px-4 py-2 font-mono text-[10px] uppercase tracking-widest transition-all disabled:opacity-50"
                      >
                        <RefreshCw size={14} className={isProcessing ? 'animate-spin' : ''} />
                        {isProcessing ? 'Processing...' : 'Process All'}
                      </button>
                      {assembleData.length > 0 && (
                        <button 
                          onClick={() => {
                            // Create frequency grid with interval 1
                            const freqGrid: number[] = [];
                            for (let f = exportMinFreq; f <= exportMaxFreq; f += 1) {
                              freqGrid.push(f);
                            }

                            // Helper for linear interpolation
                            const interpolate = (fft: FFTResult[], targetFreq: number) => {
                              if (fft.length === 0) return 0;
                              if (targetFreq <= fft[0].frequency) return fft[0].amplitude;
                              if (targetFreq >= fft[fft.length - 1].frequency) return fft[fft.length - 1].amplitude;
                              
                              const idx = _.findIndex(fft, f => f.frequency >= targetFreq);
                              if (idx === 0) return fft[0].amplitude;
                              
                              const f0 = fft[idx - 1];
                              const f1 = fft[idx];
                              const t = (targetFreq - f0.frequency) / (f1.frequency - f0.frequency);
                              return f0.amplitude + t * (f1.amplitude - f0.amplitude);
                            };

                            // Build headers: Frequency, {BLM1}, {BLM2}, ...
                            const headers = ["Frequency", ...assembleData.map(d => d.x)].join(",");
                            
                            // Build rows
                            const rows = freqGrid.map(f => {
                              const row = [f];
                              assembleData.forEach(d => {
                                row.push(interpolate(d.fft, f));
                              });
                              return row.join(",");
                            });

                            const csv = headers + "\n" + rows.join("\n");
                            const blob = new Blob([csv], { type: 'text/csv' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `fft_assemble_summary.csv`;
                            a.click();
                          }}
                          className="flex items-center gap-2 bg-[#E4E3E0] hover:bg-[#E4E3E0]/80 text-[#141414] px-4 py-2 font-mono text-[10px] uppercase tracking-widest transition-all"
                        >
                          <Download size={14} />
                          Export FFT Assemble
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {batchSummary.length === 0 && !isProcessing && (
                <div className="h-[500px] border-2 border-dashed border-[#141414]/20 flex flex-col items-center justify-center text-[#141414]/50 bg-white/30 gap-4 px-8 text-center">
                  <Activity size={48} strokeWidth={1} />
                  <p className="font-serif italic text-xl text-[#141414]">Ready to analyze — {files.length} file{files.length !== 1 ? 's' : ''} loaded</p>
                  <button
                    onClick={handleProcessAll}
                    className="bg-[#F27D26] text-[#141414] font-bold py-3 px-8 hover:bg-opacity-90 transition-all flex items-center gap-2 font-mono text-sm uppercase tracking-wider shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none"
                  >
                    <RefreshCw size={18} />
                    Process All Files
                  </button>
                  {(!selectedPrefix && selectedRColumns.length === 0) ? (
                    <p className="text-[11px] font-mono text-[#F27D26]">⚠ No R column selected — pick a prefix or column under “Columns” in the sidebar first.</p>
                  ) : (
                    <p className="text-[10px] font-mono uppercase tracking-widest opacity-40">
                      Column: {selectedPrefix ? `${selectedPrefix}_*` : selectedRColumns.join(', ')} · B-range {minB}–{maxB} T
                    </p>
                  )}
                </div>
              )}

              {isProcessing && (
                <div className="h-[500px] flex flex-col items-center justify-center bg-white/30 border border-[#141414]/5">
                  <RefreshCw className="animate-spin text-[#F27D26] mb-4" size={48} />
                  <p className="font-mono text-sm uppercase tracking-widest animate-pulse">Analyzing all files...</p>
                </div>
              )}

              {batchSummary.length > 0 && !isProcessing && (
                <div className="grid grid-cols-1 gap-12">
                  {/* Spectral Overview Section for All Files */}
                  <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                    <div className="flex items-center gap-3 mb-8">
                      <Activity size={20} className="text-[#F27D26]" />
                      <h4 className="text-sm font-mono uppercase tracking-widest">Global Spectral Overview (All Files)</h4>
                    </div>
                    
                    <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                      {/* Raw Data Overlay */}
                      <div className="h-[300px] bg-white/5 p-4 border border-white/10">
                        <h5 className="text-[10px] font-mono uppercase opacity-60 mb-4 text-center">Raw Data (R vs B)</h5>
                        <ResponsiveContainer width="100%" height="90%">
                          <LineChart margin={{ bottom: 10, left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                            <XAxis 
                              dataKey="b" 
                              type="number" 
                              domain={['auto', 'auto']}
                              tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            />
                            <YAxis tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }} />
                            <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '8px', color: '#E4E3E0' }} />
                            {allRawSeries.slice(0, 20).map((s, idx) => (
                              <Line 
                                key={idx}
                                data={s.data}
                                type="monotone"
                                dataKey="val"
                                name={s.fileName}
                                stroke={`hsl(${(idx * 360) / Math.min(20, allRawSeries.length)}, 70%, 60%)`}
                                dot={false}
                                strokeWidth={1}
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                        {allRawSeries.length > 20 && <p className="text-[8px] text-center opacity-40 mt-1">Showing first 20 series</p>}
                      </div>

                      {/* Oscillatory Data Overlay */}
                      <div className="h-[300px] bg-white/5 p-4 border border-white/10">
                        <h5 className="text-[10px] font-mono uppercase opacity-60 mb-4 text-center">Oscillatory Data (ΔR vs 1/B)</h5>
                        <ResponsiveContainer width="100%" height="90%">
                          <LineChart margin={{ bottom: 10, left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                            <XAxis 
                              dataKey="invB" 
                              type="number" 
                              domain={['auto', 'auto']}
                              tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            />
                            <YAxis tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }} />
                            <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '8px', color: '#E4E3E0' }} />
                            {allRawSeries.slice(0, 20).map((s, idx) => (
                              <Line 
                                key={idx}
                                data={s.invBData}
                                type="monotone"
                                dataKey="val"
                                name={s.fileName}
                                stroke={`hsl(${(idx * 360) / Math.min(20, allRawSeries.length)}, 70%, 60%)`}
                                dot={false}
                                strokeWidth={1}
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      {/* FFT Overlay */}
                      <div className="h-[300px] bg-white/5 p-4 border border-white/10">
                        <h5 className="text-[10px] font-mono uppercase opacity-60 mb-4 text-center">FFT Results (Amp vs Freq)</h5>
                        <ResponsiveContainer width="100%" height="90%">
                          <LineChart margin={{ bottom: 10, left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                            <XAxis
                              dataKey="frequency"
                              type="number"
                              domain={[0, autoFFTRange ? globalSuggestedMaxFreq : manualMaxFreq]}
                              allowDataOverflow={true}
                              tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            />
                            <YAxis tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }} />
                            <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '8px', color: '#E4E3E0' }} />
                            {assembleChartData.slice(0, 20).map((s, idx) => (
                              <Line
                                key={idx}
                                data={s.fftDisplay}
                                type="monotone"
                                dataKey="amplitude"
                                name={s.fileName}
                                stroke={`hsl(${(idx * 360) / Math.min(20, assembleData.length)}, 70%, 60%)`}
                                dot={false}
                                strokeWidth={1}
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  {/* Summary Charts */}
                  <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10 animate-in fade-in zoom-in duration-500">
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-12">
                      <div className="h-[450px] w-full bg-[#141414]/50 p-6 border border-[#E4E3E0]/5">
                        <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                          <h4 className="text-xs font-mono uppercase opacity-60">All Detected Frequencies vs BLM</h4>
                          <div className="flex items-center gap-1.5">
                            <span className="text-[8px] font-mono uppercase opacity-40">Live Range:</span>
                            <input type="number" step="0.1" value={dashLiveRange.minDraft}
                              ref={wheelStepRef(dashLiveRange.minDraft, dashLiveRange.changeMin)}
                              onChange={(e) => dashLiveRange.changeMin(Number(e.target.value))}
                              title="Process All 없이 즉시 미리보기할 Min B (스크롤로 조절 가능)"
                              className="w-14 bg-[#141414] border border-[#E4E3E0]/30 p-0.5 font-mono text-[9px] text-[#E4E3E0]" />
                            <span className="opacity-30 text-[9px]">–</span>
                            <input type="number" step="0.1" value={dashLiveRange.maxDraft}
                              ref={wheelStepRef(dashLiveRange.maxDraft, dashLiveRange.changeMax)}
                              onChange={(e) => dashLiveRange.changeMax(Number(e.target.value))}
                              title="Process All 없이 즉시 미리보기할 Max B (스크롤로 조절 가능)"
                              className="w-14 bg-[#141414] border border-[#E4E3E0]/30 p-0.5 font-mono text-[9px] text-[#E4E3E0]" />
                            {dashLiveRange.active && (
                              <button onClick={() => dashLiveRange.reset(typeof minB === 'number' ? minB : 0, typeof maxB === 'number' ? maxB : 1)}
                                title="Process All 결과로 되돌리기"
                                className="px-1.5 py-0.5 font-mono text-[8px] uppercase border border-[#F27D26]/50 text-[#F27D26] hover:bg-[#F27D26]/10">Reset</button>
                            )}
                          </div>
                        </div>
                        {dashLiveRange.active && (
                          <p className="text-[8px] font-mono text-[#F27D26] text-center mb-2">
                            Live preview [{dashLiveMinBEff.toFixed(2)}, {dashLiveMaxBEff.toFixed(2)}] T — not yet Process All'd
                          </p>
                        )}
                        <ResponsiveContainer width="100%" height="90%">
                          <ScatterChart margin={{ bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                            <XAxis 
                              dataKey="blm" 
                              type="number"
                              tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                              label={{ value: 'BLM', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 10 }}
                            />
                            <YAxis 
                              dataKey="freq"
                              type="number"
                              domain={[0, autoFFTRange ? globalSuggestedMaxFreq : manualMaxFreq]}
                              allowDataOverflow={true}
                              tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                              label={{ value: 'Frequency (T)', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 10 }}
                            />
                            <ZAxis dataKey="amp" range={[20, 200]} name="Amplitude" />
                            <Tooltip 
                              contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '10px', color: '#E4E3E0' }}
                              cursor={{ strokeDasharray: '3 3' }}
                              formatter={(value: any, name: string, props: any) => {
                                if (name === 'freq') return [Number(value).toFixed(4), 'Frequency (T)'];
                                if (name === 'blm') return [Number(value).toFixed(2), 'BLM'];
                                if (name === 'amp') return [Number(value).toExponential(3), 'Amplitude'];
                                return [value, name];
                              }}
                              labelFormatter={() => ''}
                              content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                  const data = payload[0].payload;
                                  return (
                                    <div className="bg-[#141414] border border-[#E4E3E0] p-2 font-mono text-[10px] text-[#E4E3E0] shadow-xl">
                                      <p className="text-[#F27D26] font-bold mb-1 border-b border-[#E4E3E0]/20 pb-1 truncate max-w-[200px]">{data.fileName}</p>
                                      <p><span className="opacity-60">Frequency:</span> {data.freq.toFixed(4)} T</p>
                                      <p><span className="opacity-60">BLM:</span> {data.blm}{blmMode === 'metadata' ? '°' : ''}</p>
                                      <p><span className="opacity-60">Amplitude:</span> {data.amp.toExponential(4)}</p>
                                      {data.temp && <p><span className="opacity-60">Temperature:</span> {data.temp.toFixed(2)} K</p>}
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <Legend verticalAlign="top" height={36}/>
                            <Scatter
                              name="Detected Peaks"
                              data={displayedBatchSummary}
                              fill="#F27D26"
                              stroke="#FFFFFF"
                              strokeWidth={1}
                            />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="h-[450px] w-full bg-[#141414]/50 p-6 border border-[#E4E3E0]/5">
                        <h4 className="text-xs font-mono uppercase opacity-60 mb-6 text-center">Peak Amplitude Distribution</h4>
                        <ResponsiveContainer width="100%" height="90%">
                          <ScatterChart margin={{ bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                            <XAxis 
                              dataKey="blm" 
                              type="number"
                              tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                              label={{ value: 'BLM', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 10 }}
                            />
                            <YAxis 
                              dataKey="amp"
                              type="number"
                              scale="log"
                              domain={['auto', 'auto']}
                              tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                              label={{ value: 'Amplitude (log)', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 10 }}
                            />
                            <Tooltip 
                              contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '10px', color: '#E4E3E0' }}
                              cursor={{ strokeDasharray: '3 3' }}
                              content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                  const data = payload[0].payload;
                                  return (
                                    <div className="bg-[#141414] border border-[#E4E3E0] p-2 font-mono text-[10px] text-[#E4E3E0] shadow-xl">
                                      <p className="text-[#F27D26] font-bold mb-1 border-b border-[#E4E3E0]/20 pb-1 truncate max-w-[200px]">{data.fileName}</p>
                                      <p><span className="opacity-60">Amplitude:</span> {data.amp.toExponential(4)}</p>
                                      <p><span className="opacity-60">Frequency:</span> {data.freq.toFixed(4)} T</p>
                                      <p><span className="opacity-60">BLM:</span> {data.blm}{blmMode === 'metadata' ? '°' : ''}</p>
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <Legend verticalAlign="top" height={36}/>
                            <Scatter 
                              name="Peak Amplitudes" 
                              data={batchSummary} 
                              fill="#FFFFFF" 
                              stroke="#F27D26"
                              strokeWidth={1}
                            />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="overflow-x-auto">
                      <div className="flex justify-between items-end mb-4">
                        <h4 className="text-sm font-mono uppercase tracking-widest text-[#F27D26]">All Detected Peaks</h4>
                        <button 
                          onClick={() => {
                            const csv = `File Name,BLM,Frequency (T),Amplitude\n` + 
                              batchSummary.map(item => `"${item.fileName}",${item.blm},${item.freq},${item.amp}`).join('\n');
                            const blob = new Blob([csv], { type: 'text/csv' });
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.setAttribute('hidden', '');
                            a.setAttribute('href', url);
                            a.setAttribute('download', 'sdh_all_peaks_summary.csv');
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                          }}
                          className="flex items-center gap-2 text-[10px] font-mono uppercase bg-[#E4E3E0]/10 hover:bg-[#E4E3E0]/20 px-3 py-1 transition-all"
                        >
                          <Download size={12} />
                          Export All Peaks CSV
                        </button>
                      </div>
                      <table className="w-full text-left font-mono text-[10px]">
                        <thead>
                          <tr className="border-b border-[#E4E3E0]/20 text-[#F27D26]">
                            <th className="py-2 px-4 uppercase tracking-wider">File Name</th>
                            <th className="py-2 px-4 uppercase tracking-wider text-center">BLM</th>
                            <th className="py-2 px-4 uppercase tracking-wider text-right">Frequency (T)</th>
                            <th className="py-2 px-4 uppercase tracking-wider text-right">Amplitude</th>
                          </tr>
                        </thead>
                        <tbody>
                          {batchSummary.map((item) => (
                            <tr key={item.id} className="border-b border-[#E4E3E0]/5 hover:bg-[#E4E3E0]/10 transition-colors">
                              <td className="py-2 px-4 opacity-70 truncate max-w-[200px]">{item.fileName}</td>
                              <td className="py-2 px-4 font-bold text-center">{item.blm}{blmMode === 'metadata' ? '°' : ''}</td>
                              <td className="py-2 px-4 text-right">{item.freq.toFixed(4)}</td>
                              <td className="py-2 px-4 text-right font-mono">{item.amp.toExponential(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* FFT Evolution vs BLM Table */}
                </div>
              )}
            </div>
          ) : activeTab === 'raw' ? (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">

              {/* ── Control Panel ── */}
              <div className="bg-[#141414] text-[#E4E3E0] border border-[#E4E3E0]/10">
                {/* Top row */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-[#E4E3E0]/10">
                  <div className="flex items-center gap-3">
                    <Activity size={18} className="text-[#F27D26]" />
                    <h3 className="text-lg font-serif italic font-bold">Raw Data Analysis</h3>
                    <span className="text-[9px] font-mono opacity-30 ml-2">{files.length} file{files.length !== 1 ? 's' : ''} · {uniqueBlms.length} angles</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <select
                      value={selectedRawColumn}
                      onChange={(e) => setSelectedRawColumn(e.target.value)}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[10px] px-3 py-1.5 focus:outline-none focus:border-[#F27D26] cursor-pointer"
                    >
                      <option value="" className="bg-[#141414]">— select column —</option>
                      <optgroup label="Prefixes" className="bg-[#141414]">
                        {prefixes.map(p => <option key={p} value={p}>{p} (all)</option>)}
                      </optgroup>
                      <optgroup label="Columns" className="bg-[#141414]">
                        {files[selectedFileIndex]?.columns.map(c => <option key={c} value={c}>{c}</option>)}
                      </optgroup>
                    </select>
                    <div className="flex items-center gap-1.5 px-3 py-1.5 border border-[#4ADE80]/30 text-[#4ADE80] text-[10px] font-mono">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#4ADE80]" />
                      Cos Fit
                    </div>
                  </div>
                </div>

                {/* BLM filter row */}
                <div className="flex items-start gap-4 px-6 py-3 border-b border-[#E4E3E0]/10">
                  <div className="flex items-center gap-1.5 pt-0.5 flex-none">
                    <span className="text-[9px] font-mono uppercase opacity-40 mr-1">BLM</span>
                    <button onClick={() => setActiveBlms(new Set(uniqueBlms))} className="px-1.5 py-0.5 font-mono text-[8px] uppercase border border-[#E4E3E0]/20 hover:bg-white/10">All</button>
                    <button onClick={() => setActiveBlms(new Set())} className="px-1.5 py-0.5 font-mono text-[8px] uppercase border border-[#E4E3E0]/20 hover:bg-white/10">None</button>
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-16 overflow-y-auto">
                    {uniqueBlms.map((blm, idx) => (
                      <button
                        key={blm}
                        onClick={() => { const n = new Set(activeBlms); n.has(blm) ? n.delete(blm) : n.add(blm); setActiveBlms(n); }}
                        className={`px-1.5 py-0.5 font-mono text-[8px] border transition-all ${activeBlms.has(blm) ? 'text-[#141414] border-transparent' : 'border-[#E4E3E0]/15 text-[#E4E3E0] opacity-35 hover:opacity-70'}`}
                        style={{ backgroundColor: activeBlms.has(blm) ? `hsl(${(idx * 137) % 360}, 55%, 55%)` : undefined }}
                      >
                        {blm}°
                      </button>
                    ))}
                  </div>
                </div>

                {/* Export row */}
                <div className="flex items-center gap-2 px-6 py-2.5">
                  <span className="text-[8px] font-mono uppercase opacity-30 mr-1">Export</span>
                  <button onClick={exportAllColumnsCSV} className="px-2.5 py-1 font-mono text-[8px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"><Download size={9} />Raw CSV</button>
                  <button onClick={exportProcessedCSV} className="px-2.5 py-1 font-mono text-[8px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"><Download size={9} />ΔR</button>
                  <button onClick={exportFFTSpectraCSV} className="px-2.5 py-1 font-mono text-[8px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"><Download size={9} />FFT</button>
                  <button onClick={exportPeaksWideCSV} className="px-2.5 py-1 font-mono text-[8px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"><Download size={9} />Peaks</button>
                  <button onClick={exportOscillatory1overBCSV} className="px-2.5 py-1 font-mono text-[8px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"><Download size={9} />Osc 1/B</button>
                  <span className="w-px h-4 bg-[#E4E3E0]/15 mx-1" />
                  <button onClick={exportRawAlignedCSV} title="FFT Assemble처럼 하나의 B(T) 그리드에 보간, 각도/온도순 정렬" className="px-2.5 py-1 font-mono text-[8px] uppercase border border-[#F27D26]/40 text-[#F27D26] hover:bg-[#F27D26]/10 flex items-center gap-1"><Download size={9} />Raw (Aligned)</button>
                  <button onClick={exportBackgroundSubtractedAlignedCSV} title="B(T) 그리드에 보간, polynomial 배경 제거된 ΔR, 각도/온도순 정렬" className="px-2.5 py-1 font-mono text-[8px] uppercase border border-[#F27D26]/40 text-[#F27D26] hover:bg-[#F27D26]/10 flex items-center gap-1"><Download size={9} />ΔR (Aligned)</button>
                  <button onClick={exportOscillatoryAlignedCSV} title="FFT Assemble처럼 하나의 1/B 그리드에 보간, 각도/온도순 정렬" className="px-2.5 py-1 font-mono text-[8px] uppercase border border-[#F27D26]/40 text-[#F27D26] hover:bg-[#F27D26]/10 flex items-center gap-1"><Download size={9} />Osc (Aligned)</button>
                  {activeBlms.size > 30 && <span className="ml-2 text-[8px] font-mono text-[#F27D26] animate-pulse">{activeBlms.size} series active — performance may suffer</span>}
                </div>
              </div>

              <div className="grid grid-cols-1 gap-6">

                {/* ── 1. Angular Distribution ── */}
                <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Angular Distribution</h4>
                      <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">
                        {selectedRawColumn || '—'} <span className="text-[#E4E3E0] opacity-40">@ {rawBSliceTarget === null ? 'Max B' : `${rawBSliceTarget} T`}</span>
                      </p>
                      <div className="flex items-center gap-1.5 mt-2">
                        <span className="text-[8px] font-mono uppercase opacity-30 mr-0.5">Harmonics</span>
                        {([1, 2, 3, 4, 6, 12] as const).map(h => (
                          <button
                            key={h}
                            onClick={() => {
                              const next = new Set(selectedHarmonics);
                              next.has(h) ? next.delete(h) : next.add(h);
                              setSelectedHarmonics(next);
                            }}
                            className={`px-1.5 py-0.5 font-mono text-[8px] border transition-all ${selectedHarmonics.has(h) ? 'border-[#4ADE80]/60 text-[#4ADE80] bg-[#4ADE80]/10' : 'border-[#E4E3E0]/15 text-[#E4E3E0] opacity-30 hover:opacity-60'}`}
                          >
                            n={h}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="number" step="0.5" placeholder="B (T)"
                        value={rawBSliceInput}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setRawBSliceInput(e.target.value)}
                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter') { const v = parseFloat(rawBSliceInput); setRawBSliceTarget(isFinite(v) ? v : null); } }}
                        className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[10px] w-20 px-2 py-1 focus:outline-none focus:border-[#F27D26] text-center"
                      />
                      <button onClick={() => { const v = parseFloat(rawBSliceInput); setRawBSliceTarget(isFinite(v) ? v : null); }} className="px-2 py-1 font-mono text-[10px] border border-[#F27D26]/50 text-[#F27D26] hover:bg-[#F27D26]/10">Set</button>
                      <button onClick={() => { setRawBSliceTarget(null); setRawBSliceInput(''); }} className={`px-2 py-1 font-mono text-[10px] border transition-all ${rawBSliceTarget === null ? 'border-[#F27D26] text-[#F27D26] bg-[#F27D26]/10' : 'border-[#E4E3E0]/15 opacity-40 hover:opacity-70'}`}>Max B</button>
                      <button
                        onClick={() => { const csv = "BLM(deg),R,FileName,Column,B(T)\n" + rawPlotDisplayData.filter((d: any) => activeBlms.size === 0 || activeBlms.has(d.x)).map((d: any) => `${d.x},${d.y},"${d.fileName}","${d.columnName}",${rawBSliceTarget ?? 'MaxB'}`).join('\n'); downloadCSV(csv, `R_vs_BLM_${rawBSliceTarget ?? 'MaxB'}T_${selectedRawColumn}.csv`); }}
                        className="px-2 py-1 font-mono text-[9px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"
                      ><Download size={9} />CSV</button>
                    </div>
                  </div>
                  <div className="h-[380px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart margin={{ bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                        <XAxis
                          dataKey="x"
                          type="number"
                          name="BLM"
                          tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                          label={{ value: 'BLM (°)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 10 }}
                        />
                        <YAxis
                          dataKey="y"
                          tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                          label={{ value: selectedRawColumn, angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 10 }}
                          tickFormatter={(v: number) => v.toExponential(2)}
                        />
                        <Tooltip
                          cursor={{ strokeDasharray: '3 3' }}
                          contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '10px', color: '#E4E3E0' }}
                          formatter={(val: any) => typeof val === 'number' ? val.toExponential(4) : val}
                        />
                        <Legend wrapperStyle={{ paddingTop: '20px', fontSize: '10px', fontFamily: 'monospace' }} />
                        <Scatter
                          name={`${selectedRawColumn} @ ${rawBSliceTarget === null ? 'Max B' : rawBSliceTarget + ' T'}`}
                          data={rawPlotFitData}
                          fill="#F27D26"
                          isAnimationActive={false}
                          shape={(props: any) => {
                            const { cx, cy } = props;
                            return <circle cx={cx} cy={cy} r={7} fill="#F27D26" fillOpacity={0.85} />;
                          }}
                        />
                        {cosFitCurve.length > 0 && (
                          <Scatter
                            name="Cos Fit"
                            data={cosFitCurve}
                            line={{ stroke: '#4ADE80', strokeWidth: 2 }}
                            lineType="joint"
                            shape={(props: any) => <circle cx={props.cx} cy={props.cy} r={0} />}
                            fill="#4ADE80"
                            isAnimationActive={false}
                          />
                        )}
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                  {cosFitAtSlice && (
                    <div className="mt-3 flex flex-wrap gap-2 items-center">
                      {([
                        [1, cosFitAtSlice.A1], [2, cosFitAtSlice.A2], [3, cosFitAtSlice.A3],
                        [4, cosFitAtSlice.A4], [6, cosFitAtSlice.A6], [12, cosFitAtSlice.A12],
                      ] as [number, number][]).filter(([h]) => selectedHarmonics.has(h)).map(([h, val]) => (
                        <div key={h} className="border border-[#4ADE80]/25 px-3 py-1.5 bg-[#4ADE80]/5 flex items-center gap-2">
                          <span className="text-[8px] font-mono uppercase text-[#4ADE80] opacity-70">A{h}</span>
                          <span className="text-[11px] font-mono text-[#4ADE80]">{val.toExponential(3)}</span>
                        </div>
                      ))}
                      <span className="text-[8px] font-mono opacity-25 text-[#E4E3E0] ml-1">y₀={cosFitAtSlice.y0.toExponential(2)}</span>
                      <div className="ml-auto flex gap-3 items-center border-l border-[#E4E3E0]/10 pl-3">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[8px] font-mono uppercase opacity-40 text-[#E4E3E0]">R²</span>
                          <span className="text-[11px] font-mono text-[#4ADE80]">{cosFitAtSlice.r2.toFixed(5)}</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[8px] font-mono uppercase opacity-40 text-[#E4E3E0]">RMSE</span>
                          <span className="text-[11px] font-mono text-[#4ADE80]">{cosFitAtSlice.rmse.toExponential(3)}</span>
                        </div>
                      </div>
                    </div>
                  )}
                  {!cosFitAtSlice && rawPlotFitData.length > 0 && selectedHarmonics.size > 0 && (
                    <p className="mt-2 text-[9px] font-mono text-[#4ADE80]/50">Need ≥{1 + 2 * selectedHarmonics.size} data points for {selectedHarmonics.size} harmonics (currently {rawPlotFitData.length})</p>
                  )}
                </div>

                {/* ── 2. R vs B-Field ── */}
                <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Field Sweep</h4>
                      <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">{selectedRawColumn || '—'} <span className="text-[#E4E3E0] opacity-40">vs B-Field</span></p>
                    </div>
                    <button onClick={exportRawDataCSV} className="px-2 py-1 font-mono text-[9px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"><Download size={9} />CSV</button>
                  </div>
                  <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                    <div className="lg:col-span-3 h-[420px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart margin={{ bottom: 20, left: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                          <XAxis 
                            dataKey="b" 
                            type="number" 
                            domain={['auto', 'auto']} 
                            tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            label={{ value: 'B-Field (T)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 10 }}
                          />
                          <YAxis 
                            tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            label={{ value: selectedRawColumn, angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 10 }}
                          />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '10px', color: '#E4E3E0' }}
                          />
                          <Legend 
                            wrapperStyle={{ paddingTop: '20px', fontSize: '10px', fontFamily: 'monospace' }}
                          />
                          {allRawSeries.filter(s => activeBlms.has(s.blm)).slice(0, 50).map((series) => {
                            const blmIdx = uniqueBlms.indexOf(series.blm);
                            const color = series.fileName === files[selectedFileIndex]?.name ? "#F27D26" : `hsl(${(blmIdx * 137) % 360}, 50%, 50%)`;
                            return (
                              <Line 
                                key={`${series.fileName}-${series.columnName}`} 
                                name={`${series.blm}°`} 
                                data={series.data} 
                                dataKey="val"
                                stroke={color} 
                                strokeWidth={series.fileName === files[selectedFileIndex]?.name ? 2 : 1}
                                dot={false}
                                isAnimationActive={false}
                              />
                            );
                          })}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="border border-[#E4E3E0]/10 p-3 overflow-hidden flex flex-col h-[420px]">
                      <h4 className="text-[9px] font-mono uppercase opacity-40 mb-3 text-[#E4E3E0]">Data (first 100 pts)</h4>
                      <div className="flex-1 overflow-auto">
                        <table className="w-full text-left border-collapse">
                          <thead className="sticky top-0 bg-[#141414] z-10">
                            <tr className="border-b border-[#E4E3E0]/20">
                              <th className="p-1 text-[9px] font-mono uppercase opacity-60 text-[#E4E3E0]">Idx</th>
                              <th className="p-1 text-[9px] font-mono uppercase opacity-60 text-[#E4E3E0]">B (T)</th>
                              <th className="p-1 text-[9px] font-mono uppercase opacity-60 text-[#E4E3E0]">R (Ω)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {files.length > 0 && files[selectedFileIndex].data.slice(0, 100).map((row, idx) => {
                              const bCol = selectedBColumn || Object.keys(row)[0];
                              const rCol = selectedRawColumn || Object.keys(row)[1];
                              return (
                                <tr key={idx} className="border-b border-[#E4E3E0]/5 hover:bg-white/5">
                                  <td className="p-1 text-[9px] font-mono text-[#E4E3E0] opacity-40">{idx + 1}</td>
                                  <td className="p-1 text-[9px] font-mono text-[#F27D26]">{parseFloat(String(row[bCol])).toFixed(4)}</td>
                                  <td className="p-1 text-[9px] font-mono text-[#E4E3E0]">{parseFloat(String(row[rCol])).toFixed(4)}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      <div className="mt-2 text-[8px] font-mono opacity-40 text-[#E4E3E0] truncate">
                        {files.length > 0 ? files[selectedFileIndex].name : 'No file'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* ── 3. Advanced: 2-col side by side ── */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                    <div className="flex justify-between items-center mb-4">
                      <div>
                        <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Peak Transition Field</h4>
                        <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">B<sub>max dR/dB</sub> <span className="text-[#E4E3E0] opacity-40">vs BLM</span></p>
                      </div>
                      <div className="flex items-center gap-2">
                        <input type="number" value={maxDerivMinB} onChange={(e) => setMaxDerivMinB(parseFloat(e.target.value) || 0)} className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-10 px-1 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                        <span className="text-[9px] opacity-30">–</span>
                        <input type="number" value={maxDerivMaxB} onChange={(e) => setMaxDerivMaxB(parseFloat(e.target.value) || 0)} className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-10 px-1 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                        <span className="text-[9px] opacity-30">T</span>
                        <button onClick={() => handleProcessAll()} className="text-[#F27D26] hover:opacity-70 ml-1" title="Recalculate"><RefreshCw size={11} /></button>
                        <button onClick={() => { const filtered = maxDerivPlotData.filter(d => activeBlms.has(d.x)); downloadCSV("BLM(°),Field(T),FileName,Column\n" + filtered.map(d => `${d.x},${d.y},${d.fileName},${d.columnName}`).join('\n'), `max_deriv_vs_blm_${maxDerivMinB}T_${maxDerivMaxB}T.csv`); }} className="px-2 py-0.5 font-mono text-[8px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"><Download size={8} />CSV</button>
                      </div>
                    </div>
                    <div className="h-[340px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#222" />
                          <XAxis dataKey="x" type="number" name="BLM" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }} label={{ value: 'BLM (°)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }} />
                          <YAxis dataKey="y" name="B peak" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }} label={{ value: 'B (T)', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 9 }} />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '9px', color: '#E4E3E0' }} />
                          <Scatter name="B at max dR/dB" data={maxDerivPlotData.filter(d => activeBlms.has(d.x))} fill="#F27D26" shape="star" isAnimationActive={false} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                    <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50 mb-4">Individual Oscillatory Curves <span className="opacity-50">(ΔR vs 1/B)</span></h4>
                    <div className="grid grid-cols-2 gap-2 max-h-[340px] overflow-y-auto">
                      {allRawSeries.filter(s => activeBlms.has(s.blm)).slice(0, 30).map((series) => (
                        <div key={`${series.fileName}-${series.columnName}`} className="border border-[#E4E3E0]/8 p-2">
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-[8px] font-mono truncate max-w-[80px] opacity-40">{series.fileName}</span>
                            <span className="text-[8px] font-mono text-[#F27D26]">{series.blm}°</span>
                          </div>
                          <div className="h-[70px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                              <AreaChart data={series.invBData}>
                                <XAxis dataKey="invB" hide type="number" domain={[maxB && Number(maxB) !== 0 ? 1/Number(maxB) : 'auto', minB && Number(minB) !== 0 ? 1/Number(minB) : 'auto']} allowDataOverflow={true} />
                                <YAxis hide />
                                <Area type="monotone" dataKey="val" stroke="#F27D26" fill="#F27D26" fillOpacity={0.2} dot={false} isAnimationActive={false} strokeWidth={1} />
                              </AreaChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      ))}
                    </div>
                    {allRawSeries.filter(s => activeBlms.has(s.blm)).length > 30 && (
                      <p className="text-[8px] font-mono opacity-30 mt-2 text-center">Showing first 30. Filter BLMs to see others.</p>
                    )}
                  </div>
                </div>

                {/* ── 5. R vs BLM at Fixed Field ── */}
                <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Angular Dependence at Fixed Field</h4>
                      <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">{selectedRawColumn || '—'} <span className="text-[#E4E3E0] opacity-40">vs BLM @ multiple B</span></p>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          step="0.5"
                          placeholder="B value (T)"
                          value={fieldSliceInput}
                          onChange={e => setFieldSliceInput(e.target.value)}
                          onKeyDown={e => {
                            if (e.key === 'Enter') {
                              const v = parseFloat(fieldSliceInput);
                              if (isFinite(v) && !fieldSliceValues.includes(v))
                                setFieldSliceValues(prev => [...prev, v].sort((a, b) => a - b));
                              setFieldSliceInput('');
                            }
                          }}
                          className="bg-transparent border-b border-[#E4E3E0]/20 text-[#E4E3E0] font-mono text-[10px] w-24 focus:outline-none focus:border-[#F27D26] text-center"
                        />
                        <button
                          onClick={() => {
                            const v = parseFloat(fieldSliceInput);
                            if (isFinite(v) && !fieldSliceValues.includes(v))
                              setFieldSliceValues(prev => [...prev, v].sort((a, b) => a - b));
                            setFieldSliceInput('');
                          }}
                          className="text-[#F27D26] border border-[#F27D26]/40 px-2 py-0.5 font-mono text-[10px] hover:bg-[#F27D26]/10"
                        >
                          + Add (T)
                        </button>
                      </div>
                      <button
                        onClick={() => setFieldSliceValues([1, 5, 10, 15, 20, 25, 30])}
                        className="px-2 py-0.5 font-mono text-[9px] border border-[#E4E3E0]/20 opacity-50 hover:opacity-80"
                      >
                        Reset
                      </button>
                      <button
                        onClick={() => {
                          const rows: string[] = ['BLM(deg),Field(T),R,FileName'];
                          fieldSliceValues.forEach(b => {
                            (fieldSliceData[b] || []).forEach(pt => {
                              rows.push(`${pt.blm},${b},${pt.val},"${pt.fileName}"`);
                            });
                          });
                          downloadCSV(rows.join('\n'), `R_vs_BLM_fixed_field_${selectedRawColumn || 'all'}.csv`);
                        }}
                        className="bg-[#F27D26] text-white px-3 py-1 font-mono text-[10px] uppercase tracking-widest hover:bg-[#F27D26]/80 flex items-center gap-2"
                      >
                        <Download size={12} />
                        Export CSV
                      </button>
                    </div>
                  </div>

                  {/* Active field chips */}
                  <div className="flex flex-wrap gap-2 mb-6">
                    {fieldSliceValues.map((v, idx) => (
                      <span
                        key={v}
                        className="flex items-center gap-1 px-2 py-0.5 font-mono text-[9px] border border-[#E4E3E0]/15"
                        style={{ borderLeft: `3px solid hsl(${(idx * 51) % 360}, 60%, 55%)` }}
                      >
                        {v} T
                        <button
                          onClick={() => setFieldSliceValues(prev => prev.filter(x => x !== v))}
                          className="opacity-40 hover:opacity-100 hover:text-red-400 ml-1 leading-none"
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>

                  <div className="h-[420px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart margin={{ bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                        <XAxis
                          dataKey="blm"
                          type="number"
                          allowDuplicatedCategory={false}
                          tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                          label={{ value: 'BLM (°)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 10 }}
                        />
                        <YAxis
                          tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                          label={{ value: selectedRawColumn || 'R', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 10 }}
                          tickFormatter={(v: number) => v.toExponential(2)}
                        />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '10px', color: '#E4E3E0' }}
                          formatter={(val: any) => typeof val === 'number' ? val.toExponential(4) : val}
                        />
                        <Legend wrapperStyle={{ paddingTop: '20px', fontSize: '10px', fontFamily: 'monospace' }} />
                        {fieldSliceValues.map((b, idx) => (
                          <Line
                            key={b}
                            name={`B = ${b} T`}
                            data={fieldSliceData[b] || []}
                            dataKey="val"
                            stroke={`hsl(${(idx * 51) % 360}, 60%, 55%)`}
                            strokeWidth={1.5}
                            dot={{ r: 3, fill: `hsl(${(idx * 51) % 360}, 60%, 55%)` }}
                            isAnimationActive={false}
                            connectNulls={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* ── 6. Harmonic Amplitudes vs H² ── */}
                <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Harmonic Amplitudes vs Field</h4>
                      <p className="text-[10px] font-mono mt-0.5">
                        <span style={{ color: '#4ADE80' }}>A₂, A₃, A₄, A₆</span>
                        <span className="text-[#E4E3E0] opacity-40"> vs H² (T²) · 0.1 T intervals</span>
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      {cosHarmonicScanData.length > 0 && <span className="text-[9px] font-mono opacity-30">{cosHarmonicScanData.length} pts</span>}
                      <button
                        onClick={() => { if (cosHarmonicScanData.length === 0) return; downloadCSV('B(T),A1,A2,A3,A4,A6\n' + cosHarmonicScanData.map(d => `${d.b},${d.A1},${d.A2},${d.A3},${d.A4},${d.A6}`).join('\n'), `harmonic_amplitudes_vs_B_${selectedRawColumn || 'all'}.csv`); }}
                        className={`px-2 py-1 font-mono text-[9px] uppercase border flex items-center gap-1 transition-all ${cosHarmonicScanData.length > 0 ? 'border-[#4ADE80]/40 text-[#4ADE80] hover:bg-[#4ADE80]/10' : 'border-[#E4E3E0]/10 opacity-20 cursor-not-allowed'}`}
                      ><Download size={9} />CSV</button>
                    </div>
                  </div>
                  {cosHarmonicScanDataB2.length > 0 ? (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {([
                          { key: 'A1' as const, label: 'A₁ (n=1)', color: '#C084FC', h: 1 },
                          { key: 'A2' as const, label: 'A₂ (n=2)', color: '#60A5FA', h: 2 },
                          { key: 'A3' as const, label: 'A₃ (n=3)', color: '#F472B6', h: 3 },
                          { key: 'A4' as const, label: 'A₄ (n=4)', color: '#FBBF24', h: 4 },
                          { key: 'A6' as const, label: 'A₆ (n=6)', color: '#4ADE80', h: 6 },
                        ] as { key: 'A1' | 'A2' | 'A3' | 'A4' | 'A6'; label: string; color: string; h: number }[])
                        .filter(({ h }) => selectedHarmonics.has(h))
                        .map(({ key, label, color }) => {
                          const vals = cosHarmonicScanDataB2.map(d => d[key]);
                          const maxVal = Math.max(...vals);
                          const maxPt = cosHarmonicScanDataB2[vals.indexOf(maxVal)];
                          return (
                            <div key={key} className="border border-[#E4E3E0]/10 p-4" style={{ borderLeft: `3px solid ${color}` }}>
                              <div className="flex justify-between items-center mb-3">
                                <span className="text-[10px] font-mono uppercase" style={{ color }}>{label}</span>
                                <span className="text-[9px] font-mono opacity-40 text-[#E4E3E0]">
                                  max {maxVal.toExponential(3)} @ {maxPt?.b} T ({maxPt ? maxPt.b2.toFixed(1) : '—'} T²)
                                </span>
                              </div>
                              <div className="h-[280px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                  <LineChart data={cosHarmonicScanDataB2} margin={{ bottom: 20, left: 10, right: 10 }}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#222" />
                                    <XAxis
                                      dataKey="b2"
                                      type="number"
                                      domain={['dataMin', 'dataMax']}
                                      tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                      tickFormatter={(v: number) => v.toFixed(0)}
                                      label={{ value: 'H² (T²)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }}
                                    />
                                    <YAxis
                                      tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                      label={{ value: label, angle: -90, position: 'insideLeft', fill: color, fontSize: 9 }}
                                      tickFormatter={(v: number) => v.toExponential(1)}
                                      width={60}
                                    />
                                    <Tooltip
                                      contentStyle={{ backgroundColor: '#141414', border: `1px solid ${color}44`, fontSize: '9px', color: '#E4E3E0' }}
                                      formatter={(val: any) => [typeof val === 'number' ? val.toExponential(4) : val, label]}
                                      labelFormatter={(b2: number) => `H² = ${b2.toFixed(2)} T²  (H = ${Math.sqrt(b2).toFixed(2)} T)`}
                                    />
                                    <Line
                                      dataKey={key}
                                      stroke={color}
                                      strokeWidth={1.5}
                                      dot={false}
                                      isAnimationActive={false}
                                    />
                                  </LineChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-32 px-8 text-[10px] font-mono text-center leading-relaxed text-[#E4E3E0] opacity-50">
                        {harmonicScanDiag ?? 'Computing…'}
                      </div>
                    )}
                  </div>

              </div>
            </div>
          ) : activeTab === 'torque' ? (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">

              {/* ── Control Panel ── */}
              <div className="bg-[#141414] text-[#E4E3E0] border border-[#E4E3E0]/10">
                <div className="flex items-center justify-between px-6 py-4">
                  <div className="flex items-center gap-3">
                    <Gauge size={18} className="text-[#F27D26]" />
                    <h3 className="text-lg font-serif italic font-bold">Torque Analysis</h3>
                    <span className="text-[9px] font-mono opacity-30 ml-2">{files.length} file{files.length !== 1 ? 's' : ''} · {uniqueBlms.length} angles</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-[9px] font-mono uppercase opacity-40">Column</span>
                    <select
                      value={selectedRawColumn}
                      onChange={(e) => setSelectedRawColumn(e.target.value)}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[10px] px-3 py-1.5 focus:outline-none focus:border-[#F27D26] cursor-pointer"
                    >
                      <option value="" className="bg-[#141414]">— select column —</option>
                      <optgroup label="Prefixes" className="bg-[#141414]">
                        {prefixes.map(p => <option key={p} value={p}>{p} (all)</option>)}
                      </optgroup>
                      <optgroup label="Columns" className="bg-[#141414]">
                        {files[selectedFileIndex]?.columns.map(c => <option key={c} value={c}>{c}</option>)}
                      </optgroup>
                    </select>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-6">

                {/* ── 1. Field-Sweep Decomposition ── */}
                <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Field-Sweep Decomposition</h4>
                      <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">τ(H) = a·H + b·H² + c·H³</p>
                    </div>
                    <div className="flex items-center gap-3 flex-wrap justify-end">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[8px] font-mono opacity-40">H_sf limit</span>
                        <input type="number" step="0.5" value={tauHsfLimit}
                          onChange={e => setTauHsfLimit(parseFloat(e.target.value) || 10)}
                          className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-14 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                        <span className="text-[8px] font-mono opacity-30">T</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[8px] font-mono opacity-40">Threshold</span>
                        <input type="number" step="0.01" value={tauResidThreshold}
                          onChange={e => setTauResidThreshold(parseFloat(e.target.value) || 0.05)}
                          className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-14 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                      </div>
                      <div className="flex items-center gap-1.5 border-l border-[#E4E3E0]/10 pl-3">
                        <span className="text-[8px] font-mono opacity-40">Plot</span>
                        {(['fast', 'full'] as const).map(mode => (
                          <button
                            key={mode}
                            onClick={() => setSweepDensity(mode)}
                            title={mode === 'fast'
                              ? `Draw ~${SWEEP_PLOT_MAX} points per angle (min/max per bucket — same envelope). Exports stay full resolution.`
                              : 'Draw every raw point. Accurate but slow when many angles are loaded.'}
                            className={`px-1.5 py-0.5 font-mono text-[8px] uppercase border transition-all ${sweepDensity === mode ? 'border-[#F27D26] text-[#F27D26] bg-[#F27D26]/10' : 'border-[#E4E3E0]/15 opacity-40 hover:opacity-70'}`}
                          >{mode}</button>
                        ))}
                        <span className="text-[8px] font-mono opacity-30 ml-0.5">
                          {fieldSweepPlot.drawn.toLocaleString()}
                          {fieldSweepPlot.drawn !== fieldSweepPlot.total && ` / ${fieldSweepPlot.total.toLocaleString()}`} pts
                        </span>
                      </div>
                    </div>
                  </div>

                  {fieldSweepByAngle.length > 0 ? (<>
                    {/* τ/H vs H */}
                    <p className="text-[9px] font-mono opacity-40 mb-1">τ/H vs H — intercept → a (spontaneous), slope → b (χ anisotropy)</p>
                    <div className="h-[320px] w-full mb-6">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart margin={{ bottom: 20, left: 10, right: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#222" />
                          <XAxis dataKey="h" type="number" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            label={{ value: 'H (T)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }} />
                          <YAxis dataKey="v" type="number" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            label={{ value: 'τ/H', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 9 }}
                            tickFormatter={(v: number) => v.toExponential(1)} width={60} />
                          <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '9px', color: '#E4E3E0' }}
                            formatter={(v: any) => [typeof v === 'number' ? v.toExponential(4) : v]} />
                          <Legend wrapperStyle={{ paddingTop: '16px', fontSize: '9px', fontFamily: 'monospace' }} />
                          {/* One <Line> (a single SVG path) per angle instead of a
                              Scatter whose per-point shape emitted one <circle> DOM
                              node per data point — that made this chart alone render
                              tens of thousands of nodes and freeze the tab. */}
                          {fieldSweepPlot.series.map(({ blm, pts, color }) => (
                            <Line key={blm} name={`${blm}°`} data={pts} dataKey="v"
                              stroke={color} strokeWidth={1} dot={false}
                              isAnimationActive={false} />
                          ))}
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>

                    {/* 2×2 parameter grid */}
                    <div className="grid grid-cols-2 gap-4">
                      {/* a(θ) */}
                      <div className="border border-[#E4E3E0]/10 p-3">
                        <p className="text-[9px] font-mono opacity-40 mb-2">a(θ) — spontaneous moment term</p>
                        <div className="h-[180px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ bottom: 16, left: 8 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                              <XAxis dataKey="blm" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                label={{ value: 'θ (°)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 8 }} />
                              <YAxis dataKey="a" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                tickFormatter={(v: number) => v.toExponential(1)} width={55} />
                              <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '8px', color: '#E4E3E0' }} />
                              <Scatter data={fieldSweepByAngle.filter(e => e.lowFit).map(e => ({ blm: e.blm, a: e.lowFit!.a }))}
                                fill="#F27D26" shape="circle" isAnimationActive={false} />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                      {/* b(θ) */}
                      <div className="border border-[#E4E3E0]/10 p-3">
                        <p className="text-[9px] font-mono opacity-40 mb-2">b(θ) — susceptibility anisotropy</p>
                        <div className="h-[180px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ bottom: 16, left: 8 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                              <XAxis dataKey="blm" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                label={{ value: 'θ (°)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 8 }} />
                              <YAxis dataKey="b" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                tickFormatter={(v: number) => v.toExponential(1)} width={55} />
                              <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '8px', color: '#E4E3E0' }} />
                              <Scatter data={fieldSweepByAngle.filter(e => e.lowFit).map(e => ({ blm: e.blm, b: e.lowFit!.b }))}
                                fill="#60A5FA" shape="circle" isAnimationActive={false} />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                      {/* H_sf(θ) */}
                      <div className="border border-[#E4E3E0]/10 p-3">
                        <p className="text-[9px] font-mono opacity-40 mb-2">H_sf(θ) — spin-flop field</p>
                        <div className="h-[180px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ bottom: 16, left: 8 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                              <XAxis dataKey="blm" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                label={{ value: 'θ (°)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 8 }} />
                              <YAxis dataKey="hSf" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                label={{ value: 'H_sf (T)', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 8 }} width={45} />
                              <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '8px', color: '#E4E3E0' }} />
                              <Scatter data={fieldSweepByAngle.map(e => ({ blm: e.blm, hSf: e.hSf }))}
                                fill="#4ADE80" shape="diamond" isAnimationActive={false} />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                      {/* τ_sat(θ) */}
                      <div className="border border-[#E4E3E0]/10 p-3">
                        <p className="text-[9px] font-mono opacity-40 mb-2">τ_sat(θ) — high-field saturation</p>
                        <div className="h-[180px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ bottom: 16, left: 8 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                              <XAxis dataKey="blm" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                label={{ value: 'θ (°)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 8 }} />
                              <YAxis dataKey="tauSat" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                tickFormatter={(v: number) => v.toExponential(1)} width={55} />
                              <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '8px', color: '#E4E3E0' }} />
                              <Scatter data={fieldSweepByAngle.filter(e => e.tauSat !== null).map(e => ({ blm: e.blm, tauSat: e.tauSat }))}
                                fill="#F472B6" shape="circle" isAnimationActive={false} />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>
                  </>) : (
                    <div className="flex items-center justify-center h-32 text-[10px] font-mono opacity-30">Load data and select a column</div>
                  )}
                </div>

                {/* ── 2. Angle-Sweep Fourier Decomposition ── */}
                <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Angle-Sweep Fourier Decomposition</h4>
                      <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">τ(α) = Σ Aₙ sin(nα + φₙ), n = 1..{tauFourierNMax}</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[8px] font-mono opacity-40">n_max</span>
                        <input type="number" step="1" min="1" max="12" value={tauFourierNMax}
                          onChange={e => setTauFourierNMax(Math.max(1, Math.min(12, parseInt(e.target.value) || 6)))}
                          className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-10 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[8px] font-mono opacity-40">V (mm³)</span>
                        <input type="number" step="0.1" value={tauSampleVol}
                          onChange={e => setTauSampleVol(parseFloat(e.target.value) || 1)}
                          className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-14 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                      </div>
                    </div>
                  </div>

                  {fourierScanData.length > 0 ? (() => {
                    const a6Sat = (() => {
                      if (fourierScanData.length === 0) return null;
                      const n = Math.max(1, Math.floor(fourierScanData.length * 0.1));
                      const last = [...fourierScanData].sort((a, b) => b.h - a.h).slice(0, n);
                      const vals = last.map(d => d.coeffs.find(c => c.n === 6)?.A ?? 0);
                      return vals.reduce((s, v) => s + v, 0) / vals.length;
                    })();
                    const k6Est = a6Sat !== null ? a6Sat / (6 * tauSampleVol) : null;
                    const a2OverA6 = fourierScanData.map(d => {
                      const a2 = d.coeffs.find(c => c.n === 2)?.A ?? 0;
                      const a6 = d.coeffs.find(c => c.n === 6)?.A ?? 0;
                      return { h: d.h, ratio: a6 > 0 ? a2/a6 : null };
                    }).filter(d => d.ratio !== null);

                    return (
                      <div className="space-y-4">
                        {/* Amplitude chart */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                          <div className="border border-[#E4E3E0]/10 p-3">
                            <p className="text-[9px] font-mono opacity-40 mb-2">Amplitude Aₙ vs H</p>
                            <div className="h-[240px]">
                              <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart margin={{ bottom: 16, left: 8 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                                  <XAxis dataKey="h" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    label={{ value: 'H (T)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 8 }} />
                                  <YAxis dataKey="A" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    tickFormatter={(v: number) => v.toExponential(1)} width={55} />
                                  <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '8px', color: '#E4E3E0' }} />
                                  <Legend wrapperStyle={{ fontSize: '8px', fontFamily: 'monospace' }} />
                                  {([2,4,6] as const).map((n, ni) => {
                                    const color = ['#60A5FA','#FBBF24','#4ADE80'][ni];
                                    return (
                                      <Line key={n} name={`A${n}`}
                                        data={fourierScanData.map(d => ({ h: d.h, A: d.coeffs.find(c => c.n === n)?.A ?? 0 }))}
                                        dataKey="A" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                                    );
                                  })}
                                </ComposedChart>
                              </ResponsiveContainer>
                            </div>
                          </div>

                          {/* Phase chart */}
                          <div className="border border-[#E4E3E0]/10 p-3">
                            <p className="text-[9px] font-mono opacity-40 mb-2">Phase φₙ (°) vs H</p>
                            <div className="h-[240px]">
                              <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart margin={{ bottom: 16, left: 8 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                                  <XAxis dataKey="h" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    label={{ value: 'H (T)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 8 }} />
                                  <YAxis dataKey="phi" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    label={{ value: 'φ (°)', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 8 }} width={40} />
                                  <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '8px', color: '#E4E3E0' }} />
                                  <Legend wrapperStyle={{ fontSize: '8px', fontFamily: 'monospace' }} />
                                  {([2,4,6] as const).map((n, ni) => {
                                    const color = ['#60A5FA','#FBBF24','#4ADE80'][ni];
                                    return (
                                      <Line key={n} name={`φ${n}`}
                                        data={fourierScanData.map(d => ({ h: d.h, phi: d.coeffs.find(c => c.n === n)?.phiDeg ?? 0 }))}
                                        dataKey="phi" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                                    );
                                  })}
                                </ComposedChart>
                              </ResponsiveContainer>
                            </div>
                          </div>

                          {/* A₂/A₆ ratio + K₆ */}
                          <div className="border border-[#E4E3E0]/10 p-3">
                            <div className="flex justify-between items-start mb-2">
                              <p className="text-[9px] font-mono opacity-40">A₂/A₆ ratio vs H</p>
                              {k6Est !== null && (
                                <div className="text-right">
                                  <span className="text-[8px] font-mono opacity-40 block">K₆ ≈</span>
                                  <span className="text-[10px] font-mono text-[#4ADE80]">{k6Est.toExponential(3)}</span>
                                  <span className="text-[7px] font-mono opacity-30 ml-1">τ·mm⁻³</span>
                                </div>
                              )}
                            </div>
                            <div className="h-[240px]">
                              <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart margin={{ bottom: 16, left: 8 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                                  <XAxis dataKey="h" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    label={{ value: 'H (T)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 8 }} />
                                  <YAxis dataKey="ratio" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    label={{ value: 'A₂/A₆', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 8 }} width={40} />
                                  <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '8px', color: '#E4E3E0' }} />
                                  <Line name="A₂/A₆" data={a2OverA6} dataKey="ratio"
                                    stroke="#C084FC" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                                </ComposedChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })() : (
                    <div className="flex items-center justify-center h-32 text-[10px] font-mono opacity-30">Load data with ≥{2*tauFourierNMax} BLM angles</div>
                  )}
                </div>

                {/* ── 3. Scaling Verification ── */}
                <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Scaling Verification</h4>
                      <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">τ/H² vs α (collapse test) and τ vs α</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[8px] font-mono opacity-40">H values (T)</span>
                      <input value={tauScaleFieldsStr}
                        onChange={e => setTauScaleFieldsStr(e.target.value)}
                        placeholder="2,5,10,15,20"
                        className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-36 px-2 py-0.5 focus:outline-none focus:border-[#F27D26]" />
                    </div>
                  </div>

                  {scalingVerifData.length > 0 ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* τ/H² vs α */}
                      <div className="border border-[#E4E3E0]/10 p-3">
                        <p className="text-[9px] font-mono opacity-40 mb-2">τ/H² vs α — collapses at low field if τ ∝ H²</p>
                        <div className="h-[300px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart margin={{ bottom: 20, left: 10 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                              <XAxis dataKey="alpha" type="number" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                label={{ value: 'α (°)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }} />
                              <YAxis dataKey="tauOverH2" type="number" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                tickFormatter={(v: number) => v.toExponential(1)} width={60} />
                              <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '9px', color: '#E4E3E0' }} />
                              <Legend wrapperStyle={{ paddingTop: '12px', fontSize: '9px', fontFamily: 'monospace' }} />
                              {scalingVerifData.map(({ h, pts }, i) => {
                                const color = `hsl(${(i * 55) % 360}, 65%, 58%)`;
                                return (
                                  <Scatter key={h} name={`${h} T`} data={pts} fill={color}
                                    line={{ stroke: color, strokeWidth: 1.5 }} lineType="joint"
                                    shape={(p: any) => <circle cx={p.cx} cy={p.cy} r={3} fill={color} />}
                                    isAnimationActive={false} />
                                );
                              })}
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                      {/* τ vs α */}
                      <div className="border border-[#E4E3E0]/10 p-3">
                        <p className="text-[9px] font-mono opacity-40 mb-2">τ vs α (unnormalized)</p>
                        <div className="h-[300px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart margin={{ bottom: 20, left: 10 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                              <XAxis dataKey="alpha" type="number" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                label={{ value: 'α (°)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }} />
                              <YAxis dataKey="tau" type="number" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                tickFormatter={(v: number) => v.toExponential(1)} width={60} />
                              <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '9px', color: '#E4E3E0' }} />
                              <Legend wrapperStyle={{ paddingTop: '12px', fontSize: '9px', fontFamily: 'monospace' }} />
                              {scalingVerifData.map(({ h, pts }, i) => {
                                const color = `hsl(${(i * 55) % 360}, 65%, 58%)`;
                                return (
                                  <Scatter key={h} name={`${h} T`} data={pts} fill={color}
                                    line={{ stroke: color, strokeWidth: 1.5 }} lineType="joint"
                                    shape={(p: any) => <circle cx={p.cx} cy={p.cy} r={3} fill={color} />}
                                    isAnimationActive={false} />
                                );
                              })}
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-32 text-[10px] font-mono opacity-30">Load data and enter H values above</div>
                  )}
                </div>

              </div>
            </div>
          ) : activeTab === 'landau' ? (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">

              {/* ── Control Panel ── */}
              <div className="bg-[#141414] text-[#E4E3E0] border border-[#E4E3E0]/10">
                <div className="flex items-center justify-between px-6 py-4 border-b border-[#E4E3E0]/10">
                  <div className="flex items-center gap-3">
                    <GitBranch size={18} className="text-[#F27D26]" />
                    <h3 className="text-lg font-serif italic font-bold">Landau Fan &amp; Berry Phase</h3>
                    <span className="text-[9px] font-mono opacity-30 ml-2">n = F/B − ½ + β + δ</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-[9px] font-mono uppercase opacity-40">Series</span>
                    <select
                      value={landauSelected?.key ?? ''}
                      onChange={(e) => setLandauResultKey(e.target.value)}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[10px] px-3 py-1.5 focus:outline-none focus:border-[#F27D26] cursor-pointer max-w-[280px]"
                    >
                      {landauSeriesList.map(s => (
                        <option key={s.key} value={s.key} className="bg-[#141414]">{s.blm}° · {s.col}</option>
                      ))}
                    </select>
                  </div>
                </div>
                {/* Controls */}
                <div className="flex items-center gap-4 px-6 py-3 flex-wrap">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[8px] font-mono uppercase opacity-40">Integer n at</span>
                    {(['max', 'min'] as const).map(t => (
                      <button key={t} onClick={() => setLandauIntegerAt(t)}
                        className={`px-1.5 py-0.5 font-mono text-[8px] uppercase border transition-all ${landauIntegerAt === t ? 'border-[#F27D26] text-[#F27D26] bg-[#F27D26]/10' : 'border-[#E4E3E0]/15 opacity-40 hover:opacity-70'}`}
                      >{t === 'max' ? 'ΔR max' : 'ΔR min'}</button>
                    ))}
                  </div>
                  <button onClick={() => setLandauUseBoth(v => !v)}
                    className={`px-2 py-0.5 font-mono text-[8px] uppercase border transition-all ${landauUseBoth ? 'border-[#4ADE80]/60 text-[#4ADE80] bg-[#4ADE80]/10' : 'border-[#E4E3E0]/15 opacity-40 hover:opacity-70'}`}
                    title="Use both maxima and minima (half-integer indices between integers)"
                  >max+min</button>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[8px] font-mono uppercase opacity-40">n₀</span>
                    <input type="number" step="1" value={landauN0}
                      onChange={e => setLandauN0(parseInt(e.target.value) || 0)}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-12 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[8px] font-mono uppercase opacity-40">Smooth</span>
                    <input type="number" step="2" min="1" value={landauSmoothWin}
                      onChange={e => setLandauSmoothWin(Math.max(1, parseInt(e.target.value) || 1))}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-12 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[8px] font-mono uppercase opacity-40">Prominence</span>
                    <input type="number" step="0.01" min="0" value={landauProm}
                      onChange={e => setLandauProm(Math.max(0, parseFloat(e.target.value) || 0))}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-14 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[8px] font-mono uppercase opacity-40">δ</span>
                    <select value={landauDelta} onChange={e => setLandauDelta(parseFloat(e.target.value))}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] px-1.5 py-0.5 focus:outline-none focus:border-[#F27D26] cursor-pointer">
                      <option value={0}>0 (2D)</option>
                      <option value={0.125}>+1/8 (3D)</option>
                      <option value={-0.125}>−1/8 (3D)</option>
                    </select>
                  </div>
                </div>
                {/* Band-pass row: FFT → keep one peak → IFFT → index */}
                <div className="flex items-center gap-4 px-6 py-3 border-t border-[#E4E3E0]/10 flex-wrap">
                  <button onClick={() => setLandauBandpass(v => !v)}
                    className={`px-2 py-0.5 font-mono text-[8px] uppercase border transition-all ${landauBandpass ? 'border-[#4ADE80]/60 text-[#4ADE80] bg-[#4ADE80]/10' : 'border-[#E4E3E0]/15 opacity-40 hover:opacity-70'}`}
                    title="FFT the oscillation, keep only the selected frequency band, inverse-FFT, then index the single component"
                  >band-pass (FFT→IFFT)</button>
                  <div className={`flex items-center gap-1.5 ${landauBandpass ? '' : 'opacity-30 pointer-events-none'}`}>
                    <span className="text-[8px] font-mono uppercase opacity-40">Band</span>
                    <input type="number" step="10" value={landauBandLo || ''} placeholder="lo"
                      onChange={e => setLandauBandLo(parseFloat(e.target.value) || 0)}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-16 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                    <span className="text-[8px] font-mono opacity-30">–</span>
                    <input type="number" step="10" value={landauBandHi || ''} placeholder="hi"
                      onChange={e => setLandauBandHi(parseFloat(e.target.value) || 0)}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-16 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                    <span className="text-[8px] font-mono opacity-30">T</span>
                    <button onClick={() => { setLandauBandLo(0); setLandauBandHi(0); }}
                      className={`px-1.5 py-0.5 font-mono text-[8px] uppercase border transition-all ${landauBand === null ? 'border-[#F27D26] text-[#F27D26] bg-[#F27D26]/10' : 'border-[#E4E3E0]/15 opacity-40 hover:opacity-70'}`}
                      title="Auto: center the band on the dominant peak">Auto</button>
                  </div>
                  <div className={`flex items-center gap-1.5 ${landauBandpass ? '' : 'opacity-30 pointer-events-none'}`}>
                    <span className="text-[8px] font-mono uppercase opacity-40">Edge</span>
                    <input type="number" step="0.05" min="0" max="1" value={landauEdge}
                      onChange={e => setLandauEdge(Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-14 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                  </div>
                </div>
              </div>

              {landauSeriesList.length === 0 ? (
                <div className="bg-[#141414] text-[#E4E3E0] border border-[#E4E3E0]/10 flex items-center justify-center h-40 text-[11px] font-mono opacity-50 text-center px-8">
                  No processed ΔR(1/B) yet. Go to the Dashboard, pick a column prefix, and click “Process All” — the Landau fan reads the background-subtracted oscillations from there.
                </div>
              ) : (() => {
                const fan = landauFan;
                const hasFit = !!fan && isFinite(fan.slope) && fan.points.length >= 2;
                const maxInvB = landauSelected && landauSelected.series.length
                  ? landauSelected.series[landauSelected.series.length - 1].invB : 0;
                const beta = fan ? fan.intercept + 0.5 - landauDelta : NaN;
                const berryOverPi = 2 * beta;
                const berryMod = ((berryOverPi % 2) + 2) % 2; // into [0,2)
                const fitLine = hasFit ? [{ invB: 0, n: fan!.intercept }, { invB: maxInvB, n: fan!.slope * maxInvB + fan!.intercept }] : [];
                const intPts = fan ? fan.points.filter(p => Number.isInteger(p.n)) : [];
                const halfPts = fan ? fan.points.filter(p => !Number.isInteger(p.n)) : [];
                const band = landauFiltered?.band;
                const specMax = landauFiltered ? Math.max((band?.hi ?? 0) * 1.6, landauFiltered.peakF * 3.5, 100) : 0;
                const specView = landauFiltered ? landauFiltered.spectrum.filter(p => p.frequency <= specMax) : [];
                return (
                  <>
                    {/* Chart 0: FFT spectrum + band picker */}
                    {landauBandpass && specView.length > 0 && (
                      <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                        <div className="flex justify-between items-start mb-3">
                          <div>
                            <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">FFT Spectrum — keep one peak</h4>
                            <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">dominant F ≈ {landauFiltered!.peakF.toFixed(1)} T <span className="opacity-40 text-[#E4E3E0]">· band {band!.lo.toFixed(0)}–{band!.hi.toFixed(0)} T {landauBand === null ? '(auto)' : '(manual)'}</span></p>
                          </div>
                        </div>
                        <div className="h-[200px] w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={specView} margin={{ bottom: 16, left: 10, right: 10 }}>
                              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#222" />
                              <XAxis dataKey="frequency" type="number" domain={[0, specMax]}
                                tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                tickFormatter={(v: number) => v.toFixed(0)}
                                label={{ value: 'F (T)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 9 }} />
                              <YAxis tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                tickFormatter={(v: number) => v.toExponential(0)} width={50} />
                              <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '9px', color: '#E4E3E0' }}
                                formatter={(v: any) => [typeof v === 'number' ? v.toExponential(3) : v, 'Amp']}
                                labelFormatter={(v: number) => `F = ${Number(v).toFixed(1)} T`} />
                              {band && band.hi > band.lo && <ReferenceArea x1={band.lo} x2={band.hi} fill="#F27D26" fillOpacity={0.15} stroke="#F27D26" strokeOpacity={0.4} />}
                              <Area type="linear" dataKey="amplitude" stroke="#4ADE80" fill="#4ADE80" fillOpacity={0.15} strokeWidth={1} dot={false} isAnimationActive={false} />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    )}

                    {/* Chart 1: Landau level picking */}
                    <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Landau Levels — {landauFiltered?.filtered ? 'band-passed ΔR' : 'ΔR'} vs 1/B</h4>
                          <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">{landauSelected?.blm}° · {landauSelected?.col} <span className="opacity-40 text-[#E4E3E0]">· {fan?.points.length ?? 0} levels tagged{landauFiltered?.filtered ? ` · single component ${band!.lo.toFixed(0)}–${band!.hi.toFixed(0)} T` : ''}</span></p>
                        </div>
                        <div className="flex items-center gap-3 text-[9px] font-mono">
                          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#F27D26]" />integer n</span>
                          {landauUseBoth && <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#60A5FA]" />half-integer</span>}
                        </div>
                      </div>
                      <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart margin={{ bottom: 20, left: 10, right: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#222" />
                            <XAxis dataKey="invB" type="number" domain={['dataMin', 'dataMax']}
                              tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                              tickFormatter={(v: number) => v.toFixed(3)}
                              label={{ value: '1/B (T⁻¹)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }} />
                            <YAxis dataKey="val" type="number" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                              tickFormatter={(v: number) => v.toExponential(1)} width={60}
                              label={{ value: 'ΔR', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 9 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '9px', color: '#E4E3E0' }}
                              formatter={(v: any) => [typeof v === 'number' ? v.toExponential(4) : v]}
                              labelFormatter={(v: number) => `1/B = ${v.toFixed(4)} (B = ${(1 / v).toFixed(2)} T)`} />
                            <Line data={landauLine} dataKey="val" stroke="#E4E3E0" strokeOpacity={0.5} strokeWidth={1} dot={false} isAnimationActive={false} />
                            <Scatter data={intPts} dataKey="val" fill="#F27D26" isAnimationActive={false}
                              shape={(p: any) => <circle cx={p.cx} cy={p.cy} r={4} fill="#F27D26" />} />
                            {landauUseBoth && <Scatter data={halfPts} dataKey="val" fill="#60A5FA" isAnimationActive={false}
                              shape={(p: any) => <circle cx={p.cx} cy={p.cy} r={3} fill="#60A5FA" />} />}
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Chart 2: fan + results */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                      <div className="lg:col-span-2 bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                        <div className="flex justify-between items-start mb-3">
                          <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Landau Fan — n vs 1/B</h4>
                          <button
                            onClick={() => { if (!fan || fan.points.length === 0) return; downloadCSV('n,1/B(1/T),B(T),type,DeltaR\n' + fan.points.map(p => `${p.n},${p.invB},${p.b},${p.type},${p.val}`).join('\n'), `landau_fan_${landauSelected?.blm}deg_${landauSelected?.col}.csv`); }}
                            className="px-2 py-1 font-mono text-[9px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"><Download size={9} />CSV</button>
                        </div>
                        <div className="h-[320px] w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart margin={{ bottom: 20, left: 10, right: 10 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                              <XAxis dataKey="invB" type="number" domain={[0, 'dataMax']}
                                tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                tickFormatter={(v: number) => v.toFixed(3)}
                                label={{ value: '1/B (T⁻¹)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }} />
                              <YAxis dataKey="n" type="number" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                label={{ value: 'Landau index n', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 9 }} width={50} />
                              <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '9px', color: '#E4E3E0' }}
                                labelFormatter={(v: number) => `1/B = ${Number(v).toFixed(4)}`} />
                              {hasFit && <Line data={fitLine} dataKey="n" stroke="#4ADE80" strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} />}
                              <Scatter data={intPts} dataKey="n" fill="#F27D26" isAnimationActive={false}
                                shape={(p: any) => <circle cx={p.cx} cy={p.cy} r={4} fill="#F27D26" />} />
                              {landauUseBoth && <Scatter data={halfPts} dataKey="n" fill="#60A5FA" isAnimationActive={false}
                                shape={(p: any) => <circle cx={p.cx} cy={p.cy} r={3} fill="#60A5FA" />} />}
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      {/* Results panel */}
                      <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10 flex flex-col justify-center gap-4">
                        {hasFit ? (<>
                          <div className="border border-[#F27D26]/30 bg-[#F27D26]/5 p-4">
                            <p className="text-[8px] font-mono uppercase opacity-50 mb-1">Berry phase φ_B</p>
                            <p className="text-2xl font-mono text-[#F27D26]">{berryMod.toFixed(3)}<span className="text-sm opacity-60"> π</span></p>
                            <p className="text-[8px] font-mono opacity-40 mt-1">β = {beta.toFixed(3)} · intercept = {fan!.intercept.toFixed(3)}</p>
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <div className="border border-[#E4E3E0]/10 p-3">
                              <p className="text-[8px] font-mono uppercase opacity-50">F (slope)</p>
                              <p className="text-[13px] font-mono text-[#4ADE80]">{fan!.slope.toFixed(1)} T</p>
                            </div>
                            <div className="border border-[#E4E3E0]/10 p-3">
                              <p className="text-[8px] font-mono uppercase opacity-50">F (FFT)</p>
                              <p className="text-[13px] font-mono text-[#E4E3E0]">{landauSelected?.freqFFT != null ? `${landauSelected.freqFFT.toFixed(1)} T` : '—'}</p>
                            </div>
                            <div className="border border-[#E4E3E0]/10 p-3">
                              <p className="text-[8px] font-mono uppercase opacity-50">R²</p>
                              <p className="text-[13px] font-mono text-[#E4E3E0]">{fan!.r2.toFixed(5)}</p>
                            </div>
                            <div className="border border-[#E4E3E0]/10 p-3">
                              <p className="text-[8px] font-mono uppercase opacity-50">Levels</p>
                              <p className="text-[13px] font-mono text-[#E4E3E0]">{fan!.points.length}</p>
                            </div>
                          </div>
                          <p className="text-[8px] font-mono opacity-30 leading-relaxed">
                            φ_B = 2π(intercept + ½ − δ). Trivial → 0; topological → π. Flipping “integer at max/min” shifts φ_B by π — set it to match your convention (ρ minima vs σ maxima).
                          </p>
                        </>) : (
                          <div className="text-[10px] font-mono opacity-40 text-center py-8">
                            Not enough Landau levels detected. Lower the prominence, adjust smoothing, or widen the B-range (ROI) on the Dashboard.
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Multi-angle summary */}
                    <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Angle Dependence</h4>
                          <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">Berry phase &amp; frequency per angle <span className="opacity-40 text-[#E4E3E0]">· same detection settings · {landauByAngle.length} angles fit</span></p>
                        </div>
                        <button
                          onClick={() => { if (landauByAngle.length === 0) return; downloadCSV('BLM(deg),Column,F(T),intercept,BerryPhase/pi,R2,nLevels\n' + landauByAngle.map(d => `${d.blm},"${d.col}",${d.slope},${d.intercept},${(((d.berryOverPi % 2) + 2) % 2)},${d.r2},${d.nPts}`).join('\n'), 'landau_berryphase_vs_angle.csv'); }}
                          className="px-2 py-1 font-mono text-[9px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"><Download size={9} />CSV</button>
                      </div>
                      {landauByAngle.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="border border-[#E4E3E0]/10 p-3">
                            <p className="text-[9px] font-mono opacity-40 mb-2">Berry phase φ_B / π vs angle</p>
                            <div className="h-[220px]">
                              <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ bottom: 16, left: 8 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                                  <XAxis dataKey="blm" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    label={{ value: 'θ (°)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 8 }} />
                                  <YAxis dataKey="berry" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    label={{ value: 'φ_B / π', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 8 }} width={45} />
                                  <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '8px', color: '#E4E3E0' }} />
                                  <Scatter data={landauByAngle.map(d => ({ blm: d.blm, berry: ((d.berryOverPi % 2) + 2) % 2 }))}
                                    fill="#F27D26" shape="circle" isAnimationActive={false} />
                                </ScatterChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                          <div className="border border-[#E4E3E0]/10 p-3">
                            <p className="text-[9px] font-mono opacity-40 mb-2">Frequency F vs angle (fan slope)</p>
                            <div className="h-[220px]">
                              <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ bottom: 16, left: 8 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                                  <XAxis dataKey="blm" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    label={{ value: 'θ (°)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 8 }} />
                                  <YAxis dataKey="slope" type="number" tick={{ fontSize: 8, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    tickFormatter={(v: number) => v.toFixed(0)} width={50}
                                    label={{ value: 'F (T)', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 8 }} />
                                  <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '8px', color: '#E4E3E0' }} />
                                  <Scatter data={landauByAngle.map(d => ({ blm: d.blm, slope: d.slope }))}
                                    fill="#4ADE80" shape="diamond" isAnimationActive={false} />
                                </ScatterChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-center justify-center h-24 text-[10px] font-mono opacity-30">No angles produced a valid fan with current settings.</div>
                      )}
                    </div>
                  </>
                );
              })()}
            </div>
          ) : activeTab === 'symmetry' ? (
            <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-3">
                    <RefreshCw size={24} className="text-[#F27D26]" />
                    <h3 className="text-2xl font-serif italic font-bold">Symmetry Analysis</h3>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex flex-col">
                      <label className="text-[10px] font-mono uppercase opacity-60 mb-1">Symmetry Center (°)</label>
                      <input 
                        type="number" 
                        step="0.1"
                        value={symmetryCenter} 
                        onChange={(e) => setSymmetryCenter(parseFloat(e.target.value) || 0)}
                        className="bg-transparent border-b border-[#E4E3E0]/20 text-[#E4E3E0] font-mono text-xs focus:outline-none focus:border-[#F27D26] w-24"
                      />
                    </div>
                    <div className="flex flex-col">
                      <label className="text-[10px] font-mono uppercase opacity-60 mb-1">Min Freq (T)</label>
                      <input 
                        type="number" 
                        value={symmetryMinFreq} 
                        onChange={(e) => setSymmetryMinFreq(parseFloat(e.target.value) || 0)}
                        className="bg-transparent border-b border-[#E4E3E0]/20 text-[#E4E3E0] font-mono text-xs focus:outline-none focus:border-[#F27D26] w-20"
                      />
                    </div>
                    <div className="flex flex-col">
                      <label className="text-[10px] font-mono uppercase opacity-60 mb-1">Max Freq (T)</label>
                      <input 
                        type="number" 
                        value={symmetryMaxFreq} 
                        onChange={(e) => setSymmetryMaxFreq(parseFloat(e.target.value) || 0)}
                        className="bg-transparent border-b border-[#E4E3E0]/20 text-[#E4E3E0] font-mono text-xs focus:outline-none focus:border-[#F27D26] w-20"
                      />
                    </div>
                    <div className="flex gap-2">
                      <button onClick={() => setSymmetryCenter(0)} className="px-2 py-1 font-mono text-[10px] border border-white/20 hover:bg-white/10">0°</button>
                      <button onClick={() => setSymmetryCenter(90)} className="px-2 py-1 font-mono text-[10px] border border-white/20 hover:bg-white/10">90°</button>
                      <button onClick={() => setSymmetryCenter(180)} className="px-2 py-1 font-mono text-[10px] border border-white/20 hover:bg-white/10">180°</button>
                      <button 
                        onClick={() => {
                          setSymmetryMinFreq(0);
                          setSymmetryMaxFreq(globalSuggestedMaxFreq);
                        }}
                        className="px-2 py-1 font-mono text-[10px] border border-[#F27D26]/40 text-[#F27D26] hover:bg-[#F27D26]/10"
                      >
                        Auto Range
                      </button>
                      <button 
                        onClick={() => {
                          let csv = "Branch,Original Angle (deg),Original Freq (T),Original Amp,Mirrored Angle (deg),Mirrored Freq (T),Mirrored Amp\n";
                          branchesWithFits.forEach((branch) => {
                            const filtered = branch.peaks.filter(p => p.freq >= symmetryMinFreq && p.freq <= symmetryMaxFreq);
                            filtered.forEach(p => {
                              const mirroredBlm = 2 * symmetryCenter - p.blm;
                              csv += `${branch.name},${p.blm},${p.freq},${p.amp},${mirroredBlm.toFixed(2)},${p.freq},${p.amp}\n`;
                            });
                          });
                          const blob = new Blob([csv], { type: 'text/csv' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `symmetry_analysis_center_${symmetryCenter}.csv`;
                          a.click();
                        }}
                        className="ml-4 bg-[#F27D26] text-white px-3 py-1 font-mono text-[10px] uppercase tracking-widest hover:bg-[#F27D26]/80 flex items-center gap-2"
                      >
                        <Download size={12} />
                        Export Symmetry CSV
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
                <div className="xl:col-span-8 bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                  <h4 className="text-xs font-mono uppercase opacity-60 mb-6">Mirrored Branch Visualization (Center: {symmetryCenter}°)</h4>
                  <div className="h-[600px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                        <XAxis 
                          dataKey="blm" 
                          type="number" 
                          domain={['auto', 'auto']} 
                          tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                          label={{ value: 'BLM', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 10 }}
                        />
                        <YAxis 
                          dataKey="freq" 
                          type="number" 
                          domain={[symmetryMinFreq, symmetryMaxFreq]}
                          tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                          label={{ value: 'Frequency (T)', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 10 }}
                        />
                        <Tooltip 
                          cursor={{ strokeDasharray: '3 3' }}
                          contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '10px', color: '#E4E3E0' }}
                        />
                        <Legend wrapperStyle={{ paddingTop: '20px', fontSize: '10px', fontFamily: 'monospace' }} />
                        
                        {/* Reference line for symmetry center */}
                        <ReferenceLine x={symmetryCenter} stroke="#F27D26" strokeDasharray="5 5" label={{ value: 'Center', position: 'top', fill: '#F27D26', fontSize: 10 }} />

                        {branchesWithFits.map((branch, idx) => {
                          const color = `hsl(${(idx * 137) % 360}, 70%, 60%)`;
                          const mirroredData = branch.peaks.map(p => ({
                            ...p,
                            blm: 2 * symmetryCenter - p.blm,
                            isMirrored: true
                          }));
                          
                          const filteredOriginal = branch.peaks.filter(p => p.freq >= symmetryMinFreq && p.freq <= symmetryMaxFreq);
                          const filteredMirrored = mirroredData.filter(p => p.freq >= symmetryMinFreq && p.freq <= symmetryMaxFreq);

                          return (
                            <React.Fragment key={branch.id}>
                              <Scatter 
                                name={`${branch.name} (Original)`} 
                                data={filteredOriginal} 
                                fill={color} 
                                shape="circle" 
                                isAnimationActive={false}
                              />
                              <Scatter 
                                name={`${branch.name} (Mirrored)`} 
                                data={filteredMirrored} 
                                fill={color} 
                                shape="triangle" 
                                opacity={0.4}
                                isAnimationActive={false}
                              />
                            </React.Fragment>
                          );
                        })}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="text-[10px] font-mono mt-4 opacity-40 italic">
                    Triangles represent data points mirrored around {symmetryCenter}°. 
                    Overlap between circles and triangles indicates symmetry.
                  </p>
                </div>

                <div className="xl:col-span-4 bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10 flex flex-col">
                  <h4 className="text-xs font-mono uppercase opacity-60 mb-6">Symmetry Fit Summary</h4>
                  <div className="flex-1 overflow-auto">
                    <table className="w-full text-[10px] font-mono">
                      <thead>
                        <tr className="border-b border-white/10 text-left opacity-60">
                          <th className="pb-2 font-normal">Branch</th>
                          <th className="pb-2 font-normal">Opt. Center</th>
                          <th className="pb-2 font-normal">R²</th>
                          <th className="pb-2 font-normal text-right">F₀ (T)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {branchesWithFits.filter(b => selectedBranchIds.has(b.id)).map((branch) => (
                          <tr key={branch.id} className="border-b border-white/5 hover:bg-white/5">
                            <td className="py-2 flex items-center gap-2">
                              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: `hsl(${(branches.findIndex(br => br.id === branch.id) * 137) % 360}, 70%, 60%)` }} />
                              {branch.name}
                            </td>
                            <td className="py-2">{branch.fit?.cosOffset ?? '-'}°</td>
                            <td className="py-2">{branch.fit?.r2.toFixed(4) ?? '-'}</td>
                            <td className="py-2 text-right">{branch.fit?.f0.toFixed(1) ?? '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-8 pt-6 border-t border-white/10">
                    <div className="bg-white/5 p-4 border border-white/10">
                      <h5 className="text-[10px] font-mono uppercase opacity-60 mb-2">Analysis Note</h5>
                      <p className="text-[10px] leading-relaxed opacity-80 italic">
                        "Opt. Center" is the angle that maximizes the 1/cos(θ-φ) fit for each branch within the current frequency range. 
                        If this value deviates significantly from the global center ({symmetryCenter}°), it may indicate local misalignment or complex band structure.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : activeTab === 'branch' ? (
            <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-3">
                    <GitBranch size={24} className="text-[#F27D26]" />
                    <h3 className="text-2xl font-serif italic font-bold">Branch Analysis</h3>
                    <div className="flex items-center gap-3 ml-4 pl-4 border-l border-[#E4E3E0]/15">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[8px] font-mono uppercase opacity-40">Freq Tol (T)</span>
                        <input type="number" step="0.1" min="0.01" value={branchTolerance}
                          onChange={(e) => setBranchTolerance(parseFloat(e.target.value) || 0.5)}
                          className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[10px] w-16 px-2 py-1 text-center focus:outline-none focus:border-[#F27D26]" />
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[8px] font-mono uppercase opacity-40">Min Peaks</span>
                        <input type="number" min="2" value={minPeaksPerBranch}
                          onChange={(e) => setMinPeaksPerBranch(parseInt(e.target.value) || 3)}
                          className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[10px] w-14 px-2 py-1 text-center focus:outline-none focus:border-[#F27D26]" />
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-6">
                    <button
                      onClick={() => {
                        let csv = "";
                        branchesWithFits.forEach((b, i) => {
                          if (i > 0) csv += ",";
                          csv += `${b.name} Angle (deg),${b.name} Freq (T),${b.name} Amp`;
                        });
                        csv += ",Others Angle (deg),Others Freq (T),Others Amp\n";

                        branchComparisonTable.forEach(row => {
                          branchesWithFits.forEach((b, i) => {
                            if (i > 0) csv += ",";
                            if (row[b.id]) {
                              csv += `${row.blm},${row[b.id].freq},${row[b.id].amp}`;
                            } else {
                              csv += ",,";
                            }
                          });
                          // Others
                          csv += ",";
                          if (row.others) {
                            csv += `${row.blm},${row.others.freq},${row.others.amp}`;
                          } else {
                            csv += ",,";
                          }
                          csv += "\n";
                        });

                        const blob = new Blob([csv], { type: 'text/csv' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `branch_analysis_wide.csv`;
                        a.click();
                      }}
                      className="bg-white/10 text-white px-4 py-2 font-mono text-[10px] uppercase tracking-widest hover:bg-white/20 border border-white/10"
                    >
                      <Download size={14} className="mr-2 inline" />
                      Export Wide CSV
                    </button>
                    <button 
                      onClick={() => {
                        let csv = "Branch,BLM Angle,Frequency (T),Amplitude,FileName,F0,R2\n";
                        branchesWithFits.forEach((branch, idx) => {
                          branch.peaks.forEach(p => {
                            csv += `Branch ${idx + 1},${p.blm},${p.freq},${p.amp},${p.fileName},${branch.fit?.f0 || ''},${branch.fit?.r2 || ''}\n`;
                          });
                        });
                        others.forEach(p => {
                          csv += `Others,${p.blm},${p.freq},${p.amp},${p.fileName},,\n`;
                        });
                        const blob = new Blob([csv], { type: 'text/csv' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `branch_analysis_combined.csv`;
                        a.click();
                      }}
                      className="bg-[#F27D26] text-white px-4 py-2 font-mono text-[10px] uppercase tracking-widest hover:bg-opacity-80"
                    >
                      <Download size={14} className="mr-2 inline" />
                      Export Combined CSV
                    </button>
                  </div>
                </div>
              </div>

              {branches.length === 0 && others.length === 0 ? (
                <div className="h-[500px] border-2 border-dashed border-[#141414]/20 flex flex-col items-center justify-center text-[#141414]/40 bg-white/30">
                  <GitBranch size={48} strokeWidth={1} />
                  <p className="mt-4 font-serif italic text-xl">No Branches Detected</p>
                  <p className="text-xs font-mono uppercase tracking-widest mt-2">Process all files to run branch analysis</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-12">
                  <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                    <div className="h-[600px] w-full bg-[#141414]/50 p-6 border border-[#E4E3E0]/5">
                      <h4 className="text-xs font-mono uppercase opacity-60 mb-6 text-center">Frequency vs BLM (Branches)</h4>
                      <ResponsiveContainer width="100%" height="90%">
                        <ScatterChart margin={{ bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                          <XAxis 
                            dataKey="blm" 
                            type="number"
                            tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            label={{ value: 'BLM', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 10 }}
                          />
                          <YAxis 
                            dataKey="freq"
                            type="number"
                            domain={[0, autoFFTRange ? globalSuggestedMaxFreq : manualMaxFreq]}
                            allowDataOverflow={true}
                            tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            label={{ value: 'Frequency (T)', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 10 }}
                          />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '10px', color: '#E4E3E0' }}
                            cursor={{ strokeDasharray: '3 3' }}
                            content={({ active, payload }) => {
                              if (active && payload && payload.length) {
                                const data = payload[0].payload;
                                return (
                                  <div className="bg-[#141414] border border-[#E4E3E0] p-2 font-mono text-[10px] text-[#E4E3E0] shadow-xl">
                                    <p className="text-[#F27D26] font-bold mb-1 border-b border-[#E4E3E0]/20 pb-1">{data.branchName || 'Others'}</p>
                                    <p><span className="opacity-60">Frequency:</span> {data.freq.toFixed(4)} T</p>
                                    <p><span className="opacity-60">BLM:</span> {data.blm}°</p>
                                    <p><span className="opacity-60">Amplitude:</span> {data.amp.toExponential(4)}</p>
                                    <p className="text-[8px] opacity-40 mt-1 truncate max-w-[150px]">{data.fileName}</p>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Legend verticalAlign="top" height={36}/>
                          
                          {/* Others */}
                          <Scatter 
                            name="Others" 
                            data={others.map(p => ({ ...p, branchName: 'Others' }))} 
                            fill="#444" 
                            shape="circle"
                            opacity={0.4}
                          />

                          {/* Branches */}
                          {branchesWithFits.map((branch, idx) => {
                            const color = `hsl(${(idx * 137) % 360}, 70%, 60%)`;
                            const fitPoints = [];
                            if (branch.fit) {
                              const blmValues = branch.peaks.map(p => p.blm);
                              const minBlm = Math.min(...blmValues);
                              const maxBlm = Math.max(...blmValues);
                              for (let b = minBlm; b <= maxBlm; b += 1) {
                                const rad = ((b - branch.fit.cosOffset) * Math.PI) / 180;
                                const cosVal = Math.cos(rad);
                                if (Math.abs(cosVal) > 0.01) {
                                  fitPoints.push({ blm: b, freq: branch.fit.f0 / cosVal });
                                }
                              }
                            }
                            return (
                              <React.Fragment key={branch.id}>
                                <Scatter 
                                  name={branch.name} 
                                  data={branch.peaks.map(p => ({ ...p, branchName: branch.name }))} 
                                  fill={color}
                                  line={{ stroke: color, strokeWidth: 1, strokeDasharray: '5 5' }}
                                />
                                {fitPoints.length > 0 && (
                                  <Scatter 
                                    name={`${branch.name} Fit`}
                                    data={fitPoints}
                                    fill="none"
                                    line={{ stroke: color, strokeWidth: 2 }}
                                    shape={() => null}
                                    legendType="none"
                                  />
                                )}
                              </React.Fragment>
                            );
                          })}
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                    <h4 className="text-sm font-mono uppercase tracking-widest text-[#F27D26] mb-8">Branch Comparison Table (Wide Format)</h4>
                    <div className="overflow-x-auto border border-white/10 max-h-[600px]">
                      <table className="w-full text-left font-mono text-[10px] border-collapse">
                        <thead>
                          <tr className="border-b border-white/20 bg-white/5">
                            <th className="py-3 px-4 uppercase sticky left-0 bg-[#141414] z-20 border-r border-white/10">BLM</th>
                            {branchesWithFits.map(b => (
                              <th key={b.id} className="py-3 px-4 uppercase text-center border-r border-white/10 min-w-[120px]">
                                {b.name}
                              </th>
                            ))}
                            <th className="py-3 px-4 uppercase text-center opacity-40 min-w-[120px]">Others</th>
                          </tr>
                        </thead>
                        <tbody>
                          {branchComparisonTable.map((row, rIdx) => (
                            <tr key={rIdx} className="border-b border-white/5 hover:bg-white/5">
                              <td className="py-2 px-4 font-bold sticky left-0 bg-[#141414] z-10 border-r border-white/10">{row.blm}°</td>
                              {branchesWithFits.map(b => (
                                <td key={b.id} className="py-2 px-4 text-center border-r border-white/10">
                                  {row[b.id] !== null ? (
                                    <span className="text-[#F27D26] font-bold">{row[b.id].freq.toFixed(4)}</span>
                                  ) : (
                                    <span className="opacity-20">-</span>
                                  )}
                                </td>
                              ))}
                              <td className="py-2 px-4 text-center opacity-40">
                                {row.others !== null ? row.others.freq.toFixed(4) : '-'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                        {branchesWithFits.some(b => b.fit) && (
                          <tfoot className="sticky bottom-0 bg-[#141414] border-t-2 border-[#F27D26]/30">
                            <tr>
                              <td className="py-3 px-4 font-bold uppercase text-[9px] opacity-60 border-r border-white/10">F₀ (Fit)</td>
                              {branchesWithFits.map(b => (
                                <td key={b.id} className="py-3 px-4 text-center border-r border-white/10">
                                  {b.fit ? (
                                    <div className="flex flex-col">
                                      <span className="text-[#F27D26] font-bold">{b.fit.f0.toFixed(4)} T</span>
                                      <span className="text-[8px] opacity-40">R²: {b.fit.r2.toFixed(3)}</span>
                                    </div>
                                  ) : '-'}
                                </td>
                              ))}
                              <td className="py-3 px-4 border-l border-white/10"></td>
                            </tr>
                          </tfoot>
                        )}
                      </table>
                    </div>
                  </div>

                  <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-12">
                      {branchesWithFits.map((branch, idx) => {
                        const color = `hsl(${(idx * 137) % 360}, 70%, 60%)`;
                        const blmValues = branch.peaks.map(p => p.blm);
                        const minBlm = Math.min(...blmValues);
                        const maxBlm = Math.max(...blmValues);
                        const freqValues = branch.peaks.map(p => p.freq);
                        const minFreq = Math.min(...freqValues);
                        const maxFreq = Math.max(...freqValues);
                        const padding = (maxFreq - minFreq) * 0.2 || 10;

                        const fitPoints = [];
                        if (branch.fit) {
                          for (let b = minBlm - 5; b <= maxBlm + 5; b += 1) {
                            const rad = ((b - branch.fit.cosOffset) * Math.PI) / 180;
                            const cosVal = Math.cos(rad);
                            if (Math.abs(cosVal) > 0.01) {
                              fitPoints.push({ blm: b, freq: branch.fit.f0 / cosVal });
                            }
                          }
                        }

                        return (
                          <div key={branch.id} className="bg-white/5 p-6 border border-white/10">
                            <div className="flex justify-between items-center mb-6">
                              <div className="flex items-center gap-4">
                                <input 
                                  type="checkbox"
                                  checked={selectedBranchIds.has(branch.id)}
                                  onChange={() => {
                                    const next = new Set(selectedBranchIds);
                                    if (next.has(branch.id)) next.delete(branch.id);
                                    else next.add(branch.id);
                                    setSelectedBranchIds(next);
                                  }}
                                  className="w-4 h-4 accent-[#F27D26]"
                                />
                                <h5 className="text-xl font-serif italic font-bold">{branch.name}</h5>
                              </div>
                              <div className="flex gap-4 text-[10px] font-mono uppercase tracking-widest">
                                <span className="opacity-40">{branch.peaks.length} Points</span>
                              </div>
                            </div>
                            
                            <div className="h-[300px] w-full bg-black/20 p-4 border border-white/5">
                              <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ bottom: 20, left: 20 }}>
                                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                                  <XAxis 
                                    dataKey="blm" 
                                    type="number"
                                    domain={[Math.floor(minBlm - 5), Math.ceil(maxBlm + 5)]}
                                    tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    label={{ value: 'BLM', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }}
                                  />
                                  <YAxis 
                                    dataKey="freq"
                                    type="number"
                                    domain={[Math.max(0, minFreq - padding), maxFreq + padding]}
                                    tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                    label={{ value: 'Freq (T)', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 9 }}
                                  />
                                  <Tooltip 
                                    contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '9px', color: '#E4E3E0' }}
                                    cursor={{ strokeDasharray: '3 3' }}
                                  />
                                  <Scatter 
                                    name={branch.name} 
                                    data={branch.peaks} 
                                    fill={color}
                                    line={{ stroke: color, strokeWidth: 1, strokeDasharray: '5 5' }}
                                  />
                                  {fitPoints.length > 0 && (
                                    <Scatter 
                                      name="Fit"
                                      data={fitPoints}
                                      fill="none"
                                      line={{ stroke: color, strokeWidth: 2 }}
                                      shape={() => null}
                                    />
                                  )}
                                </ScatterChart>
                              </ResponsiveContainer>
                            </div>
                            
                            {branch.fit && (
                              <div className="mt-4 p-3 bg-[#F27D26]/10 border border-[#F27D26]/20 flex justify-around font-mono text-[10px] uppercase tracking-widest">
                                <div><span className="opacity-60">F₀:</span> <span className="text-[#F27D26] font-bold">{branch.fit.f0.toFixed(4)} T</span></div>
                                <div><span className="opacity-60">R²:</span> <span className="text-[#F27D26] font-bold">{branch.fit.r2.toFixed(4)}</span></div>
                                <div><span className="opacity-60">Model:</span> F₀/cos(θ-{branch.fit.cosOffset}°)</div>
                              </div>
                            )}
                            
                            <div className="mt-6 overflow-x-auto">
                              <table className="w-full text-left font-mono text-[9px]">
                                <thead>
                                  <tr className="border-b border-white/10 text-white/40">
                                    <th className="py-1 px-2 uppercase">File</th>
                                    <th className="py-1 px-2 uppercase text-center">BLM</th>
                                    <th className="py-1 px-2 uppercase text-right">Freq</th>
                                    <th className="py-1 px-2 uppercase text-right">Amp</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {branch.peaks.map((p, pIdx) => (
                                    <tr key={pIdx} className="border-b border-white/5 hover:bg-white/5">
                                      <td className="py-1 px-2 truncate max-w-[120px] opacity-60">{p.fileName}</td>
                                      <td className="py-1 px-2 text-center font-bold">{p.blm}°</td>
                                      <td className="py-1 px-2 text-right">{p.freq.toFixed(4)}</td>
                                      <td className="py-1 px-2 text-right opacity-60">{p.amp.toExponential(2)}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                    <h4 className="text-sm font-mono uppercase tracking-widest text-[#F27D26] mb-6">Branch Summary</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {branchesWithFits.map((branch, idx) => (
                        <div key={branch.id} className={`bg-white/5 p-4 border transition-all ${selectedBranchIds.has(branch.id) ? 'border-[#F27D26]' : 'border-white/10'} flex flex-col justify-between`}>
                          <div>
                            <div className="flex justify-between items-center mb-2">
                              <div className="flex items-center gap-2">
                                <input 
                                  type="checkbox"
                                  checked={selectedBranchIds.has(branch.id)}
                                  onChange={() => {
                                    const next = new Set(selectedBranchIds);
                                    if (next.has(branch.id)) next.delete(branch.id);
                                    else next.add(branch.id);
                                    setSelectedBranchIds(next);
                                  }}
                                  className="w-3 h-3 accent-[#F27D26]"
                                />
                                <h5 className="font-bold font-serif italic text-lg">{branch.name}</h5>
                              </div>
                              <span className="text-[9px] font-mono bg-[#F27D26] text-[#141414] px-2 py-0.5 uppercase">{branch.peaks.length} Points</span>
                            </div>
                            <div className="space-y-1 mb-4">
                              <div className="flex justify-between text-[10px] font-mono">
                                <span className="opacity-40">Avg Freq:</span>
                                <span>{_.meanBy(branch.peaks, 'freq').toFixed(2)} T</span>
                              </div>
                              <div className="flex justify-between text-[10px] font-mono">
                                <span className="opacity-40">BLM Range:</span>
                                <span>
                                  {Math.min(...branch.peaks.map(p => p.blm))}° - {Math.max(...branch.peaks.map(p => p.blm))}°
                                </span>
                              </div>
                              {branch.fit && (
                                <div className="pt-2 mt-2 border-t border-white/10 space-y-1">
                                  <div className="flex justify-between text-[10px] font-mono text-[#F27D26]">
                                    <span className="opacity-60">F₀:</span>
                                    <span className="font-bold">{branch.fit.f0.toFixed(4)} T</span>
                                  </div>
                                  <div className="flex justify-between text-[10px] font-mono text-[#F27D26]">
                                    <span className="opacity-60">R²:</span>
                                    <span className="font-bold">{branch.fit.r2.toFixed(4)}</span>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {others.length > 0 && (
                    <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                      <h4 className="text-sm font-mono uppercase tracking-widest text-gray-500 mb-6">Unassigned Peaks (Others)</h4>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left font-mono text-[10px]">
                          <thead>
                            <tr className="border-b border-[#E4E3E0]/20 text-gray-500">
                              <th className="py-2 px-4 uppercase tracking-wider">File Name</th>
                              <th className="py-2 px-4 uppercase tracking-wider text-center">BLM</th>
                              <th className="py-2 px-4 uppercase tracking-wider text-right">Frequency (T)</th>
                              <th className="py-2 px-4 uppercase tracking-wider text-right">Amplitude</th>
                            </tr>
                          </thead>
                          <tbody>
                            {others.map((item, idx) => (
                              <tr key={idx} className="border-b border-[#E4E3E0]/5 hover:bg-[#E4E3E0]/10 transition-colors opacity-60">
                                <td className="py-2 px-4 truncate max-w-[200px]">{item.fileName}</td>
                                <td className="py-2 px-4 font-bold text-center">{item.blm}°</td>
                                <td className="py-2 px-4 text-right">{item.freq.toFixed(4)}</td>
                                <td className="py-2 px-4 text-right font-mono">{item.amp.toExponential(4)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : activeTab === 'harmonics' ? (
            <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                <div className="flex justify-between items-center mb-8">
                  <div className="flex items-center gap-3">
                    <Layers size={24} className="text-[#F27D26]" />
                    <h3 className="text-2xl font-serif italic font-bold">Single File Harmonic Analysis</h3>
                  </div>
                  <div className="flex items-center gap-6 bg-white/5 px-6 py-3 border border-white/10">
                    <div className="flex flex-col gap-1">
                      <label className="text-[9px] font-mono uppercase opacity-60">Select BLM File:</label>
                      <select 
                        value={selectedFileForHarmonics}
                        onChange={(e) => {
                          setSelectedFileForHarmonics(e.target.value);
                          const fileData = assembleData.find(d => d.fileName === e.target.value);
                          if (fileData && fileData.peaks && fileData.peaks.length > 0) {
                            // Default to strongest peak
                            const strongest = _.maxBy(fileData.peaks, 'amplitude') as unknown as FFTResult | undefined;
                            if (strongest) setFundamentalFreq(strongest.frequency);
                          }
                        }}
                        className="bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0] min-w-[200px]"
                      >
                        <option value="">-- Select File --</option>
                        {assembleData.map(d => (
                          <option key={d.fileName} value={d.fileName}>{d.fileName}</option>
                        ))}
                      </select>
                    </div>
                    
                    {fundamentalFreq !== null && (
                      <div className="flex flex-col gap-1 border-l border-white/10 pl-4">
                        <label className="text-[9px] font-mono uppercase opacity-60">Fundamental (F₁):</label>
                        <div className="flex items-center gap-2">
                          <input 
                            type="number" 
                            step="0.1"
                            value={fundamentalFreq} 
                            onChange={(e) => setFundamentalFreq(Number(e.target.value))}
                            className="w-20 bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0]"
                          />
                          <span className="text-[9px] font-mono opacity-40">T</span>
                        </div>
                      </div>
                    )}

                    <div className="flex flex-col gap-1 border-l border-white/10 pl-4">
                      <label className="text-[9px] font-mono uppercase opacity-60">Match Tol:</label>
                      <div className="flex items-center gap-2">
                        <input 
                          type="number" 
                          step="0.005"
                          min="0.001"
                          max="0.1"
                          value={harmonicTolerance} 
                          onChange={(e) => setHarmonicTolerance(Number(e.target.value))}
                          className="w-16 bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0]"
                        />
                        <span className="text-[9px] font-mono opacity-40">rel</span>
                      </div>
                    </div>
                  </div>
                </div>

                {(() => {
                  const fileData = assembleData.find(d => d.fileName === selectedFileForHarmonics);
                  if (!fileData) {
                    return (
                      <div className="h-[300px] border-2 border-dashed border-white/10 flex flex-col items-center justify-center text-white/20">
                        <FileText size={48} strokeWidth={1} />
                        <p className="mt-4 font-serif italic text-xl">No File Selected</p>
                        <p className="text-xs font-mono uppercase tracking-widest mt-2">Select a BLM file to analyze its harmonics</p>
                      </div>
                    );
                  }

                  const peaks = fileData.peaks || [];
                  const harmonics = [1, 2, 3, 4, 5, 6, 7, 8];
                  const f1 = fundamentalFreq || 0;
                  
                  const harmonicMatches = harmonics.map(h => {
                    const target = h * f1;
                    const match = peaks.find(p => Math.abs(p.frequency - target) / target < harmonicTolerance);
                    return { h, target, match };
                  });

                  const maxFreq = Math.max(f1 * 8.5, ...peaks.map(p => p.frequency)) * 1.05;
                  const maxAmp = Math.max(...peaks.map(p => p.amplitude)) * 1.1;

                  return (
                    <div className="space-y-8">
                      <div className="bg-white/5 p-6 border border-white/10">
                        <div className="flex justify-between items-center mb-6">
                          <div>
                            <h5 className="text-xl font-serif italic font-bold">{selectedFileForHarmonics}</h5>
                            <p className="text-[10px] font-mono uppercase opacity-40 mt-1">
                              {blmMode === 'step' ? 'Step' : 'Angle'}: {fileData.x.toFixed(2)}{blmMode === 'step' ? '' : '°'} | Detected Peaks: {peaks.length}
                            </p>
                          </div>
                        </div>

                        <div className="h-[500px] w-full bg-black/20 p-4 border border-white/5">
                          <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={peaks} margin={{ bottom: 20, left: 20 }}>
                              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                              <XAxis 
                                dataKey="frequency" 
                                type="number"
                                domain={[0, maxFreq]}
                                tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                label={{ value: 'Frequency (T)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }}
                              />
                              <YAxis 
                                dataKey="amplitude"
                                type="number"
                                domain={[0, maxAmp]}
                                tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                label={{ value: 'Amplitude', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 9 }}
                              />
                              <Tooltip 
                                contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '9px', color: '#E4E3E0' }}
                                cursor={{ strokeDasharray: '3 3' }}
                              />
                              
                              {/* All Peaks as Bars */}
                              <Scatter 
                                name="All Peaks" 
                                data={peaks} 
                                fill="#E4E3E0" 
                                opacity={0.3}
                              />

                              {/* Harmonic Markers */}
                              {harmonicMatches.map(hm => (
                                <ReferenceLine 
                                  key={hm.h}
                                  x={hm.target} 
                                  stroke={hm.match ? "#F27D26" : "#333"} 
                                  strokeDasharray="5 5"
                                  label={{ 
                                    value: `${hm.h}F`, 
                                    position: 'top', 
                                    fill: hm.match ? "#F27D26" : "#333", 
                                    fontSize: 10,
                                    fontFamily: 'monospace'
                                  }}
                                />
                              ))}

                              {/* Highlighted Matches */}
                              <Scatter 
                                name="Harmonic Matches" 
                                data={harmonicMatches.filter(hm => hm.match).map(hm => hm.match!)} 
                                fill="#F27D26" 
                                shape="star"
                                size={100}
                              />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>

                        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
                          {harmonicMatches.map(hm => (
                            <div 
                              key={hm.h} 
                              className={`p-3 border transition-colors ${hm.match ? 'bg-[#F27D26]/10 border-[#F27D26]/30' : 'bg-white/5 border-white/10 opacity-40'}`}
                              onClick={() => hm.match && setFundamentalFreq(hm.match.frequency / hm.h)}
                              style={{ cursor: hm.match ? 'pointer' : 'default' }}
                            >
                              <span className="text-[10px] font-mono uppercase opacity-60 block">{hm.h}F Target</span>
                              <span className="text-sm font-mono font-bold block">{hm.target.toFixed(2)}</span>
                              {hm.match && (
                                <div className="mt-2 pt-2 border-t border-[#F27D26]/20">
                                  <span className="text-[9px] font-mono uppercase opacity-60 block">Found</span>
                                  <span className="text-[11px] font-mono font-bold text-[#F27D26]">{hm.match.frequency.toFixed(2)}</span>
                                  <span className="text-[9px] font-mono opacity-40 block">Err: {((Math.abs(hm.match.frequency - hm.target) / hm.target) * 100).toFixed(2)}%</span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>

                        <div className="mt-8 border-t border-white/10 pt-6">
                          <h6 className="text-[10px] font-mono uppercase tracking-widest opacity-60 mb-4">Peak List for {selectedFileForHarmonics}</h6>
                          <div className="max-h-64 overflow-y-auto border border-white/5 bg-black/20">
                            <table className="w-full text-left border-collapse">
                              <thead className="sticky top-0 bg-[#141414] border-b border-white/10">
                                <tr>
                                  <th className="p-2 text-[9px] font-mono uppercase opacity-40">Frequency (T)</th>
                                  <th className="p-2 text-[9px] font-mono uppercase opacity-40">Amplitude</th>
                                  <th className="p-2 text-[9px] font-mono uppercase opacity-40">Classification</th>
                                  <th className="p-2 text-[9px] font-mono uppercase opacity-40">Action</th>
                                </tr>
                              </thead>
                              <tbody>
                                {_.orderBy(peaks, ['frequency']).map((p, pIdx) => {
                                  const hMatch = harmonicMatches.find(hm => hm.match === p);
                                  return (
                                    <tr key={pIdx} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                      <td className="p-2 text-[10px] font-mono">{p.frequency.toFixed(4)}</td>
                                      <td className="p-2 text-[10px] font-mono">{p.amplitude.toExponential(2)}</td>
                                      <td className="p-2 text-[10px] font-mono">
                                        {hMatch ? (
                                          <span className="px-2 py-0.5 bg-[#F27D26]/20 text-[#F27D26] rounded-full text-[9px] font-bold">
                                            {hMatch.h}F Harmonic
                                          </span>
                                        ) : (
                                          <span className="opacity-30">-</span>
                                        )}
                                      </td>
                                      <td className="p-2">
                                        <button 
                                          onClick={() => setFundamentalFreq(p.frequency)}
                                          className="text-[9px] font-mono uppercase tracking-widest bg-white/5 hover:bg-white/10 px-2 py-1 border border-white/10 transition-colors"
                                        >
                                          Set as F₁
                                        </button>
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          ) : activeTab === 'assemble' ? (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {(() => {
                const assembleSuggestedMax = assembleFreqMax > 0 ? Math.ceil(assembleFreqMax * 1.2 / 100) * 100 : 'auto';

                return (
                  <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                    <div className="flex justify-between items-center mb-6">
                      <div className="flex items-center gap-3">
                        <Activity size={24} className="text-[#F27D26]" />
                        <h3 className="text-2xl font-serif italic font-bold">FFT Assemble</h3>
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="flex items-center gap-4 bg-white/5 px-4 py-2 border border-white/10">
                          <div className="flex items-center gap-2">
                            <label className="text-[9px] font-mono uppercase opacity-60">Export Range:</label>
                            <input 
                              type="number" 
                              value={exportMinFreq} 
                              onChange={(e) => setExportMinFreq(Number(e.target.value))}
                              className="w-16 bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0]"
                              placeholder="Min"
                            />
                            <span className="text-[#E4E3E0]/40">-</span>
                            <input 
                              type="number" 
                              value={exportMaxFreq} 
                              onChange={(e) => setExportMaxFreq(Number(e.target.value))}
                              className="w-16 bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0]"
                              placeholder="Max"
                            />
                          </div>
                        </div>
                        <div className="flex items-center gap-4 bg-white/5 px-4 py-2 border border-white/10">
                          <div className="flex items-center gap-2">
                            <label className="text-[9px] font-mono uppercase opacity-60">Auto Range</label>
                            <input 
                              type="checkbox" 
                              checked={autoFFTRange} 
                              onChange={(e) => setAutoFFTRange(e.target.checked)}
                              className="accent-[#F27D26]"
                            />
                          </div>
                          {!autoFFTRange && (
                            <div className="flex items-center gap-2">
                              <label className="text-[9px] font-mono uppercase opacity-60">Max Freq:</label>
                              <input 
                                type="number" 
                                value={manualMaxFreq} 
                                onChange={(e) => setManualMaxFreq(Number(e.target.value))}
                                className="w-20 bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0]"
                              />
                            </div>
                          )}
                          <div className="flex items-center gap-2 border-l border-white/10 pl-4 ml-2">
                            <label className="text-[9px] font-mono uppercase opacity-60">Show Peaks</label>
                            <input 
                              type="checkbox" 
                              checked={showPeaksInAssemble} 
                              onChange={(e) => setShowPeaksInAssemble(e.target.checked)}
                              className="accent-[#F27D26]"
                            />
                          </div>
                        </div>
                        <button 
                          onClick={() => {
                            if (assembleData.length === 0) return;
                            
                            // Create frequency grid with interval 1
                            const freqGrid: number[] = [];
                            for (let f = exportMinFreq; f <= exportMaxFreq; f += 1) {
                              freqGrid.push(f);
                            }

                            // Helper for linear interpolation
                            const interpolate = (fft: FFTResult[], targetFreq: number) => {
                              if (fft.length === 0) return 0;
                              if (targetFreq <= fft[0].frequency) return fft[0].amplitude;
                              if (targetFreq >= fft[fft.length - 1].frequency) return fft[fft.length - 1].amplitude;
                              
                              const idx = _.findIndex(fft, f => f.frequency >= targetFreq);
                              if (idx === 0) return fft[0].amplitude;
                              
                              const f0 = fft[idx - 1];
                              const f1 = fft[idx];
                              const t = (targetFreq - f0.frequency) / (f1.frequency - f0.frequency);
                              return f0.amplitude + t * (f1.amplitude - f0.amplitude);
                            };

                            // Build headers: Frequency, {BLM1}, {BLM2}, ...
                            const headers = ["Frequency", ...assembleData.map(d => d.x)].join(",");
                            
                            // Build rows
                            const rows = freqGrid.map(f => {
                              const row = [f];
                              assembleData.forEach(d => {
                                row.push(interpolate(d.fft, f));
                              });
                              return row.join(",");
                            });

                            const csv = headers + "\n" + rows.join("\n");
                            const blob = new Blob([csv], { type: 'text/csv' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `fft_assemble_simplified.csv`;
                            a.click();
                          }}
                          className="bg-[#F27D26] text-white px-4 py-2 font-mono text-[10px] uppercase tracking-widest hover:bg-opacity-80"
                        >
                          Export FFT Assemble
                        </button>
                        <div className="text-right">
                          <span className="text-[10px] font-mono uppercase opacity-40 block">Waterfall View</span>
                          <span className="text-[10px] font-mono uppercase opacity-40 block">{assembleData.length} Files Assembled</span>
                        </div>
                      </div>
                    </div>

                    {assembleData.length === 0 && !isProcessing && (
                      <div className="h-[400px] border-2 border-dashed border-[#E4E3E0]/20 flex flex-col items-center justify-center text-[#E4E3E0]/40">
                        <Activity size={48} strokeWidth={1} />
                        <p className="mt-4 font-serif italic text-xl">No Data to Assemble</p>
                        <p className="text-xs font-mono uppercase tracking-widest mt-2">Click "Process All Files" to generate assembly</p>
                      </div>
                    )}

                    {assembleData.length > 0 && (
                      <div className="space-y-12">
                         <div className="h-[600px] w-full bg-[#141414]/50 p-6 border border-[#E4E3E0]/5">
                            <h4 className="text-xs font-mono uppercase opacity-60 mb-6 text-center">FFT Evolution vs {blmMode === 'step' ? 'Steps' : 'BLM'}</h4>
                            <ResponsiveContainer width="100%" height="90%">
                              <LineChart margin={{ bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                                <XAxis 
                                  dataKey="frequency" 
                                  type="number"
                                  domain={[0, autoFFTRange ? assembleSuggestedMax : manualMaxFreq]}
                                  allowDataOverflow={true}
                                  tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                  label={{ value: 'Frequency (T)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 10 }}
                                />
                                <YAxis 
                                  tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                  label={{ value: 'Amplitude', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 10 }}
                                />
                                <Tooltip 
                                  contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '10px', color: '#E4E3E0' }}
                                />
                                <Legend verticalAlign="top" height={36}/>
                                {assembleChartData.map((item, idx) => (
                                  <Line
                                    key={item.fileName}
                                    data={item.fftDisplay}
                                    type="linear"
                                    dataKey="amplitude"
                                    stroke={`hsl(${(idx * 360) / assembleChartData.length}, 70%, 60%)`}
                                    strokeWidth={1}
                                    dot={false}
                                    name={`${blmMode === 'step' ? 'Step' : 'BLM'}: ${item.x}`}
                                    isAnimationActive={false}
                                  />
                                ))}
                                {/* Peaks in Assemble */}
                                {showPeaksInAssemble && assembleData.map((item, idx) => (
                                  <Scatter
                                    key={`peaks-${item.fileName}`}
                                    data={(item.peaks || []).map(p => ({ frequency: p.frequency, amplitude: p.amplitude }))}
                                    fill="#FFFFFF"
                                    stroke={`hsl(${(idx * 360) / assembleData.length}, 70%, 60%)`}
                                    strokeWidth={1}
                                    name={`Peaks ${item.fileName}`}
                                    legendType="none"
                                    isAnimationActive={false}
                                  >
                                    <LabelList 
                                      dataKey="frequency" 
                                      position="top" 
                                      offset={10} 
                                      formatter={(val: number) => val.toFixed(1)} 
                                      style={{ fontSize: '8px', fontFamily: 'monospace', fill: '#FFFFFF', opacity: 0.8 }}
                                    />
                                  </Scatter>
                                ))}
                              </LineChart>
                            </ResponsiveContainer>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {assembleChartData.map((item) => (
                              <div key={item.fileName} className="bg-white/5 p-4 border border-[#E4E3E0]/10">
                                <div className="flex justify-between items-center mb-2">
                                  <span className="text-[10px] font-mono truncate max-w-[150px] opacity-60">{item.fileName}</span>
                                  <span className="text-[10px] font-mono font-bold text-[#F27D26]">{blmMode === 'step' ? 'Step' : 'BLM'}: {item.x}</span>
                                </div>
                                <div className="h-[100px] w-full">
                                  <MiniFFT
                                    data={item.fft}
                                    freqMax={autoFFTRange ? assembleSuggestedMax : manualMaxFreq}
                                    peaks={item.peaks}
                                    showPeaks={showPeaksInAssemble}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          ) : activeTab === 'lkfit' ? (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                <div className="flex justify-between items-center mb-8 flex-wrap gap-4">
                  <div className="flex items-center gap-3">
                    <Activity size={24} className="text-[#F27D26]" />
                    <div>
                      <h3 className="text-2xl font-serif italic font-bold">Lifshitz-Kosevich Mass Fit</h3>
                      {activeBucket === 'temperature' && (
                        <p className="text-[10px] font-mono opacity-40 mt-0.5">
                          T-Sweep Set: {orientations.find(o => o.id === activeOrientationId)?.temperature.groups.find(g => g.id === activeTempGroupId)?.label ?? '—'}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-6 bg-white/5 px-6 py-3 border border-white/10 flex-wrap">
                    <div className="flex flex-col gap-1">
                      <label className="text-[9px] font-mono uppercase opacity-60">Column:</label>
                      <select
                        value={lkColumn}
                        onChange={(e) => setLkColumn(e.target.value)}
                        className="bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0] min-w-[140px]"
                      >
                        <option value="">-- First available --</option>
                        {lkColumnOptions.map(c => <option key={c} value={c}>{c}</option>)}
                      </select>
                    </div>
                    <div className="flex flex-col gap-1 border-l border-white/10 pl-4">
                      <label className="text-[9px] font-mono uppercase opacity-60">Frequency (T):</label>
                      <input
                        type="number"
                        step="0.1"
                        value={lkFreq ?? ''}
                        onChange={(e) => setLkFreq(e.target.value === '' ? null : Number(e.target.value))}
                        className="w-24 bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0]"
                        placeholder="Manual"
                      />
                    </div>
                    <button
                      onClick={exportLKFitCSV}
                      disabled={!lkFitResult || lkFitResult.points.length < 3}
                      title="온도별 진폭/피팅값 + m*, A0, R²를 CSV로 저장"
                      className="px-2.5 py-1.5 font-mono text-[9px] uppercase border border-[#F27D26]/40 text-[#F27D26] hover:bg-[#F27D26]/10 disabled:opacity-30 disabled:cursor-not-allowed self-end flex items-center gap-1"
                    >
                      <Download size={11} />Export CSV
                    </button>
                  </div>
                </div>

                {lkFreqClusters.length > 0 && (
                  <div className="flex items-center gap-2 flex-wrap mb-8">
                    <span className="text-[9px] font-mono uppercase opacity-40 mr-2">Frequently detected:</span>
                    {lkFreqClusters.map(fc => (
                      <button
                        key={fc.freq}
                        onClick={() => setLkFreq(fc.freq)}
                        className={`px-2.5 py-1 font-mono text-[10px] border transition-colors ${
                          lkFreq != null && Math.abs(lkFreq - fc.freq) / fc.freq < 0.02
                            ? 'bg-[#F27D26]/20 border-[#F27D26] text-[#F27D26]'
                            : 'bg-white/5 border-white/10 hover:bg-white/10'
                        }`}
                      >
                        {fc.freq.toFixed(1)} T <span className="opacity-40">×{fc.count}</span>
                      </button>
                    ))}
                  </div>
                )}

                {lkFreq == null ? (
                  <div className="h-[300px] border-2 border-dashed border-white/10 flex flex-col items-center justify-center text-white/20">
                    <Activity size={48} strokeWidth={1} />
                    <p className="mt-4 font-serif italic text-xl">No Frequency Selected</p>
                    <p className="text-xs font-mono uppercase tracking-widest mt-2">Pick a detected frequency above, or type one in manually</p>
                  </div>
                ) : !lkFitResult || lkFitResult.points.length < 3 ? (
                  <div className="h-[300px] border-2 border-dashed border-white/10 flex flex-col items-center justify-center text-white/20 text-center px-8">
                    <AlertCircle size={48} strokeWidth={1} />
                    <p className="mt-4 font-serif italic text-xl">Not Enough Points</p>
                    <p className="text-xs font-mono uppercase tracking-widest mt-2">
                      Need at least 3 files with a resolvable temperature and processed results ({lkFitResult?.points.length ?? 0} found)
                    </p>
                  </div>
                ) : (() => {
                  const { points, mStar, A0, r2 } = lkFitResult;
                  const avgBeff = points.reduce((s, p) => s + p.bEff, 0) / points.length;
                  const curvePoints = lkFitCurve;

                  return (
                    <div className="space-y-8">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-white/5 p-4 border border-white/10">
                          <span className="text-[9px] font-mono uppercase opacity-60 block">Effective Mass m*</span>
                          <span className="text-xl font-mono font-bold text-[#F27D26]">{mStar != null ? `${mStar.toFixed(3)} m₀` : '—'}</span>
                        </div>
                        <div className="bg-white/5 p-4 border border-white/10">
                          <span className="text-[9px] font-mono uppercase opacity-60 block">R²</span>
                          <span className="text-xl font-mono font-bold">{r2 != null ? r2.toFixed(4) : '—'}</span>
                        </div>
                        <div className="bg-white/5 p-4 border border-white/10">
                          <span className="text-[9px] font-mono uppercase opacity-60 block">A0</span>
                          <span className="text-xl font-mono font-bold">{A0 != null ? A0.toExponential(3) : '—'}</span>
                        </div>
                        <div className="bg-white/5 p-4 border border-white/10">
                          <span className="text-[9px] font-mono uppercase opacity-60 block">Points / Avg B_eff</span>
                          <span className="text-xl font-mono font-bold">{points.length} <span className="text-xs opacity-50">/ {avgBeff.toFixed(2)} T</span></span>
                        </div>
                      </div>

                      <div className="h-[400px] w-full bg-black/20 p-4 border border-white/5">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart margin={{ bottom: 20, left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                            <XAxis
                              dataKey="temp"
                              type="number"
                              domain={['auto', 'auto']}
                              tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                              label={{ value: 'Temperature (K)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }}
                            />
                            <YAxis
                              dataKey="amp"
                              type="number"
                              domain={['auto', 'auto']}
                              tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                              label={{ value: 'Amplitude', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 9 }}
                            />
                            <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #E4E3E0', fontSize: '9px', color: '#E4E3E0' }} />
                            <Legend wrapperStyle={{ fontSize: 9, fontFamily: 'monospace' }} />
                            {curvePoints.length > 0 && (
                              <Line data={curvePoints} dataKey="ampFit" type="monotone" dot={false} stroke="#F27D26" strokeWidth={2} name="LK Fit" isAnimationActive={false} />
                            )}
                            <Scatter data={points} dataKey="amp" fill="#E4E3E0" name="Data" />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="border-t border-white/10 pt-6">
                        <div className="flex justify-between items-center mb-4">
                          <h6 className="text-[10px] font-mono uppercase tracking-widest opacity-60">Temperature Series ({lkFreq.toFixed(1)} T{lkColumn ? `, ${lkColumn}` : ''})</h6>
                          <button onClick={exportLKFitCSV} className="px-2.5 py-1 font-mono text-[8px] uppercase border border-[#E4E3E0]/15 hover:bg-white/5 flex items-center gap-1"><Download size={9} />CSV</button>
                        </div>
                        <div className="max-h-64 overflow-y-auto border border-white/5 bg-black/20">
                          <table className="w-full text-left border-collapse">
                            <thead className="sticky top-0 bg-[#141414] border-b border-white/10">
                              <tr>
                                <th className="p-2 text-[9px] font-mono uppercase opacity-40">File</th>
                                <th className="p-2 text-[9px] font-mono uppercase opacity-40">T (K)</th>
                                <th className="p-2 text-[9px] font-mono uppercase opacity-40">B_eff (T)</th>
                                <th className="p-2 text-[9px] font-mono uppercase opacity-40">Amplitude</th>
                                <th className="p-2 text-[9px] font-mono uppercase opacity-40">Predicted</th>
                              </tr>
                            </thead>
                            <tbody>
                              {points.map((p, pIdx) => {
                                const X = mStar != null ? 14.69 * mStar * p.temp / p.bEff : NaN;
                                const pred = mStar != null && A0 != null ? A0 * lkDampingFactor(X) : NaN;
                                return (
                                  <tr key={pIdx} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                    <td className="p-2 text-[10px] font-mono">{p.fileName}</td>
                                    <td className="p-2 text-[10px] font-mono">{p.temp.toFixed(2)}</td>
                                    <td className="p-2 text-[10px] font-mono">{p.bEff.toFixed(2)}</td>
                                    <td className="p-2 text-[10px] font-mono">{p.amp.toExponential(3)}</td>
                                    <td className="p-2 text-[10px] font-mono">{isFinite(pred) ? pred.toExponential(3) : '—'}</td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          ) : activeTab === 'fftrange' ? (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-[#141414] text-[#E4E3E0] p-8 border border-[#E4E3E0]/10">
                <div className="flex justify-between items-center mb-8 flex-wrap gap-4">
                  <div className="flex items-center gap-3">
                    <Activity size={24} className="text-[#F27D26]" />
                    <h3 className="text-2xl font-serif italic font-bold">FFT Range vs Detected Peak</h3>
                  </div>
                  <div className="flex items-center gap-6 bg-white/5 px-6 py-3 border border-white/10 flex-wrap">
                    <div className="flex flex-col gap-1">
                      <label className="text-[9px] font-mono uppercase opacity-60">Candidate Min B (T):</label>
                      <input type="number" step="0.1" value={rangeScanMinBDraft}
                        onChange={(e) => handleRangeScanMinBChange(Number(e.target.value))}
                        className="w-24 bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0]" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[9px] font-mono uppercase opacity-60">Candidate Max B (T):</label>
                      <input type="number" step="0.1" value={rangeScanMaxBDraft}
                        onChange={(e) => handleRangeScanMaxBChange(Number(e.target.value))}
                        className="w-24 bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0]" />
                    </div>
                    <button
                      onClick={() => { setTempMinB(rangeScanMinBDraft); setTempMaxB(rangeScanMaxBDraft); setMinB(rangeScanMinBDraft); setMaxB(rangeScanMaxBDraft); }}
                      className="px-2.5 py-1.5 font-mono text-[9px] uppercase border border-[#F27D26] text-[#F27D26] hover:bg-[#F27D26]/10 self-end"
                    >
                      Use This Range
                    </button>
                  </div>
                </div>
                <p className="text-[10px] font-mono opacity-40 mb-6 leading-relaxed">
                  현재 적용된 범위 [{isFinite(committedMinB) ? committedMinB.toFixed(2) : '−∞'}, {isFinite(committedMaxB) ? committedMaxB.toFixed(2) : '∞'}] T (파란색) 대비, 위에서 바꾼 후보 범위 (주황색) 로 각 파일의 detected peak이 어떻게 움직이는지 각도별로 비교합니다. 값을 바꾸면 타이핑을 멈춘 직후 자동 갱신되고, "Process All"은 누를 필요 없습니다.
                </p>

                {rangeScanBatch.length === 0 ? (
                  <div className="h-[300px] border-2 border-dashed border-white/10 flex flex-col items-center justify-center text-white/20 text-center px-8">
                    <FileText size={48} strokeWidth={1} />
                    <p className="mt-4 font-serif italic text-xl">No Data</p>
                    <p className="text-xs font-mono uppercase tracking-widest mt-2">
                      {files.length === 0
                        ? 'Upload files first'
                        : filesByAngle.every(({ file }) => fftSkip.has(file.id))
                        ? 'FFT Files 목록에서 파일을 포함시키세요 (사이드바)'
                        : 'Neither range found a peak above threshold for any included file — try widening the range or lowering the peak threshold'}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {rangeScanStability && (
                      <div className="grid grid-cols-3 gap-4">
                        <div className="bg-white/5 p-4 border border-white/10">
                          <span className="text-[9px] font-mono uppercase opacity-60 block">Files Compared</span>
                          <span className="text-xl font-mono font-bold">{rangeScanStability.n}</span>
                        </div>
                        <div className="bg-white/5 p-4 border border-white/10">
                          <span className="text-[9px] font-mono uppercase opacity-60 block">Mean |ΔPeak|</span>
                          <span className="text-xl font-mono font-bold">{rangeScanStability.meanDelta.toFixed(2)} T</span>
                        </div>
                        <div className="bg-white/5 p-4 border border-white/10">
                          <span className="text-[9px] font-mono uppercase opacity-60 block">Max |ΔPeak|</span>
                          <span className={`text-xl font-mono font-bold ${rangeScanStability.maxDelta > 20 ? 'text-red-400' : 'text-[#4ADE80]'}`}>{rangeScanStability.maxDelta.toFixed(2)} T</span>
                        </div>
                      </div>
                    )}

                    <div className="h-[420px] w-full bg-black/20 p-4 border border-white/5">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart margin={{ bottom: 20, left: 10, right: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                          <XAxis dataKey="blm" type="number" domain={['auto', 'auto']}
                            tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            label={{ value: 'BLM / angle (°)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }} />
                          <YAxis type="number" domain={['auto', 'auto']}
                            tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            label={{ value: 'Peak F (T)', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 9 }} width={55} />
                          <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '9px', color: '#E4E3E0' }}
                            formatter={(v: any, name: string) => [typeof v === 'number' ? v.toFixed(2) : v, name]}
                            labelFormatter={(v: number) => `angle = ${Number(v).toFixed(2)}°`} />
                          <Legend wrapperStyle={{ fontSize: 9, fontFamily: 'monospace' }} />
                          <Scatter data={rangeScanCurrentPeaks} dataKey="frequency" fill="#60A5FA" name="Current Range" isAnimationActive={false} />
                          <Scatter data={rangeScanCandidatePeaks} dataKey="frequency" fill="#F27D26" name="Candidate Range" isAnimationActive={false} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="border border-white/5 bg-black/20 max-h-64 overflow-y-auto">
                      <table className="w-full text-left border-collapse">
                        <thead className="sticky top-0 bg-[#141414] border-b border-white/10">
                          <tr>
                            <th className="p-2 text-[9px] font-mono uppercase opacity-40">Angle</th>
                            <th className="p-2 text-[9px] font-mono uppercase opacity-40">File</th>
                            <th className="p-2 text-[9px] font-mono uppercase opacity-40">Current Peak (T)</th>
                            <th className="p-2 text-[9px] font-mono uppercase opacity-40">Candidate Peak (T)</th>
                            <th className="p-2 text-[9px] font-mono uppercase opacity-40">ΔPeak (T)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {rangeScanBatch.map((d, i) => {
                            const delta = d.committedPeak !== null && d.candidatePeak !== null ? Math.abs(d.candidatePeak - d.committedPeak) : null;
                            return (
                              <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                <td className="p-2 text-[10px] font-mono">{d.blm}°</td>
                                <td className="p-2 text-[10px] font-mono truncate max-w-[200px]">{d.fileName}</td>
                                <td className="p-2 text-[10px] font-mono text-[#60A5FA]">{d.committedPeak?.toFixed(2) ?? '—'}</td>
                                <td className="p-2 text-[10px] font-mono text-[#F27D26]">{d.candidatePeak?.toFixed(2) ?? '—'}</td>
                                <td className={`p-2 text-[10px] font-mono ${delta !== null && delta > 20 ? 'text-red-400' : ''}`}>{delta?.toFixed(2) ?? '—'}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : activeTab === 'ifft' ? (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-[#141414] text-[#E4E3E0] border border-[#E4E3E0]/10">
                <div className="flex items-center justify-between px-6 py-4 border-b border-[#E4E3E0]/10 flex-wrap gap-3">
                  <div className="flex items-center gap-3">
                    <Activity size={18} className="text-[#F27D26]" />
                    <div>
                      <h3 className="text-lg font-serif italic font-bold">IFFT Verification</h3>
                      <p className="text-[9px] font-mono opacity-40 mt-0.5">특정 주파수가 실제 신호인지, 재구성한 성분을 원본 ΔR(1/B)와 겹쳐서 확인합니다</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-[9px] font-mono uppercase opacity-40">File · Column</span>
                    <select
                      value={ifftSelected?.key ?? ''}
                      onChange={(e) => setIfftResultKey(e.target.value)}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[10px] px-3 py-1.5 focus:outline-none focus:border-[#F27D26] cursor-pointer max-w-[280px]"
                    >
                      {landauSeriesList.map(s => (
                        <option key={s.key} value={s.key} className="bg-[#141414]">{s.blm}° · {s.col}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {ifftSelected && ifftSelected.result.detectedPeaks.length > 0 && (
                  <div className="flex items-center gap-2 flex-wrap px-6 py-3 border-b border-[#E4E3E0]/10">
                    <span className="text-[9px] font-mono uppercase opacity-40 mr-1">Detected peaks:</span>
                    {ifftSelected.result.detectedPeaks.slice(0, 8).map((p, i) => {
                      const isSelected = !!ifftBand && p.frequency >= ifftBand.lo && p.frequency <= ifftBand.hi;
                      return (
                        <button
                          key={i}
                          onClick={() => {
                            const halfWidth = Math.max(2, p.frequency * 0.08);
                            setIfftBandLo(Math.max(0, p.frequency - halfWidth));
                            setIfftBandHi(p.frequency + halfWidth);
                          }}
                          className={`px-2.5 py-1 font-mono text-[10px] border transition-colors ${
                            isSelected ? 'bg-[#F27D26]/20 border-[#F27D26] text-[#F27D26]' : 'bg-white/5 border-white/10 hover:bg-white/10'
                          }`}
                        >
                          {p.frequency.toFixed(1)} T
                        </button>
                      );
                    })}
                  </div>
                )}

                <div className="flex items-center gap-4 px-6 py-3 flex-wrap">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[8px] font-mono uppercase opacity-40">Band</span>
                    <input type="number" step="1" value={ifftBandLo || ''} placeholder="lo"
                      ref={wheelStepRef(ifftBandLo, (v: number) => setIfftBandLo(Math.max(0, v)), 1)}
                      onChange={e => setIfftBandLo(Math.max(0, parseFloat(e.target.value) || 0))}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-16 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                    <span className="text-[8px] font-mono opacity-30">–</span>
                    <input type="number" step="1" value={ifftBandHi || ''} placeholder="hi"
                      ref={wheelStepRef(ifftBandHi, (v: number) => setIfftBandHi(Math.max(0, v)), 1)}
                      onChange={e => setIfftBandHi(Math.max(0, parseFloat(e.target.value) || 0))}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-16 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                    <span className="text-[8px] font-mono opacity-30">T</span>
                    <button onClick={() => { setIfftBandLo(0); setIfftBandHi(0); }}
                      className={`px-1.5 py-0.5 font-mono text-[8px] uppercase border transition-all ${ifftBand === null ? 'border-[#F27D26] text-[#F27D26] bg-[#F27D26]/10' : 'border-[#E4E3E0]/15 opacity-40 hover:opacity-70'}`}
                      title="Auto: center the band on the dominant peak">Auto</button>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[8px] font-mono uppercase opacity-40">Edge</span>
                    <input type="number" step="0.05" min="0" max="1" value={ifftEdge}
                      onChange={e => setIfftEdge(Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
                      className="bg-[#1a1a1a] border border-[#E4E3E0]/15 text-[#E4E3E0] font-mono text-[9px] w-14 px-1.5 py-0.5 text-center focus:outline-none focus:border-[#F27D26]" />
                  </div>
                </div>
              </div>

              {landauSeriesList.length === 0 ? (
                <div className="bg-[#141414] text-[#E4E3E0] border border-[#E4E3E0]/10 flex items-center justify-center h-40 text-[11px] font-mono opacity-50 text-center px-8">
                  No processed ΔR(1/B) yet. Go to the Dashboard, pick a column prefix, and click “Process All” first.
                </div>
              ) : (() => {
                const band = ifftFiltered?.band;
                const specMax = ifftFiltered ? Math.max((band?.hi ?? 0) * 1.6, ifftFiltered.peakF * 3.5, 100) : 0;
                const specView = ifftFiltered ? ifftFiltered.spectrum.filter(p => p.frequency <= specMax) : [];
                return (
                  <>
                    {specView.length > 0 && (
                      <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                        <div className="flex justify-between items-start mb-3">
                          <div>
                            <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">FFT Spectrum — pick the band to test</h4>
                            <p className="text-[10px] font-mono mt-0.5 text-[#F27D26]">band {band!.lo.toFixed(1)}–{band!.hi.toFixed(1)} T <span className="opacity-40 text-[#E4E3E0]">{ifftBand === null ? '(auto: dominant peak)' : '(manual)'}</span></p>
                          </div>
                        </div>
                        <div className="h-[180px] w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={specView} margin={{ bottom: 16, left: 10, right: 10 }}>
                              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#222" />
                              <XAxis dataKey="frequency" type="number" domain={[0, specMax]}
                                tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                tickFormatter={(v: number) => v.toFixed(0)}
                                label={{ value: 'F (T)', position: 'insideBottom', offset: 8, fill: '#E4E3E0', fontSize: 9 }} />
                              <YAxis tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                                tickFormatter={(v: number) => v.toExponential(0)} width={50} />
                              <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '9px', color: '#E4E3E0' }}
                                formatter={(v: any) => [typeof v === 'number' ? v.toExponential(3) : v, 'Amp']}
                                labelFormatter={(v: number) => `F = ${Number(v).toFixed(1)} T`} />
                              {band && band.hi > band.lo && <ReferenceArea x1={band.lo} x2={band.hi} fill="#F27D26" fillOpacity={0.15} stroke="#F27D26" strokeOpacity={0.4} />}
                              <Area type="linear" dataKey="amplitude" stroke="#4ADE80" fill="#4ADE80" fillOpacity={0.15} strokeWidth={1} dot={false} isAnimationActive={false} />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    )}

                    <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#E4E3E0]/10">
                      <div className="flex justify-between items-start mb-3 flex-wrap gap-2">
                        <div>
                          <h4 className="text-[11px] font-mono uppercase tracking-widest opacity-50">Raw ΔR vs Reconstructed Single Component</h4>
                          <p className="text-[10px] font-mono mt-0.5 opacity-60">
                            {ifftSelected?.blm}° · {ifftSelected?.col} — 재구성된 성분(주황)이 실제 데이터(회색)의 굴곡과 맞아떨어지면 진짜 신호, 그렇지 않으면 노이즈일 가능성이 높습니다.
                          </p>
                        </div>
                        <div className="flex items-center gap-3 text-[9px] font-mono shrink-0">
                          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[#E4E3E0]/50 inline-block" />raw ΔR</span>
                          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[#F27D26] inline-block" />reconstructed</span>
                        </div>
                      </div>
                      <div className="h-[340px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart margin={{ bottom: 20, left: 10, right: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#222" />
                            <XAxis dataKey="invB" type="number" domain={['dataMin', 'dataMax']}
                              tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                              tickFormatter={(v: number) => v.toFixed(3)}
                              label={{ value: '1/B (T⁻¹)', position: 'insideBottom', offset: 10, fill: '#E4E3E0', fontSize: 9 }} />
                            <YAxis dataKey="val" type="number" tick={{ fontSize: 9, fontFamily: 'monospace', fill: '#E4E3E0' }}
                              tickFormatter={(v: number) => v.toExponential(1)} width={60}
                              label={{ value: 'ΔR', angle: -90, position: 'insideLeft', fill: '#E4E3E0', fontSize: 9 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', fontSize: '9px', color: '#E4E3E0' }}
                              formatter={(v: any) => [typeof v === 'number' ? v.toExponential(4) : v]}
                              labelFormatter={(v: number) => `1/B = ${Number(v).toFixed(4)} (B = ${(1 / v).toFixed(2)} T)`} />
                            <Line data={ifftOriginalLine} dataKey="val" stroke="#E4E3E0" strokeOpacity={0.5} strokeWidth={1} dot={false} isAnimationActive={false} name="Raw ΔR" />
                            <Line data={ifftReconLine} dataKey="val" stroke="#F27D26" strokeWidth={1.75} dot={false} isAnimationActive={false} name="Reconstructed" />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </>
                );
              })()}
            </div>
          ) : (
            <>
              {files.length > 0 && analysisResults.length === 0 && !isProcessing && (
                currentFileResults && fftSkip.has(currentFileResults.id) ? (
                  <div className="h-[600px] border-2 border-dashed border-[#141414]/20 flex flex-col items-center justify-center text-[#141414]/40 text-center px-8">
                    <Activity size={48} strokeWidth={1} />
                    <p className="mt-4 font-serif italic text-xl">이 파일은 FFT에서 제외되어 있습니다</p>
                    <p className="text-xs font-mono uppercase tracking-widest mt-2">
                      oscillation score {(oscScores[currentFileResults.id] ?? 0).toFixed(1)} · 사이드바 “FFT Files” 에서 체크하면 분석됩니다
                    </p>
                  </div>
                ) : (
                  <div className="h-[600px] border-2 border-dashed border-[#141414]/20 flex flex-col items-center justify-center text-[#141414]/40">
                    <Settings2 size={48} strokeWidth={1} />
                    <p className="mt-4 font-serif italic text-xl">No matching columns found for this file</p>
                    <p className="text-xs font-mono uppercase tracking-widest mt-2">Please select B and R columns in the sidebar</p>
                  </div>
                )
              )}

              {files.length === 0 && !isProcessing && (
                <div className="h-[600px] border-2 border-dashed border-[#141414]/20 flex flex-col items-center justify-center text-[#141414]/40">
                  <FileText size={48} strokeWidth={1} />
                  <p className="mt-4 font-serif italic text-xl">Upload data and select columns to begin</p>
                  <p className="text-xs font-mono uppercase tracking-widest mt-2">Expected format: B (Tesla), R (Ohm)</p>
                </div>
              )}

              {isProcessing && (
                <div className="h-[600px] flex items-center justify-center">
                  <RefreshCw className="animate-spin text-[#141414]/40" size={32} />
                </div>
              )}

              {analysisResults.map(res => (
                <div key={res.columnName} className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="border-l-4 border-[#141414] pl-4">
                    <h2 className="text-2xl font-serif italic font-bold uppercase tracking-tight">
                      Analysis: {res.columnName}
                    </h2>
                  </div>

                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                    {/* Plot 1: Raw Data & Background */}
                    <div className="bg-white p-6 border border-[#141414]">
                      <div className="flex justify-between items-center mb-6">
                        <h3 className="font-serif italic font-bold text-lg">1. Raw Data & Fit</h3>
                        <span className="text-[10px] font-mono uppercase opacity-40">R (Ω) vs B (T)</span>
                      </div>
                      <div className="h-[250px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={res.plotData}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                            <XAxis dataKey="b" type="number" domain={['auto', 'auto']} hide />
                            <YAxis tick={{ fontSize: 10, fontFamily: 'monospace' }} domain={['auto', 'auto']} />
                            <Tooltip contentStyle={{ borderRadius: 0, border: '1px solid #141414', fontFamily: 'monospace', fontSize: '10px' }} />
                            <Line type="monotone" dataKey="r" stroke="#141414" strokeWidth={1} dot={false} name="Measured" />
                            <Line type="monotone" dataKey="bg" stroke="#F27D26" strokeWidth={1} strokeDasharray="5 5" dot={false} name="Fit" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Plot 2: Oscillatory Component */}
                    <div className="bg-white p-6 border border-[#141414]">
                      <div className="flex justify-between items-center mb-6">
                        <h3 className="font-serif italic font-bold text-lg">2. Oscillatory (ΔR)</h3>
                        <span className="text-[10px] font-mono uppercase opacity-40">ΔR (Ω) vs 1/B (T⁻¹)</span>
                      </div>
                      <div className="h-[250px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={res.plotData}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                            <XAxis 
                              dataKey="invB" 
                              type="number" 
                              domain={[
                                maxB && Number(maxB) !== 0 ? 1 / Number(maxB) : 'auto', 
                                minB && Number(minB) !== 0 ? 1 / Number(minB) : 'auto'
                              ]} 
                              tick={{ fontSize: 9, fontFamily: 'monospace' }}
                              label={{ value: '1/B', position: 'insideBottom', offset: -2, fontSize: 9 }}
                            />
                            <YAxis tick={{ fontSize: 10, fontFamily: 'monospace' }} />
                            <Tooltip contentStyle={{ borderRadius: 0, border: '1px solid #141414', fontFamily: 'monospace', fontSize: '10px' }} />
                            <Line type="monotone" dataKey="deltaR" stroke="#141414" strokeWidth={1} dot={false} name="ΔR" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  {/* FFT Spectrum */}
                  <div className="bg-[#141414] text-[#E4E3E0] p-6">
                    {(() => {
                      // Live preview swaps in a range-overridden result only once
                      // fileLiveRange is actively being edited; otherwise this is
                      // exactly `res`, so nothing here changes from before.
                      const displayRes = fileLiveResults?.get(res.columnName) ?? res;
                      return (
                    <>
                    <div className="flex justify-between items-center mb-6">
                      <div className="flex flex-col">
                        <h3 className="font-serif italic font-bold text-xl">3. FFT Power Spectrum</h3>
                        {displayRes.peakFrequency && (
                          <div className="flex flex-col">
                            <span className="text-[10px] font-mono text-[#F27D26]">
                              Peak: {displayRes.peakFrequency.frequency.toFixed(1)} T (Amp: {displayRes.peakFrequency.amplitude.toExponential(2)})
                            </span>
                            <span className="text-[9px] font-mono opacity-40">
                              Max Amp (excl. DC): {(() => { let m = 0; for (let i = 1; i < displayRes.fftResults.length; i++) if (displayRes.fftResults[i].amplitude > m) m = displayRes.fftResults[i].amplitude; return displayRes.fftResults.length > 1 ? m.toExponential(2) : 'N/A'; })()}
                            </span>
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[8px] font-mono uppercase opacity-40">Live Range:</span>
                          <input
                            type="number"
                            step="0.1"
                            value={fileLiveRange.minDraft}
                            ref={wheelStepRef(fileLiveRange.minDraft, fileLiveRange.changeMin)}
                            onChange={(e) => fileLiveRange.changeMin(Number(e.target.value))}
                            title="Process All 없이 즉시 미리보기할 Min B (스크롤로 조절 가능)"
                            className="w-14 bg-[#1a1a1a] border border-[#E4E3E0]/30 p-0.5 font-mono text-[9px] text-[#E4E3E0]"
                          />
                          <span className="opacity-30 text-[9px]">–</span>
                          <input
                            type="number"
                            step="0.1"
                            value={fileLiveRange.maxDraft}
                            ref={wheelStepRef(fileLiveRange.maxDraft, fileLiveRange.changeMax)}
                            onChange={(e) => fileLiveRange.changeMax(Number(e.target.value))}
                            title="Process All 없이 즉시 미리보기할 Max B (스크롤로 조절 가능)"
                            className="w-14 bg-[#1a1a1a] border border-[#E4E3E0]/30 p-0.5 font-mono text-[9px] text-[#E4E3E0]"
                          />
                          {fileLiveRange.active && (
                            <button
                              onClick={() => fileLiveRange.reset(typeof minB === 'number' ? minB : 0, typeof maxB === 'number' ? maxB : 1)}
                              title="Process All 결과로 되돌리기"
                              className="px-1.5 py-0.5 font-mono text-[8px] uppercase border border-[#F27D26]/50 text-[#F27D26] hover:bg-[#F27D26]/10"
                            >
                              Reset
                            </button>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <label className="text-[9px] font-mono uppercase opacity-60">Auto Range</label>
                          <input
                            type="checkbox"
                            checked={autoFFTRange}
                            onChange={(e) => setAutoFFTRange(e.target.checked)}
                            className="accent-[#F27D26]"
                          />
                        </div>
                        {!autoFFTRange && (
                          <div className="flex items-center gap-2">
                            <label className="text-[9px] font-mono uppercase opacity-60">Max Freq:</label>
                            <input
                              type="number"
                              value={manualMaxFreq}
                              onChange={(e) => setManualMaxFreq(Number(e.target.value))}
                              className="w-20 bg-[#141414] border border-[#E4E3E0]/30 p-1 font-mono text-[10px] text-[#E4E3E0]"
                            />
                          </div>
                        )}
                        <span className="text-[10px] font-mono uppercase opacity-40 text-[#E4E3E0]">Amplitude vs Frequency (T)</span>
                      </div>
                    </div>
                    {fileLiveRange.active && (
                      <p className="text-[8px] font-mono text-[#F27D26] text-right -mt-4 mb-4">
                        Live preview [{fileLiveMinBEff.toFixed(2)}, {fileLiveMaxBEff.toFixed(2)}] T — not yet Process All'd
                      </p>
                    )}
                    <div className="h-[300px] w-full">
                      <ResponsiveContainer
                        width="100%"
                        height="100%"
                        key={`${res.columnName}-${autoFFTRange}-${manualMaxFreq}`}
                      >
                        <ComposedChart data={downsampleFFT(displayRes.fftResults, 1000)}>
                          <defs>
                            <linearGradient id={`colorAmp-${res.columnName}`} x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#F27D26" stopOpacity={0.8}/>
                              <stop offset="95%" stopColor="#F27D26" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                          <XAxis
                            dataKey="frequency"
                            type="number"
                            domain={[0, autoFFTRange ? (batchSummary.length > 0 ? globalSuggestedMaxFreq : displayRes.suggestedMaxFreq) : manualMaxFreq]}
                            allowDataOverflow={true}
                            tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }}
                            label={{ value: 'Frequency (T)', position: 'insideBottom', offset: -5, fill: '#E4E3E0', fontSize: 10 }}
                          />
                          <YAxis tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#E4E3E0' }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#141414', borderRadius: 0, border: '1px solid #E4E3E0', fontFamily: 'monospace', fontSize: '10px', color: '#E4E3E0' }}
                            itemStyle={{ color: '#F27D26' }}
                            formatter={(value: any, name: string) => {
                              if (name === 'Amplitude') return [value.toExponential(4), name];
                              if (name === 'Threshold') return [value.toExponential(4), name];
                              return [value, name];
                            }}
                          />
                          {autoFFTRange && batchSummary.length > 0 && (
                            <ReferenceLine
                              x={globalSuggestedMaxFreq / 1.5}
                              stroke="#F27D26"
                              strokeDasharray="3 3"
                              label={{ value: 'Max Peak', position: 'top', fill: '#F27D26', fontSize: 9, fontFamily: 'monospace' }}
                              opacity={0.5}
                            />
                          )}
                          <Area type="monotone" dataKey="amplitude" stroke="#F27D26" fillOpacity={1} fill={`url(#colorAmp-${res.columnName})`} name="Amplitude" />
                          <Line type="stepAfter" dataKey="threshold" stroke="#FFFFFF" strokeDasharray="5 5" dot={false} name="Threshold" opacity={0.5} />
                          <Scatter
                            data={displayRes.detectedPeaks}
                            name="Peaks"
                            fill="#FFFFFF"
                            stroke="#F27D26"
                            strokeWidth={2}
                            dataKey="amplitude"
                            isAnimationActive={false}
                          >
                            <LabelList
                              dataKey="frequency"
                              position="top"
                              offset={10}
                              formatter={(val: number) => val.toFixed(2)}
                              style={{ fontSize: '10px', fontFamily: 'monospace', fill: '#FFFFFF' }}
                            />
                          </Scatter>
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                    </>
                      );
                    })()}

                    {/* Detected Peaks Table */}
                    <div className="mt-8 border-t border-[#E4E3E0]/10 pt-6">
                      <div className="flex justify-between items-center mb-4">
                        <h4 className="font-serif italic font-bold text-lg">Detected Peaks (Prominence & Width Filter)</h4>
                        <div className="flex items-center gap-4">
                          <button 
                            onClick={() => copyPeaksToClipboard(res.detectedPeaks)}
                            className="flex items-center gap-2 bg-[#E4E3E0]/10 hover:bg-[#E4E3E0]/20 px-3 py-1 text-[10px] font-mono uppercase transition-all"
                          >
                            <Copy size={12} />
                            Copy Peaks
                          </button>
                          <div className="flex flex-col items-end">
                            <span className="text-[9px] font-mono uppercase opacity-40">Frequency-Dependent Thresholds Active</span>
                            <span className="text-[9px] font-mono uppercase opacity-40">Sorted by Amplitude</span>
                          </div>
                        </div>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left font-mono text-[11px]">
                          <thead>
                            <tr className="border-b border-[#E4E3E0]/20 text-[#F27D26]">
                              <th className="py-2 px-4 uppercase tracking-wider">Rank</th>
                              <th className="py-2 px-4 uppercase tracking-wider">Frequency (T)</th>
                              <th className="py-2 px-4 uppercase tracking-wider">Amplitude</th>
                              <th className="py-2 px-4 uppercase tracking-wider">% of Max</th>
                            </tr>
                          </thead>
                          <tbody>
                            {res.detectedPeaks.length > 0 ? (
                              res.detectedPeaks.slice(0, 10).map((peak, idx) => (
                                <tr key={idx} className="border-b border-[#E4E3E0]/5 hover:bg-[#E4E3E0]/5 transition-colors">
                                  <td className="py-2 px-4 opacity-60">#{idx + 1}</td>
                                  <td className="py-2 px-4 font-bold">{peak.frequency.toFixed(2)}</td>
                                  <td className="py-2 px-4">{peak.amplitude.toExponential(3)}</td>
                                  <td className="py-2 px-4">
                                    {res.peakFrequency 
                                      ? ((peak.amplitude / res.peakFrequency.amplitude) * 100).toFixed(1) 
                                      : '100'}%
                                  </td>
                                </tr>
                              ))
                            ) : (
                              <tr>
                                <td colSpan={4} className="py-4 px-4 text-center opacity-40 italic">No peaks detected above threshold</td>
                              </tr>
                            )}
                          </tbody>
                        </table>
                        {res.detectedPeaks.length > 10 && (
                          <p className="mt-2 text-[9px] opacity-40 italic text-right">Showing top 10 peaks</p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      </main>

      {/* Footer Info */}
      <footer className="max-w-7xl mx-auto mt-12 pt-8 border-t border-[#141414]/10 flex flex-col md:flex-row justify-between gap-8 pb-12">
        <div className="max-w-md">
          <div className="flex items-center gap-2 mb-2">
            <Info size={16} />
            <h4 className="font-serif italic font-bold">About SdH Analysis</h4>
          </div>
          <p className="text-xs leading-relaxed opacity-60">
            Shubnikov-de Haas (SdH) oscillations are periodic variations in the magnetoresistance of a material at low temperatures and high magnetic fields. 
            This tool performs a standard processing pipeline: background subtraction using polynomial fitting, 
            conversion to inverse magnetic field (1/B), linear interpolation for uniform sampling, and Fast Fourier Transform (FFT) to extract the oscillation frequencies.
          </p>
        </div>
        <div className="text-right">
          <p className="text-[10px] font-mono uppercase tracking-widest opacity-40">
            Built for Scientific Data Processing
          </p>
        </div>
      </footer>
    </div>
  );
}
