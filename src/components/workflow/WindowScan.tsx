// How do the peaks depend on the B window? Re-run the pipeline for a few lower
// field bounds and compare. Two things come out of it:
//
//  1. Where the oscillation is still visible. On the widest window each file's
//     dominant component is band-pass reconstructed (IFFT); walking down from
//     the top field, the lower bound is where its envelope sinks below twice
//     that of nearby equal-width bands. The median over files is the B min —
//     going lower only adds noise (and background the polynomial can't follow).
//  2. Which peaks are robust. A real 1/B frequency stays put when the window
//     changes; leakage from the background or window edge moves with the
//     window's resolution 1/Δ(1/B). A peak found (within max(1.5·res, 4 %)) in
//     every window where it is above the 6-cycle limit is marked robust
//     (tolerance max(0.75·(res₁+res₂), 4 %·F)).
import { useState } from 'react';
import { AngleContour, type ContourRow } from '../AngleContour';
import { MIN_CYCLES, fLoOf, resolutionOf, thresholdPeaks } from '../../lib/peaks';
import { reconstruct, toUniform } from '../../lib/verify';
import type { AnalysisResult } from '../../types';
import { Pill, PrimaryPill, fmt, type Series } from './shared';

export interface WindowScanProps {
  rows: Series[];
  relMin: number;
  fMax: number;
  normalize: boolean;
  reprocess: (s: Series, min: number, max: number) => AnalysisResult | null;
  field: { lo: number; hi: number } | null;   // measured field extent, all files
  curMin: number;
  curMax: number;
  onApply: (bMin: number) => void;
}

interface WinPeak { fileName: string; angle: number; F: number; robust: boolean; seen: number; eligible: number }
interface Win { bMin: number; res: number; fLo: number; rows: ContourRow[]; peaks: WinPeak[]; tag?: string }
interface ScanOut { bHi: number; bVis: number; bVisSpread: [number, number]; nVis: number; wins: Win[] }

const tick = () => new Promise(r => setTimeout(r, 0));
const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[s.length >> 1] : NaN; };
const quant = (v: number[], q: number) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : NaN; };

// Lowest field where the component at F (the file's dominant frequency in the
// current window — on the wide window the background leftovers can outgrow
// it) still stands out locally: its IFFT envelope is compared with the
// envelopes of equally wide bands at 0.6, 0.75, 1.33 and 1.6 F, whose median
// is the local noise (the median shrugs off one side band that happens to
// hold another branch). Plain residual rms doesn't work: the low-field data
// is sparse in 1/B, interpolation smooths its noise away and the walk would
// never stop. Segments are ~3 periods wide in 1/B, walked from the top field
// down; one weak segment is tolerated (a beat node), two in a row end it.
const SIDE = [0.6, 0.75, 1.33, 1.6];
function visibleFrom(r: AnalysisResult, F: number): number | null {
  const u = toUniform(r.plotData.map(d => ({ invB: d.invB, val: d.deltaR })));
  if (!u) return null;
  const rec = reconstruct(u, F);
  const sides = SIDE.map(m => reconstruct(u, m * F));
  const x0 = rec[0].x, x1 = rec[rec.length - 1].x;
  const w = Math.max(3 / F, (x1 - x0) / 80);
  const meanEnv = (list: { x: number; env: number }[], a: number) => {
    let s = 0, n = 0;
    for (const p of list) if (p.x >= a && p.x < a + w) { s += p.env; n++; }
    return n ? s / n : NaN;
  };
  let lastGood = x0, misses = 0;
  for (let a = x0, k = 0; a < x1; a += w, k++) {
    const env = meanEnv(rec, a);
    if (!Number.isFinite(env)) continue;
    const noise = median(sides.map(sd => meanEnv(sd, a)));
    // the first segment (top field) sits on the Hilbert edge: don't let it end the walk
    if (k > 0 && env < 2 * noise) { if (++misses >= 2) break; continue; }
    misses = 0;
    lastGood = Math.min(x1, a + w);
  }
  return 1 / lastGood;
}

export function WindowScan({ rows, relMin, fMax, normalize, reprocess, field, curMin, curMax, onApply }: WindowScanProps) {
  const [out, setOut] = useState<ScanOut | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const run = async () => {
    if (!field) return;
    const bHi = Math.min(curMax, field.hi);
    const floor = Math.max(field.lo, 0.1 * bHi, 0.5);
    // 1. visibility on the widest window
    const vis: number[] = [];
    for (let i = 0; i < rows.length; i++) {
      setBusy(`진동이 보이는 field 찾는 중 ${i + 1}/${rows.length}`);
      await tick();
      const F0 = thresholdPeaks(rows[i].result, { relMin: 0.5, fMin: 0, fMax, maxPerFile: 1 })[0]?.frequency;
      const r = F0 ? reprocess(rows[i], floor, bHi) : null;
      const v = r && visibleFrom(r, F0!);
      if (v && Number.isFinite(v)) vis.push(v);
    }
    const bVis = Math.max(floor, Math.round((median(vis) || curMin) * 2) / 2);
    // 2. four lower bounds, evenly spaced in 1/B, from the narrow end down to bVis
    const start = curMin > bVis * 1.1 ? curMin : Math.min(bHi * 0.7, bVis * 1.6);
    const n = start - bVis < 1 ? 1 : 4;
    const bMins = Array.from({ length: n }, (_, k) => {
      const inv = 1 / start + ((1 / bVis - 1 / start) * k) / Math.max(1, n - 1);
      return Math.round((n === 1 ? bVis : 1 / inv) * 10) / 10;
    });
    const per: { bMin: number; res: number; fLo: number; rows: ContourRow[]; byFile: Map<string, { F: number; fLo: number; res: number }[]>; fLoByFile: Map<string, { fLo: number; res: number }> }[] = [];
    for (const [wi, bMin] of bMins.entries()) {
      const contour: ContourRow[] = [];
      const byFile = new Map<string, { F: number; fLo: number; res: number }[]>();
      const fLoByFile = new Map<string, { fLo: number; res: number }>();
      const ress: number[] = [];
      for (let i = 0; i < rows.length; i++) {
        setBusy(`창 ${wi + 1}/${bMins.length} (B ≥ ${bMin} T) · 파일 ${i + 1}/${rows.length}`);
        if (i % 4 === 0) await tick();
        const s = rows[i];
        const r = reprocess(s, bMin, bHi);
        if (!r) continue;
        const res = resolutionOf(r), fLo = fLoOf(r);
        ress.push(res);
        contour.push({ angle: s.blm, fileName: s.fileName, fft: r.fftResults, fLo });
        fLoByFile.set(s.fileName, { fLo, res });
        byFile.set(s.fileName, thresholdPeaks(r, { relMin, fMin: 0, fMax, maxPerFile: 3 }).map(p => ({ F: p.frequency, fLo, res })));
      }
      per.push({ bMin, res: median(ress), fLo: MIN_CYCLES * median(ress), rows: contour, byFile, fLoByFile });
    }
    // 3. robustness across windows
    const angleOf = new Map(rows.map(s => [s.fileName, s.blm]));
    const wins: Win[] = per.map((w, wi) => {
      const peaks: WinPeak[] = [];
      for (const [file, list] of w.byFile) for (const p of list) {
        let seen = 0, eligible = 0;
        per.forEach((o, oi) => {
          if (oi === wi) return;
          const lim = o.fLoByFile.get(file);
          if (!lim || p.F < lim.fLo) return;
          eligible++;
          // a peak's F is uncertain by about half a resolution in each window
          const tol = Math.max(0.75 * (lim.res + p.res), 0.04 * p.F);
          if ((o.byFile.get(file) ?? []).some(q => Math.abs(q.F - p.F) <= tol)) seen++;
        });
        peaks.push({ fileName: file, angle: angleOf.get(file) ?? 0, F: p.F, seen, eligible, robust: eligible > 0 && seen === eligible });
      }
      const tag = Math.abs(w.bMin - curMin) < 0.05 ? '현재' : wi === per.length - 1 ? '진동이 보이는 하한' : undefined;
      return { bMin: w.bMin, res: w.res, fLo: w.fLo, rows: w.rows, peaks, tag };
    });
    setOut({ bHi, bVis, bVisSpread: [quant(vis, 0.2), quant(vis, 0.8)], nVis: vis.length, wins });
    setBusy(null);
  };

  return (
    <div className="mt-8 border-t border-black/[0.06] pt-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-3xl">
          <h3 className="text-[17px] font-semibold">B 범위에 따라 봉우리가 어떻게 바뀌나</h3>
          <p className="mt-1 text-[13px] leading-relaxed text-[#6e6e73]">
            B 하한을 몇 단계로 낮추며 FFT 를 다시 하고 같은 높이 기준({Math.round(relMin * 100)}%)으로 봉우리를 고릅니다.
            가장 낮은 하한은 <b>진동이 잡음보다 크게 보이는 field</b>까지만 내려갑니다 — 주된 봉우리의 IFFT 포락선이 이웃 주파수 대역(같은 폭)의 2배 이상인 가장 낮은 field, 파일 중앙값.
            진짜 1/B 진동은 창을 바꿔도 F 가 그대로이고, 배경·창 끝에서 새는 가짜 봉우리는 창 폭(해상도)을 따라 움직입니다 —
            <span className="text-[#248a3d]"> 모든 창에서 유지된 봉우리</span>를 초록으로 표시합니다.
          </p>
        </div>
        <PrimaryPill onClick={run} disabled={!!busy || !field || rows.length === 0}>{out ? '다시 스캔' : 'B 범위 스캔'}</PrimaryPill>
      </div>
      {busy && <p className="mt-3 text-[13px] text-[#0071e3] tabular-nums">{busy}…</p>}
      {!field && <p className="mt-3 text-[13px] text-[#6e6e73]">필드 컬럼을 읽을 수 없습니다.</p>}
      {out && !busy && (
        <>
          <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl bg-[#f5f5f7] px-4 py-3 text-[13px]">
            <span>진동이 보이는 하한 ≈ <b className="tabular-nums">{out.bVis} T</b>
              <span className="text-[#86868b] tabular-nums"> (파일 {out.nVis}개 중앙값, 20–80%: {fmt(out.bVisSpread[0], 3)}–{fmt(out.bVisSpread[1], 3)} T)</span></span>
            <span className="text-[#6e6e73]">상한 {fmt(out.bHi, 3)} T · 지금 하한 {curMin} T</span>
          </div>
          <div className="mt-4 grid gap-5 xl:grid-cols-2">
            {out.wins.map(w => {
              const nRob = w.peaks.filter(p => p.robust).length;
              return (
                <div key={w.bMin} className="rounded-2xl border border-black/[0.06] p-4">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <div className="text-[15px] font-semibold tabular-nums">
                        B ≥ {w.bMin} T {w.tag && <span className="ml-1 rounded-full bg-[#0071e3]/10 px-2 py-0.5 text-[11px] font-medium text-[#0071e3]">{w.tag}</span>}
                      </div>
                      <div className="text-[11px] text-[#86868b] tabular-nums">
                        해상도 {fmt(w.res, 3)} T · 검증 가능 F ≥ {fmt(w.fLo, 3)} T · 봉우리 {w.peaks.length}개 중 유지 {nRob}개
                      </div>
                    </div>
                    {Math.abs(w.bMin - curMin) >= 0.05 && <Pill onClick={() => onApply(w.bMin)} title="사이드바 Min B 를 이 값으로 바꾸고 전체 파일을 다시 처리">이 하한 적용</Pill>}
                  </div>
                  <AngleContour rows={w.rows} fMax={fMax} normalize={normalize} height={300}
                    peaks={w.peaks.map((p, i) => ({
                      id: `${p.fileName}-${i}`, angle: p.angle, F: p.F, status: 'unchecked',
                      fill: p.robust ? '#34c759' : '#ff9500',
                      title: `${p.angle}° · ${fmt(p.F, 4)} T · 다른 창 ${p.eligible}개 중 ${p.seen}개에서 보임`,
                    }))}
                    legend={<>
                      <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full border border-white bg-[#34c759]" />모든 창에서 유지</span>
                      <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full border border-white bg-[#ff9500]" />이 창에서만</span>
                    </>} />
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
