// STEP 2 — find candidate frequencies across angles and pick them.
// The FFT of every file sits in one angle × F heatmap. Every spectral maximum
// whose SNR clears a threshold is picked automatically (and re-picked when the
// threshold moves); clicking the map adds one by hand, snapped to the
// strongest maximum within ±4 %. Steps 3–5 work on the resulting list.
import { useEffect, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { FFTHeatmap } from '../FFTHeatmap';
import { Pill, StepCard, STATUS_LABEL, STATUS_STYLE, fmt, type PickedPeak, type Series } from './shared';

function snapPeak(s: Series, F: number) {
  const fr = s.result.fftResults;
  let best: { frequency: number; amplitude: number } | null = null;
  for (const p of fr) if (Math.abs(p.frequency - F) <= 0.04 * F + 1 && (!best || p.amplitude > best.amplitude)) best = p;
  return best;
}

// FFT resolution 1/Δ(1/B) of the processed window
function resolutionOf(s: Series) {
  let lo = Infinity, hi = -Infinity;
  for (const d of s.result.plotData) if (Number.isFinite(d.invB)) { lo = Math.min(lo, d.invB); hi = Math.max(hi, d.invB); }
  return hi > lo ? 1 / (hi - lo) : NaN;
}

// Spectral maxima with amplitude / noise floor ≥ snrMin. The floor is the
// median of the spectrum between 4·resolution (below is leftover background)
// and fMax. Below minCycles·resolution there are too
// few oscillations in the window to verify, so those are skipped. Maxima
// closer than one resolution are one peak (zero-padding ripples).
export function thresholdPeaks(s: Series, opt: { snrMin: number; fMin: number; fMax: number; minCycles: number; maxPerFile: number }) {
  const fr = s.result.fftResults;
  const res = resolutionOf(s);
  if (!fr.length || !Number.isFinite(res)) return [];
  // only up to fMax: far above it the (smoothed, interpolated) spectrum is
  // nearly empty and a median there would make every bump look significant
  const floorAmps = fr.filter(p => p.frequency >= 4 * res && p.frequency <= opt.fMax).map(p => p.amplitude).sort((a, b) => a - b);
  const floor = floorAmps[floorAmps.length >> 1];
  if (!(floor > 0)) return [];
  const lo = Math.max(opt.fMin, opt.minCycles * res);
  const cand: { frequency: number; amplitude: number; snr: number }[] = [];
  for (let i = 1; i < fr.length - 1; i++) {
    const p = fr[i];
    if (p.frequency < lo || p.frequency > opt.fMax) continue;
    if (p.amplitude >= fr[i - 1].amplitude && p.amplitude > fr[i + 1].amplitude && p.amplitude / floor >= opt.snrMin)
      cand.push({ ...p, snr: p.amplitude / floor });
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

const peakId = (s: Series, F: number) => `${s.fileName}::${s.col}::${F.toFixed(2)}`;

// threshold settings outlive the step being unmounted (switching steps must
// not reset them and silently re-pick with the defaults)
// SNR 5 by default: generous on purpose, step 3 sorts real from spurious
const saved = { on: true, snrMin: 5, fMin: 0, maxPerFile: 3 };

export function PeakPickStep({ series, column, columns, onColumn, picked, setPicked }: {
  series: Series[];
  column: string;
  columns: string[];
  onColumn: (c: string) => void;
  picked: PickedPeak[];
  setPicked: (fn: (prev: PickedPeak[]) => PickedPeak[]) => void;
}) {
  const rows = useMemo(() => series.filter(s => s.col === column).sort((a, b) => a.blm - b.blm), [series, column]);
  const autoMax = useMemo(() => {
    const m = Math.max(0, ...rows.map(s => s.result.suggestedMaxFreq || 0));
    return Math.round(Math.min(3000, m > 0 ? m : 2000));
  }, [rows]);
  const [fMaxSel, setFMax] = useState<number | null>(null);
  const fMax = fMaxSel ?? autoMax;
  const [normRows, setNormRows] = useState(true);
  const [autoOn, setAutoOn] = useState(saved.on);
  const [snrMin, setSnrMin] = useState(saved.snrMin);
  const [fMin, setFMin] = useState(saved.fMin);
  const [maxPerFile, setMaxPerFile] = useState(saved.maxPerFile);
  useEffect(() => { Object.assign(saved, { on: autoOn, snrMin, fMin, maxPerFile }); }, [autoOn, snrMin, fMin, maxPerFile]);

  const add = (s: Series, F: number) => {
    const p = snapPeak(s, F);
    if (!p) return;
    setPicked(prev => prev.some(q => q.fileName === s.fileName && q.col === s.col && Math.abs(q.F - p.frequency) < 0.02 * p.frequency)
      ? prev
      : [...prev, { id: peakId(s, p.frequency), fileName: s.fileName, col: s.col, angle: s.blm, F: p.frequency, amp: p.amplitude, status: 'unchecked' }]);
  };

  // peaks above the threshold, every file of this channel
  const above = useMemo(() => rows.flatMap(s =>
    thresholdPeaks(s, { snrMin, fMin, fMax, minCycles: 6, maxPerFile }).map(p => ({
      id: peakId(s, p.frequency), fileName: s.fileName, col: s.col, angle: s.blm,
      F: p.frequency, amp: p.amplitude, snr: p.snr, status: 'unchecked' as const, src: 'threshold' as const,
    }))), [rows, snrMin, fMin, fMax, maxPerFile]);

  // Sync the threshold picks of this channel with `above`: drop the ones that
  // fell below (unless already confirmed/rejected in step 3), add the new
  // ones, never duplicate a hand-picked peak. Angles are refreshed too, since
  // step 1 may have recalibrated them.
  const syncAbove = () => setPicked(prev => {
    const want = new Map(above.map(p => [p.id, p]));
    const kept = prev.filter(q => q.col !== column || q.src !== 'threshold' || want.has(q.id) || q.status !== 'unchecked')
      .map(q => {
        const w = q.src === 'threshold' ? want.get(q.id) : undefined;
        return w && w.angle !== q.angle ? { ...q, angle: w.angle } : q;
      });
    const added = above.filter(p => !kept.some(q => q.fileName === p.fileName && q.col === p.col && Math.abs(q.F - p.F) < 0.02 * p.F));
    const next = [...kept, ...added];
    return next.length === prev.length && next.every((q, i) => q === prev[i]) ? prev : next;
  });
  useEffect(() => { if (autoOn) syncAbove(); }, [autoOn, above]); // eslint-disable-line react-hooks/exhaustive-deps

  const mine = picked.filter(p => p.col === column);
  const heatRows = rows.map(s => ({
    x: s.blm, fileName: s.fileName, fft: s.result.fftResults,
    peaks: mine.filter(p => p.fileName === s.fileName).map(p => ({ frequency: p.F, amplitude: p.amp })),
  }));

  return (
    <StepCard
      step={2}
      title="FFT — 각도별로 봉우리 고르기"
      lead={<>파일마다의 FFT 를 각도 순으로 쌓은 지도입니다. 각도를 따라 이어지는 밝은 줄기가 진동 주파수 가지입니다. SNR 이 임계값을 넘는 봉우리는 <b>자동으로</b> 목록에 들어가고(빨간 점), 임계값을 바꾸면 바로 다시 골라집니다. 놓친 봉우리는 <b>지도를 클릭</b>해 직접 추가하세요(±4% 안 가장 큰 봉우리). 다음 단계에서 하나씩 진짜인지 검증합니다.</>}
      actions={<>
        <select value={column} onChange={e => onColumn(e.target.value)} className="rounded-lg border border-black/10 px-2 py-1 text-[13px]">
          {columns.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <label className="flex items-center gap-1.5 text-[12px] text-[#6e6e73]">
          F 최대 <input type="number" value={fMax} step={100} onChange={e => setFMax(Math.max(50, Number(e.target.value) || 0))}
            className="w-20 rounded-lg border border-black/10 px-2 py-1 text-[13px] tabular-nums" />
        </label>
        <label className="flex items-center gap-1.5 text-[12px] text-[#6e6e73]">
          <input type="checkbox" checked={normRows} onChange={e => setNormRows(e.target.checked)} className="accent-[#0071e3]" /> 행마다 정규화
        </label>
      </>}
    >
      {rows.length === 0 ? (
        <p className="text-[14px] text-[#6e6e73]">이 채널로 처리된 파일이 없습니다. Process All Files 를 먼저 실행하세요.</p>
      ) : (
        <>
          <div className="pt-6">
            <FFTHeatmap rows={heatRows} fMax={fMax} normalizeRows={normRows} showPeaks rowLabel="θ (°)"
              onPick={(row, F) => { const s = rows.find(r => r.fileName === row.fileName); if (s) add(s, F); }} />
          </div>
          <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl bg-[#f5f5f7] px-4 py-3 text-[12px] text-[#6e6e73]">
            <label className="flex items-center gap-1.5 font-medium text-[#1d1d1f]">
              <input type="checkbox" checked={autoOn} onChange={e => setAutoOn(e.target.checked)} className="accent-[#0071e3]" /> 임계값 넘으면 자동 선택
            </label>
            <label className="flex items-center gap-1.5" title="봉우리 진폭 / 스펙트럼 중앙값(잡음 바닥). 3단계 '진짜 진동' 기준이 SNR ≥ 10">
              SNR ≥ <input type="number" value={snrMin} min={2} step={1} onChange={e => setSnrMin(Math.max(2, Number(e.target.value) || 0))}
                className="w-16 rounded-lg border border-black/10 bg-white px-2 py-1 text-[13px] tabular-nums" />
            </label>
            <label className="flex items-center gap-1.5" title="이보다 낮은 F 는 무시. 창 안 진동 6회 미만(검증 불가)도 자동으로 제외">
              F ≥ <input type="number" value={fMin} min={0} step={10} onChange={e => setFMin(Math.max(0, Number(e.target.value) || 0))}
                className="w-20 rounded-lg border border-black/10 bg-white px-2 py-1 text-[13px] tabular-nums" /> T
            </label>
            <label className="flex items-center gap-1.5">
              파일마다 최대
              <select value={maxPerFile} onChange={e => setMaxPerFile(Number(e.target.value))} className="rounded-lg border border-black/10 bg-white px-2 py-1 text-[12px]">
                {[1, 2, 3, 5, 8].map(n => <option key={n} value={n}>{n}개</option>)}
              </select>
            </label>
            <span className="tabular-nums">기준 통과 {above.length}개 · {new Set(above.map(p => p.fileName)).size}/{rows.length} 파일</span>
            {!autoOn && <Pill onClick={syncAbove} className="bg-white">이 기준으로 추가</Pill>}
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <span className="text-[13px] font-semibold">고른 봉우리 {mine.length}개</span>
            <span className="text-[12px] text-[#86868b]">자동 {mine.filter(p => p.src === 'threshold').length} · 직접 {mine.filter(p => p.src !== 'threshold').length}</span>
            {mine.length > 0 && <Pill onClick={() => { setAutoOn(false); setPicked(prev => prev.filter(p => p.col !== column)); }}>이 채널 목록 비우기</Pill>}
          </div>
          {mine.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {[...mine].sort((a, b) => a.angle - b.angle || a.F - b.F).map(p => (
                <span key={p.id} className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[12px] tabular-nums ${STATUS_STYLE[p.status]}`} title={`${p.fileName} · ${STATUS_LABEL[p.status]}${p.snr ? ` · SNR ${p.snr.toFixed(1)}` : ''} · ${p.src === 'threshold' ? '임계값 자동' : '직접 선택'}`}>
                  {p.src !== 'threshold' && <span className="h-1.5 w-1.5 rounded-full bg-[#0071e3]" />}
                  {p.angle}° · {fmt(p.F, 4)} T
                  <button onClick={() => setPicked(prev => prev.filter(q => q.id !== p.id))} className="opacity-50 hover:opacity-100"><X size={11} /></button>
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </StepCard>
  );
}
