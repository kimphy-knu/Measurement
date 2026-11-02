// STEP 2 — find candidate frequencies across angles and pick them.
// The FFT of every file sits in one θ × F contour map with the picked peaks
// on it. Every spectral maximum that clears a relative-height threshold is
// picked automatically (and re-picked when the threshold moves); clicking the
// map adds one by hand, snapped to the strongest maximum within ±4 %. Steps
// 3–5 work on the resulting list.
import { useEffect, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { AngleContour } from '../AngleContour';
import { fLoOf, thresholdPeaks } from '../../lib/peaks';
import { WindowScan, type WindowScanProps } from './WindowScan';
import { Pill, StepCard, STATUS_LABEL, STATUS_STYLE, fmt, type PickedPeak, type Series } from './shared';

function snapPeak(s: Series, F: number) {
  const fr = s.result.fftResults;
  let best: { frequency: number; amplitude: number } | null = null;
  for (const p of fr) if (Math.abs(p.frequency - F) <= 0.04 * F + 1 && (!best || p.amplitude > best.amplitude)) best = p;
  return best;
}

const peakId = (s: Series, F: number) => `${s.fileName}::${s.col}::${F.toFixed(2)}`;

// threshold settings outlive the step being unmounted (switching steps must
// not reset them and silently re-pick with the defaults)
// 30 % by default: generous on purpose, step 3 sorts real from spurious
const saved = { on: true, relPct: 30, fMin: 0, maxPerFile: 3 };

export function PeakPickStep({ series, column, columns, onColumn, picked, setPicked, scan }: {
  series: Series[];
  // B-window scan hooks from the app (re-run the pipeline on another window)
  scan?: Pick<WindowScanProps, 'reprocess' | 'field' | 'curMin' | 'curMax' | 'onApply'>;
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
  const [relPct, setRelPct] = useState(saved.relPct);
  const [fMin, setFMin] = useState(saved.fMin);
  const [maxPerFile, setMaxPerFile] = useState(saved.maxPerFile);
  useEffect(() => { Object.assign(saved, { on: autoOn, relPct, fMin, maxPerFile }); }, [autoOn, relPct, fMin, maxPerFile]);

  const add = (s: Series, F: number) => {
    const p = snapPeak(s, F);
    if (!p) return;
    setPicked(prev => prev.some(q => q.fileName === s.fileName && q.col === s.col && Math.abs(q.F - p.frequency) < 0.02 * p.frequency)
      ? prev
      : [...prev, { id: peakId(s, p.frequency), fileName: s.fileName, col: s.col, angle: s.blm, F: p.frequency, amp: p.amplitude, status: 'unchecked' }]);
  };

  // peaks above the threshold, every file of this channel
  const above = useMemo(() => rows.flatMap(s =>
    thresholdPeaks(s.result, { relMin: relPct / 100, fMin, fMax, maxPerFile }).map(p => ({
      id: peakId(s, p.frequency), fileName: s.fileName, col: s.col, angle: s.blm,
      F: p.frequency, amp: p.amplitude, rel: p.rel, status: 'unchecked' as const, src: 'threshold' as const,
    }))), [rows, relPct, fMin, fMax, maxPerFile]);

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
  const contourRows = useMemo(() => rows.map(s => ({ angle: s.blm, fileName: s.fileName, fft: s.result.fftResults, fLo: fLoOf(s.result) })), [rows]);
  const peakTitle = (p: PickedPeak) =>
    `${p.angle}° · ${fmt(p.F, 4)} T · ${STATUS_LABEL[p.status]}${p.rel ? ` · 높이 ${(p.rel * 100).toFixed(0)}%` : ''} · ${p.src === 'threshold' ? '임계값 자동' : '직접 선택'}\n${p.fileName}`;

  return (
    <StepCard
      step={2}
      title="FFT — 각도별로 봉우리 고르기"
      lead={<>각도(가로, 실제 간격) × 주파수(세로) FFT 등고선 지도입니다. 각도를 따라 이어지는 진한 줄기가 진동 주파수 가지이고, 고른 봉우리가 같은 축 위에 점으로 찍혀 F(θ) 가 바로 보입니다. 각도마다 가장 큰 봉우리 대비 <b>높이가 기준을 넘는 봉우리는 자동으로</b> 골라지고, 기준을 바꾸면 바로 다시 골라집니다. 놓친 봉우리는 <b>지도를 클릭</b>해 추가하세요(그 각도 파일들에서 ±4% 안 가장 큰 봉우리).</>}
      actions={<>
        <select value={column} onChange={e => onColumn(e.target.value)} className="rounded-lg border border-black/10 px-2 py-1 text-[13px]">
          {columns.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <label className="flex items-center gap-1.5 text-[12px] text-[#6e6e73]">
          F 최대 <input type="number" value={fMax} step={100} onChange={e => setFMax(Math.max(50, Number(e.target.value) || 0))}
            className="w-20 rounded-lg border border-black/10 px-2 py-1 text-[13px] tabular-nums" />
        </label>
        <label className="flex items-center gap-1.5 text-[12px] text-[#6e6e73]">
          <input type="checkbox" checked={normRows} onChange={e => setNormRows(e.target.checked)} className="accent-[#0071e3]" /> 각도마다 정규화
        </label>
      </>}
    >
      {rows.length === 0 ? (
        <p className="text-[14px] text-[#6e6e73]">이 채널로 처리된 파일이 없습니다. Process All Files 를 먼저 실행하세요.</p>
      ) : (
        <>
          <AngleContour rows={contourRows} fMax={fMax} normalize={normRows}
            peaks={mine.map(p => ({ id: p.id, angle: p.angle, F: p.F, status: p.status, manual: p.src !== 'threshold', title: peakTitle(p) }))}
            onPick={(files, F) => { for (const s of rows) if (files.includes(s.fileName)) add(s, F); }} />
          <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl bg-[#f5f5f7] px-4 py-3 text-[12px] text-[#6e6e73]">
            <label className="flex items-center gap-1.5 font-medium text-[#1d1d1f]">
              <input type="checkbox" checked={autoOn} onChange={e => setAutoOn(e.target.checked)} className="accent-[#0071e3]" /> 임계값 넘으면 자동 선택
            </label>
            <label className="flex items-center gap-1.5" title="같은 파일에서 가장 큰 봉우리 대비 높이 (사용 가능 범위 안). 낮출수록 약한 가지까지 골라짐">
              높이 ≥ <input type="range" min={5} max={95} step={5} value={relPct} onChange={e => setRelPct(Number(e.target.value))} className="w-28 accent-[#0071e3]" />
              <span className="w-9 tabular-nums text-[#1d1d1f]">{relPct}%</span>
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
                <span key={p.id} className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[12px] tabular-nums ${STATUS_STYLE[p.status]}`} title={peakTitle(p)}>
                  {p.src !== 'threshold' && <span className="h-1.5 w-1.5 rounded-full bg-[#0071e3]" />}
                  {p.angle}° · {fmt(p.F, 4)} T
                  <button onClick={() => setPicked(prev => prev.filter(q => q.id !== p.id))} className="opacity-50 hover:opacity-100"><X size={11} /></button>
                </span>
              ))}
            </div>
          )}
          {scan && <WindowScan rows={rows} relMin={relPct / 100} fMax={fMax} normalize={normRows} {...scan} />}
        </>
      )}
    </StepCard>
  );
}
