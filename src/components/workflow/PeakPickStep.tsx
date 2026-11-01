// STEP 2 — find candidate frequencies across angles and pick them.
// The FFT of every file sits in one angle × F heatmap; clicking it snaps to the
// strongest spectral maximum within ±4 % and adds that peak to the list that
// steps 3–5 work on.
import { useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { FFTHeatmap } from '../FFTHeatmap';
import { Pill, StepCard, STATUS_LABEL, STATUS_STYLE, fmt, type PickedPeak, type Series } from './shared';

function snapPeak(s: Series, F: number) {
  const fr = s.result.fftResults;
  let best: { frequency: number; amplitude: number } | null = null;
  for (const p of fr) if (Math.abs(p.frequency - F) <= 0.04 * F + 1 && (!best || p.amplitude > best.amplitude)) best = p;
  return best;
}

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
  const [autoN, setAutoN] = useState(1);

  const add = (s: Series, F: number) => {
    const p = snapPeak(s, F);
    if (!p) return;
    setPicked(prev => prev.some(q => q.fileName === s.fileName && q.col === s.col && Math.abs(q.F - p.frequency) < 0.02 * p.frequency)
      ? prev
      : [...prev, { id: `${s.fileName}::${s.col}::${p.frequency.toFixed(2)}`, fileName: s.fileName, col: s.col, angle: s.blm, F: p.frequency, amp: p.amplitude, status: 'unchecked' }]);
  };

  const addTopPeaks = () => {
    for (const s of rows) for (const p of s.result.detectedPeaks.filter(q => q.frequency <= fMax).slice(0, autoN)) add(s, p.frequency);
  };

  const mine = picked.filter(p => p.col === column);
  const heatRows = rows.map(s => ({
    x: s.blm, fileName: s.fileName, fft: s.result.fftResults,
    peaks: mine.filter(p => p.fileName === s.fileName).map(p => ({ frequency: p.F, amplitude: p.amp })),
  }));

  return (
    <StepCard
      step={2}
      title="FFT — 각도별로 봉우리 고르기"
      lead={<>파일마다의 FFT 를 각도 순으로 쌓은 지도입니다. 각도를 따라 이어지는 밝은 줄기가 진동 주파수 가지입니다. <b>지도를 클릭</b>하면 그 근처(±4%)의 가장 큰 봉우리가 목록에 들어가고, 빨간 점으로 표시됩니다. 다음 단계에서 하나씩 진짜인지 검증합니다.</>}
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
          <div className="mt-5 flex flex-wrap items-center gap-3">
            <span className="text-[13px] font-semibold">고른 봉우리 {mine.length}개</span>
            <span className="text-[12px] text-[#86868b]">빠르게 시작하려면 →</span>
            <select value={autoN} onChange={e => setAutoN(Number(e.target.value))} className="rounded-lg border border-black/10 px-2 py-1 text-[12px]">
              {[1, 2, 3].map(n => <option key={n} value={n}>파일마다 상위 {n}개</option>)}
            </select>
            <Pill onClick={addTopPeaks}>자동으로 추가</Pill>
            {mine.length > 0 && <Pill onClick={() => setPicked(prev => prev.filter(p => p.col !== column))}>이 채널 목록 비우기</Pill>}
          </div>
          {mine.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {[...mine].sort((a, b) => a.angle - b.angle || a.F - b.F).map(p => (
                <span key={p.id} className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[12px] tabular-nums ${STATUS_STYLE[p.status]}`} title={`${p.fileName} · ${STATUS_LABEL[p.status]}`}>
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
