// STEP 3 — is each picked peak a real oscillation? Band-pass it, inverse-FFT,
// lay the reconstruction over ΔR(1/B), and read the automatic verdict
// (src/lib/verify.ts). The user confirms or rejects; the verdict is advice.
import { useEffect, useMemo, useState } from 'react';
import { CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { checkPeak, reconstruct, toUniform, VERDICT_LABEL } from '../../lib/verify';
import { Metric, Pill, PrimaryPill, StepCard, STATUS_LABEL, STATUS_STYLE, TICK, TOOLTIP, fmt, seriesOf, type PickedPeak, type Series } from './shared';

const VERDICT_TONE: Record<string, string> = {
  real: 'text-[#248a3d]', doubtful: 'text-[#a05a00]', artifact: 'text-red-600', lowF: 'text-[#6e6e73]',
};

export function VerifyStep({ series, picked, setPicked }: {
  series: Series[];
  picked: PickedPeak[];
  setPicked: (fn: (prev: PickedPeak[]) => PickedPeak[]) => void;
}) {
  const list = useMemo(() => [...picked].sort((a, b) => a.col.localeCompare(b.col) || a.angle - b.angle || a.F - b.F), [picked]);

  // uniform 1/B grids are shared by every peak of the same file+column
  const uniforms = useMemo(() => {
    const m = new Map<string, ReturnType<typeof toUniform>>();
    for (const p of picked) {
      const k = `${p.fileName}::${p.col}`;
      if (m.has(k)) continue;
      const s = seriesOf(series, p);
      m.set(k, s ? toUniform(s.result.plotData.map(d => ({ invB: d.invB, val: d.deltaR }))) : null);
    }
    return m;
  }, [picked, series]);

  const checks = useMemo(() => {
    const m = new Map<string, ReturnType<typeof checkPeak>>();
    for (const p of picked) {
      const u = uniforms.get(`${p.fileName}::${p.col}`);
      if (u) m.set(p.id, checkPeak(u, p.F));
    }
    return m;
  }, [picked, uniforms]);

  // remember the automatic verdict on the peak itself (shown in steps 4–5)
  useEffect(() => {
    setPicked(prev => {
      let changed = false;
      const next = prev.map(p => {
        const v = checks.get(p.id)?.verdict;
        if (v && v !== p.auto) { changed = true; return { ...p, auto: v }; }
        return p;
      });
      return changed ? next : prev;
    });
  }, [checks, setPicked]);

  const [selId, setSelId] = useState<string | null>(null);
  const sel = list.find(p => p.id === selId) ?? list.find(p => p.status === 'unchecked') ?? list[0] ?? null;
  const c = sel ? checks.get(sel.id) : undefined;
  const recon = useMemo(() => {
    if (!sel) return [];
    const u = uniforms.get(`${sel.fileName}::${sel.col}`);
    return u ? reconstruct(u, sel.F) : [];
  }, [sel, uniforms]);

  const setStatus = (id: string, status: PickedPeak['status']) => {
    setPicked(prev => prev.map(p => (p.id === id ? { ...p, status } : p)));
    // move on to the next undecided peak
    const idx = list.findIndex(p => p.id === id);
    const next = list.slice(idx + 1).find(p => p.status === 'unchecked' && p.id !== id);
    if (next) setSelId(next.id);
  };

  const applyAuto = () => setPicked(prev => prev.map(p => {
    const v = checks.get(p.id)?.verdict;
    if (p.status !== 'unchecked' || !v) return p;
    return v === 'real' ? { ...p, status: 'confirmed' } : v === 'artifact' ? { ...p, status: 'rejected' } : p;
  }));

  const counts = { confirmed: 0, rejected: 0, unchecked: 0 } as Record<PickedPeak['status'], number>;
  for (const p of picked) counts[p.status]++;

  return (
    <StepCard
      step={3}
      title="IFFT — 고른 봉우리가 진짜 신호인지"
      lead={<>봉우리 주변만 남기고 역 FFT 한 성분(파랑)을 ΔR(1/B)(회색) 위에 겹칩니다. 진짜 진동은 1/B 전 구간에서 위상이 맞고(주파수 흔들림 &lt; 3%), 진폭이 B 와 함께 커집니다. 자동 판정은 참고용 — 확정·제외는 직접 고르세요.</>}
      actions={<>
        <span className="text-[12px] text-[#86868b] tabular-nums">확정 {counts.confirmed} · 제외 {counts.rejected} · 미확인 {counts.unchecked}</span>
        <Pill onClick={applyAuto} disabled={!counts.unchecked} title="미확인 봉우리 중 '진짜 진동'은 확정, '인공 신호 의심'은 제외">자동 판정대로 일괄 처리</Pill>
      </>}
    >
      {list.length === 0 ? (
        <p className="text-[14px] text-[#6e6e73]">2단계에서 봉우리를 먼저 고르세요.</p>
      ) : (
        <div className="grid gap-5 lg:grid-cols-[300px_1fr]">
          <div className="max-h-[560px] overflow-y-auto pr-1">
            {list.map(p => {
              const ck = checks.get(p.id);
              return (
                <button key={p.id} onClick={() => setSelId(p.id)}
                  className={`mb-1 flex w-full items-center justify-between gap-2 rounded-xl px-3 py-2 text-left transition-colors ${sel?.id === p.id ? 'bg-[#0071e3]/[0.08]' : 'hover:bg-black/[0.03]'}`}>
                  <div className="min-w-0">
                    <div className="text-[13px] font-medium tabular-nums">{p.angle}° · {fmt(p.F, 4)} T</div>
                    <div className="truncate text-[11px] text-[#86868b]">{p.col} · {p.fileName}</div>
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-0.5">
                    <span className={`rounded-full px-2 py-0.5 text-[11px] ${STATUS_STYLE[p.status]}`}>{STATUS_LABEL[p.status]}</span>
                    {ck && <span className={`text-[10px] ${VERDICT_TONE[ck.verdict]}`}>{VERDICT_LABEL[ck.verdict]}</span>}
                  </div>
                </button>
              );
            })}
          </div>

          {sel && (
            <div>
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-[17px] font-semibold tabular-nums">{sel.angle}° · F = {fmt(sel.F, 5)} T</div>
                  <div className="text-[12px] text-[#86868b]">{sel.col} · {sel.fileName}</div>
                </div>
                <div className="flex gap-2">
                  <PrimaryPill onClick={() => setStatus(sel.id, 'confirmed')}>확정</PrimaryPill>
                  <Pill onClick={() => setStatus(sel.id, 'rejected')}>제외</Pill>
                  {sel.status !== 'unchecked' && <Pill onClick={() => setStatus(sel.id, 'unchecked')}>되돌리기</Pill>}
                </div>
              </div>
              <div className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={recon} margin={{ top: 8, right: 16, bottom: 18, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5ea" />
                    <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} tick={TICK} tickFormatter={(v: number) => v.toFixed(3)}
                      label={{ value: '1/B (1/T)', position: 'insideBottom', offset: -10, fontSize: 11, fill: '#6e6e73' }} />
                    <YAxis tick={TICK} width={70} tickFormatter={(v: number) => fmt(v, 3)} />
                    <Tooltip contentStyle={TOOLTIP} formatter={(v) => fmt(Number(v))} labelFormatter={(v) => `1/B ${Number(v).toFixed(4)}`} />
                    <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11, paddingBottom: 6 }} />
                    <Line dataKey="val" name="ΔR (배경 제거)" stroke="#c7c7cc" dot={false} strokeWidth={1} isAnimationActive={false} />
                    <Line dataKey="comp" name={`IFFT 성분 (${fmt(sel.F, 4)} T)`} stroke="#0072B2" dot={false} strokeWidth={1.8} isAnimationActive={false} />
                    <Line dataKey="env" name="포락선" stroke="#D55E00" dot={false} strokeWidth={1} strokeDasharray="4 3" isAnimationActive={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              {c && (
                <>
                  <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
                    <Metric label="자동 판정" value={<span className={VERDICT_TONE[c.verdict]}>{VERDICT_LABEL[c.verdict]}</span>} />
                    <Metric label="창 안 진동 수" value={c.cycles.toFixed(1)} hint="6회 미만이면 1/B 주기성을 잴 수 없음" />
                    <Metric label="SNR" value={c.snr.toFixed(1)} />
                    <Metric label="주파수 흔들림" value={Number.isFinite(c.drift) ? `${(c.drift * 100).toFixed(1)}%` : '—'} hint="1/B 3구간 국소 주파수 차이 / F. 진짜 < 3%, B·시간 주기 잡음 ≳ 10%" />
                    <Metric label="포락선 기울기" value={c.envSlope.toFixed(1)} hint="ln|진폭| vs 1/B. 음수 = B 가 클수록 커짐 (Dingle)" />
                  </div>
                  <p className="mt-2 text-[12px] text-[#6e6e73]">{c.reasons.join(' · ')} · 이 성분이 ΔR 분산의 {(c.varFrac * 100).toFixed(0)}% 를 설명</p>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </StepCard>
  );
}
