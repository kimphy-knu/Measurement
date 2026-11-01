// STEP 1 — fix the angle axis from the torque itself.
//
// Rotator readouts (BLM steps) have no built-in degree scale or zero. The
// torque's 2-fold term τ ≈ A·cos(2θ + φ) does: one full period in BLM units is
// 180°, and τ = 0 marks the two symmetry axes. So: take τ at the highest field
// of every file, fit C + a·cos(2πx/P) + b·sin(2πx/P) over a scan of periods P
// (x = raw BLM), keep the best P → BLM per degree = P/180; then the zero
// crossings of the fitted 2-fold term give the axis to put at 0°.
import { useMemo, useState } from 'react';
import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
import { Metric, Pill, PrimaryPill, StepCard, TICK, TOOLTIP, fmt, type Series } from './shared';

type AngleSource = 'title' | 'blm' | 'order';

function fitPeriod(x: number[], y: number[], P: number) {
  // least squares for C + a cos + b sin (3×3 normal equations)
  const w = (2 * Math.PI) / P;
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const t = [0, 0, 0];
  for (let i = 0; i < x.length; i++) {
    const r = [1, Math.cos(w * x[i]), Math.sin(w * x[i])];
    for (let a = 0; a < 3; a++) { t[a] += r[a] * y[i]; for (let b = 0; b < 3; b++) S[a][b] += r[a] * r[b]; }
  }
  // solve via Cramer (3×3)
  const det = (m: number[][]) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det(S);
  if (Math.abs(D) < 1e-300) return null;
  const col = (k: number) => S.map((row, i) => row.map((v, j) => (j === k ? t[i] : v)));
  const [C, a, b] = [0, 1, 2].map(k => det(col(k)) / D);
  const my = y.reduce((s, v) => s + v, 0) / y.length;
  let ssr = 0, sst = 0;
  for (let i = 0; i < x.length; i++) {
    const f = C + a * Math.cos(w * x[i]) + b * Math.sin(w * x[i]);
    ssr += (y[i] - f) ** 2; sst += (y[i] - my) ** 2;
  }
  return { C, a, b, r2: sst > 0 ? 1 - ssr / sst : 0 };
}

export function AngleStep({
  series, column, columns, onColumn,
  angleSource, onAngleSource, blmPerDegree, blmOffset, onApply,
}: {
  series: Series[];
  column: string;
  columns: string[];
  onColumn: (c: string) => void;
  angleSource: AngleSource;
  onAngleSource: (s: AngleSource) => void;
  blmPerDegree: number;
  blmOffset: number;
  onApply: (perDegree: number, offset: number) => void;
}) {
  const [topFrac, setTopFrac] = useState(0.05);

  // τ at the top of each sweep: mean raw signal over the highest `topFrac` of |B|
  const pts = useMemo(() => series.filter(s => s.col === column).map(s => {
    const d = s.result.plotData;
    const bmax = Math.max(...d.map(p => Math.abs(p.b)));
    const top = d.filter(p => Math.abs(p.b) >= bmax * (1 - topFrac));
    const tau = top.reduce((a, p) => a + p.r, 0) / Math.max(1, top.length);
    const raw = (s.blm - blmOffset) * blmPerDegree; // undo the current scale → rotator units
    return { angle: s.blm, raw, tau, bmax, file: s.fileName };
  }).filter(p => Number.isFinite(p.tau)).sort((a, b) => a.raw - b.raw), [series, column, topFrac, blmOffset, blmPerDegree]);

  const fit = useMemo(() => {
    if (pts.length < 5) return null;
    const x = pts.map(p => p.raw), y = pts.map(p => p.tau);
    const span = x[x.length - 1] - x[0];
    if (!(span > 0)) return null;
    let best: { P: number; C: number; a: number; b: number; r2: number } | null = null;
    // periods from 1/2 of the span (two 2-fold periods visible) to 6× (barely curved)
    for (let lp = Math.log(span / 2); lp <= Math.log(span * 6); lp += 0.004) {
      const P = Math.exp(lp);
      const f = fitPeriod(x, y, P);
      if (f && (!best || f.r2 > best.r2)) best = { P, ...f };
    }
    if (!best) return null;
    const perDeg = best.P / 180;
    // in the suggested angle θ = raw/perDeg: τ = C + A cos(2θ + φ)
    const A = Math.hypot(best.a, best.b);
    const phi = (Math.atan2(-best.b, best.a) * 180) / Math.PI;
    const norm = (t: number) => ((t % 180) + 270) % 180 - 90;           // → (−90, 90]
    const axisStable = norm((90 - phi) / 2);   // τ = 0 with dτ/dθ < 0
    const axisOther = norm((-90 - phi) / 2);   // τ = 0 with dτ/dθ > 0
    return { ...best, perDeg, A, phi, axisStable, axisOther, coversHalf: span >= best.P / 2 };
  }, [pts]);

  // angles under the SUGGESTED calibration with the stable axis at 0°
  const preview = fit ? pts.map(p => ({ ...p, theta: p.raw / fit.perDeg - fit.axisStable })) : [];
  const curve = fit ? Array.from({ length: 241 }, (_, i) => {
    const th = -180 + (i * 360) / 240;
    const raw = (th + fit.axisStable) * fit.perDeg;
    const w = (2 * Math.PI) / fit.P;
    return { theta: th, fit: fit.C + fit.a * Math.cos(w * raw) + fit.b * Math.sin(w * raw) };
  }) : [];

  return (
    <StepCard
      step={1}
      title="각도 축 정하기 — 토크로"
      lead={<>회전기 BLM 값에는 도 단위와 0°가 없습니다. 최고 자기장에서의 토크 τ(BLM)를 2회 대칭 τ ≈ A·cos(2θ+φ)로 맞추면 한 주기 = 180° 로 <b>BLM/°</b> 를, τ = 0 교차점으로 <b>대칭축</b>을 얻습니다. 이 축을 0° 로 놓고 다음 단계로 갑니다.</>}
      actions={<>
        <span className="text-[12px] text-[#86868b]">각도 출처</span>
        <div className="flex bg-black/5 rounded-full p-0.5">
          {([['blm', 'BLM 컬럼'], ['title', '파일명 숫자'], ['order', '파일 순서']] as const).map(([v, l]) => (
            <button key={v} onClick={() => onAngleSource(v)}
              className={`px-3 py-1 text-[12px] rounded-full ${angleSource === v ? 'bg-white font-medium shadow-[0_1px_3px_rgba(0,0,0,.12)]' : 'text-[#1d1d1f]/60'}`}>{l}</button>
          ))}
        </div>
      </>}
    >
      {series.length === 0 ? (
        <p className="text-[14px] text-[#6e6e73]">먼저 파일을 올리고 <b>Process All Files</b> 를 눌러 주세요. 처리된 신호로 각도를 맞춥니다.</p>
      ) : (
        <>
          <div className="mb-4 flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-[13px]">
              <span className="text-[#6e6e73]">토크 채널</span>
              <select value={column} onChange={e => onColumn(e.target.value)} className="rounded-lg border border-black/10 px-2 py-1 text-[13px]">
                {columns.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2 text-[13px]">
              <span className="text-[#6e6e73]">최고 자기장 구간</span>
              <select value={topFrac} onChange={e => setTopFrac(Number(e.target.value))} className="rounded-lg border border-black/10 px-2 py-1 text-[13px]">
                {[0.02, 0.05, 0.1, 0.2].map(f => <option key={f} value={f}>상위 {f * 100}%</option>)}
              </select>
            </label>
            <span className="text-[12px] text-[#86868b]">파일 {pts.length}개 · 현재 BLM/° = {fmt(blmPerDegree)}, 오프셋 = {fmt(blmOffset)}°</span>
          </div>

          {angleSource !== 'blm' && (
            <p className="mb-4 rounded-xl bg-[#fff8e6] px-4 py-2.5 text-[13px] text-[#8a5a00]">
              각도 출처가 "{angleSource === 'title' ? '파일명 숫자' : '파일 순서'}" 입니다. NHMFL 시퀀스 파일은 파일명 숫자가 순번이라 각도가 아닙니다 — 회전기 값을 쓰려면 <b>BLM 컬럼</b>을 고르세요.
            </p>
          )}

          <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
            <div className="h-[340px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart margin={{ top: 10, right: 16, bottom: 18, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e5ea" />
                  <XAxis dataKey="theta" type="number" domain={['dataMin', 'dataMax']} tick={TICK}
                    label={{ value: fit ? 'θ (제안 보정, 안정축 = 0°)' : 'θ', position: 'insideBottom', offset: -10, fontSize: 11, fill: '#6e6e73' }} />
                  <YAxis tick={TICK} width={70} tickFormatter={(v: number) => fmt(v, 3)} label={{ value: `τ @ Bmax (${column})`, angle: -90, position: 'insideLeft', fontSize: 11, fill: '#6e6e73' }} />
                  <Tooltip contentStyle={TOOLTIP} formatter={(v) => fmt(Number(v))} />
                  {fit && <Line data={curve} dataKey="fit" stroke="#009E73" dot={false} strokeWidth={2} strokeDasharray="5 4" isAnimationActive={false} name="2회 대칭 피팅" />}
                  <Scatter data={fit ? preview.map(p => ({ theta: p.theta, tau: p.tau })) : pts.map(p => ({ theta: p.angle, tau: p.tau }))}
                    dataKey="tau" fill="#0072B2" isAnimationActive={false} name="τ" />
                  {fit && <ReferenceLine x={0} stroke="#D55E00" strokeDasharray="4 3" label={{ value: '안정축', fontSize: 10, fill: '#D55E00', position: 'top' }} />}
                  {fit && <ReferenceLine x={fit.axisOther - fit.axisStable} stroke="#8e8e93" strokeDasharray="4 3" />}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-2">
              {fit ? (
                <>
                  <Metric label="BLM / ° (2회 대칭 주기 / 180)" value={fmt(fit.perDeg)} />
                  <Metric label="안정축 위치 (제안 보정 전 θ)" value={`${fit.axisStable.toFixed(1)}°`} hint="τ = 0 이고 dτ/dθ < 0 — 자기 이방성의 안정 방향" />
                  <Metric label="2회 대칭 진폭 · 피팅 R²" value={`${fmt(fit.A, 3)} · ${fit.r2.toFixed(3)}`} />
                  {!fit.coversHalf && <p className="text-[12px] text-[#a05a00]">회전 범위가 반 주기(90°)에 못 미칩니다 — 주기 추정이 불확실하니 BLM/° 는 아는 값을 직접 넣는 게 낫습니다.</p>}
                  <PrimaryPill className="w-full" onClick={() => onApply(fit.perDeg, -fit.axisStable)}>이 보정 적용 (안정축 = 0°)</PrimaryPill>
                  {/* θ = raw/perDeg + offset; the axis sits at raw = axisStable·fit.perDeg */}
                  <Pill className="w-full" onClick={() => onApply(blmPerDegree, -(fit.axisStable * fit.perDeg) / blmPerDegree)}>BLM/° 는 그대로, 0° 만 맞추기</Pill>
                </>
              ) : (
                <p className="text-[13px] text-[#6e6e73]">파일이 5개 이상 있어야 2회 대칭을 맞출 수 있습니다.</p>
              )}
            </div>
          </div>
        </>
      )}
    </StepCard>
  );
}
