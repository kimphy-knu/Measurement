// STEP 4 — does a verified frequency evolve continuously with angle? Confirmed
// peaks are grouped into branches (auto-tracked, editable), and each branch is
// fitted to F(θ) = F₀ / |cos(θ − θ₀)| — the 2D / cylindrical Fermi-surface
// form. A real extremal orbit traces a smooth curve; a stray peak doesn't.
import { useMemo, useState } from 'react';
import { CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
import { BRANCH_COLORS, Metric, Pill, StepCard, TICK, TOOLTIP, fmt, type PickedPeak } from './shared';

function fitInvCos(pts: { angle: number; F: number }[]) {
  if (pts.length < 3) return null;
  let best: { F0: number; th0: number; rms: number } | null = null;
  for (let th0 = -90; th0 <= 90; th0 += 0.25) {
    const w = pts.map(p => 1 / Math.abs(Math.cos(((p.angle - th0) * Math.PI) / 180)));
    if (w.some(v => !Number.isFinite(v) || v > 20)) continue; // within ~87° of θ₀ only
    const F0 = pts.reduce((s, p, i) => s + p.F * w[i], 0) / w.reduce((s, v) => s + v * v, 0);
    const rms = Math.sqrt(pts.reduce((s, p, i) => s + (p.F - F0 * w[i]) ** 2, 0) / pts.length);
    if (!best || rms < best.rms) best = { F0, th0, rms };
  }
  return best;
}

// Link peaks angle by angle: each joins the branch whose last point is within
// `tol` (relative) in F, nearest first; otherwise it starts a new branch. Two
// peaks of the same FILE never share a branch, but repeated sweeps at the same
// angle (separate files) do.
function autoBranches(peaks: PickedPeak[], tol: number): Map<string, string> {
  const sorted = [...peaks].sort((a, b) => a.angle - b.angle || a.F - b.F);
  const branches: { name: string; last: PickedPeak; files: Set<string> }[] = [];
  const out = new Map<string, string>();
  for (const p of sorted) {
    let best: (typeof branches)[number] | null = null;
    for (const b of branches) {
      if (b.files.has(p.fileName)) continue;
      const rel = Math.abs(b.last.F - p.F) / p.F;
      if (rel <= tol && (!best || Math.abs(best.last.F - p.F) > Math.abs(b.last.F - p.F))) best = b;
    }
    if (best) { best.last = p; best.files.add(p.fileName); out.set(p.id, best.name); }
    else {
      // α, β, γ, … (Greek lower case from U+03B1)
      const name = String.fromCharCode(0x3b1 + (branches.length % 24));
      branches.push({ name, last: p, files: new Set([p.fileName]) });
      out.set(p.id, name);
    }
  }
  return out;
}

export function EvolutionStep({ picked, setPicked }: {
  picked: PickedPeak[];
  setPicked: (fn: (prev: PickedPeak[]) => PickedPeak[]) => void;
}) {
  const [includeUnchecked, setIncludeUnchecked] = useState(false);
  const [tol, setTol] = useState(0.12);
  const use = picked.filter(p => p.status === 'confirmed' || (includeUnchecked && p.status === 'unchecked'));
  const names = useMemo(() => [...new Set(use.map(p => p.branch ?? '—'))].sort(), [use]);
  const colorOf = (n: string) => BRANCH_COLORS[Math.max(0, names.indexOf(n)) % BRANCH_COLORS.length];

  const fits = useMemo(() => names.filter(n => n !== '—').map(n => {
    const pts = use.filter(p => p.branch === n).map(p => ({ angle: p.angle, F: p.F }));
    return { name: n, n: pts.length, fit: fitInvCos(pts), span: pts.length ? [Math.min(...pts.map(p => p.angle)), Math.max(...pts.map(p => p.angle))] : [0, 0] };
  }), [names, use]);

  const runAuto = () => {
    const m = autoBranches(use, tol);
    setPicked(prev => prev.map(p => (m.has(p.id) ? { ...p, branch: m.get(p.id) } : p)));
  };
  const setBranch = (id: string, b: string) => setPicked(prev => prev.map(p => (p.id === id ? { ...p, branch: b || undefined } : p)));

  return (
    <StepCard
      step={4}
      title="각도 의존성 — 주파수 가지 추적"
      lead={<>확정한 봉우리를 각도에 따라 이어 가지로 묶고, 가지마다 F(θ) = F₀/|cos(θ−θ₀)| 를 맞춥니다. 각도를 따라 매끄럽게 이어지는 가지가 실제 페르미면 궤도이고, 혼자 튀는 점은 다시 3단계로 가서 확인할 대상입니다.</>}
      actions={<>
        <label className="flex items-center gap-1.5 text-[12px] text-[#6e6e73]">
          <input type="checkbox" checked={includeUnchecked} onChange={e => setIncludeUnchecked(e.target.checked)} className="accent-[#0071e3]" /> 미확인도 포함
        </label>
        <label className="flex items-center gap-1.5 text-[12px] text-[#6e6e73]">
          이웃 각도 F 차이 ≤
          <select value={tol} onChange={e => setTol(Number(e.target.value))} className="rounded-lg border border-black/10 px-1.5 py-0.5 text-[12px]">
            {[0.05, 0.08, 0.12, 0.2].map(t => <option key={t} value={t}>{t * 100}%</option>)}
          </select>
        </label>
        <Pill onClick={runAuto} disabled={use.length < 2}>가지 자동 나누기</Pill>
      </>}
    >
      {use.length > 0 && Math.max(...use.map(p => p.angle)) - Math.min(...use.map(p => p.angle)) > 360 && (
        <p className="mb-4 rounded-xl bg-[#fff8e6] px-4 py-2.5 text-[13px] text-[#8a5a00]">
          각도 범위가 360° 를 넘습니다 — 아직 회전기 원시값(BLM)인 것 같습니다. 1단계에서 BLM/° 보정을 적용해야 1/cos 피팅이 의미가 있습니다.
        </p>
      )}
      {use.length === 0 ? (
        <p className="text-[14px] text-[#6e6e73]">3단계에서 봉우리를 확정하면 여기 나타납니다.</p>
      ) : (
        <div className="grid gap-5 lg:grid-cols-[1fr_340px]">
          <div className="h-[380px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart margin={{ top: 8, right: 16, bottom: 18, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e5ea" />
                <XAxis dataKey="angle" type="number" domain={['dataMin - 5', 'dataMax + 5']} tick={TICK}
                  label={{ value: 'θ (°)', position: 'insideBottom', offset: -10, fontSize: 11, fill: '#6e6e73' }} allowDuplicatedCategory={false} />
                <YAxis dataKey="F" type="number" domain={['auto', 'auto']} tick={TICK} width={60}
                  label={{ value: 'F (T)', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#6e6e73' }} />
                <Tooltip contentStyle={TOOLTIP} formatter={(v) => fmt(Number(v))} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11, paddingBottom: 6 }} />
                {names.map(n => (
                  <Scatter key={n} name={n === '—' ? '가지 없음' : n} data={use.filter(p => (p.branch ?? '—') === n).map(p => ({ angle: p.angle, F: p.F }))}
                    fill={n === '—' ? '#c7c7cc' : colorOf(n)} isAnimationActive={false} />
                ))}
                {fits.filter(f => f.fit).map(f => (
                  <Line key={`fit-${f.name}`} name={`${f.name} 피팅`} dataKey="F" stroke={colorOf(f.name)} strokeDasharray="5 4" dot={false} strokeWidth={1.5} isAnimationActive={false} legendType="none"
                    data={Array.from({ length: 120 }, (_, i) => {
                      const a = f.span[0] + ((f.span[1] - f.span[0]) * i) / 119;
                      const c = Math.abs(Math.cos(((a - f.fit!.th0) * Math.PI) / 180));
                      // break the curve near the 1/cos pole instead of drawing a spike
                      return { angle: a, F: c >= 0.15 ? f.fit!.F0 / c : null };
                    })} />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2">
            {fits.length === 0 && <p className="text-[13px] text-[#6e6e73]">"가지 자동 나누기"를 누르거나 아래 표에서 가지 이름을 직접 정하세요.</p>}
            {fits.map(f => (
              <div key={f.name} className="rounded-xl bg-[#f5f5f7] px-3.5 py-2.5">
                <div className="flex items-center gap-2 text-[13px] font-semibold"><span className="h-2.5 w-2.5 rounded-full" style={{ background: colorOf(f.name) }} />{f.name} <span className="font-normal text-[#86868b]">· 점 {f.n}개 · {f.span[0]}°–{f.span[1]}°</span></div>
                {f.fit ? (
                  <div className="mt-1 grid grid-cols-3 gap-2 text-[12px] tabular-nums">
                    <span>F₀ <b>{fmt(f.fit.F0, 4)}</b> T</span><span>θ₀ <b>{f.fit.th0.toFixed(1)}</b>°</span><span>rms <b>{fmt(f.fit.rms, 3)}</b></span>
                  </div>
                ) : <div className="mt-1 text-[12px] text-[#86868b]">점 3개 이상이면 1/cos 피팅</div>}
              </div>
            ))}
            <Metric label="가지로 묶인 봉우리" value={`${use.filter(p => p.branch).length} / ${use.length}`} />
          </div>
        </div>
      )}

      {use.length > 0 && (
        <div className="mt-5 max-h-64 overflow-y-auto">
          <table className="w-full text-[12px] tabular-nums">
            <thead><tr className="text-left text-[11px] text-[#86868b]"><th className="py-1 pr-3 font-normal">θ</th><th className="pr-3 font-normal">F (T)</th><th className="pr-3 font-normal">채널</th><th className="font-normal">가지</th></tr></thead>
            <tbody>
              {[...use].sort((a, b) => a.angle - b.angle || a.F - b.F).map(p => (
                <tr key={p.id} className="border-t border-black/[0.06]">
                  <td className="py-1 pr-3">{p.angle}°</td><td className="pr-3">{fmt(p.F, 5)}</td><td className="pr-3 text-[#6e6e73]">{p.col}</td>
                  <td>
                    <input value={p.branch ?? ''} onChange={e => setBranch(p.id, e.target.value.trim())} placeholder="—"
                      className="w-20 rounded-md border border-black/10 px-1.5 py-0.5 text-[12px]" style={{ color: p.branch ? colorOf(p.branch) : undefined }} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </StepCard>
  );
}
