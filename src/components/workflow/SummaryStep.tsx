// STEP 5 — everything the earlier steps decided, in one table: what was
// picked, what the automatic check said, what was confirmed, which branch.
import { Download } from 'lucide-react';
import { VERDICT_LABEL } from '../../lib/verify';
import { saveText } from '../../lib/download';
import { BRANCH_COLORS, Metric, Pill, StepCard, STATUS_LABEL, STATUS_STYLE, fmt, type PickedPeak } from './shared';

export function SummaryStep({ picked, sampleId, angleNote, onGo }: {
  picked: PickedPeak[];
  sampleId: string;
  angleNote: string;
  onGo: (step: number) => void;
}) {
  const conf = picked.filter(p => p.status === 'confirmed');
  const branches = [...new Set(conf.map(p => p.branch).filter(Boolean))] as string[];
  const sorted = [...picked].sort((a, b) => (a.branch ?? '~').localeCompare(b.branch ?? '~') || a.angle - b.angle || a.F - b.F);

  const exportCsv = () => {
    const head = ['status', 'auto_verdict', 'branch', 'angle_deg', 'F_T', 'amplitude', 'column', 'file'];
    const rows = sorted.map(p => [p.status, p.auto ?? '', p.branch ?? '', p.angle, p.F.toFixed(3), p.amp, p.col, p.fileName].join(','));
    saveText([`# ${sampleId} · ${angleNote}`, head.join(','), ...rows].join('\n'), `${sampleId}_peaks.csv`, 'text/csv');
  };

  return (
    <StepCard
      step={5}
      title="정리"
      lead={<>지금까지 고르고 검증한 봉우리 전체입니다. 이 목록은 <b>Save</b> 로 세션 파일에 같이 저장되고, 불러오면 그대로 이어서 작업할 수 있습니다.</>}
      actions={<Pill onClick={exportCsv} disabled={!picked.length}><span className="inline-flex items-center gap-1.5"><Download size={13} /> CSV</span></Pill>}
    >
      <div className="mb-5 grid grid-cols-2 gap-2 sm:grid-cols-5">
        <Metric label="고른 봉우리" value={picked.length} />
        <Metric label="확정" value={conf.length} />
        <Metric label="제외" value={picked.filter(p => p.status === 'rejected').length} />
        <Metric label="미확인" value={picked.filter(p => p.status === 'unchecked').length} />
        <Metric label="가지" value={branches.length ? branches.join(', ') : '—'} />
      </div>
      <p className="mb-4 text-[12px] text-[#86868b]">각도 기준: {angleNote}</p>

      {picked.length === 0 ? (
        <p className="text-[14px] text-[#6e6e73]">아직 고른 봉우리가 없습니다. <button className="text-[#0066cc]" onClick={() => onGo(2)}>2단계로 →</button></p>
      ) : (
        <div className="max-h-[480px] overflow-y-auto">
          <table className="w-full text-[12px] tabular-nums">
            <thead className="sticky top-0 bg-white"><tr className="text-left text-[11px] text-[#86868b]">
              <th className="py-1.5 pr-3 font-normal">상태</th><th className="pr-3 font-normal">자동 판정</th><th className="pr-3 font-normal">가지</th>
              <th className="pr-3 font-normal">θ</th><th className="pr-3 font-normal">F (T)</th><th className="pr-3 font-normal">진폭</th><th className="pr-3 font-normal">채널</th><th className="font-normal">파일</th>
            </tr></thead>
            <tbody>
              {sorted.map(p => (
                <tr key={p.id} className="border-t border-black/[0.06]">
                  <td className="py-1.5 pr-3"><span className={`rounded-full px-2 py-0.5 text-[11px] ${STATUS_STYLE[p.status]}`}>{STATUS_LABEL[p.status]}</span></td>
                  <td className="pr-3 text-[#6e6e73]">{p.auto ? VERDICT_LABEL[p.auto] : '—'}</td>
                  <td className="pr-3 font-semibold" style={{ color: p.branch ? BRANCH_COLORS[branches.indexOf(p.branch) % BRANCH_COLORS.length] : undefined }}>{p.branch ?? '—'}</td>
                  <td className="pr-3">{p.angle}°</td><td className="pr-3">{fmt(p.F, 5)}</td><td className="pr-3">{fmt(p.amp, 3)}</td>
                  <td className="pr-3">{p.col}</td><td className="max-w-[220px] truncate text-[#6e6e73]" title={p.fileName}>{p.fileName}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-5 text-[12px] text-[#86868b]">
        다음으로: 온도 sweep 이 있으면 <b>L-K Fit</b> 에서 확정한 F 의 유효질량을, <b>Landau Fan</b> 에서 Berry 위상을 볼 수 있습니다 (아래 "개별 도구").
      </p>
    </StepCard>
  );
}
