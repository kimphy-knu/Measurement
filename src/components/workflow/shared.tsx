import type { ReactNode } from 'react';
import type { AnalysisResult } from '../../types';
import type { Verdict } from '../../lib/verify';

// A peak the user picked in step 2. It travels through verification (step 3),
// angle evolution (step 4) and the summary (step 5), and is saved with the
// session. Keyed by file name + column rather than upload index so it survives
// reloading a session.
export interface PickedPeak {
  id: string;
  fileName: string;
  col: string;
  angle: number;
  F: number;
  amp: number;
  status: 'unchecked' | 'confirmed' | 'rejected';
  // 'threshold' = added by step 2's SNR threshold and replaced when it changes;
  // absent = picked by hand
  src?: 'threshold';
  snr?: number;
  auto?: Verdict;
  branch?: string;
  note?: string;
}

// One processed (file, column) — the app's landauSeriesList entries.
export interface Series {
  key: string;
  fileIdx: number;
  fileName: string;
  blm: number;
  col: string;
  result: AnalysisResult;
}

export const STATUS_LABEL: Record<PickedPeak['status'], string> = {
  unchecked: '미확인',
  confirmed: '확정',
  rejected: '제외',
};
export const STATUS_STYLE: Record<PickedPeak['status'], string> = {
  unchecked: 'bg-black/5 text-[#6e6e73]',
  confirmed: 'bg-[#248a3d]/10 text-[#248a3d]',
  rejected: 'bg-red-500/10 text-red-600',
};

// Okabe–Ito order, fixed: branch colours stay the same from step to step.
export const BRANCH_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#1d1d1f', '#8e8e93'];

export function seriesOf(list: Series[], p: Pick<PickedPeak, 'fileName' | 'col'>): Series | undefined {
  return list.find(s => s.fileName === p.fileName && s.col === p.col);
}

export function StepCard({ step, title, lead, actions, children }: {
  step: number; title: string; lead: ReactNode; actions?: ReactNode; children: ReactNode;
}) {
  return (
    <section className="bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,.04),0_12px_32px_-20px_rgba(0,0,0,.12)]">
      <header className="flex flex-wrap items-start justify-between gap-4 px-6 pt-6 pb-4 border-b border-black/[0.06]">
        <div className="max-w-3xl">
          <p className="text-[12px] font-semibold text-[#0071e3]">STEP {step}</p>
          <h2 className="text-[24px] font-bold tracking-[-0.02em] leading-tight">{title}</h2>
          <p className="mt-1.5 text-[13px] text-[#6e6e73] leading-relaxed">{lead}</p>
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </header>
      <div className="p-6">{children}</div>
    </section>
  );
}

export function Pill({ className = '', children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button {...rest} className={`px-3.5 py-1.5 rounded-full text-[13px] border border-black/10 hover:bg-black/5 transition-colors disabled:opacity-40 ${className}`}>
      {children}
    </button>
  );
}

export function PrimaryPill({ className = '', children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button {...rest} className={`px-4 py-1.5 rounded-full text-[13px] font-medium bg-[#0071e3] text-white hover:brightness-110 transition-all disabled:opacity-40 ${className}`}>
      {children}
    </button>
  );
}

export function Metric({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="rounded-xl bg-[#f5f5f7] px-3.5 py-2.5" title={hint}>
      <div className="text-[11px] text-[#86868b]">{label}</div>
      <div className="text-[16px] font-semibold tabular-nums">{value}</div>
    </div>
  );
}

export const TOOLTIP = { backgroundColor: '#ffffff', borderRadius: 12, border: '1px solid #e5e5ea', boxShadow: '0 8px 24px rgba(0,0,0,.08)', fontSize: 11 };
export const TICK = { fontSize: 10, fill: '#6e6e73' };

export function fmt(v: number, d = 4) {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(d - 1) : String(parseFloat(v.toPrecision(d)));
}
