import { useEffect, useState, type ReactNode } from 'react';
import { Lock } from 'lucide-react';
import { getStoredAccount, storeAccount, verifyCredentials } from '../lib/auth';

/**
 * Wraps the app in the access gate. See src/lib/auth.ts for what this gate does
 * and does not protect.
 *
 * Styled to match the sample-ID start screen in App.tsx, so signing in reads as
 * the first step of the same flow rather than a bolted-on door.
 */
export function LoginGate({ children }: { children: ReactNode }) {
  const [account, setAccount] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);
  const [idDraft, setIdDraft] = useState('');
  const [pwDraft, setPwDraft] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    setAccount(getStoredAccount());
    setChecked(true);
  }, []);

  // Render nothing until localStorage has been read, so a signed-in reload does
  // not flash the login form before the dashboard appears.
  if (!checked) return null;
  if (account) return <>{children}</>;

  const submit = () => {
    if (!verifyCredentials(idDraft, pwDraft)) {
      setError('Incorrect ID or password.');
      setPwDraft('');
      return;
    }
    const id = idDraft.trim().toUpperCase();
    storeAccount(id);
    setAccount(id);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') submit();
  };

  return (
    <div className="min-h-screen bg-[#E4E3E0] text-[#141414] font-sans flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-[#141414] text-[#E4E3E0] border border-[#E4E3E0]/10 p-10">
        <h1 className="text-3xl font-bold tracking-tighter uppercase italic font-serif mb-1">
          Quantum Analysis
        </h1>
        <p className="text-xs opacity-50 font-mono uppercase tracking-widest mb-8">
          SdH Analysis Suite
        </p>

        <label className="block text-[10px] font-mono uppercase opacity-60 mb-2">Account ID</label>
        <input
          type="text"
          autoFocus
          autoComplete="username"
          value={idDraft}
          onChange={(e) => { setIdDraft(e.target.value); setError(''); }}
          onKeyDown={onKeyDown}
          className="w-full bg-[#0a0a0a] border border-[#E4E3E0]/30 p-3 font-mono text-sm text-[#E4E3E0] focus:outline-none focus:border-[#F27D26] mb-4"
        />

        <label className="block text-[10px] font-mono uppercase opacity-60 mb-2">Password</label>
        <input
          type="password"
          autoComplete="current-password"
          value={pwDraft}
          onChange={(e) => { setPwDraft(e.target.value); setError(''); }}
          onKeyDown={onKeyDown}
          className="w-full bg-[#0a0a0a] border border-[#E4E3E0]/30 p-3 font-mono text-sm text-[#E4E3E0] focus:outline-none focus:border-[#F27D26] mb-4"
        />

        <button
          onClick={submit}
          disabled={!idDraft.trim() || !pwDraft}
          className="w-full flex items-center justify-center gap-2 bg-[#F27D26] text-[#141414] font-mono text-xs uppercase tracking-widest font-bold py-3 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-opacity-90 transition-all"
        >
          <Lock size={13} />
          Sign In
        </button>

        {error && <p className="mt-4 text-[11px] font-mono text-red-400">{error}</p>}

        <p className="mt-8 pt-6 border-t border-white/10 text-[9px] font-mono opacity-40 leading-relaxed">
          All processing runs in your browser. Measurement files are opened from
          your own machine and are never uploaded.
        </p>
      </div>
    </div>
  );
}
