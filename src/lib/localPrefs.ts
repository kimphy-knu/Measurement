// ---------- localStorage helpers ----------
const LS_PREFS_KEY = 'qoa_preferences';

export function lsGetPrefs(): Record<string, any> {
  try { return JSON.parse(localStorage.getItem(LS_PREFS_KEY) || '{}'); } catch { return {}; }
}
export function lsSavePrefs(data: Record<string, any>) {
  // Storage is not always reachable: a private window, blocked site data, or
  // the thumbnail capture a published Artifact goes through can make the
  // accessor itself throw. Losing a preference is fine; taking the app down
  // on the way past is not.
  try {
    localStorage.setItem(LS_PREFS_KEY, JSON.stringify(data));
  } catch {
    /* preferences simply do not persist in this context */
  }
}
