// ---------- localStorage helpers ----------
const LS_PREFS_KEY = 'qoa_preferences';

export function lsGetPrefs(): Record<string, any> {
  try { return JSON.parse(localStorage.getItem(LS_PREFS_KEY) || '{}'); } catch { return {}; }
}
export function lsSavePrefs(data: Record<string, any>) {
  localStorage.setItem(LS_PREFS_KEY, JSON.stringify(data));
}
