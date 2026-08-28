// Lock-in channels come in pairs: in-phase (Output1, X) and quadrature
// (Output2, Y). Only the in-phase channel is ever analysed — the quadrature one
// is dropped from every selection / fit / plot / export path as soon as its
// partner is present. A file carrying only the quadrature channel is left alone;
// there'd be nothing left to fit.
const OUTPUT1_RE = /output[\s_-]*1(?!\d)/i;
const OUTPUT2_RE = /output[\s_-]*2(?!\d)/i;

// A "Y" that stands alone as one part of the name — "1.7K_95deg_Y" (last part)
// or "PrAlGe_Y_006" (middle part, followed by a run number). Never "YIELD",
// "Y2", "Rxy" or "B_y_offset".
export function quadraturePartners(col: string): string[] {
  const parts = col.split(/([\s_-])/);          // keeps the separators
  const out: string[] = [];
  for (let i = 0; i < parts.length; i += 2) {
    if (parts[i].toLowerCase() !== 'y') continue;
    // "…_Y_006" → "…_006": the in-phase channel has no letter at all
    const dropped = parts.filter((_, j) => j !== i && j !== (i > 0 ? i - 1 : 1)).join('');
    // "…_Y" → "…_X": the pair is named by axis
    const swapped = parts.map((p, j) => (j === i ? (p === 'Y' ? 'X' : 'x') : p)).join('');
    out.push(dropped, swapped);
  }
  return out;
}

export function dropQuadratureStrict(cols: string[], pool: string[] = cols): string[] {
  const hasOutput1 = pool.some(c => OUTPUT1_RE.test(c));
  // A Y channel is only dropped when its in-phase partner is actually present —
  // a lone Y column is still someone's only data.
  const poolLower = new Set(pool.map(c => c.toLowerCase()));
  return cols.filter(c => {
    if (hasOutput1 && OUTPUT2_RE.test(c)) return false;
    if (quadraturePartners(c).some(p => poolLower.has(p.toLowerCase()))) return false;
    return true;
  });
}
