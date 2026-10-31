/**
 * Fold `dist/` into one self-contained HTML file for publishing as a Claude
 * Artifact.
 *
 * Artifacts serve a single page and block every external request, so the
 * separate .js and .css Vite emits cannot be fetched — they have to travel
 * inside the document. The published file also omits <!doctype>/<html>/<head>/
 * <body>: the platform wraps what we write.
 *
 * Run `npm run build` first, then `node scripts/build-artifact.mjs`.
 */
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const assetDir = join(root, 'dist', 'assets');

let names;
try {
  names = readdirSync(assetDir);
} catch {
  console.error('dist/assets not found — run `npm run build` first.');
  process.exit(1);
}

const jsName = names.find((n) => n.endsWith('.js'));
const cssName = names.find((n) => n.endsWith('.css'));
if (!jsName || !cssName) {
  console.error(`Expected one .js and one .css in dist/assets, found: ${names.join(', ')}`);
  process.exit(1);
}

const css = readFileSync(join(assetDir, cssName), 'utf8');
const js = readFileSync(join(assetDir, jsName), 'utf8');

// `</style>` inside a <style> block ends it, and CSS has no escape that works
// in every position — so refuse rather than emit a silently broken page.
if (/<\/style/i.test(css)) {
  console.error('Bundled CSS contains "</style>" and cannot be inlined safely.');
  process.exit(1);
}
// In JS the sequence only matters inside string literals, where `<\/script` is
// exactly equivalent and parses identically.
const safeJs = js.replace(/<\/script/gi, '<\\/script');

const html = `<title>SdH Analysis Suite</title>
<style>
/* The viewer paints its own ground behind the page; the app's canvas colour
   has to be stated here or a dark-themed viewer shows through. */
body { background: #E4E3E0; }
${css}
</style>
<div id="root"></div>
<script type="module">
${safeJs}
</script>
`;

const outPath = join(root, 'dist', 'artifact.html');
writeFileSync(outPath, html, 'utf8');

const mb = (statSync(outPath).size / 1024 / 1024).toFixed(2);
console.log(`Wrote ${outPath} (${mb} MB) from ${cssName} + ${jsName}`);
