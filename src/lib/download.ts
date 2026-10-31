/**
 * One way out for every generated file, because the two places this app runs
 * hand files to the user differently.
 *
 * Served as an ordinary static site, an anchor with a `download` attribute is
 * the whole story. Published as a Claude Artifact the page is framed in a
 * sandbox that makes exactly that inert: the click is swallowed, no file
 * appears, and nothing is reported — an export button that looks like it worked
 * and silently did nothing. There the file has to be offered through the host's
 * `downloads` capability, which asks the viewer to confirm before saving.
 *
 * So: ask the host first, fall back to the anchor when there is no host. The
 * capability resolves asynchronously and never on the first synchronous run, so
 * callers stay fire-and-forget and failures surface here rather than at eight
 * separate call sites.
 */

type SaveResult = { status: string };
type DownloadsNamespace = {
  save(req: { filename: string; data: Blob | string }): Promise<SaveResult>;
};

let downloadsPromise: Promise<DownloadsNamespace | null> | null = null;

function hostDownloads(): Promise<DownloadsNamespace | null> {
  if (downloadsPromise) return downloadsPromise;
  const claude = (window as any).claude;
  downloadsPromise =
    claude && typeof claude.use === 'function'
      ? Promise.resolve(claude.use('downloads')).catch(() => null)
      : Promise.resolve(null);
  return downloadsPromise;
}

function anchorDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function deliver(blob: Blob, filename: string): Promise<void> {
  const downloads = await hostDownloads();
  if (!downloads) {
    anchorDownload(blob, filename);
    return;
  }
  try {
    await downloads.save({ filename, data: blob });
  } catch (err: any) {
    // The viewer declining is a normal outcome, not a failure to report.
    if (err?.code === 'declined') return;
    console.error('Download failed:', err);
    window.alert(`파일을 저장하지 못했습니다 (${err?.code ?? 'unknown error'}).`);
  }
}

/** Offer `blob` to the user as `filename`. Fire-and-forget. */
export function saveBlob(blob: Blob, filename: string): void {
  void deliver(blob, filename);
}

/** Convenience for the common text-export case. */
export function saveText(content: string, filename: string, mime: string): void {
  saveBlob(new Blob([content], { type: mime }), filename);
}
