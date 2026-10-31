/**
 * Client-side access gate for the published build.
 *
 * What this is, and what it is not:
 *
 * This keeps the app out of reach of anyone who merely stumbles onto the URL.
 * It is NOT access control. The whole app ships to the browser as static files,
 * so anyone willing to read the bundle can bypass the check. That is acceptable
 * here only because the server holds nothing worth taking — every measurement
 * file is opened locally and never leaves the machine it was opened on, so
 * getting past this gate yields an empty analysis tool and nothing else.
 *
 * If the app itself ever has to be genuinely private, the gate has to move in
 * front of the server: an edge authenticator such as Cloudflare Access refuses
 * to serve the bundle at all until the visitor proves who they are. Tightening
 * anything in this file cannot achieve that.
 *
 * The credential pair is therefore stored as a SHA-256 digest rather than in
 * plaintext. That does not make the gate strong, but it does stop the bundle
 * from handing over a password that someone may have reused elsewhere.
 */

const CREDENTIAL_HASH =
  'c59ad0ae43c94349f167218c0b5289576e9dbd4c01c1997142fa451c2c414b58';

const STORAGE_KEY = 'qa-auth-account';

// SHA-256, implemented here rather than via crypto.subtle on purpose.
// crypto.subtle exists only in secure contexts, so it is undefined when the dev
// server is reached over plain http on a LAN address — which `npm run dev` does
// by default (--host=0.0.0.0). Relying on it would make the login silently fail
// in exactly that case. One hash of a short string costs nothing, so this runs
// everywhere and stays synchronous.
const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function sha256Hex(message: string): string {
  const msg = new TextEncoder().encode(message);
  const bitLen = msg.length * 8;

  // 64-byte blocks: message, a single 0x80 marker, then a big-endian 64-bit
  // length in the final eight bytes.
  const padded = new Uint8Array((((msg.length + 8) >> 6) + 1) * 64);
  padded.set(msg);
  padded[msg.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(bitLen / 0x100000000), false);
  view.setUint32(padded.length - 4, bitLen >>> 0, false);

  const H = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const w = new Uint32Array(64);

  for (let off = 0; off < padded.length; off += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(off + i * 4, false);
    for (let i = 16; i < 64; i++) {
      const x = w[i - 15];
      const y = w[i - 2];
      const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
      const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let [a, b, c, d, e, f, g, h] = H;
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + S1 + ch + K[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0;
      d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }

    H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0;
    H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0;
    H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0;
    H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
  }

  return Array.from(H, (v) => v.toString(16).padStart(8, '0')).join('');
}

/**
 * The account id is compared case-insensitively — it is a lab-shared name, not
 * a personal identity, and a capitalisation slip is a support question nobody
 * needs. The password is compared exactly.
 */
export function verifyCredentials(accountId: string, password: string): boolean {
  const id = accountId.trim().toUpperCase();
  if (!id || !password) return false;
  return sha256Hex(`${id}:${password}`) === CREDENTIAL_HASH;
}

/** The signed-in account id, or null when the gate has not been passed. */
export function getStoredAccount(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    // Private-mode browsers can throw on access rather than return null.
    return null;
  }
}

export function storeAccount(accountId: string): void {
  try {
    localStorage.setItem(STORAGE_KEY, accountId);
  } catch {
    // Non-fatal: the gate simply reappears on the next reload.
  }
}

export function clearAccount(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Non-fatal.
  }
}
