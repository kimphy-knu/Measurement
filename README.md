# Quantum Analysis — SdH Analysis Suite

Browser-based analysis of quantum oscillations (SdH / dHvA, torque): background
subtraction, 1/B resampling, FFT, Landau fan, L-K fit, angular harmonics.
All processing runs locally in the browser; measurement files are never uploaded.

## Run locally

Prerequisite: Node.js 20+

```
npm install
npm run dev
```

or double-click `start.bat`. Opens at http://localhost:3000.

## Deploy (GitHub Pages)

Pushing to `main`/`master` builds and publishes https://kimphy-knu.github.io/Measurement/
via [.github/workflows/deploy.yml](.github/workflows/deploy.yml):

| Path | Source |
|---|---|
| `/` | [site/index.html](site/index.html) — landing page |
| `/qo/` | this app |
| `/transport/` | prebuilt from `..\transport data analysis` |
| `/arres/` | `..\ARRES\tools\arres_combined_demo.html` (synthetic demo data) |

After changing Transport or ARRES, refresh their copies with
`pwsh scripts/sync-apps.ps1`, then commit and push.

Measurement data (`/*.json`, `/*.txt`, `/*.dat`, `/*.csv` at the repo root) is
git-ignored and is never committed or deployed.
