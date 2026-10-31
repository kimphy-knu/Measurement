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

Pushing to `main` builds and publishes the site via
[.github/workflows/deploy.yml](.github/workflows/deploy.yml).

One-time setup: repo **Settings → Pages → Source: GitHub Actions**.
The site is then served at `https://<user>.github.io/<repo>/`.

Measurement data (`/*.json`, `/*.txt`, `/*.dat`, `/*.csv` at the repo root) is
git-ignored and is never committed or deployed.
