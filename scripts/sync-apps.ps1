# Refresh the sibling tools published under site/ from their own project folders.
#
#   site/transport/  <- vite build of ..\transport data analysis (relative base)
#   site/arres/      <- ..\ARRES\tools\arres_combined_demo.html (synthetic data only)
#
# Only build output is copied, never local_data/ or measurement files. ARRES
# deliberately uses the demo page: arres_combined.html embeds real scans.
#
# Run from anywhere:  pwsh scripts/sync-apps.ps1   then commit and push.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$programDir = Split-Path -Parent $root
$transportSrc = Join-Path $programDir 'transport data analysis'
$arresDemo = Join-Path $programDir 'ARRES\tools\arres_combined_demo.html'

$transportOut = Join-Path $root 'site\transport'
$arresOut = Join-Path $root 'site\arres'

Write-Host "Building Transport from $transportSrc ..."
Push-Location $transportSrc
try {
  if (-not (Test-Path node_modules)) { npm install }
  npx vite build --base=./ --outDir $transportOut --emptyOutDir
  if ($LASTEXITCODE -ne 0) { throw 'Transport build failed.' }
} finally {
  Pop-Location
}

Write-Host "Copying ARRES demo from $arresDemo ..."
New-Item -ItemType Directory -Force $arresOut | Out-Null
Copy-Item $arresDemo (Join-Path $arresOut 'index.html') -Force

Write-Host 'Done. Review with `git status`, then commit and push.'
