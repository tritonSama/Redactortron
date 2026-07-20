# Install Redactortron (Python deps + Poppler on Windows)
# Usage:
#   .\scripts\install.ps1
#   .\scripts\install.ps1 -Api -Dev
#   .\scripts\install.ps1 -WithPoppler

param(
    [switch]$Api,
    [switch]$Dev,
    [switch]$WithPoppler,
    [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Redactortron installer" -ForegroundColor Cyan
Write-Host "Repo: $Root"

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "Python was not found on PATH. Install Python 3.9+ and retry."
}

$installArgs = @("scripts/install_deps.py")
if ($Api) { $installArgs += "--api" }
if ($Dev) { $installArgs += "--dev" }
if ($WithPoppler) { $installArgs += "--with-poppler" }
if ($SkipVerify) { $installArgs += "--skip-verify" }

# Default: pull Poppler on Windows when missing
if (-not $WithPoppler) {
    $pdftoppm = Get-Command pdftoppm -ErrorAction SilentlyContinue
    if (-not $pdftoppm) {
        Write-Host "Poppler not on PATH — will download into .tools\poppler" -ForegroundColor Yellow
        $installArgs += "--with-poppler"
    }
}

& python @installArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

# Prepend local Poppler to this session if present
$envFile = Join-Path $Root ".env.redactortron"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^REDACTORTRON_POPPLER_PATH=(.+)$') {
            $bin = $Matches[1]
            if (Test-Path $bin) {
                $env:REDACTORTRON_POPPLER_PATH = $bin
                $env:PATH = "$bin;$env:PATH"
                Write-Host "Session PATH includes Poppler: $bin" -ForegroundColor Green
            }
        }
    }
}

Write-Host ""
Write-Host "Next:" -ForegroundColor Cyan
Write-Host "  python -m redactortron ui"
Write-Host "  python -m redactortron scan --input document.pdf"
