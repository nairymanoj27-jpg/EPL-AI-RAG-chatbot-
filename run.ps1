# Starts the FastAPI app with the project's virtual environment.
#
# Usage:
#   .\run.ps1
#   .\run.ps1 -Reload
#   .\run.ps1 -BindHost 0.0.0.0 -Port 8080
[CmdletBinding()]
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Virtual environment not found. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt"
    exit 1
}

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".env"))) {
    Write-Error "Missing .env. Copy .env.example to .env and fill in the required values."
    exit 1
}

$uvicornArgs = @(
    "-m", "uvicorn", "app.asgi:app",
    "--host", $BindHost,
    "--port", "$Port"
)
if ($Reload) {
    $uvicornArgs += "--reload"
    $uvicornArgs += "--reload-dir"
    $uvicornArgs += (Join-Path $PSScriptRoot "app")
}

Write-Host ""
Write-Host "  UI:   http://${BindHost}:${Port}/ui/"
Write-Host "  Docs: http://${BindHost}:${Port}/docs"
Write-Host "  Stop: Ctrl+C"
Write-Host ""

& $python @uvicornArgs
exit $LASTEXITCODE
