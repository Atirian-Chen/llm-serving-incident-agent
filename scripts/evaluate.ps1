$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv-standard\Scripts\python.exe"
$RuntimePath = Join-Path $ProjectRoot ".runtime"
$TmpPath = Join-Path $RuntimePath "tmp"

if (-not (Test-Path $VenvPython)) {
    throw "E:\.venv-standard is missing. Run .\scripts\setup.ps1 once first."
}
New-Item -ItemType Directory -Force -Path $TmpPath | Out-Null
$env:TEMP = $TmpPath
$env:TMP = $TmpPath
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTHONUNBUFFERED = "1"
if (-not $env:AGENT_TIMEOUT_SECONDS) {
    $env:AGENT_TIMEOUT_SECONDS = "180"
}

& $VenvPython -u (Join-Path $ProjectRoot "scripts\evaluate.py") @args
exit $LASTEXITCODE
