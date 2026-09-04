param(
    [switch]$ForceRecreate,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv-standard"
$RuntimePath = Join-Path $ProjectRoot ".runtime"
$TmpPath = Join-Path $RuntimePath "tmp"
$PipCachePath = Join-Path $RuntimePath "pip-cache"
$HfHomePath = Join-Path $RuntimePath "huggingface"
$TorchHomePath = Join-Path $RuntimePath "torch"

New-Item -ItemType Directory -Force -Path $TmpPath, $PipCachePath, $HfHomePath, $TorchHomePath | Out-Null

$env:TEMP = $TmpPath
$env:TMP = $TmpPath
$env:PIP_CACHE_DIR = $PipCachePath
$env:HF_HOME = $HfHomePath
$env:TORCH_HOME = $TorchHomePath
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

if ($ForceRecreate -and (Test-Path $VenvPath)) {
    Remove-Item -LiteralPath $VenvPath -Recurse -Force
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    throw "Python 3.11+ was not found on PATH. Install Python, then rerun this script."
}

if (-not (Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
    & $Python.Source -m venv $VenvPath
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip --cache-dir $PipCachePath
# Install CUDA torch before sentence-transformers resolves its dependency from
# PyPI. This keeps strict RAG_DEVICE=cuda from silently ending up on CPU.
& $VenvPython -m pip install --upgrade --force-reinstall "torch==2.14.0+cu126" --index-url $TorchIndexUrl --cache-dir $PipCachePath
& $VenvPython -m pip install --requirement (Join-Path $ProjectRoot "requirements.lock") --cache-dir $PipCachePath
& $VenvPython -m pip install --no-deps --editable $ProjectRoot --cache-dir $PipCachePath

Write-Host "Ready: $VenvPython"
Write-Host "Packages: $VenvPath"
Write-Host "Temp/cache: $RuntimePath"
Write-Host "Run tests with: .\scripts\test.ps1"
