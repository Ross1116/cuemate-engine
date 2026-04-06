$ErrorActionPreference = "Stop"

$delegateScript = Join-Path $PSScriptRoot "build-essentia-semantics-image.ps1"
if (-not (Test-Path $delegateScript)) {
    throw "Shared TensorFlow/Essentia build script was not found at $delegateScript"
}

if ($env:CUEMATE_TEMPOCNN_IMAGE -and -not $env:CUEMATE_ESSENTIA_SEMANTIC_IMAGE) {
    $env:CUEMATE_ESSENTIA_SEMANTIC_IMAGE = $env:CUEMATE_TEMPOCNN_IMAGE
}

Write-Output "TempoCNN now uses the shared TensorFlow/Essentia image."
& powershell -ExecutionPolicy Bypass -File $delegateScript

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}