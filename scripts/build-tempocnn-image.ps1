$ErrorActionPreference = "Stop"

$delegateScript = Join-Path $PSScriptRoot "build-essentia-semantics-image.ps1"
if (-not (Test-Path $delegateScript)) {
    throw "Shared TensorFlow/Essentia build script was not found at $delegateScript"
}

Write-Output "TempoCNN now uses the shared TensorFlow/Essentia image."
& powershell -ExecutionPolicy Bypass -File $delegateScript
