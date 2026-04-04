$ErrorActionPreference = "Stop"

$delegateScript = Join-Path $PSScriptRoot "start-essentia-semantics-service.ps1"
if (-not (Test-Path $delegateScript)) {
    throw "Shared TensorFlow/Essentia start script was not found at $delegateScript"
}

Write-Output "TempoCNN now runs through the shared TensorFlow/Essentia service."
& powershell -ExecutionPolicy Bypass -File $delegateScript
