$ErrorActionPreference = "Stop"

$delegateScript = Join-Path $PSScriptRoot "start-essentia-semantics-service.ps1"
if (-not (Test-Path $delegateScript)) {
    throw "Shared TensorFlow/Essentia start script was not found at $delegateScript"
}

if (-not $env:CUEMATE_ESSENTIA_SEMANTIC_SERVICE_PORT) {
    if ($env:CUEMATE_TEMPOCNN_SERVICE_PORT) {
        $env:CUEMATE_ESSENTIA_SEMANTIC_SERVICE_PORT = $env:CUEMATE_TEMPOCNN_SERVICE_PORT
    }
    else {
        $env:CUEMATE_ESSENTIA_SEMANTIC_SERVICE_PORT = "47833"
    }
}

if (-not $env:CUEMATE_ESSENTIA_SEMANTIC_SERVICE_NAME -and $env:CUEMATE_TEMPOCNN_SERVICE_NAME) {
    $env:CUEMATE_ESSENTIA_SEMANTIC_SERVICE_NAME = $env:CUEMATE_TEMPOCNN_SERVICE_NAME
}

Write-Output "TempoCNN now runs through the shared TensorFlow/Essentia service."
& powershell -ExecutionPolicy Bypass -File $delegateScript

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}