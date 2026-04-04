$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$imageTag = if ($env:CUEMATE_MUSICALKEYCNN_IMAGE) { $env:CUEMATE_MUSICALKEYCNN_IMAGE } else { "cuemate-musicalkeycnn:local" }

Push-Location $repoRoot
try {
    & docker build --tag $imageTag --file docker/musicalkeycnn/Dockerfile docker/musicalkeycnn
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
