$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$imageTag = if ($env:CUEMATE_TEMPOCNN_IMAGE) { $env:CUEMATE_TEMPOCNN_IMAGE } else { "cuemate-tempocnn:local" }

Push-Location $repoRoot
try {
    & docker build --tag $imageTag --file docker/tempocnn/Dockerfile docker/tempocnn
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
