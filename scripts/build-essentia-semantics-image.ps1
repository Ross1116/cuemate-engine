$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$imageTag = if ($env:CUEMATE_ESSENTIA_SEMANTIC_IMAGE) { $env:CUEMATE_ESSENTIA_SEMANTIC_IMAGE } else { "cuemate-essentia-semantics:local" }

Push-Location $repoRoot
try {
    & docker build --tag $imageTag --file docker/essentia_semantics/Dockerfile docker/essentia_semantics
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
