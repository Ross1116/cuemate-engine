param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ComposeArgs
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$dockerConfig = Join-Path $repoRoot '.docker-local'
New-Item -ItemType Directory -Force $dockerConfig | Out-Null
$env:DOCKER_CONFIG = $dockerConfig

Push-Location $repoRoot
try {
  & docker-compose @ComposeArgs
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
