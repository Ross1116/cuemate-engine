param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ComposeArgs
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$dockerConfig = Join-Path $repoRoot '.docker-local'
New-Item -ItemType Directory -Force $dockerConfig | Out-Null
$oldDockerConfig = $env:DOCKER_CONFIG
$env:DOCKER_CONFIG = $dockerConfig

Push-Location $repoRoot
try {
  & docker-compose @ComposeArgs
  exit $LASTEXITCODE
} finally {
  if ($null -eq $oldDockerConfig) {
    Remove-Item Env:DOCKER_CONFIG -ErrorAction SilentlyContinue
  } else {
    $env:DOCKER_CONFIG = $oldDockerConfig
  }
  Pop-Location
}
