param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$DbmateArgs
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$candidates = @(
  (Join-Path $repoRoot '.local-tools\dbmate.exe'),
  (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links\dbmate.exe'),
  (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages\amacneil.dbmate_Microsoft.Winget.Source_8wekyb3d8bbwe\dbmate.exe')
)

$dbmate = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $dbmate) {
  throw 'Could not find a usable dbmate executable.'
}

Push-Location $repoRoot
try {
  & $dbmate '--migrations-dir' 'db/migrations' '--schema-file' 'db/schema.sql' @DbmateArgs
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
