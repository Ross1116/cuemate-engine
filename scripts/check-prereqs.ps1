param()

$checks = @(
  @{ Name = 'git'; Command = 'git'; Args = @('--version') },
  @{ Name = 'go'; Command = 'go'; Args = @('version') },
  @{ Name = 'python'; Command = 'python'; Args = @('--version') },
  @{ Name = 'protoc'; Command = 'protoc'; Args = @('--version') },
  @{ Name = 'buf'; Command = 'buf'; Args = @('--version') },
  @{ Name = 'docker'; Command = 'docker'; Args = @('--version') },
  @{ Name = 'docker-compose'; Command = 'docker-compose'; Args = @('version') }
)

foreach ($check in $checks) {
  $cmd = Get-Command $check.Command -ErrorAction SilentlyContinue
  if (-not $cmd) {
    Write-Output ('[missing] ' + $check.Name)
    continue
  }

  try {
    $output = & $cmd.Source @($check.Args) 2>&1 | Select-Object -First 1
    Write-Output ('[ok] ' + $check.Name + ' -> ' + $output)
  } catch {
    Write-Output ('[warning] ' + $check.Name + ' found but failed to execute: ' + $_.Exception.Message)
  }
}

if (Test-Path '.env') {
  Write-Output '[ok] .env present'
} else {
  Write-Output '[warning] .env missing (copy .env.example to .env)'
}

try {
  $dockerInfo = docker info 2>&1 | Select-Object -First 1
  Write-Output ('[ok] docker daemon reachable -> ' + $dockerInfo)
} catch {
  Write-Output '[warning] docker daemon not reachable'
}

if (Test-Path 'C:\Program Files\Tailscale\tailscale.exe') {
  try {
    $ts = & 'C:\Program Files\Tailscale\tailscale.exe' status 2>&1 | Select-Object -First 1
    Write-Output ('[ok] tailscale status -> ' + $ts)
  } catch {
    Write-Output '[warning] tailscale installed but status could not be read from this shell'
  }
} else {
  Write-Output '[warning] tailscale not installed'
}
