param()

function Invoke-ExternalCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [string[]]$Arguments = @()
  )

  $stdoutPath = [System.IO.Path]::GetTempFileName()
  $stderrPath = [System.IO.Path]::GetTempFileName()

  try {
    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $stdout = Get-Content $stdoutPath -Raw -ErrorAction SilentlyContinue
    $stderr = Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue

    if ($null -eq $stdout) { $stdout = '' }
    if ($null -eq $stderr) { $stderr = '' }

    $combined = ((@($stdout.TrimEnd(), $stderr.TrimEnd())) | Where-Object { $_ }) -join "`n"
    $firstLine = ($combined -split "`r?`n" | Where-Object { $_ } | Select-Object -First 1)
    if (-not $firstLine) {
      $firstLine = '[no output]'
    }

    return [pscustomobject]@{
      ExitCode = $process.ExitCode
      Output = $firstLine
    }
  } finally {
    Remove-Item $stdoutPath, $stderrPath -ErrorAction SilentlyContinue
  }
}

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
    $result = Invoke-ExternalCommand -FilePath $cmd.Source -Arguments $check.Args
    $output = $result.Output
    if ($result.ExitCode -ne 0) {
      Write-Output ('[warning] ' + $check.Name + ' found but failed to execute (exit ' + $result.ExitCode + '): ' + $output)
    } else {
      Write-Output ('[ok] ' + $check.Name + ' -> ' + $output)
    }
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
  $dockerResult = Invoke-ExternalCommand -FilePath 'docker' -Arguments @('info')
  $dockerInfo = $dockerResult.Output
  if ($dockerResult.ExitCode -eq 0) {
    Write-Output ('[ok] docker daemon reachable -> ' + $dockerInfo)
  } else {
    Write-Output ('[warning] docker daemon not reachable -> ' + $dockerInfo)
  }
} catch {
  $dockerInfo = $_.Exception.Message
  Write-Output ('[warning] docker daemon not reachable -> ' + $dockerInfo)
}

$tailscaleCmd = if (Test-Path 'C:\Program Files\Tailscale\tailscale.exe' -PathType Leaf) {
  'C:\Program Files\Tailscale\tailscale.exe'
} else {
  $tsCommand = Get-Command tailscale -ErrorAction SilentlyContinue
  if ($tsCommand) { $tsCommand.Source }
}

if ($tailscaleCmd) {
  try {
    $tailscaleResult = Invoke-ExternalCommand -FilePath $tailscaleCmd -Arguments @('status')
    $ts = $tailscaleResult.Output
    if ($tailscaleResult.ExitCode -eq 0) {
      Write-Output ('[ok] tailscale status -> ' + $ts)
    } else {
      Write-Output ('[warning] tailscale installed but status returned exit ' + $tailscaleResult.ExitCode + ' -> ' + $ts)
    }
  } catch {
    Write-Output '[warning] tailscale installed but status could not be read from this shell'
  }
} else {
  Write-Output '[warning] tailscale not installed'
}
