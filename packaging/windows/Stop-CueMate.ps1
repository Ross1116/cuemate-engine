[CmdletBinding(SupportsShouldProcess=$true)]
param(
    [string]$InstallDir = $PSScriptRoot,
    [string]$AppDataRoot = (Join-Path $env:LOCALAPPDATA "CueMate"),
    [int]$ApiPid = 0,
    [int]$DelaySeconds = 0,
    [switch]$SkipDocker
)

$ErrorActionPreference = "Continue"

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = $PSScriptRoot
}
if ([string]::IsNullOrWhiteSpace($AppDataRoot)) {
    $AppDataRoot = Join-Path $env:LOCALAPPDATA "CueMate"
}

if ($DelaySeconds -gt 0) {
    Start-Sleep -Seconds $DelaySeconds
}

$logDir = Join-Path $AppDataRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "shutdown.log"

function Write-ShutdownLog {
    param([string]$Message)
    $timestamp = (Get-Date).ToUniversalTime().ToString("o")
    "$timestamp $Message" | Add-Content -Path $logPath -Encoding UTF8
}

function Stop-CueMateProcess {
    [CmdletBinding(SupportsShouldProcess=$true)]
    param([int]$ProcessId, [string]$Reason)
    if ($ProcessId -le 0 -or $ProcessId -eq $PID) {
        return
    }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) {
        return
    }
    $target = "PID $ProcessId ($Reason)"
    if (-not $PSCmdlet.ShouldProcess($target, "Stop process")) {
        Write-ShutdownLog "WhatIf: Would stop $target"
        return
    }
    Write-ShutdownLog "Stopping $target"
    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    } catch {
        Write-ShutdownLog "Failed to stop PID ${ProcessId}: $($_.Exception.Message)"
    }
}

function Stop-ProcessesByCommand {
    [CmdletBinding(SupportsShouldProcess=$true)]
    param()
    $installNeedle = [System.IO.Path]::GetFullPath($InstallDir)
    $appDataNeedle = [System.IO.Path]::GetFullPath($AppDataRoot)
    $targets = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $command = [string]$_.CommandLine
            if (-not $command) {
                return $false
            }
            $isCueMateCommand = (
                $command -match "cuemate_analysis\s+serve-scoring" -or
                $command -match "cuemate_analysis\s+run-analysis-worker" -or
                $command -match "cuemate_analysis\s+run-feedback-worker"
            )
            if (-not $isCueMateCommand) {
                return $false
            }
            return (
                $command.Contains($installNeedle) -or
                $command.Contains($appDataNeedle) -or
                $command -match "cuemate_analysis\s+(serve-scoring|run-analysis-worker|run-feedback-worker)"
            )
        } |
        Sort-Object ParentProcessId -Descending

    foreach ($target in $targets) {
        Stop-CueMateProcess -ProcessId ([int]$target.ProcessId) -Reason "CueMate Python service/worker" -WhatIf:$WhatIfPreference
    }
}

function Stop-Container {
    [CmdletBinding(SupportsShouldProcess=$true)]
    param([string]$Name)
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $docker) {
        Write-ShutdownLog "Docker CLI not found; skipping container $Name"
        return
    }
    try {
        $exists = & docker ps -a --filter "name=^/${Name}$" --format "{{.Names}}" 2>$null
        if ($exists -contains $Name) {
            if (-not $PSCmdlet.ShouldProcess("Docker container $Name", "Remove forcefully")) {
                Write-ShutdownLog "WhatIf: Would stop Docker container $Name"
                return
            }
            Write-ShutdownLog "Stopping Docker container $Name"
            & docker rm -f $Name 2>&1 | Add-Content -Path $logPath -Encoding UTF8
        }
    } catch {
        Write-ShutdownLog "Failed to stop Docker container ${Name}: $($_.Exception.Message)"
    }
}

Write-ShutdownLog "CueMate shutdown requested."

Stop-ProcessesByCommand -WhatIf:$WhatIfPreference

if (-not $SkipDocker) {
    Stop-Container -Name "cuemate-essentia-semantics-service" -WhatIf:$WhatIfPreference
    Stop-Container -Name "cuemate-musicalkeycnn-service" -WhatIf:$WhatIfPreference
}

if ($ApiPid -gt 0) {
    Stop-CueMateProcess -ProcessId $ApiPid -Reason "CueMate API" -WhatIf:$WhatIfPreference
} else {
    $apiTargets = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $command = [string]$_.CommandLine
            $command -and (
                $command.Contains("apiserver.exe") -or
                $command -match "go(\.exe)?\s+run\s+\./go/cmd/apiserver"
            )
        }
    foreach ($target in $apiTargets) {
        Stop-CueMateProcess -ProcessId ([int]$target.ProcessId) -Reason "CueMate API" -WhatIf:$WhatIfPreference
    }
}

Write-ShutdownLog "CueMate shutdown complete."
