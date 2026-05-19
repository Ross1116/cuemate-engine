param(
    [string]$InstallDir = $PSScriptRoot,
    [string]$AppDataRoot = (Join-Path $env:LOCALAPPDATA "CueMate"),
    [int]$ApiPid = 0,
    [int]$DelaySeconds = 0,
    [switch]$SkipDocker
)

$ErrorActionPreference = "Continue"

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
    param([int]$ProcessId, [string]$Reason)
    if ($ProcessId -le 0 -or $ProcessId -eq $PID) {
        return
    }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) {
        return
    }
    Write-ShutdownLog "Stopping PID $ProcessId ($Reason)"
    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    } catch {
        Write-ShutdownLog "Failed to stop PID ${ProcessId}: $($_.Exception.Message)"
    }
}

function Stop-ProcessesByCommand {
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
        Stop-CueMateProcess -ProcessId ([int]$target.ProcessId) -Reason "CueMate Python service/worker"
    }
}

function Stop-Container {
    param([string]$Name)
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $docker) {
        Write-ShutdownLog "Docker CLI not found; skipping container $Name"
        return
    }
    try {
        $exists = & docker ps -a --filter "name=^/${Name}$" --format "{{.Names}}" 2>$null
        if ($exists -contains $Name) {
            Write-ShutdownLog "Stopping Docker container $Name"
            & docker rm -f $Name 2>&1 | Add-Content -Path $logPath -Encoding UTF8
        }
    } catch {
        Write-ShutdownLog "Failed to stop Docker container ${Name}: $($_.Exception.Message)"
    }
}

Write-ShutdownLog "CueMate shutdown requested."

Stop-ProcessesByCommand

if (-not $SkipDocker) {
    Stop-Container -Name "cuemate-essentia-semantics-service"
    Stop-Container -Name "cuemate-musicalkeycnn-service"
}

if ($ApiPid -gt 0) {
    Stop-CueMateProcess -ProcessId $ApiPid -Reason "CueMate API"
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
        Stop-CueMateProcess -ProcessId ([int]$target.ProcessId) -Reason "CueMate API"
    }
}

Write-ShutdownLog "CueMate shutdown complete."
