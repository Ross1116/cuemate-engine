param(
    [string]$InstallDir = $PSScriptRoot,
    [string]$AppDataRoot = (Join-Path $env:LOCALAPPDATA "CueMate"),
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
$AppDataRoot = [System.IO.Path]::GetFullPath($AppDataRoot)
$dataDir = Join-Path $AppDataRoot "data"
$logDir = Join-Path $AppDataRoot "logs"
$venvPython = Join-Path $AppDataRoot ".venv\Scripts\python.exe"
$databasePath = Join-Path $dataDir "cuemate.db"
$cachePath = Join-Path $dataDir "inference-cache.db"
$apiUrl = "http://127.0.0.1:8080"

New-Item -ItemType Directory -Force -Path $dataDir, $logDir | Out-Null

function Get-FixedDriveLetters {
    $letters = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" |
        ForEach-Object { $_.DeviceID.TrimEnd(":").ToLowerInvariant() }
    if (-not $letters) {
        return "c"
    }
    return ($letters -join ",")
}

function Set-CueMateEnvironment {
    $env:DATABASE_URL = "sqlite:$databasePath"
    $env:CUEMATE_INFERENCE_CACHE_PATH = $cachePath
    $env:WEB_DIST_DIR = (Join-Path $InstallDir "web\dist")
    $env:CUEMATE_PYTHON = $venvPython
    $env:SCORING_GRPC_ADDR = "127.0.0.1:47834"
    $env:GO_API_ADDR = "127.0.0.1:8080"
    $env:CUEMATE_MUSICALKEYCNN_DRIVES = Get-FixedDriveLetters
    $env:CUEMATE_ESSENTIA_SEMANTIC_DRIVES = $env:CUEMATE_MUSICALKEYCNN_DRIVES
}

function Test-HttpOk {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Test-TcpPort {
    param(
        [string]$HostName,
        [int]$Port
    )
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(1000)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Wait-Until {
    param(
        [scriptblock]$Predicate,
        [int]$TimeoutSeconds,
        [string]$Description
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Predicate) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    Write-Warning "Timed out waiting for $Description."
    return $false
}

function Start-LoggedProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory = $InstallDir
    )

    $stdout = Join-Path $logDir "$Name.out.log"
    $stderr = Join-Path $logDir "$Name.err.log"
    Start-Process -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr | Out-Null
}

function Ensure-BootstrapComplete {
    if ((Test-Path $venvPython -PathType Leaf) -and (Test-Path $databasePath -PathType Leaf)) {
        return
    }
    $bootstrap = Join-Path $InstallDir "Bootstrap-CueMate.ps1"
    if (-not (Test-Path $bootstrap -PathType Leaf)) {
        throw "CueMate bootstrap script is missing at $bootstrap"
    }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $bootstrap -InstallDir $InstallDir -AppDataRoot $AppDataRoot
    if ($LASTEXITCODE -ne 0) {
        throw "CueMate bootstrap failed. See logs in $logDir."
    }
}

Set-CueMateEnvironment
Ensure-BootstrapComplete
Set-CueMateEnvironment

Push-Location $InstallDir
try {
    if (-not (Test-TcpPort -HostName "127.0.0.1" -Port 47834)) {
        Start-LoggedProcess -Name "scorer" -FilePath $venvPython -Arguments @("-m", "cuemate_analysis", "serve-scoring", "--host", "127.0.0.1", "--port", "47834")
        Wait-Until -Predicate { Test-TcpPort -HostName "127.0.0.1" -Port 47834 } -TimeoutSeconds 45 -Description "CueMate scorer" | Out-Null
    }

    if (-not (Test-HttpOk -Url "$apiUrl/healthz")) {
        $apiExe = Join-Path $InstallDir "apiserver.exe"
        if (-not (Test-Path $apiExe -PathType Leaf)) {
            throw "CueMate API executable is missing at $apiExe"
        }
        Start-LoggedProcess -Name "apiserver" -FilePath $apiExe -Arguments @()
        Wait-Until -Predicate { Test-HttpOk -Url "$apiUrl/healthz" } -TimeoutSeconds 45 -Description "CueMate API" | Out-Null
    }

    if (-not $NoBrowser) {
        Start-Process $apiUrl | Out-Null
    }
} finally {
    Pop-Location
}
