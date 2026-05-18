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
$remoteConfigPath = Join-Path $AppDataRoot "remote.json"
$apiUrl = "http://127.0.0.1:8080"

New-Item -ItemType Directory -Force -Path $dataDir, $logDir | Out-Null
Start-Transcript -Path (Join-Path $logDir "launcher.log") -Append | Out-Null

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
    $env:CUEMATE_SETUP_STATE_PATH = Join-Path $AppDataRoot "setup-state.json"
    $env:CUEMATE_LOG_DIR = $logDir
    $env:WEB_DIST_DIR = (Join-Path $InstallDir "web\dist")
    $env:CUEMATE_PYTHON = $venvPython
    $env:SCORING_GRPC_ADDR = "127.0.0.1:47834"
    $env:GO_API_ADDR = "127.0.0.1:8080"
    $env:CUEMATE_MUSICALKEYCNN_DRIVES = Get-FixedDriveLetters
    $env:CUEMATE_ESSENTIA_SEMANTIC_DRIVES = $env:CUEMATE_MUSICALKEYCNN_DRIVES
    $remoteUrl = Get-CueMateRemoteUrl
    if ($remoteUrl) {
        $env:CUEMATE_REMOTE_URL = $remoteUrl
    }
}

function Get-TailscaleCommand {
    $candidates = @(
        "$env:ProgramFiles\Tailscale\tailscale.exe",
        "${env:ProgramFiles(x86)}\Tailscale\tailscale.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate -PathType Leaf) {
            return $candidate
        }
    }
    $cmd = Get-Command "tailscale.exe" -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return $null
}

function Get-CueMateRemoteUrl {
    if (-not (Test-Path $remoteConfigPath -PathType Leaf)) {
        return ""
    }
    try {
        $payload = Get-Content $remoteConfigPath -Raw | ConvertFrom-Json
        if ($payload.enabled -and $payload.remote_url) {
            return [string]$payload.remote_url
        }
    } catch {
        return ""
    }
    return ""
}

function Save-CueMateRemoteConfig {
    param(
        [bool]$Enabled,
        [string]$RemoteUrl,
        [string]$Message
    )
    [pscustomobject]@{
        enabled = $Enabled
        mode = "tailscale"
        remote_url = $RemoteUrl
        message = $Message
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $remoteConfigPath -Encoding UTF8
}

function Get-TailscaleRemoteUrl {
    param([string]$Tailscale)
    $json = & $Tailscale status --json 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $json) {
        return ""
    }
    try {
        $status = $json | ConvertFrom-Json
        $dnsName = [string]$status.Self.DNSName
        if (-not $dnsName) {
            return ""
        }
        return "https://" + $dnsName.TrimEnd(".")
    } catch {
        return ""
    }
}

function Ensure-TailscaleServe {
    $tailscale = Get-TailscaleCommand
    if (-not $tailscale) {
        Save-CueMateRemoteConfig -Enabled $false -RemoteUrl "" -Message "Tailscale is not installed."
        return
    }
    $remoteUrl = Get-TailscaleRemoteUrl -Tailscale $tailscale
    if (-not $remoteUrl) {
        Save-CueMateRemoteConfig -Enabled $false -RemoteUrl "" -Message "Tailscale is installed but not authenticated."
        return
    }

    $serveAttempts = @(
        @("serve", "--bg", "http://127.0.0.1:8080"),
        @("serve", "--bg", "8080")
    )
    foreach ($args in $serveAttempts) {
        & $tailscale @args *> (Join-Path $logDir "tailscale-serve.log")
        if ($LASTEXITCODE -eq 0) {
            Save-CueMateRemoteConfig -Enabled $true -RemoteUrl $remoteUrl -Message "Tailscale Serve configured."
            $env:CUEMATE_REMOTE_URL = $remoteUrl
            return
        }
    }
    Save-CueMateRemoteConfig -Enabled $false -RemoteUrl $remoteUrl -Message "Tailscale is logged in, but Serve could not be configured. See tailscale-serve.log."
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
    if (-not (Test-Path $FilePath -PathType Leaf)) {
        throw "Cannot start $Name because $FilePath is missing. Reinstall CueMate or rerun setup."
    }
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
    Write-Host "CueMate needs to finish first-time setup. Logs are in $logDir."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $bootstrap -InstallDir $InstallDir -AppDataRoot $AppDataRoot
    if ($LASTEXITCODE -ne 0) {
        throw "CueMate setup could not finish. Check bootstrap.log in $logDir, fix any Python/Docker prompt, then launch CueMate again."
    }
    if ((-not (Test-Path $venvPython -PathType Leaf)) -or (-not (Test-Path $databasePath -PathType Leaf))) {
        throw "CueMate core setup is not ready yet. Finish any required prerequisite prompts, then launch CueMate again. See logs in $logDir."
    }
}

try {
    Set-CueMateEnvironment
    Ensure-BootstrapComplete
    Set-CueMateEnvironment
    Ensure-TailscaleServe
    Set-CueMateEnvironment

    Push-Location $InstallDir
    try {
        if (-not (Test-TcpPort -HostName "127.0.0.1" -Port 47834)) {
            Write-Host "Starting CueMate scorer..."
            Start-LoggedProcess -Name "scorer" -FilePath $venvPython -Arguments @("-m", "cuemate_analysis", "serve-scoring", "--host", "127.0.0.1", "--port", "47834")
            Wait-Until -Predicate { Test-TcpPort -HostName "127.0.0.1" -Port 47834 } -TimeoutSeconds 45 -Description "CueMate scorer" | Out-Null
        }

        if (-not (Test-HttpOk -Url "$apiUrl/healthz")) {
            Write-Host "Starting CueMate API..."
            $apiExe = Join-Path $InstallDir "apiserver.exe"
            if (-not (Test-Path $apiExe -PathType Leaf)) {
                throw "CueMate API executable is missing at $apiExe"
            }
            Start-LoggedProcess -Name "apiserver" -FilePath $apiExe -Arguments @()
            Wait-Until -Predicate { Test-HttpOk -Url "$apiUrl/healthz" } -TimeoutSeconds 45 -Description "CueMate API" | Out-Null
        }

        if (-not $NoBrowser) {
            Write-Host "Opening CueMate at $apiUrl"
            Start-Process $apiUrl | Out-Null
        }
    } finally {
        Pop-Location
    }
} finally {
    Stop-Transcript | Out-Null
}
