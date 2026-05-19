function Get-FixedDriveLetters {
    $letters = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" |
        ForEach-Object { $_.DeviceID.TrimEnd(":").ToLowerInvariant() }
    if (-not $letters) {
        return "c"
    }
    return ($letters -join ",")
}

function Set-CueMateEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallDir,
        [Parameter(Mandatory = $true)]
        [string]$DatabasePath,
        [Parameter(Mandatory = $true)]
        [string]$CachePath,
        [Parameter(Mandatory = $true)]
        [string]$VenvPython,
        [string]$SetupStatePath,
        [string]$LogDir,
        [string]$RemoteUrl
    )

    $env:DATABASE_URL = "sqlite:$DatabasePath"
    $env:CUEMATE_REPO_ROOT = $InstallDir
    $env:CUEMATE_INFERENCE_CACHE_PATH = $CachePath
    if ($SetupStatePath) {
        $env:CUEMATE_SETUP_STATE_PATH = $SetupStatePath
    }
    if ($LogDir) {
        $env:CUEMATE_LOG_DIR = $LogDir
    }
    $env:WEB_DIST_DIR = (Join-Path $InstallDir "web\dist")
    $env:CUEMATE_PYTHON = $VenvPython
    $env:SCORING_GRPC_ADDR = "127.0.0.1:47834"
    $env:GO_API_ADDR = "127.0.0.1:8080"
    $env:CUEMATE_MUSICALKEYCNN_DRIVES = Get-FixedDriveLetters
    $env:CUEMATE_ESSENTIA_SEMANTIC_DRIVES = $env:CUEMATE_MUSICALKEYCNN_DRIVES
    if ($RemoteUrl) {
        $env:CUEMATE_REMOTE_URL = $RemoteUrl
    } else {
        Remove-Item Env:\CUEMATE_REMOTE_URL -ErrorAction SilentlyContinue
    }
}

Export-ModuleMember -Function Get-FixedDriveLetters, Set-CueMateEnvironment
