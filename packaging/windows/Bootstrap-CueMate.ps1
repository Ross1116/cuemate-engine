param(
    [string]$InstallDir = $PSScriptRoot,
    [string]$AppDataRoot = (Join-Path $env:LOCALAPPDATA "CueMate"),
    [switch]$SkipWingetInstall,
    [switch]$SkipDockerSetup,
    [switch]$SkipModelSetup
)

$ErrorActionPreference = "Stop"

$InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
$AppDataRoot = [System.IO.Path]::GetFullPath($AppDataRoot)
$dataDir = Join-Path $AppDataRoot "data"
$logDir = Join-Path $AppDataRoot "logs"
$statePath = Join-Path $AppDataRoot "setup-state.json"
$venvDir = Join-Path $AppDataRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$databasePath = Join-Path $dataDir "cuemate.db"
$cachePath = Join-Path $dataDir "inference-cache.db"
$bootstrapLog = Join-Path $logDir "bootstrap.log"

New-Item -ItemType Directory -Force -Path $AppDataRoot, $dataDir, $logDir | Out-Null
Start-Transcript -Path $bootstrapLog -Append | Out-Null

function Write-SetupState {
    param(
        [string]$Step,
        [string]$Status,
        [string]$Message = ""
    )

    [pscustomobject]@{
        step = $Step
        status = $Status
        message = $Message
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $statePath -Encoding UTF8
}

function Get-SetupStateStatus {
    if (-not (Test-Path $statePath -PathType Leaf)) {
        return ""
    }
    try {
        return (Get-Content $statePath -Raw | ConvertFrom-Json).status
    } catch {
        return ""
    }
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$ScriptBlock
    )

    Write-SetupState -Step $Name -Status "running"
    Write-Host "[CueMate setup] $Name"
    & $ScriptBlock
    if ((Get-SetupStateStatus) -eq "blocked") {
        return
    }
    Write-SetupState -Step $Name -Status "complete"
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $InstallDir
    )

    Push-Location $WorkingDirectory
    try {
        Write-Host ">> $FilePath $($Arguments -join ' ')"
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$FilePath exited with code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

function Ensure-WingetPackage {
    param(
        [string]$CommandName,
        [string]$PackageId
    )

    if (Get-Command $CommandName -ErrorAction SilentlyContinue) {
        return
    }
    if ($SkipWingetInstall) {
        throw "$CommandName is required but was not found. Rerun setup without -SkipWingetInstall."
    }
    $winget = Get-Command "winget.exe" -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget is required to install $PackageId automatically."
    }
    Invoke-External -FilePath $winget.Source -Arguments @(
        "install",
        "--id", $PackageId,
        "--source", "winget",
        "--accept-package-agreements",
        "--accept-source-agreements"
    )
}

function Get-PythonCommand {
    $python = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if (-not $python) {
        return $null
    }
    $versionText = & $python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0 -or [version]$versionText -lt [version]"3.12") {
        return $null
    }
    return $python.Source
}

function Wait-DockerReady {
    param([int]$TimeoutSeconds = 300)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        & docker info *> $null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        Start-Sleep -Seconds 5
    }
    return $false
}

function Start-DockerDesktop {
    $candidates = @(
        "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
        "${env:ProgramFiles(x86)}\Docker\Docker\Docker Desktop.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate -PathType Leaf) {
            Start-Process -FilePath $candidate | Out-Null
            return
        }
    }
}

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

function Initialize-Database {
    if (Test-Path $databasePath -PathType Leaf) {
        return
    }
    $schemaPath = Join-Path $InstallDir "db\schema.sql"
    if (-not (Test-Path $schemaPath -PathType Leaf)) {
        throw "Missing database schema at $schemaPath"
    }
    $code = @"
import pathlib
import sqlite3

db_path = pathlib.Path(r"$databasePath")
schema_path = pathlib.Path(r"$schemaPath")
db_path.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(db_path) as conn:
    conn.executescript(schema_path.read_text(encoding="utf-8"))
"@
    Invoke-External -FilePath $venvPython -Arguments @("-c", $code)
}

try {
    Invoke-Step "install-prerequisites" {
        if (-not (Get-PythonCommand)) {
            if ($SkipWingetInstall) {
                throw "Python 3.12 is required but was not found. Rerun setup without -SkipWingetInstall."
            }
            $winget = Get-Command "winget.exe" -ErrorAction SilentlyContinue
            if (-not $winget) {
                throw "winget is required to install Python 3.12 automatically."
            }
            Invoke-External -FilePath $winget.Source -Arguments @(
                "install",
                "--id", "Python.Python.3.12",
                "--source", "winget",
                "--accept-package-agreements",
                "--accept-source-agreements"
            )
        }
        Ensure-WingetPackage -CommandName "docker.exe" -PackageId "Docker.DockerDesktop"
        Ensure-WingetPackage -CommandName "tailscale.exe" -PackageId "Tailscale.Tailscale"
    }

    Invoke-Step "prepare-python" {
        $python = Get-PythonCommand
        if (-not $python) {
            throw "Python 3.12 was not found after prerequisite installation. Open a new PowerShell and rerun CueMate."
        }
        if (-not (Test-Path $venvPython -PathType Leaf)) {
            Invoke-External -FilePath $python -Arguments @("-m", "venv", $venvDir)
        }
        Invoke-External -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-External -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", (Join-Path $InstallDir "python"))
    }

    Invoke-Step "initialize-data" {
        Set-CueMateEnvironment
        Initialize-Database
        @"
DATABASE_URL=sqlite:$databasePath
DBMATE_MIGRATIONS_DIR=db/migrations
DBMATE_SCHEMA_FILE=db/schema.sql
GO_API_ADDR=127.0.0.1:8080
SCORING_GRPC_ADDR=127.0.0.1:47834
SCORING_RPC_TIMEOUT_MS=250
CUEMATE_INFERENCE_CACHE_PATH=$cachePath
"@ | Set-Content -Path (Join-Path $InstallDir ".env") -Encoding UTF8
    }

    if (-not $SkipDockerSetup) {
        Invoke-Step "prepare-docker" {
            Start-DockerDesktop
            if (-not (Wait-DockerReady -TimeoutSeconds 300)) {
                Write-SetupState -Step "prepare-docker" -Status "blocked" -Message "Docker Desktop is not ready. Start Docker Desktop, finish any required login/restart, then launch CueMate again."
                Write-Warning "Docker Desktop is not ready yet. CueMate setup will resume on next launch."
                return
            }
            Invoke-External -FilePath "powershell.exe" -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $InstallDir "scripts\build-essentia-semantics-image.ps1"))
            Invoke-External -FilePath "powershell.exe" -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $InstallDir "scripts\build-musicalkeycnn-image.ps1"))
        }
    }

    if (-not $SkipModelSetup) {
        Invoke-Step "prepare-models" {
            Set-CueMateEnvironment
            if (-not $SkipDockerSetup -and -not (Wait-DockerReady -TimeoutSeconds 30)) {
                Write-SetupState -Step "prepare-models" -Status "blocked" -Message "Docker Desktop is not ready. Model setup will resume on next launch."
                Write-Warning "Skipping model setup until Docker Desktop is ready."
                return
            }
            Invoke-External -FilePath $venvPython -Arguments @("-m", "cuemate_analysis", "download-essentia-semantic-models")
            Invoke-External -FilePath $venvPython -Arguments @("-m", "cuemate_analysis", "prewarm-model-services")
        }
    }

    Write-SetupState -Step "complete" -Status "complete"
    Write-Host "CueMate setup complete."
} catch {
    Write-SetupState -Step "failed" -Status "failed" -Message $_.Exception.Message
    Write-Error $_
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
