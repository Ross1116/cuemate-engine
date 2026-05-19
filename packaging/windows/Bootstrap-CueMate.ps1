param(
    [string]$InstallDir = $PSScriptRoot,
    [string]$AppDataRoot = (Join-Path $env:LOCALAPPDATA "CueMate"),
    [switch]$SkipWingetInstall,
    [switch]$SkipTailscaleInstall,
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
$script:SetupBlocked = $false
$script:DockerInstallDeferred = $false

New-Item -ItemType Directory -Force -Path $AppDataRoot, $dataDir, $logDir | Out-Null
Import-Module (Join-Path $InstallDir "CueMate-Common.psm1") -Force
Start-Transcript -Path $bootstrapLog -Append | Out-Null

function Read-SetupState {
    if (-not (Test-Path $statePath -PathType Leaf)) {
        return [pscustomobject]@{
            core_ready = $false
            docker_ready = $false
            model_ready = $false
            mobile_ready = $false
        }
    }
    try {
        return Get-Content $statePath -Raw | ConvertFrom-Json
    } catch {
        return [pscustomobject]@{
            core_ready = $false
            docker_ready = $false
            model_ready = $false
            mobile_ready = $false
        }
    }
}

function Write-SetupState {
    param(
        [string]$Step,
        [string]$Status,
        [string]$Message = "",
        [Nullable[bool]]$CoreReady = $null,
        [Nullable[bool]]$DockerReady = $null,
        [Nullable[bool]]$ModelReady = $null,
        [Nullable[bool]]$MobileReady = $null
    )

    $previous = Read-SetupState
    $coreReadyValue = if ($null -ne $CoreReady) { [bool]$CoreReady } else { [bool]$previous.core_ready }
    $dockerReadyValue = if ($null -ne $DockerReady) { [bool]$DockerReady } else { [bool]$previous.docker_ready }
    $modelReadyValue = if ($null -ne $ModelReady) { [bool]$ModelReady } else { [bool]$previous.model_ready }
    $mobileReadyValue = if ($null -ne $MobileReady) { [bool]$MobileReady } else { [bool]$previous.mobile_ready }

    if ($Status -eq "blocked") {
        $script:SetupBlocked = $true
    }

    [pscustomobject]@{
        step = $Step
        status = $Status
        message = $Message
        core_ready = $coreReadyValue
        docker_ready = $dockerReadyValue
        model_ready = $modelReadyValue
        mobile_ready = $mobileReadyValue
        log_dir = $logDir
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

function Test-SetupBlocked {
    return $script:SetupBlocked -or ((Get-SetupStateStatus) -eq "blocked")
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
        $script:SetupBlocked = $true
        return
    }
    Write-SetupState -Step $Name -Status "complete"
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $InstallDir,
        [int]$TimeoutSeconds = 0
    )

    Push-Location $WorkingDirectory
    try {
        Write-Host ">> $FilePath $($Arguments -join ' ')"
        if ($TimeoutSeconds -gt 0) {
            $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory -NoNewWindow -PassThru
            if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
                $taskkill = Get-Command "taskkill.exe" -ErrorAction SilentlyContinue
                if ($taskkill) {
                    & $taskkill.Source /PID $process.Id /T /F *> $null
                } else {
                    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                }
                throw "$FilePath timed out after $TimeoutSeconds seconds"
            }
            if ($process.ExitCode -ne 0) {
                throw "$FilePath exited with code $($process.ExitCode)"
            }
        } else {
            & $FilePath @Arguments
            if ($LASTEXITCODE -ne 0) {
                throw "$FilePath exited with code $LASTEXITCODE"
            }
        }
    } finally {
        Pop-Location
    }
}

function Update-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (($machinePath, $userPath) | Where-Object { $_ }) -join ";"
}

function Get-InstalledCommand {
    param(
        [string]$CommandName,
        [string[]]$CandidatePaths = @()
    )

    $cmd = Get-Command $CommandName -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    foreach ($candidate in $CandidatePaths) {
        if ($candidate -and (Test-Path $candidate -PathType Leaf)) {
            return $candidate
        }
    }
    return $null
}

function Invoke-WingetInstall {
    param(
        [string]$PackageId,
        [int]$TimeoutSeconds = 1800
    )

    $winget = Get-Command "winget.exe" -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget is required to install $PackageId automatically."
    }
    Invoke-External -FilePath $winget.Source -TimeoutSeconds $TimeoutSeconds -Arguments @(
        "install",
        "--id", $PackageId,
        "--source", "winget",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity",
        "--silent"
    )
    Update-ProcessPath
}

function Ensure-WingetPackage {
    param(
        [string]$CommandName,
        [string]$PackageId,
        [string[]]$CandidatePaths = @(),
        [switch]$Optional
    )

    if (Get-InstalledCommand -CommandName $CommandName -CandidatePaths $CandidatePaths) {
        return $true
    }
    if ($SkipWingetInstall) {
        if ($Optional) {
            Write-Warning "$CommandName was not found. Skipping automatic $PackageId install because -SkipWingetInstall was supplied."
            return $false
        }
        throw "$CommandName is required but was not found. Rerun setup without -SkipWingetInstall."
    }
    try {
        Invoke-WingetInstall -PackageId $PackageId
    } catch {
        if ($Optional) {
            Write-Warning "Could not install $PackageId automatically: $($_.Exception.Message)"
            return $false
        }
        throw
    }
    return [bool](Get-InstalledCommand -CommandName $CommandName -CandidatePaths $CandidatePaths)
}

function Get-DockerCommand {
    return Get-InstalledCommand -CommandName "docker.exe" -CandidatePaths @(
        "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe",
        "${env:ProgramFiles(x86)}\Docker\Docker\resources\bin\docker.exe"
    )
}

function Get-PythonCommand {
    $python = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if (-not $python) {
        return $null
    }
    $versionText = & $python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    $versionMatch = [regex]::Match([string]$versionText, '^\s*(\d+\.\d+)')
    if (-not $versionMatch.Success) {
        return $null
    }
    try {
        $parsedVersion = [version]$versionMatch.Groups[1].Value
    } catch {
        return $null
    }
    if ($parsedVersion -lt [version]"3.12") {
        return $null
    }
    return $python.Source
}

function Wait-DockerReady {
    param([int]$TimeoutSeconds = 300)

    $docker = Get-DockerCommand
    if (-not $docker) {
        return $false
    }
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            & $docker info *> $null
            if ($LASTEXITCODE -eq 0) {
                return $true
            }
        } catch {
            # Docker Desktop can be installed but still starting, logged out, or waiting
            # for WSL/reboot. Treat that as "not ready" so setup can resume later.
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

function Initialize-Database {
    if (Test-Path $databasePath -PathType Leaf) {
        return
    }
    $schemaPath = Join-Path $InstallDir "db\schema.sql"
    if (-not (Test-Path $schemaPath -PathType Leaf)) {
        throw "Missing database schema at $schemaPath"
    }
    $initScript = Join-Path $AppDataRoot "initialize-db.py"
    $code = @"
import pathlib
import sqlite3

db_path = pathlib.Path(r"$databasePath")
schema_path = pathlib.Path(r"$schemaPath")
db_path.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(db_path) as conn:
    conn.executescript(schema_path.read_text(encoding="utf-8"))
"@
    Set-Content -Path $initScript -Value $code -Encoding UTF8
    Invoke-External -FilePath $venvPython -Arguments @($initScript)
}

try {
    Invoke-Step "install-prerequisites" {
        if (-not (Get-PythonCommand)) {
            if ($SkipWingetInstall) {
                throw "CueMate needs Python 3.12. Install Python 3.12, or rerun setup without -SkipWingetInstall so CueMate can install it with winget."
            }
            $winget = Get-Command "winget.exe" -ErrorAction SilentlyContinue
            if (-not $winget) {
                throw "CueMate could not find winget, so it cannot install Python 3.12 automatically. Install Python 3.12 manually, then launch CueMate again."
            }
            Write-SetupState -Step "install-prerequisites" -Status "running" -Message "Installing Python 3.12. If Windows updates PATH, launch CueMate again after this step finishes."
            Invoke-WingetInstall -PackageId "Python.Python.3.12"
        }
        $dockerReady = Ensure-WingetPackage -CommandName "docker.exe" -PackageId "Docker.DockerDesktop" -CandidatePaths @(
            "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe",
            "${env:ProgramFiles(x86)}\Docker\Docker\resources\bin\docker.exe"
        ) -Optional
        if (-not $dockerReady) {
            $script:DockerInstallDeferred = $true
            Write-SetupState -Step "install-prerequisites" -Status "running" -DockerReady $false -Message "Docker Desktop was not installed automatically. Install Docker Desktop manually, then launch CueMate again to resume full setup."
        }
        if (-not $SkipTailscaleInstall) {
            $tailscaleReady = Ensure-WingetPackage -CommandName "tailscale.exe" -PackageId "Tailscale.Tailscale" -CandidatePaths @(
                "$env:ProgramFiles\Tailscale\tailscale.exe",
                "${env:ProgramFiles(x86)}\Tailscale\tailscale.exe"
            ) -Optional
            if ($tailscaleReady) {
                Write-SetupState -Step "install-prerequisites" -Status "running" -MobileReady $true
            } else {
                Write-SetupState -Step "install-prerequisites" -Status "running" -MobileReady $false -Message "Tailscale was not installed automatically; mobile access can be enabled later."
            }
        } else {
            Write-SetupState -Step "install-prerequisites" -Status "running" -MobileReady $false -Message "Tailscale installation was skipped; mobile access can be enabled later."
        }
    }
    if (Test-SetupBlocked) { exit 0 }

    Invoke-Step "prepare-python" {
        $python = Get-PythonCommand
        if (-not $python) {
            throw "Python 3.12 was installed but is not visible to this shell yet. Close this window and launch CueMate again from the Start Menu."
        }
        if (-not (Test-Path $venvPython -PathType Leaf)) {
            Write-SetupState -Step "prepare-python" -Status "running" -Message "Creating CueMate's private Python environment."
            Invoke-External -FilePath $python -Arguments @("-m", "venv", $venvDir)
        }
        Write-SetupState -Step "prepare-python" -Status "running" -Message "Installing CueMate's Python analysis package. This can take a few minutes on first setup."
        Invoke-External -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-External -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", (Join-Path $InstallDir "python"))
    }
    if (Test-SetupBlocked) { exit 0 }

    Invoke-Step "initialize-data" {
        Set-CueMateEnvironment -InstallDir $InstallDir -DatabasePath $databasePath -CachePath $cachePath -VenvPython $venvPython -SetupStatePath $statePath -LogDir $logDir
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
        Write-SetupState -Step "initialize-data" -Status "running" -CoreReady $true
    }
    if (Test-SetupBlocked) { exit 0 }

    if (-not $SkipDockerSetup) {
        Invoke-Step "prepare-docker" {
            if ($script:DockerInstallDeferred -or -not (Get-DockerCommand)) {
                Write-SetupState -Step "prepare-docker" -Status "blocked" -DockerReady $false -Message "Docker Desktop is not installed. Install Docker Desktop, finish any required restart or WSL prompt, then launch CueMate again. Setup will resume automatically."
                Write-Warning "Docker Desktop is not installed. CueMate setup will resume on next launch."
                return
            }
            Write-SetupState -Step "prepare-docker" -Status "running" -Message "Starting Docker Desktop. If Docker asks you to sign in, update WSL, or restart Windows, finish that prompt and launch CueMate again."
            Start-DockerDesktop
            if (-not (Wait-DockerReady -TimeoutSeconds 300)) {
                Write-SetupState -Step "prepare-docker" -Status "blocked" -Message "Docker Desktop is not ready yet. Open Docker Desktop, finish any login, WSL, or restart prompt, then launch CueMate again. Setup will resume automatically."
                Write-Warning "Docker Desktop is not ready yet. CueMate setup will resume on next launch."
                return
            }
            Write-SetupState -Step "prepare-docker" -Status "running" -Message "Building CueMate model-service Docker images."
            Invoke-External -FilePath "powershell.exe" -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $InstallDir "scripts\build-essentia-semantics-image.ps1"))
            Invoke-External -FilePath "powershell.exe" -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $InstallDir "scripts\build-musicalkeycnn-image.ps1"))
            Write-SetupState -Step "prepare-docker" -Status "running" -DockerReady $true
        }
    } else {
        Write-SetupState -Step "prepare-docker" -Status "skipped" -DockerReady $false -Message "Docker setup was skipped; full model analysis will be unavailable until Docker is ready."
    }
    if (Test-SetupBlocked) { exit 0 }

    if (-not $SkipModelSetup) {
        Invoke-Step "prepare-models" {
            Set-CueMateEnvironment -InstallDir $InstallDir -DatabasePath $databasePath -CachePath $cachePath -VenvPython $venvPython -SetupStatePath $statePath -LogDir $logDir
            if (-not $SkipDockerSetup -and -not (Wait-DockerReady -TimeoutSeconds 30)) {
                Write-SetupState -Step "prepare-models" -Status "blocked" -Message "Docker Desktop is not ready yet. Model setup will resume the next time you launch CueMate."
                Write-Warning "Skipping model setup until Docker Desktop is ready."
                return
            }
            Write-SetupState -Step "prepare-models" -Status "running" -Message "Downloading analysis models. This needs internet access and can take several minutes."
            Invoke-External -FilePath $venvPython -Arguments @("-m", "cuemate_analysis", "download-essentia-semantic-models")
            if ($SkipDockerSetup) {
                Write-SetupState -Step "prepare-models" -Status "running" -ModelReady $false -Message "Model downloads finished. Docker prewarm was deferred because Docker setup was skipped."
                return
            }
            Write-SetupState -Step "prepare-models" -Status "running" -Message "Prewarming Docker model services."
            Invoke-External -FilePath $venvPython -Arguments @("-m", "cuemate_analysis", "prewarm-model-services")
            Write-SetupState -Step "prepare-models" -Status "running" -ModelReady $true
        }
    } else {
        Write-SetupState -Step "prepare-models" -Status "skipped" -ModelReady $false -Message "Model setup was skipped."
    }
    if (Test-SetupBlocked) { exit 0 }

    Write-SetupState -Step "complete" -Status "complete" -CoreReady $true -DockerReady (-not $SkipDockerSetup) -ModelReady (-not $SkipModelSetup -and -not $SkipDockerSetup) -MobileReady (-not $SkipTailscaleInstall)
    Write-Host "CueMate setup complete."
} catch {
    Write-SetupState -Step "failed" -Status "failed" -Message $_.Exception.Message
    Write-Error $_
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
