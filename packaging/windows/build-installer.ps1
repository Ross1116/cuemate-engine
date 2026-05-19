param(
    [string]$Version = "0.1.0",
    [switch]$SkipInstaller,
    [switch]$SkipPrerequisiteInstall
)

$ErrorActionPreference = "Stop"

$packagingRoot = Split-Path -Parent $PSCommandPath
$repoRoot = Resolve-Path (Join-Path $packagingRoot "..\..")
$distRoot = Join-Path $repoRoot "dist\windows-installer"
$stageRoot = Join-Path $distRoot "stage\CueMate"
$outputRoot = Join-Path $distRoot "output"
$goExe = Join-Path $stageRoot "apiserver.exe"

function Get-RequiredCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$InstallHint
    )

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Missing required command '$Name'. $InstallHint"
    }
    return $cmd.Source
}

function Assert-SourcePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path $Path)) {
        throw "Cannot build installer because $Description is missing at $Path"
    }
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $repoRoot
    )

    Write-Host ">> $FilePath $($Arguments -join ' ')"
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$FilePath exited with code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

function Invoke-RobocopyChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination,
        [string[]]$ExtraArgs = @()
    )

    Assert-SourcePath -Path $Source -Description "source folder"
    $robocopy = Get-RequiredCommand -Name "robocopy.exe" -InstallHint "Robocopy is included with Windows; run this from a normal Windows PowerShell session."
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & $robocopy $Source $Destination /MIR /NFL /NDL /NJH /NJS /NP @ExtraArgs | Out-Host
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed copying $Source to $Destination with exit code $LASTEXITCODE"
    }
}

function Invoke-PackagingSmoke {
    param([switch]$RequireInstaller)

    $powershell = Get-RequiredCommand -Name "powershell.exe" -InstallHint "PowerShell is required to run CueMate packaging checks."
    $invokeArgs = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $packagingRoot "Test-PackagingSmoke.ps1")
    )
    if ($RequireInstaller) {
        $invokeArgs += "-RequireInstaller"
    }
    Invoke-LoggedCommand -FilePath $powershell -Arguments $invokeArgs -WorkingDirectory $repoRoot
}

function Get-InnoCompiler {
    $cmd = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate -PathType Leaf) {
            return $candidate
        }
    }

    if (-not $SkipPrerequisiteInstall) {
        $winget = Get-Command "winget.exe" -ErrorAction SilentlyContinue
        if ($winget) {
            Invoke-LoggedCommand -FilePath $winget.Source -Arguments @(
                "install",
                "--id", "JRSoftware.InnoSetup",
                "--source", "winget",
                "--accept-package-agreements",
                "--accept-source-agreements"
            )
            foreach ($candidate in $candidates) {
                if (Test-Path $candidate -PathType Leaf) {
                    return $candidate
                }
            }
        }
    }

    throw "Could not find Inno Setup compiler (ISCC.exe). Install Inno Setup 6, or rerun with -SkipInstaller to stage and smoke-test without producing CueMateSetup.exe."
}

function Assert-StagedRuntime {
    foreach ($item in @(
        @{ Path = (Join-Path $stageRoot "apiserver.exe"); Description = "Go API executable" },
        @{ Path = (Join-Path $stageRoot "web\dist\index.html"); Description = "built web app" },
        @{ Path = (Join-Path $stageRoot "python\pyproject.toml"); Description = "Python package" },
        @{ Path = (Join-Path $stageRoot "docker"); Description = "Docker runtime assets" },
        @{ Path = (Join-Path $stageRoot "config"); Description = "runtime config" },
        @{ Path = (Join-Path $stageRoot "db\schema.sql"); Description = "SQLite schema" },
        @{ Path = (Join-Path $stageRoot "scripts\docker-compose.ps1"); Description = "runtime scripts" },
        @{ Path = (Join-Path $stageRoot "Bootstrap-CueMate.ps1"); Description = "bootstrap script" },
        @{ Path = (Join-Path $stageRoot "Start-CueMate.ps1"); Description = "launcher script" },
        @{ Path = (Join-Path $stageRoot "CueMate-Common.psm1"); Description = "shared PowerShell module" },
        @{ Path = (Join-Path $stageRoot "README.md"); Description = "user README" },
        @{ Path = (Join-Path $stageRoot ".env.example"); Description = "environment example" }
    )) {
        Assert-SourcePath -Path $item.Path -Description $item.Description
    }
}

Write-Host "Preparing CueMate installer build $Version"
Get-RequiredCommand -Name "npm.cmd" -InstallHint "Install Node.js LTS, then rerun this script." | Out-Null
Get-RequiredCommand -Name "go.exe" -InstallHint "Install Go 1.24 or newer, then rerun this script." | Out-Null
Get-RequiredCommand -Name "powershell.exe" -InstallHint "PowerShell is required to build CueMate." | Out-Null

foreach ($source in @(
    @{ Path = (Join-Path $repoRoot "web\package.json"); Description = "web package manifest" },
    @{ Path = (Join-Path $repoRoot "go\go.mod"); Description = "Go module" },
    @{ Path = (Join-Path $repoRoot "python\pyproject.toml"); Description = "Python package" },
    @{ Path = (Join-Path $repoRoot "docker"); Description = "Docker assets" },
    @{ Path = (Join-Path $repoRoot "db\schema.sql"); Description = "database schema" }
)) {
    Assert-SourcePath -Path $source.Path -Description $source.Description
}

Remove-Item $distRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stageRoot, $outputRoot | Out-Null

Invoke-LoggedCommand -FilePath (Get-RequiredCommand -Name "npm.cmd" -InstallHint "Install Node.js LTS, then rerun this script.") -Arguments @("run", "build") -WorkingDirectory (Join-Path $repoRoot "web")
Invoke-LoggedCommand -FilePath (Get-RequiredCommand -Name "go.exe" -InstallHint "Install Go 1.24 or newer, then rerun this script.") -Arguments @("build", "-o", $goExe, "./cmd/apiserver") -WorkingDirectory (Join-Path $repoRoot "go")

Invoke-RobocopyChecked -Source (Join-Path $repoRoot "python") -Destination (Join-Path $stageRoot "python") -ExtraArgs @(
    "/XD", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info",
    "/XF", "*.pyc", "*.pyo"
)
Invoke-RobocopyChecked -Source (Join-Path $repoRoot "docker") -Destination (Join-Path $stageRoot "docker")
Invoke-RobocopyChecked -Source (Join-Path $repoRoot "config") -Destination (Join-Path $stageRoot "config")
Invoke-RobocopyChecked -Source (Join-Path $repoRoot "db") -Destination (Join-Path $stageRoot "db")
Invoke-RobocopyChecked -Source (Join-Path $repoRoot "scripts") -Destination (Join-Path $stageRoot "scripts")
Invoke-RobocopyChecked -Source (Join-Path $repoRoot "web\dist") -Destination (Join-Path $stageRoot "web\dist")

Copy-Item (Join-Path $repoRoot ".env.example") (Join-Path $stageRoot ".env.example") -Force
Copy-Item (Join-Path $repoRoot "README.md") (Join-Path $stageRoot "README.md") -Force
Copy-Item (Join-Path $packagingRoot "Bootstrap-CueMate.ps1") (Join-Path $stageRoot "Bootstrap-CueMate.ps1") -Force
Copy-Item (Join-Path $packagingRoot "Start-CueMate.ps1") (Join-Path $stageRoot "Start-CueMate.ps1") -Force
Copy-Item (Join-Path $packagingRoot "CueMate-Common.psm1") (Join-Path $stageRoot "CueMate-Common.psm1") -Force
Set-Content -Path (Join-Path $stageRoot "VERSION") -Value $Version -Encoding UTF8

Assert-StagedRuntime
Invoke-PackagingSmoke

if ($SkipInstaller) {
    Write-Host "Staged CueMate runtime at $stageRoot"
    exit 0
}

$iscc = Get-InnoCompiler
$iss = Join-Path $packagingRoot "CueMate.iss"
Invoke-LoggedCommand -FilePath $iscc -Arguments @(
    "/DMyAppVersion=$Version",
    "/DSourceDir=$stageRoot",
    "/DOutputDir=$outputRoot",
    $iss
) -WorkingDirectory $packagingRoot

Invoke-PackagingSmoke -RequireInstaller

Write-Host "Installer written to $outputRoot"
