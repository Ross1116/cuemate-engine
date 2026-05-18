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

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & robocopy $Source $Destination /MIR /NFL /NDL /NJH /NJS /NP @ExtraArgs | Out-Host
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed copying $Source to $Destination with exit code $LASTEXITCODE"
    }
}

function Get-InnoCompiler {
    $cmd = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
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

    throw "Could not find ISCC.exe. Install Inno Setup 6 or rerun without -SkipInstaller after winget is available."
}

Write-Host "Preparing CueMate installer build $Version"
Remove-Item $distRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stageRoot, $outputRoot | Out-Null

Invoke-LoggedCommand -FilePath "npm" -Arguments @("run", "build") -WorkingDirectory (Join-Path $repoRoot "web")
Invoke-LoggedCommand -FilePath "go" -Arguments @("build", "-o", $goExe, "./cmd/apiserver") -WorkingDirectory (Join-Path $repoRoot "go")

Invoke-RobocopyChecked -Source (Join-Path $repoRoot "python") -Destination (Join-Path $stageRoot "python") -ExtraArgs @(
    "/XD", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info",
    "/XF", "*.pyc", "*.pyo"
)
Invoke-RobocopyChecked -Source (Join-Path $repoRoot "docker") -Destination (Join-Path $stageRoot "docker")
Invoke-RobocopyChecked -Source (Join-Path $repoRoot "config") -Destination (Join-Path $stageRoot "config")
Invoke-RobocopyChecked -Source (Join-Path $repoRoot "db") -Destination (Join-Path $stageRoot "db")
Invoke-RobocopyChecked -Source (Join-Path $repoRoot "scripts") -Destination (Join-Path $stageRoot "scripts")
Invoke-RobocopyChecked -Source (Join-Path $repoRoot "web\dist") -Destination (Join-Path $stageRoot "web\dist")

New-Item -ItemType Directory -Force -Path (Join-Path $stageRoot "docs") | Out-Null
Copy-Item (Join-Path $repoRoot "docs\Decision_Engine_Plan.md") (Join-Path $stageRoot "docs\Decision_Engine_Plan.md") -Force
Copy-Item (Join-Path $repoRoot ".env.example") (Join-Path $stageRoot ".env.example") -Force
Copy-Item (Join-Path $repoRoot "README.md") (Join-Path $stageRoot "README.md") -Force
Copy-Item (Join-Path $packagingRoot "Bootstrap-CueMate.ps1") (Join-Path $stageRoot "Bootstrap-CueMate.ps1") -Force
Copy-Item (Join-Path $packagingRoot "Start-CueMate.ps1") (Join-Path $stageRoot "Start-CueMate.ps1") -Force
Set-Content -Path (Join-Path $stageRoot "VERSION") -Value $Version -Encoding UTF8

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

Write-Host "Installer written to $outputRoot"
