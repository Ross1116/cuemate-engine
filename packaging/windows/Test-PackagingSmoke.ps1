param(
    [string]$StageRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")) "dist\windows-installer\stage\CueMate"),
    [string]$InstallerPath = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")) "dist\windows-installer\output\CueMateSetup.exe"),
    [switch]$RequireInstaller
)

$ErrorActionPreference = "Stop"

function Assert-Path {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [string]$Description
    )
    if (-not (Test-Path $Path)) {
        throw "Missing $Description at $Path"
    }
    Write-Host "[ok] $Description"
}

function Assert-PowerShellParses {
    param([string]$Path)
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors.Count -gt 0) {
        throw "PowerShell parse errors in $Path`: $($errors[0].Message)"
    }
    Write-Host "[ok] parses: $Path"
}

$StageRoot = [System.IO.Path]::GetFullPath($StageRoot)

Assert-Path -Path $StageRoot -Description "staged CueMate runtime"
Assert-Path -Path (Join-Path $StageRoot "apiserver.exe") -Description "Go API executable"
Assert-Path -Path (Join-Path $StageRoot "web\dist\index.html") -Description "built web app"
Assert-Path -Path (Join-Path $StageRoot "python\pyproject.toml") -Description "Python package"
Assert-Path -Path (Join-Path $StageRoot "docker") -Description "Docker runtime assets"
Assert-Path -Path (Join-Path $StageRoot "config") -Description "runtime config"
Assert-Path -Path (Join-Path $StageRoot "db\schema.sql") -Description "SQLite schema"
Assert-Path -Path (Join-Path $StageRoot "scripts\docker-compose.ps1") -Description "runtime scripts"
Assert-Path -Path (Join-Path $StageRoot "Bootstrap-CueMate.ps1") -Description "bootstrap script"
Assert-Path -Path (Join-Path $StageRoot "Start-CueMate.ps1") -Description "launcher script"
Assert-Path -Path (Join-Path $StageRoot "CueMate-Common.psm1") -Description "shared PowerShell module"
Assert-Path -Path (Join-Path $StageRoot ".env.example") -Description "environment example"
Assert-Path -Path (Join-Path $StageRoot "README.md") -Description "user README"
Assert-Path -Path (Join-Path $StageRoot "VERSION") -Description "runtime version file"

Assert-PowerShellParses -Path (Join-Path $StageRoot "Bootstrap-CueMate.ps1")
Assert-PowerShellParses -Path (Join-Path $StageRoot "Start-CueMate.ps1")
Assert-PowerShellParses -Path (Join-Path $StageRoot "CueMate-Common.psm1")
Assert-PowerShellParses -Path (Join-Path $PSScriptRoot "build-installer.ps1")

if ($RequireInstaller) {
    Assert-Path -Path $InstallerPath -Description "CueMateSetup.exe"
}

Write-Host "CueMate packaging smoke check passed."
