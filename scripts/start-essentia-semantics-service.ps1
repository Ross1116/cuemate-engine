$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$imageTag = if ($env:CUEMATE_ESSENTIA_SEMANTIC_IMAGE) { $env:CUEMATE_ESSENTIA_SEMANTIC_IMAGE } else { "cuemate-essentia-semantics:local" }
$serviceName = if ($env:CUEMATE_ESSENTIA_SEMANTIC_SERVICE_NAME) { $env:CUEMATE_ESSENTIA_SEMANTIC_SERVICE_NAME } else { "cuemate-essentia-semantics-service" }
$servicePort = if ($env:CUEMATE_ESSENTIA_SEMANTIC_SERVICE_PORT) { $env:CUEMATE_ESSENTIA_SEMANTIC_SERVICE_PORT } else { "47833" }
$driveRoots = if ($env:CUEMATE_ESSENTIA_SEMANTIC_DRIVES) { $env:CUEMATE_ESSENTIA_SEMANTIC_DRIVES } else { "d" }
$device = if ($env:CUEMATE_ESSENTIA_SEMANTIC_DEVICE) { $env:CUEMATE_ESSENTIA_SEMANTIC_DEVICE } else { "auto" }
$familyPolicy = if ($env:CUEMATE_ESSENTIA_SEMANTIC_MODEL_FAMILY_POLICY) { $env:CUEMATE_ESSENTIA_SEMANTIC_MODEL_FAMILY_POLICY } else { "best_per_task" }
$modelRoot = if ($env:CUEMATE_ESSENTIA_SEMANTIC_MODEL_ROOT) { $env:CUEMATE_ESSENTIA_SEMANTIC_MODEL_ROOT } else { "python/models/essentia_semantics" }

function Start-EssentiaContainer {
    param(
        [string[]]$DockerCommand
    )

    $output = & docker @DockerCommand 2>&1
    if ($LASTEXITCODE -ne 0) {
        return @{
            Success = $false
            Output = $output
            ExitCode = $LASTEXITCODE
        }
    }

    $containerId = ($output | Select-Object -First 1)
    if ($null -ne $containerId) {
        $containerId = $containerId.Trim()
    }
    return @{
        Success = $true
        Output = $output
        ExitCode = 0
        ContainerId = $containerId
    }
}

function Convert-EssentiaCommandToCpu {
    param(
        [string[]]$DockerCommand,
        [string]$ImageTag
    )

    $converted = @()
    for ($i = 0; $i -lt $DockerCommand.Count; $i++) {
        if ($DockerCommand[$i] -eq "--gpus" -and $i + 1 -lt $DockerCommand.Count) {
            $i++
            continue
        }
        if ($DockerCommand[$i] -eq $ImageTag) {
            $converted += @("--env", "CUDA_VISIBLE_DEVICES=-1")
        }
        $converted += $DockerCommand[$i]
    }
    return $converted
}

function Test-HostGpuAvailable {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        return $false
    }
    & $nvidiaSmi.Source "-L" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-RepoRelativeUnixPath {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )

    if ($TargetPath.Equals($BasePath, [System.StringComparison]::OrdinalIgnoreCase)) {
        return "."
    }

    $baseWithSeparator = $BasePath.TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar
    if ($TargetPath.StartsWith($baseWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $TargetPath.Substring($baseWithSeparator.Length).Replace("\", "/")
    }

    $baseUri = [System.Uri]::new(($baseWithSeparator -replace "\\", "/"))
    $targetUri = [System.Uri]::new(($TargetPath -replace "\\", "/"))
    return $baseUri.MakeRelativeUri($targetUri).ToString().Replace("%20", " ")
}

$existingContainerId = (& docker ps -a --filter "name=^$serviceName$" --format "{{.ID}}") | Select-Object -First 1
if ($existingContainerId) {
    & docker rm -f $serviceName | Out-Null
}

$resolvedRepoRoot = [System.IO.Path]::GetFullPath([string]$repoRoot)
$resolvedModelRoot = [System.IO.Path]::GetFullPath((Join-Path $resolvedRepoRoot $modelRoot))
$modelVolumeArgs = @()
$modelPathInContainer = ""

if (
    $resolvedModelRoot.Equals($resolvedRepoRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    $resolvedModelRoot.StartsWith($resolvedRepoRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
    $resolvedModelRoot.StartsWith($resolvedRepoRoot + [System.IO.Path]::AltDirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
) {
    $relativePath = Get-RepoRelativeUnixPath -BasePath $resolvedRepoRoot -TargetPath $resolvedModelRoot
    $modelPathInContainer = "/workspace/$relativePath"
} else {
    $modelVolumeArgs = @("--volume", "$resolvedModelRoot`:/models:ro")
    $modelPathInContainer = "/models"
}

$command = @(
    "run", "-d", "--rm",
    "--name", $serviceName,
    "--publish", "127.0.0.1:$servicePort`:$servicePort",
    "--env", "ESSENTIA_SEMANTIC_SERVICE_PORT=$servicePort",
    "--env", "CUEMATE_ESSENTIA_SEMANTIC_MODEL_ROOT=$modelPathInContainer",
    "--env", "CUEMATE_ESSENTIA_SEMANTIC_DEVICE=$device",
    "--env", "CUEMATE_ESSENTIA_SEMANTIC_MODEL_FAMILY_POLICY=$familyPolicy",
    "--env", "PYTHONPATH=/workspace/python/src",
    "--volume", "$resolvedRepoRoot`:/workspace:ro"
)

if ($device -eq "cuda") {
    $command += @("--gpus", "all")
} elseif ($device -eq "auto") {
    if (Test-HostGpuAvailable) {
        $command += @("--gpus", "all")
    } else {
        $command += @("--env", "CUDA_VISIBLE_DEVICES=-1")
    }
} else {
    $command += @("--env", "CUDA_VISIBLE_DEVICES=-1")
}

foreach ($drive in ($driveRoots -split "," | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })) {
    $source = ($drive + ":\")
    $command += @("--mount", "type=bind,source=$source,target=/host/$drive,readonly")
}

$command += $modelVolumeArgs
$command += @(
    $imageTag,
    "python",
    "/workspace/docker/essentia_semantics/service.py"
)

Push-Location $repoRoot
try {
    $launch = Start-EssentiaContainer -DockerCommand $command
    if (-not $launch.Success) {
        $retryOnCpu = $false
        if ($device -eq "auto") {
            $stderrText = [string]::Join("`n", @($launch.Output))
            if (
                $stderrText -match "could not select device driver" -or
                $stderrText -match "capabilities:\s*\[\[gpu\]\]" -or
                $stderrText -match "nvidia-container-runtime" -or
                $stderrText -match "unknown flag:\s*--gpus"
            ) {
                $retryOnCpu = $true
            }
        }
        if (-not $retryOnCpu) {
            exit $launch.ExitCode
        }

        $cpuCommand = Convert-EssentiaCommandToCpu -DockerCommand $command -ImageTag $imageTag
        $launch = Start-EssentiaContainer -DockerCommand $cpuCommand
        if (-not $launch.Success) {
            exit $launch.ExitCode
        }
    }
    $containerId = $launch.ContainerId

    $deadline = (Get-Date).AddSeconds(30)
    $healthUrl = "http://127.0.0.1:$servicePort/health"
    $lastHealthError = $null
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3
            if ($response.status -eq "ok") {
                Write-Output "Essentia semantics service is healthy on $healthUrl"
                exit 0
            }
        } catch {
            $lastHealthError = $_
            Write-Verbose "Essentia semantics health probe failed: $_"
        }
        Start-Sleep -Milliseconds 500
    }

    if ($containerId) {
        try {
            & docker rm -f $containerId | Out-Null
        }
        catch {
            Write-Warning "Failed to clean up timed-out Essentia semantics container ${containerId}: $_"
        }
    }
    if ($lastHealthError) {
        Write-Error "Essentia semantics service did not become healthy on $healthUrl within 30 seconds. Last health error: $lastHealthError"
    }
    else {
        Write-Error "Essentia semantics service did not become healthy on $healthUrl within 30 seconds."
    }
    exit 1
}
finally {
    Pop-Location
}
