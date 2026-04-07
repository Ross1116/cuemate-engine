param(
    [string]$CheckpointPath
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$repoRootPath = [System.IO.Path]::GetFullPath([string]$repoRoot)
$repoRootPrefix = $repoRootPath.TrimEnd('\', '/')
$imageTag = if ($env:CUEMATE_MUSICALKEYCNN_IMAGE) { $env:CUEMATE_MUSICALKEYCNN_IMAGE } else { "cuemate-musicalkeycnn:local" }
$serviceName = if ($env:CUEMATE_MUSICALKEYCNN_SERVICE_NAME) { $env:CUEMATE_MUSICALKEYCNN_SERVICE_NAME } else { "cuemate-musicalkeycnn-service" }
$servicePort = if ($env:CUEMATE_MUSICALKEYCNN_SERVICE_PORT) { $env:CUEMATE_MUSICALKEYCNN_SERVICE_PORT } else { "47832" }
$driveRoots = if ($env:CUEMATE_MUSICALKEYCNN_DRIVES) { $env:CUEMATE_MUSICALKEYCNN_DRIVES } else { "d" }
$device = if ($env:CUEMATE_MUSICALKEYCNN_DEVICE) { $env:CUEMATE_MUSICALKEYCNN_DEVICE } else { "auto" }
$resolvedCheckpointPath = $null

if ([string]::IsNullOrWhiteSpace($CheckpointPath) -and $env:CUEMATE_MUSICALKEYCNN_MODEL) {
    $CheckpointPath = $env:CUEMATE_MUSICALKEYCNN_MODEL
}

$modelPathInContainer = "/workspace/python/models/musicalkeycnn/keynet.pt"
$modelVolumeArgs = @()

if (-not [string]::IsNullOrWhiteSpace($CheckpointPath)) {
    $resolvedCheckpointPath = Resolve-Path $CheckpointPath
    $resolvedCheckpointFullPath = [System.IO.Path]::GetFullPath([string]$resolvedCheckpointPath)

    if (
        $resolvedCheckpointFullPath.Equals($repoRootPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        $resolvedCheckpointFullPath.StartsWith($repoRootPrefix + "\", [System.StringComparison]::OrdinalIgnoreCase) -or
        $resolvedCheckpointFullPath.StartsWith($repoRootPrefix + "/", [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        $relativePath = $resolvedCheckpointFullPath.Substring($repoRootPrefix.Length).TrimStart('\', '/')
        $modelPathInContainer = "/workspace/" + ($relativePath -replace "\\", "/")
    }
    else {
        $checkpointDirectory = Split-Path -Parent $resolvedCheckpointFullPath
        $checkpointFileName = Split-Path -Leaf $resolvedCheckpointFullPath
        $modelPathInContainer = "/model/$checkpointFileName"
        $modelVolumeArgs += @("--volume", "$checkpointDirectory`:/model:ro")
    }
}

$existingContainerId = (& docker ps -a --filter "name=^$serviceName$" --format "{{.ID}}") | Select-Object -First 1
if ($existingContainerId) {
    & docker rm -f $serviceName | Out-Null
}

$command = @(
    "run", "-d", "--rm",
    "--name", $serviceName,
    "--publish", "127.0.0.1:$servicePort`:$servicePort",
    "--env", "MUSICALKEYCNN_SERVICE_PORT=$servicePort",
    "--env", "CUEMATE_MUSICALKEYCNN_DEFAULT_MODEL=$modelPathInContainer",
    "--env", "CUEMATE_MUSICALKEYCNN_DEFAULT_DEVICE=$device",
    "--env", "PYTHONPATH=/workspace/python/src",
    "--volume", "$repoRoot`:/workspace:ro"
)

if ($device -eq "cpu") {
    $command += @("--env", "CUDA_VISIBLE_DEVICES=-1")
}
elseif ($device -eq "cuda") {
    $command += @("--gpus", "all")
}
elseif ($device -eq "auto") {
    $gpuAvailable = $false
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        try {
            & $nvidiaSmi.Source "-L" | Out-Null
            $gpuAvailable = ($LASTEXITCODE -eq 0)
        }
        catch {
            $gpuAvailable = $false
        }
    }
    if ($gpuAvailable) {
        $command += @("--gpus", "all")
    }
    else {
        $command += @("--env", "CUDA_VISIBLE_DEVICES=-1")
    }
}

foreach ($modelArg in $modelVolumeArgs) {
    $command += $modelArg
}

foreach ($drive in ($driveRoots -split "," | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })) {
    $source = ($drive + ":\")
    $command += @("--mount", "type=bind,source=$source,target=/host/$drive,readonly")
}

$command += @(
    $imageTag,
    "python",
    "-m",
    "cuemate_analysis.musicalkey_service"
)

Push-Location $repoRoot
try {
    $rawOutput = & docker @command
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $containerId = ($rawOutput | Select-Object -First 1)
    if ($null -ne $containerId) {
        $containerId = $containerId.Trim()
    }

    $deadline = (Get-Date).AddSeconds(20)
    $healthUrl = "http://127.0.0.1:$servicePort/health"
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3
            if ($response.status -eq "ok") {
                Write-Output "MusicalKeyCNN service is healthy on $healthUrl"
                exit 0
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    }

    if ($containerId) {
        try {
            Write-Warning "MusicalKeyCNN service timed out; recent container logs:"
            & docker logs $containerId --tail 200
        }
        catch {
            Write-Warning "Failed to fetch MusicalKeyCNN container logs ${containerId}: $_"
        }
        try {
            & docker rm -f $containerId | Out-Null
        }
        catch {
            Write-Warning "Failed to clean up timed-out MusicalKeyCNN container ${containerId}: $_"
        }
    }
    Write-Error "MusicalKeyCNN service did not become healthy on $healthUrl within 20 seconds."
    exit 1
}
finally {
    Pop-Location
}
