$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$imageTag = if ($env:CUEMATE_TEMPOCNN_IMAGE) { $env:CUEMATE_TEMPOCNN_IMAGE } else { "cuemate-tempocnn:local" }
$serviceName = if ($env:CUEMATE_TEMPOCNN_SERVICE_NAME) { $env:CUEMATE_TEMPOCNN_SERVICE_NAME } else { "cuemate-tempocnn-service" }
$servicePort = if ($env:CUEMATE_TEMPOCNN_SERVICE_PORT) { $env:CUEMATE_TEMPOCNN_SERVICE_PORT } else { "47831" }
$driveRoots = if ($env:CUEMATE_TEMPOCNN_DRIVES) { $env:CUEMATE_TEMPOCNN_DRIVES } else { "d" }
$accelerator = if ($env:CUEMATE_TEMPOCNN_ACCELERATOR) { $env:CUEMATE_TEMPOCNN_ACCELERATOR } else { "auto" }

function Start-TempoContainer {
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

function Convert-TempoCommandToCpu {
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

$existingContainerId = (& docker ps -a --filter "name=^$serviceName$" --format "{{.ID}}") | Select-Object -First 1
if ($existingContainerId) {
    & docker rm -f $serviceName | Out-Null
}

$command = @(
    "run", "-d", "--rm",
    "--name", $serviceName,
    "--publish", "127.0.0.1:$servicePort`:$servicePort",
    "--env", "TEMPOCNN_SERVICE_PORT=$servicePort",
    "--env", "TF_CPP_MIN_LOG_LEVEL=3",
    "--env", "CUEMATE_TEMPOCNN_DEFAULT_MODEL=/workspace/python/models/essentia/deepsquare-k16-3.pb",
    "--volume", "$repoRoot`:/workspace:ro"
)

if ($accelerator -eq "auto") {
    $command += @("--gpus", "all")
} else {
    $command += @("--env", "CUDA_VISIBLE_DEVICES=-1")
}

foreach ($drive in ($driveRoots -split "," | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })) {
    $source = ($drive + ":\")
    $command += @("--mount", "type=bind,source=$source,target=/host/$drive,readonly")
}

$command += @(
    $imageTag,
    "python",
    "/workspace/docker/tempocnn/service.py"
)

Push-Location $repoRoot
try {
    $launch = Start-TempoContainer -DockerCommand $command
    if (-not $launch.Success) {
        $retryOnCpu = $false
        if ($accelerator -eq "auto") {
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

        $cpuCommand = Convert-TempoCommandToCpu -DockerCommand $command -ImageTag $imageTag
        $launch = Start-TempoContainer -DockerCommand $cpuCommand
        if (-not $launch.Success) {
            exit $launch.ExitCode
        }
    }
    $containerId = $launch.ContainerId

    $deadline = (Get-Date).AddSeconds(20)
    $healthUrl = "http://127.0.0.1:$servicePort/health"
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3
            if ($response.status -eq "ok") {
                Write-Output "TempoCNN service is healthy on $healthUrl"
                exit 0
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    }

    if ($containerId) {
        try {
            & docker rm -f $containerId | Out-Null
        }
        catch {
            Write-Warning "Failed to clean up timed-out TempoCNN container ${containerId}: $_"
        }
    }
    Write-Error "TempoCNN service did not become healthy on $healthUrl within 20 seconds."
    exit 1
}
finally {
    Pop-Location
}
