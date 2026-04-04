$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$imageTag = if ($env:CUEMATE_TEMPOCNN_IMAGE) { $env:CUEMATE_TEMPOCNN_IMAGE } else { "cuemate-tempocnn:local" }
$serviceName = if ($env:CUEMATE_TEMPOCNN_SERVICE_NAME) { $env:CUEMATE_TEMPOCNN_SERVICE_NAME } else { "cuemate-tempocnn-service" }
$servicePort = if ($env:CUEMATE_TEMPOCNN_SERVICE_PORT) { $env:CUEMATE_TEMPOCNN_SERVICE_PORT } else { "47831" }
$driveRoots = if ($env:CUEMATE_TEMPOCNN_DRIVES) { $env:CUEMATE_TEMPOCNN_DRIVES } else { "d" }
$accelerator = if ($env:CUEMATE_TEMPOCNN_ACCELERATOR) { $env:CUEMATE_TEMPOCNN_ACCELERATOR } else { "auto" }

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
    & docker @command | Out-Null
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

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

    Write-Error "TempoCNN service did not become healthy on $healthUrl within 20 seconds."
    exit 1
}
finally {
    Pop-Location
}
