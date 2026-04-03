$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$imageTag = if ($env:CUEMATE_MUSICALKEYCNN_IMAGE) { $env:CUEMATE_MUSICALKEYCNN_IMAGE } else { "cuemate-musicalkeycnn:local" }
$serviceName = if ($env:CUEMATE_MUSICALKEYCNN_SERVICE_NAME) { $env:CUEMATE_MUSICALKEYCNN_SERVICE_NAME } else { "cuemate-musicalkeycnn-service" }
$servicePort = if ($env:CUEMATE_MUSICALKEYCNN_SERVICE_PORT) { $env:CUEMATE_MUSICALKEYCNN_SERVICE_PORT } else { "47832" }
$driveRoots = if ($env:CUEMATE_MUSICALKEYCNN_DRIVES) { $env:CUEMATE_MUSICALKEYCNN_DRIVES } else { "d" }
$device = if ($env:CUEMATE_MUSICALKEYCNN_DEVICE) { $env:CUEMATE_MUSICALKEYCNN_DEVICE } else { "auto" }

$healthUrl = "http://127.0.0.1:$servicePort/health"
try {
    $existingResponse = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2
    if ($existingResponse.status -eq "ok") {
        Write-Output "MusicalKeyCNN service is healthy on $healthUrl"
        return 0
    }
} catch {
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
    "--env", "CUEMATE_MUSICALKEYCNN_DEFAULT_MODEL=/workspace/python/models/musicalkeycnn/keynet.pt",
    "--env", "CUEMATE_MUSICALKEYCNN_DEFAULT_DEVICE=$device",
    "--env", "PYTHONPATH=/workspace/python/src",
    "--volume", "$repoRoot`:/workspace:ro"
)

if ($device -eq "cpu") {
    $command += @("--env", "CUDA_VISIBLE_DEVICES=-1")
} else {
    $command += @("--gpus", "all")
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
    & docker @command | Out-Null
    if ($LASTEXITCODE -ne 0) {
        return $LASTEXITCODE
    }

    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3
            if ($response.status -eq "ok") {
                Write-Output "MusicalKeyCNN service is healthy on $healthUrl"
                return 0
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    }

    Write-Error "MusicalKeyCNN service did not become healthy on $healthUrl within 20 seconds."
    return 1
}
finally {
    Pop-Location
}
