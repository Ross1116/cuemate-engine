param(
  [string]$ProtoRoot = "proto",
  [string]$ProtoFile = "proto/djengine/scoring/v1/scoring.proto",
  [string]$DescriptorOut = "data/scoring.pb",
  [string]$PythonOut = "python/src"
)

$buf = Get-Command buf -ErrorAction SilentlyContinue
if (-not $buf) {
  throw "Could not find 'buf' on PATH."
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  throw "Could not find 'python' on PATH."
}

& $buf.Source lint
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

$descriptorDir = Split-Path -Parent $DescriptorOut
if ($descriptorDir) {
  New-Item -ItemType Directory -Force $descriptorDir | Out-Null
}

New-Item -ItemType Directory -Force $PythonOut | Out-Null

@(
  "djengine",
  "djengine/scoring",
  "djengine/scoring/v1"
) | ForEach-Object {
  $pkgDir = Join-Path $PythonOut $_
  New-Item -ItemType Directory -Force $pkgDir | Out-Null
  $initPath = Join-Path $pkgDir "__init__.py"
  if (-not (Test-Path $initPath)) {
    Set-Content -Path $initPath -Value '"""Generated scoring protobuf namespace."""'
  }
}

& $python.Source -m grpc_tools.protoc `
  "--proto_path=$ProtoRoot" `
  "--python_out=$PythonOut" `
  "--grpc_python_out=$PythonOut" `
  "--descriptor_set_out=$DescriptorOut" `
  $ProtoFile
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Output "Linted $ProtoFile, wrote descriptor set to $DescriptorOut, and generated Python gRPC stubs in $PythonOut"
