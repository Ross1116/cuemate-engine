param(
  [string]$ProtoRoot = "proto",
  [string]$ProtoFile = "proto/djengine/scoring/v1/scoring.proto",
  [string]$DescriptorOut = "data/scoring.pb",
  [string]$PythonOut = "python/src",
  [string]$GoOut = "go"
)

$buf = Get-Command buf -ErrorAction SilentlyContinue
if (-not $buf) {
  throw "Could not find 'buf' on PATH."
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  throw "Could not find 'python' on PATH."
}

$protoc = Get-Command protoc -ErrorAction SilentlyContinue
if (-not $protoc) {
  throw "Could not find 'protoc' on PATH."
}

$go = Get-Command go -ErrorAction SilentlyContinue
if (-not $go) {
  throw "Could not find 'go' on PATH."
}

$gopath = (& $go.Source env GOPATH).Trim()
if (-not $gopath) {
  throw "Could not determine GOPATH."
}

$gobin = $env:GOBIN
if ($gobin -and $gobin.Trim()) {
  $pluginBin = $gobin.Trim()
} else {
  $pluginBin = Join-Path $gopath "bin"
}
$exeSuffix = ""
if ($env:OS -eq "Windows_NT") {
  $exeSuffix = ".exe"
}
$protocGenGo = Join-Path $pluginBin "protoc-gen-go$exeSuffix"
$protocGenGoGrpc = Join-Path $pluginBin "protoc-gen-go-grpc$exeSuffix"
if (-not (Test-Path $protocGenGo)) {
  throw "Could not find protoc-gen-go at $protocGenGo. Install it with 'go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.11'."
}
if (-not (Test-Path $protocGenGoGrpc)) {
  throw "Could not find protoc-gen-go-grpc at $protocGenGoGrpc. Install it with 'go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1'."
}

$env:Path = "$pluginBin;$env:Path"

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
    Set-Content -Path $initPath -Value '"""Generated scoring protobuf namespace."""' -Encoding UTF8
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

& $protoc.Source `
  "--proto_path=$ProtoRoot" `
  "--go_out=$GoOut" `
  "--go_opt=module=github.com/Ross1116/cuemate-engine/go" `
  "--go-grpc_out=$GoOut" `
  "--go-grpc_opt=module=github.com/Ross1116/cuemate-engine/go" `
  $ProtoFile
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Output "Linted $ProtoFile, wrote descriptor set to $DescriptorOut, generated Python gRPC stubs in $PythonOut, and generated Go gRPC stubs in $GoOut\\gen"
