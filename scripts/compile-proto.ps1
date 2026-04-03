param(
  [string]$ProtoRoot = "proto",
  [string]$ProtoFile = "proto/djengine/scoring/v1/scoring.proto",
  [string]$DescriptorOut = "data/scoring.pb"
)

$protobufHome = if ($env:PROTOBUF_HOME) {
  $env:PROTOBUF_HOME
} else {
  Join-Path $env:USERPROFILE "Protobuf"
}

$protoc = Join-Path $protobufHome "bin\protoc.exe"
if (-not (Test-Path $protoc)) {
  throw "Could not find protoc.exe at $protoc"
}

$outDir = Split-Path -Parent $DescriptorOut
if ($outDir) {
  New-Item -ItemType Directory -Force $outDir | Out-Null
}

& $protoc "--proto_path=$ProtoRoot" "--descriptor_set_out=$DescriptorOut" $ProtoFile
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Output "Validated $ProtoFile and wrote descriptor set to $DescriptorOut"
