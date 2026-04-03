# Scripts

This directory holds setup and contract-maintenance helpers. No product logic lives here.

## Available helpers

- `compile-proto.ps1`: validates the protobuf contract with the local `protoc` compiler and writes a descriptor set to `data/scoring.pb`

## Usage

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
```
