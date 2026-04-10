from __future__ import annotations

import importlib
from types import ModuleType


def load_scoring_proto_modules() -> tuple[ModuleType, ModuleType]:
    """Return generated scoring protobuf modules or raise a clear runtime error."""
    try:
        pb2 = importlib.import_module("djengine.scoring.v1.scoring_pb2")
        pb2_grpc = importlib.import_module("djengine.scoring.v1.scoring_pb2_grpc")
        return pb2, pb2_grpc
    except ImportError as exc:  # pragma: no cover - exercised through CLI/service tests
        raise RuntimeError(
            "Generated scoring protobuf modules were not found. "
            "Run powershell -ExecutionPolicy Bypass -File .\\scripts\\compile-proto.ps1 "
            "before using the scoring gRPC service."
        ) from exc
