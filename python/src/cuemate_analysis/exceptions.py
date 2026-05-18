"""CueMate engine exception hierarchy."""

from __future__ import annotations


class CueMateError(Exception):
    """Base exception for all CueMate engine errors."""


class ModelServiceError(CueMateError):
    """Docker/gRPC model service communication failures."""


class AnalysisError(CueMateError):
    """DSP or feature extraction failures."""


class ConfigError(CueMateError):
    """Configuration loading or validation failures."""
