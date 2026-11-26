"""Edge Vision Bench: cross-runtime edge vision deployment and benchmarking suite."""

from edge_vision_bench.schemas import (
    BenchmarkResult,
    DetectionBox,
    HealthResponse,
    InferenceResponse,
    ParityReport,
    PipelineConfig,
    PrecisionName,
    RuntimeName,
    TrackType,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BenchmarkResult",
    "DetectionBox",
    "HealthResponse",
    "InferenceResponse",
    "ParityReport",
    "PipelineConfig",
    "PrecisionName",
    "RuntimeName",
    "TrackType",
]
