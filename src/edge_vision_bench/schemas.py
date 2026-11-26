"""Shared Pydantic schemas and pipeline configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

RuntimeName = Literal["pytorch", "pytorch_compile", "onnxruntime", "tensorrt", "openvino"]
PrecisionName = Literal["fp32", "fp16", "int8"]
TrackType = Literal["detection", "embedding"]


class DetectionBox(BaseModel):
    """Single object detection bounding box with optional tracking and world coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int
    class_name: str | None = None
    track_id: int | None = None
    world_xy: tuple[float, float] | None = None


class InferenceResponse(BaseModel):
    """Standardized inference output across detection and embedding tracks."""

    runtime: RuntimeName
    precision: PrecisionName
    detections: list[DetectionBox]
    latency_breakdown_ms: dict[str, float]
    frame_index: int | None = None
    embedding_dim: int | None = None
    embedding_norm: float | None = None


class HealthResponse(BaseModel):
    """Service health and active runtime diagnostics."""

    status: Literal["ok", "degraded"]
    active_runtime: str
    precision: str
    device: str
    engine_loaded: bool
    available_runtimes: list[str]


class ParityReport(BaseModel):
    """Numerical parity verification report between reference and target runtimes."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    reference_runtime: str
    target_runtime: str
    precision: PrecisionName
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    cosine_similarity: float | None = None
    rtol: float
    atol: float
    num_samples: int


class BenchmarkResult(BaseModel):
    """Aggregated latency, memory, accuracy, and OOD robustness benchmark metrics."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    track: TrackType = "detection"
    runtime: str
    precision: str
    hardware_target: str
    batch_size: int = 1
    p50_compute_ms: float
    p95_compute_ms: float
    p99_compute_ms: float
    p50_h2d_ms: float = 0.0
    p50_d2h_ms: float = 0.0
    p50_preproc_ms: float = 0.0
    p50_postproc_ms: float = 0.0
    end_to_end_fps: float
    peak_vram_mb: float = 0.0
    peak_ram_mb: float = 0.0
    artifact_size_mb: float = 0.0
    clean_map50_95: float | None = None
    clean_map50: float | None = None
    ood_blur_map50_95: float | None = None
    ood_occlusion_map50_95: float | None = None
    ood_lowlight_map50_95: float | None = None
    clean_mota: float | None = None
    clean_idf1: float | None = None
    ood_blur_mota: float | None = None
    embedding_cosine_sim: float | None = None
    ood_blur_cosine_sim: float | None = None
    ood_occlusion_cosine_sim: float | None = None
    ood_lowlight_cosine_sim: float | None = None


class PipelineConfig(BaseModel):
    """Configuration for model export, runtime compilation, tracking, and benchmarking."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    track: TrackType = "detection"
    input_size: tuple[int, int] = (640, 640)
    num_classes: int = 80
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    tracker_high_thresh: float = 0.5
    tracker_low_thresh: float = 0.1
    tracker_match_thresh: float = 0.8
    tracker_track_buffer: int = 30
    homography_matrix: list[list[float]] | None = None
    opset_version: int = 17
    min_batch: int = 1
    opt_batch: int = 4
    max_batch: int = 16
    warmup_runs: int = 50
    parity_rtol: float = 1e-3
    parity_atol: float = 1e-4
    parity_min_cosine: float = 0.995
    artifact_dir: str = "models/artifacts"
    weights_dir: str = "models/weights"
    results_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load and validate a PipelineConfig from a YAML file."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if raw is None:
            data: dict[str, object] = {}
        elif isinstance(raw, dict):
            data = dict(raw)
        else:
            raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(raw).__name__}")

        if "input_size" in data and isinstance(data["input_size"], list):
            if len(data["input_size"]) != 2:
                raise ValueError(
                    f"input_size must contain exactly 2 integers, got {data['input_size']!r}"
                )
            data["input_size"] = (int(data["input_size"][0]), int(data["input_size"][1]))

        return cls.model_validate(data)
