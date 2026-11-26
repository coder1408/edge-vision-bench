"""Abstract base interface for cross-runtime inference engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import psutil


@dataclass(frozen=True)
class EngineTiming:
    """Device-level timing breakdown in milliseconds for a single inference call."""

    h2d_ms: float
    compute_ms: float
    d2h_ms: float

    @property
    def total_device_ms(self) -> float:
        """Total device transfer and compute time in milliseconds."""
        return self.h2d_ms + self.compute_ms + self.d2h_ms


class BaseRuntimeEngine(ABC):
    """Unified runtime engine contract implemented by PyTorch, ORT, OpenVINO, and TensorRT."""

    def __init__(self, artifact_path: str, precision: str, device: str) -> None:
        self.artifact_path = artifact_path
        self.precision = precision
        self.device = device

    @abstractmethod
    def warmup(self, input_shape: tuple[int, int, int, int], runs: int = 50) -> None:
        """Run dummy forward passes to stabilize caches, allocators, and JIT kernels."""

    @abstractmethod
    def infer(self, tensor_nchw: np.ndarray) -> tuple[list[np.ndarray], EngineTiming]:
        """Execute inference on an NCHW numpy tensor and return outputs plus timing."""

    @abstractmethod
    def get_vram_usage_mb(self) -> float:
        """Return peak or current GPU VRAM usage in megabytes (0.0 on CPU-only runtimes)."""

    def get_ram_usage_mb(self) -> float:
        """Return current process resident set size (RSS) in megabytes."""
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)

    def close(self) -> None:
        """Release runtime resources and device handles."""
        return None
