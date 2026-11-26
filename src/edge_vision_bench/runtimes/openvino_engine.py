"""OpenVINO CPU inference engine for FP32, FP16, and NNCF INT8 IR models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from edge_vision_bench.runtimes.base import BaseRuntimeEngine, EngineTiming


class OpenVINORuntimeEngine(BaseRuntimeEngine):
    """Inference engine backed by OpenVINO Core and compiled InferRequest."""

    def __init__(
        self,
        artifact_path: str | Path,
        precision: str = "fp32",
        device: str = "cpu",
    ) -> None:
        super().__init__(artifact_path=str(artifact_path), precision=precision, device=device)

        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"OpenVINO IR artifact not found: {path}")

        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "openvino is not installed. Install with `pip install -e '.[cpu]'`."
            ) from exc

        ov_device = device.upper() if device else "CPU"
        core = ov.Core()
        ov_model = core.read_model(model=str(path))
        self._compiled_model: Any = core.compile_model(ov_model, ov_device)
        self._infer_request: Any = self._compiled_model.create_infer_request()
        self._num_outputs: int = len(self._compiled_model.outputs)

    def warmup(self, input_shape: tuple[int, int, int, int], runs: int = 50) -> None:
        """Run dummy forward passes to warm up OpenVINO thread pools and caches."""
        dummy = np.zeros(input_shape, dtype=np.float32)
        for _ in range(max(0, runs)):
            self.infer(dummy)

    def infer(self, tensor_nchw: np.ndarray) -> tuple[list[np.ndarray], EngineTiming]:
        """Run synchronous CPU inference and return copied FP32 numpy outputs."""
        if self._infer_request is None:
            raise RuntimeError("OpenVINORuntimeEngine has already been closed.")

        arr = np.ascontiguousarray(tensor_nchw, dtype=np.float32)
        t0 = time.perf_counter_ns()
        self._infer_request.infer({0: arr})
        t1 = time.perf_counter_ns()

        outputs = [
            np.asarray(self._infer_request.get_output_tensor(i).data, dtype=np.float32).copy()
            for i in range(self._num_outputs)
        ]
        compute_ms = float(t1 - t0) / 1e6
        return outputs, EngineTiming(h2d_ms=0.0, compute_ms=compute_ms, d2h_ms=0.0)

    def get_vram_usage_mb(self) -> float:
        """OpenVINO CPU engine does not use discrete GPU VRAM."""
        return 0.0

    def close(self) -> None:
        """Release compiled OpenVINO model and inference request."""
        self._infer_request = None
        self._compiled_model = None
