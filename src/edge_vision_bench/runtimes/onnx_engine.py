"""ONNX Runtime CPU and CUDA (IOBinding) inference engine."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from edge_vision_bench.runtimes.base import BaseRuntimeEngine, EngineTiming

logger = logging.getLogger(__name__)


def _parse_cuda_device_id(device: str) -> int:
    if ":" in device:
        _, idx_str = device.split(":", 1)
        return int(idx_str)
    return 0


def _query_nvml_process_vram_mb(device_id: int) -> float:
    try:
        import pynvml
    except ImportError:
        return 0.0

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        pid = os.getpid()
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == pid and proc.usedGpuMemory is not None:
                return float(proc.usedGpuMemory) / (1024.0 * 1024.0)
    except Exception:
        return 0.0
    finally:
        if initialized:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
    return 0.0


class ONNXRuntimeEngine(BaseRuntimeEngine):
    """Inference engine backed by ONNX Runtime with CUDA IOBinding support."""

    def __init__(
        self,
        artifact_path: str | Path,
        precision: str = "fp32",
        device: str = "cpu",
    ) -> None:
        super().__init__(artifact_path=str(artifact_path), precision=precision, device=device)

        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model artifact not found: {path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install with `pip install -e '.[cpu]'` "
                "or `pip install -e '.[gpu]'`."
            ) from exc

        self._ort: Any = ort
        self._is_cuda = device.lower().startswith("cuda")
        self._device_id = _parse_cuda_device_id(device) if self._is_cuda else 0

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers: list[Any]
        if self._is_cuda:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider requested for ONNXRuntimeEngine, "
                    f"but available providers are: {available}"
                )
            providers = [
                ("CUDAExecutionProvider", {"device_id": self._device_id}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        self._session: Any = ort.InferenceSession(
            str(path),
            sess_options=sess_options,
            providers=providers,
        )

        inputs = self._session.get_inputs()
        if not inputs:
            raise ValueError(f"ONNX model {path} defines zero inputs.")

        self._input_name: str = inputs[0].name
        self._input_type: str = inputs[0].type
        self._output_names: list[str] = [out.name for out in self._session.get_outputs()]
        self._input_dtype: np.dtype[Any] = (
            np.dtype(np.float16) if "float16" in self._input_type else np.dtype(np.float32)
        )

    def warmup(self, input_shape: tuple[int, int, int, int], runs: int = 50) -> None:
        """Run dummy forward passes to warm up ONNX Runtime allocators and kernels."""
        dummy = np.zeros(input_shape, dtype=np.float32)
        for _ in range(max(0, runs)):
            self.infer(dummy)

    def infer(self, tensor_nchw: np.ndarray) -> tuple[list[np.ndarray], EngineTiming]:
        """Run inference on CPU or CUDA (via IOBinding) with separated stage timing."""
        if self._session is None:
            raise RuntimeError("ONNXRuntimeEngine has already been closed.")

        arr = np.ascontiguousarray(tensor_nchw, dtype=self._input_dtype)

        if self._is_cuda:
            io_binding = self._session.io_binding()
            t0 = time.perf_counter_ns()
            ort_input = self._ort.OrtValue.ortvalue_from_numpy(arr, "cuda", self._device_id)
            io_binding.bind_ortvalue_input(self._input_name, ort_input)
            for out_name in self._output_names:
                io_binding.bind_output(out_name, "cuda", self._device_id)
            io_binding.synchronize_inputs()
            t1 = time.perf_counter_ns()

            self._session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            t2 = time.perf_counter_ns()

            ort_outputs = io_binding.get_outputs()
            np_outputs = [np.asarray(out.numpy(), dtype=np.float32) for out in ort_outputs]
            t3 = time.perf_counter_ns()

            io_binding.clear_binding_inputs()
            io_binding.clear_binding_outputs()

            timing = EngineTiming(
                h2d_ms=float(t1 - t0) / 1e6,
                compute_ms=float(t2 - t1) / 1e6,
                d2h_ms=float(t3 - t2) / 1e6,
            )
            return np_outputs, timing

        t0 = time.perf_counter_ns()
        raw_outputs = self._session.run(self._output_names, {self._input_name: arr})
        t1 = time.perf_counter_ns()

        np_outputs = [np.asarray(out, dtype=np.float32) for out in raw_outputs]
        timing = EngineTiming(h2d_ms=0.0, compute_ms=float(t1 - t0) / 1e6, d2h_ms=0.0)
        return np_outputs, timing

    def get_vram_usage_mb(self) -> float:
        """Return GPU VRAM usage via NVML when running on CUDA, else 0.0."""
        if not self._is_cuda:
            return 0.0
        return _query_nvml_process_vram_mb(self._device_id)

    def close(self) -> None:
        """Release the underlying ONNX Runtime session."""
        self._session = None
