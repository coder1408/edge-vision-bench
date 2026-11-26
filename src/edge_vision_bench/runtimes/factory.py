"""Runtime engine factory and host availability discovery."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from edge_vision_bench.runtimes.base import BaseRuntimeEngine
from edge_vision_bench.runtimes.onnx_engine import ONNXRuntimeEngine
from edge_vision_bench.runtimes.openvino_engine import OpenVINORuntimeEngine
from edge_vision_bench.runtimes.pytorch_engine import PyTorchRuntimeEngine
from edge_vision_bench.runtimes.trt_engine import TensorRTRuntimeEngine
from edge_vision_bench.schemas import PrecisionName, RuntimeName


def create_engine(
    runtime: RuntimeName,
    artifact_path: str | Path,
    precision: PrecisionName = "fp32",
    device: str = "cpu",
    model: Any | None = None,
) -> BaseRuntimeEngine:
    """Instantiate a concrete BaseRuntimeEngine for the requested runtime backend."""
    if runtime == "pytorch":
        return PyTorchRuntimeEngine(
            artifact_path=artifact_path,
            precision=precision,
            device=device,
            model=model,
            compile_model=False,
        )
    if runtime == "pytorch_compile":
        return PyTorchRuntimeEngine(
            artifact_path=artifact_path,
            precision=precision,
            device=device,
            model=model,
            compile_model=True,
        )
    if runtime == "onnxruntime":
        return ONNXRuntimeEngine(
            artifact_path=artifact_path,
            precision=precision,
            device=device,
        )
    if runtime == "openvino":
        return OpenVINORuntimeEngine(
            artifact_path=artifact_path,
            precision=precision,
            device=device,
        )
    if runtime == "tensorrt":
        return TensorRTRuntimeEngine(
            artifact_path=artifact_path,
            precision=precision,
            device=device,
        )

    raise ValueError(f"Unsupported runtime: {runtime!r}")


def _has_cuda_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _safe_find_spec(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _has_cuda_python() -> bool:
    return _safe_find_spec("cuda.bindings.runtime") or _safe_find_spec("cuda.cudart")


def list_available_runtimes() -> list[str]:
    """Return runtime backend names available in the current Python/hardware environment."""
    available: list[str] = []

    if importlib.util.find_spec("torch") is not None:
        available.extend(["pytorch", "pytorch_compile"])

    if importlib.util.find_spec("onnxruntime") is not None:
        available.append("onnxruntime")

    if importlib.util.find_spec("openvino") is not None:
        available.append("openvino")

    if importlib.util.find_spec("tensorrt") is not None and _has_cuda_python() and _has_cuda_gpu():
        available.append("tensorrt")

    return available
