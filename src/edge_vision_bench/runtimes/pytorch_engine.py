"""PyTorch eager and torch.compile runtime engine implementation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from edge_vision_bench.runtimes.base import BaseRuntimeEngine, EngineTiming

logger = logging.getLogger(__name__)


class PyTorchRuntimeEngine(BaseRuntimeEngine):
    """Inference engine backed by PyTorch eager mode or torch.compile."""

    def __init__(
        self,
        artifact_path: str | Path = "",
        precision: str = "fp32",
        device: str = "cpu",
        model: nn.Module | None = None,
        compile_model: bool = False,
    ) -> None:
        super().__init__(artifact_path=str(artifact_path), precision=precision, device=device)

        self._torch_device = torch.device(device)
        self._is_cuda = self._torch_device.type == "cuda"
        if self._is_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device}' requested for PyTorchRuntimeEngine, "
                "but CUDA is not available on this host."
            )

        resolved_model = self._resolve_model(model, artifact_path)
        resolved_model.eval()
        resolved_model.to(self._torch_device)

        self._input_dtype = torch.float32
        prec = precision.lower()
        if prec == "fp16":
            if self._is_cuda:
                resolved_model = resolved_model.half()
                self._input_dtype = torch.float16
            else:
                logger.warning(
                    "FP16 requested on CPU for PyTorchRuntimeEngine; "
                    "staying in FP32 for numerical stability and operator support."
                )
        elif prec == "int8":
            logger.warning(
                "INT8 precision is not natively supported in PyTorchRuntimeEngine; "
                "falling back to FP32 execution."
            )
        elif prec != "fp32":
            raise ValueError(f"Unsupported precision '{precision}' for PyTorchRuntimeEngine.")

        self._base_model: nn.Module = resolved_model
        self._compile_model = compile_model
        self._runner: Callable[[torch.Tensor], Any] = resolved_model

        if compile_model:
            self._runner = torch.compile(resolved_model, mode="reduce-overhead")

    @staticmethod
    def _resolve_model(model: nn.Module | None, artifact_path: str | Path) -> nn.Module:
        if model is not None:
            return model

        path_str = str(artifact_path).strip()
        if not path_str:
            raise ValueError(
                "PyTorchRuntimeEngine requires either a `model` nn.Module instance "
                "or a valid `artifact_path`."
            )

        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"PyTorch model artifact not found: {path}")

        if path.suffix in {".ts", ".torchscript"}:
            loaded_script = torch.jit.load(str(path), map_location="cpu")
            return loaded_script

        loaded: Any = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(loaded, nn.Module):
            return loaded

        raise ValueError(
            f"Artifact at {path} is not a serialized nn.Module (got {type(loaded).__name__}). "
            "Pass the instantiated `model` directly to PyTorchRuntimeEngine."
        )

    @staticmethod
    def _normalize_outputs(raw_out: Any) -> list[torch.Tensor]:
        if isinstance(raw_out, torch.Tensor):
            return [raw_out]
        if isinstance(raw_out, (tuple, list)):
            return [t for t in raw_out if isinstance(t, torch.Tensor)]
        raise TypeError(f"Unsupported PyTorch model output type: {type(raw_out).__name__}")

    def warmup(self, input_shape: tuple[int, int, int, int], runs: int = 50) -> None:
        """Run dummy forward passes to stabilize caches, allocators, and compiled graphs."""
        dummy = np.zeros(input_shape, dtype=np.float32)
        for _ in range(max(0, runs)):
            self.infer(dummy)
        if self._is_cuda:
            torch.cuda.synchronize(self._torch_device)
            torch.cuda.reset_peak_memory_stats(self._torch_device)

    def infer(self, tensor_nchw: np.ndarray) -> tuple[list[np.ndarray], EngineTiming]:
        """Run inference and return FP32 numpy outputs alongside stage timing."""
        arr = np.ascontiguousarray(tensor_nchw, dtype=np.float32)

        if self._is_cuda:
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_h2d = torch.cuda.Event(enable_timing=True)
            ev_compute = torch.cuda.Event(enable_timing=True)
            ev_d2h = torch.cuda.Event(enable_timing=True)

            cpu_tensor = torch.from_numpy(arr)
            with torch.inference_mode():
                ev_start.record()
                gpu_tensor = cpu_tensor.to(
                    device=self._torch_device,
                    dtype=self._input_dtype,
                    non_blocking=False,
                )
                ev_h2d.record()

                if self._compile_model and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                raw_out = self._runner(gpu_tensor)
                ev_compute.record()

                tensors = self._normalize_outputs(raw_out)
                np_outputs = [t.detach().float().cpu().numpy() for t in tensors]
                ev_d2h.record()
                torch.cuda.synchronize(self._torch_device)

            timing = EngineTiming(
                h2d_ms=float(ev_start.elapsed_time(ev_h2d)),
                compute_ms=float(ev_h2d.elapsed_time(ev_compute)),
                d2h_ms=float(ev_compute.elapsed_time(ev_d2h)),
            )
            return np_outputs, timing

        cpu_tensor = torch.from_numpy(arr).to(dtype=self._input_dtype)
        with torch.inference_mode():
            t0 = time.perf_counter_ns()
            raw_out = self._runner(cpu_tensor)
            t1 = time.perf_counter_ns()
            tensors = self._normalize_outputs(raw_out)
            np_outputs = [t.detach().float().cpu().numpy() for t in tensors]

        compute_ms = float(t1 - t0) / 1e6
        return np_outputs, EngineTiming(h2d_ms=0.0, compute_ms=compute_ms, d2h_ms=0.0)

    def get_vram_usage_mb(self) -> float:
        """Return peak CUDA memory allocated by PyTorch in megabytes."""
        if self._is_cuda and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated(self._torch_device)) / (1024.0 * 1024.0)
        return 0.0

    def close(self) -> None:
        """Release GPU caches when closing the engine."""
        if self._is_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
