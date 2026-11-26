"""TensorRT 10.x CUDA inference engine with lazy GPU imports and event timing."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from edge_vision_bench.runtimes.base import BaseRuntimeEngine, EngineTiming

logger = logging.getLogger(__name__)


def _import_trt_and_cudart() -> tuple[Any, Any]:
    """Lazily import TensorRT and cuda-python so CPU-only hosts never fail on import."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT runtime requires NVIDIA CUDA GPU with `tensorrt>=10.0` installed. "
            "Install GPU extras with `pip install -e '.[gpu]'` on a CUDA-capable host."
        ) from exc

    cudart: Any = None
    try:
        from cuda.bindings import runtime as cudart_mod

        cudart = cudart_mod
    except ImportError:
        try:
            from cuda import cudart as cudart_legacy

            cudart = cudart_legacy
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT runtime requires `cuda-python>=12.4` "
                "(`cuda.bindings.runtime` or `cuda.cudart`). "
                "Install GPU extras with `pip install -e '.[gpu]'`."
            ) from exc

    return trt, cudart


def _check_cuda(result: tuple[Any, ...], op_name: str) -> tuple[Any, ...]:
    """Validate cuda-python return tuple and raise a clear RuntimeError on failure."""
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA runtime call `{op_name}` failed with error code {err!r}.")
    return result[1:]


def _query_nvml_vram_mb(device_id: int) -> float:
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


class TensorRTRuntimeEngine(BaseRuntimeEngine):
    """TensorRT 10.x runtime engine using tensor-name API and CUDA event timing."""

    def __init__(
        self,
        artifact_path: str | Path,
        precision: str = "fp16",
        device: str = "cuda",
    ) -> None:
        super().__init__(artifact_path=str(artifact_path), precision=precision, device=device)

        self._closed: bool = True
        self._cudart: Any = None
        self._trt: Any = None
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._events: dict[str, Any] = {}
        self._device_buffers: dict[str, tuple[int, int]] = {}
        self._device_id: int = 0

        if ":" in device:
            _, idx_str = device.split(":", 1)
            self._device_id = int(idx_str)

        trt, cudart = _import_trt_and_cudart()
        self._trt = trt
        self._cudart = cudart

        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"TensorRT engine plan not found: {path}")

        _check_cuda(cudart.cudaSetDevice(self._device_id), "cudaSetDevice")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(trt_logger)
        with path.open("rb") as handle:
            serialized_plan = handle.read()

        self._engine = self._runtime.deserialize_cuda_engine(serialized_plan)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._tensor_dtypes: dict[str, np.dtype[Any]] = {}

        for idx in range(self._engine.num_io_tensors):
            name: str = self._engine.get_tensor_name(idx)
            mode = self._engine.get_tensor_mode(name)
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            self._tensor_dtypes[name] = dtype
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

        if not self._input_names:
            raise RuntimeError(f"TensorRT engine {path} has no input tensors.")

        self._primary_input: str = self._input_names[0]

        (self._stream,) = _check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")
        for ev_name in ("start", "h2d", "compute", "d2h"):
            (ev,) = _check_cuda(cudart.cudaEventCreate(), f"cudaEventCreate({ev_name})")
            self._events[ev_name] = ev

        self._closed = False

    def _ensure_device_buffer(self, name: str, required_nbytes: int) -> int:
        allocated_nbytes, ptr = self._device_buffers.get(name, (0, 0))
        if required_nbytes <= allocated_nbytes and ptr != 0:
            return ptr

        if ptr != 0:
            _check_cuda(self._cudart.cudaFree(ptr), f"cudaFree({name})")
            self._device_buffers.pop(name, None)

        alloc_size = max(1, required_nbytes)
        (new_ptr,) = _check_cuda(self._cudart.cudaMalloc(alloc_size), f"cudaMalloc({name})")
        self._device_buffers[name] = (alloc_size, int(new_ptr))
        return int(new_ptr)

    def warmup(self, input_shape: tuple[int, int, int, int], runs: int = 50) -> None:
        """Warm up TensorRT kernels, CUDA stream, and dynamic device buffers."""
        dummy = np.zeros(input_shape, dtype=np.float32)
        for _ in range(max(0, runs)):
            self.infer(dummy)

    def infer(self, tensor_nchw: np.ndarray) -> tuple[list[np.ndarray], EngineTiming]:
        """Execute async TensorRT v3 inference with CUDA event stage decomposition."""
        if self._closed or self._context is None:
            raise RuntimeError("TensorRTRuntimeEngine has already been closed.")

        in_name = self._primary_input
        in_dtype = self._tensor_dtypes[in_name]
        host_in = np.ascontiguousarray(tensor_nchw, dtype=in_dtype)

        if not self._context.set_input_shape(in_name, tuple(int(d) for d in host_in.shape)):
            raise RuntimeError(
                f"Failed to set TensorRT input shape {host_in.shape} for tensor '{in_name}'."
            )

        in_ptr = self._ensure_device_buffer(in_name, host_in.nbytes)
        self._context.set_tensor_address(in_name, in_ptr)

        host_outputs: list[tuple[str, np.ndarray, int]] = []
        for out_name in self._output_names:
            out_shape = tuple(int(d) for d in self._context.get_tensor_shape(out_name))
            if any(dim < 0 for dim in out_shape):
                raise RuntimeError(
                    f"Unresolved dynamic output shape {out_shape} for tensor '{out_name}'."
                )
            out_dtype = self._tensor_dtypes[out_name]
            host_out = np.empty(out_shape, dtype=out_dtype)
            out_ptr = self._ensure_device_buffer(out_name, host_out.nbytes)
            self._context.set_tensor_address(out_name, out_ptr)
            host_outputs.append((out_name, host_out, out_ptr))

        cudart = self._cudart
        stream = self._stream
        ev_start = self._events["start"]
        ev_h2d = self._events["h2d"]
        ev_compute = self._events["compute"]
        ev_d2h = self._events["d2h"]

        _check_cuda(cudart.cudaEventRecord(ev_start, stream), "cudaEventRecord(start)")
        _check_cuda(
            cudart.cudaMemcpyAsync(
                in_ptr,
                host_in.ctypes.data,
                host_in.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )
        _check_cuda(cudart.cudaEventRecord(ev_h2d, stream), "cudaEventRecord(h2d)")

        ok = self._context.execute_async_v3(int(stream))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False.")

        _check_cuda(cudart.cudaEventRecord(ev_compute, stream), "cudaEventRecord(compute)")

        for _, host_out, out_ptr in host_outputs:
            _check_cuda(
                cudart.cudaMemcpyAsync(
                    host_out.ctypes.data,
                    out_ptr,
                    host_out.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                ),
                "cudaMemcpyAsync(D2H)",
            )

        _check_cuda(cudart.cudaEventRecord(ev_d2h, stream), "cudaEventRecord(d2h)")
        _check_cuda(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

        (h2d_ms,) = _check_cuda(
            cudart.cudaEventElapsedTime(ev_start, ev_h2d), "cudaEventElapsedTime(h2d)"
        )
        (compute_ms,) = _check_cuda(
            cudart.cudaEventElapsedTime(ev_h2d, ev_compute), "cudaEventElapsedTime(compute)"
        )
        (d2h_ms,) = _check_cuda(
            cudart.cudaEventElapsedTime(ev_compute, ev_d2h), "cudaEventElapsedTime(d2h)"
        )

        np_outputs = [np.asarray(arr, dtype=np.float32) for _, arr, _ in host_outputs]
        return np_outputs, EngineTiming(
            h2d_ms=float(h2d_ms),
            compute_ms=float(compute_ms),
            d2h_ms=float(d2h_ms),
        )

    def get_vram_usage_mb(self) -> float:
        """Return process VRAM via NVML, falling back to engine workspace + IO buffers."""
        nvml_mb = _query_nvml_vram_mb(self._device_id)
        if nvml_mb > 0.0:
            return nvml_mb

        buffer_bytes = sum(nbytes for nbytes, _ in self._device_buffers.values())
        engine_bytes = 0
        if self._engine is not None:
            if hasattr(self._engine, "device_memory_size_v2"):
                engine_bytes = int(self._engine.device_memory_size_v2)
            elif hasattr(self._engine, "device_memory_size"):
                engine_bytes = int(self._engine.device_memory_size)
        return float(buffer_bytes + engine_bytes) / (1024.0 * 1024.0)

    def close(self) -> None:
        """Free CUDA device buffers, events, stream, and TensorRT handles cleanly."""
        if getattr(self, "_closed", True):
            return
        self._closed = True

        cudart = getattr(self, "_cudart", None)
        if cudart is not None:
            for _, (_, ptr) in list(self._device_buffers.items()):
                if ptr:
                    try:
                        cudart.cudaFree(ptr)
                    except Exception:
                        logger.debug("Ignoring cudaFree error during close.", exc_info=True)
            self._device_buffers.clear()

            for ev in self._events.values():
                if ev is not None:
                    try:
                        cudart.cudaEventDestroy(ev)
                    except Exception:
                        logger.debug("Ignoring cudaEventDestroy error during close.", exc_info=True)
            self._events.clear()

            if self._stream is not None:
                try:
                    cudart.cudaStreamDestroy(self._stream)
                except Exception:
                    logger.debug("Ignoring cudaStreamDestroy error during close.", exc_info=True)
                self._stream = None

        self._context = None
        self._engine = None
        self._runtime = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
