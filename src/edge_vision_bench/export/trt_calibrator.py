"""TensorRT INT8 entropy calibrator with CPU-safe lazy CUDA/TensorRT imports."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart
    except ImportError:
        cudart = None

_CalibratorBase: Any = (
    getattr(trt, "IInt8EntropyCalibrator2", object) if trt is not None else object
)


def _check_cuda(result: Any, operation: str) -> Any:
    if isinstance(result, tuple):
        err = result[0]
        rest = result[1:]
    else:
        err = result
        rest = ()

    if cudart is not None and err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error during {operation}: {err}")

    if len(rest) == 1:
        return rest[0]
    return rest if rest else None


class EntropyCalibrator(_CalibratorBase):  # type: ignore[misc]
    """TensorRT ``IInt8EntropyCalibrator2`` implementation backed by ``cuda-python``."""

    def __init__(
        self,
        calibration_frames: list[Path],
        preprocess_fn: Callable[[Path], np.ndarray],
        batch_size: int = 8,
        input_shape: tuple[int, int, int] = (3, 640, 640),
        cache_file: Path = Path("models/artifacts/calib.cache"),
    ) -> None:
        self.device_ptr: int | None = None
        if trt is None or cudart is None:
            raise RuntimeError(
                "TensorRT and cuda-python are only available on CUDA GPU hosts. "
                "Install the `gpu` extra (`pip install -e '.[gpu]'`) on a CUDA-capable host."
            )
        if not hasattr(trt, "IInt8EntropyCalibrator2"):
            raise RuntimeError(
                "TensorRT 11+ removed IInt8EntropyCalibrator2 and uses explicit Q/DQ INT8 ONNX "
                "quantization instead. Use build_trt_engine() or quantize_onnx_qdq_int8()."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        super().__init__()
        self.calibration_frames = [Path(frame) for frame in calibration_frames]
        self.preprocess_fn = preprocess_fn
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = Path(cache_file)
        self.current_index = 0

        num_elements = int(batch_size * np.prod(input_shape))
        self.batch_nbytes = num_elements * np.dtype(np.float32).itemsize
        self.device_ptr = int(_check_cuda(cudart.cudaMalloc(self.batch_nbytes), "cudaMalloc"))

    def get_batch_size(self) -> int:
        """Return the fixed calibration batch size."""
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        """Load, preprocess, and upload the next calibration batch to device memory."""
        del names
        if self.device_ptr is None:
            raise RuntimeError("EntropyCalibrator device buffer has already been freed")
        if self.current_index >= len(self.calibration_frames):
            return None

        batch_paths = self.calibration_frames[
            self.current_index : self.current_index + self.batch_size
        ]
        self.current_index += len(batch_paths)

        samples: list[np.ndarray] = []
        for frame_path in batch_paths:
            arr = np.asarray(self.preprocess_fn(frame_path), dtype=np.float32)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.shape != self.input_shape:
                raise ValueError(
                    f"Preprocessed calibration frame {frame_path} has shape {arr.shape}, "
                    f"expected {self.input_shape}"
                )
            samples.append(arr)

        # Pad trailing partial batch by repeating the last valid sample to avoid distorting stats.
        while len(samples) < self.batch_size:
            samples.append(samples[-1])

        batch = np.ascontiguousarray(np.stack(samples, axis=0), dtype=np.float32)
        _check_cuda(
            cudart.cudaMemcpy(
                self.device_ptr,
                batch.ctypes.data,
                batch.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ),
            "cudaMemcpyHostToDevice",
        )
        return [int(self.device_ptr)]

    def read_calibration_cache(self) -> bytes | None:
        """Read an existing calibration table from disk if present."""
        if self.cache_file.exists():
            logger.info("Loading TensorRT INT8 calibration cache from %s", self.cache_file)
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Persist the generated calibration table to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(bytes(cache))
        logger.info("Saved TensorRT INT8 calibration cache to %s", self.cache_file)

    def close(self) -> None:
        """Free allocated CUDA calibration memory safely."""
        ptr = getattr(self, "device_ptr", None)
        if ptr is not None and cudart is not None:
            self.device_ptr = None
            _check_cuda(cudart.cudaFree(ptr), "cudaFree")

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
