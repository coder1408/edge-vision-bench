"""TensorRT 10.x serialized engine builder supporting FP32, FP16, and INT8."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from edge_vision_bench.export.trt_calibrator import EntropyCalibrator
from edge_vision_bench.schemas import PrecisionName

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
except ImportError:
    trt = None


def _noop_preprocess(path: Path) -> np.ndarray:
    raise RuntimeError(f"preprocess_fn was not provided while attempting to calibrate using {path}")


def build_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    precision: PrecisionName = "fp16",
    input_shape: tuple[int, int, int] = (3, 640, 640),
    min_batch: int = 1,
    opt_batch: int = 4,
    max_batch: int = 16,
    calibration_frames: list[Path] | None = None,
    preprocess_fn: Callable[[Path], np.ndarray] | None = None,
    cache_file: Path | None = None,
    workspace_mb: int = 2048,
) -> Path:
    """Compile an ONNX model into a serialized TensorRT 10.x ``.engine`` plan file.

    Args:
        onnx_path: Source ``.onnx`` model file.
        engine_path: Destination serialized ``.engine`` path.
        precision: Target precision (``"fp32"``, ``"fp16"``, or ``"int8"``).
        input_shape: Spatial input shape ``(C, H, W)`` without batch dimension.
        min_batch: Minimum dynamic batch size for optimization profile.
        opt_batch: Optimal dynamic batch size for optimization profile.
        max_batch: Maximum dynamic batch size for optimization profile.
        calibration_frames: Image paths used for INT8 entropy calibration.
        preprocess_fn: Function mapping an image path to a ``(C, H, W)`` or ``(1, C, H, W)``
            float32 array matching the model's expected preprocessing.
        cache_file: Optional calibration cache file path for INT8 builds.
        workspace_mb: TensorRT builder workspace limit in MiB.

    Returns:
        Path to the serialized ``.engine`` artifact.
    """
    if trt is None:
        raise RuntimeError(
            "TensorRT is only available on CUDA GPU hosts. Install the `gpu` extra "
            "(`pip install -e '.[gpu]'`) on an NVIDIA GPU host with TensorRT 10.x."
        )

    src = Path(onnx_path)
    dest = Path(engine_path)
    if not src.exists():
        raise FileNotFoundError(f"ONNX model not found: {src}")
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError(f"Unsupported TensorRT precision: {precision!r}")
    if not (1 <= min_batch <= opt_batch <= max_batch):
        raise ValueError(
            f"Batch bounds must satisfy 1 <= min_batch <= opt_batch <= max_batch, "
            f"got ({min_batch}, {opt_batch}, {max_batch})"
        )

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)

    explicit_batch = (
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH")
        else 0
    )
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(src)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"Failed to parse ONNX graph {src}: {'; '.join(errors)}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        min=(min_batch, *input_shape),
        opt=(opt_batch, *input_shape),
        max=(max_batch, *input_shape),
    )
    config.add_optimization_profile(profile)

    calibrator: EntropyCalibrator | None = None
    try:
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            resolved_cache = (
                Path(cache_file) if cache_file is not None else dest.with_suffix(".calib.cache")
            )
            has_cache = resolved_cache.exists()
            has_frames = bool(calibration_frames) and preprocess_fn is not None
            if not has_cache and not has_frames:
                raise ValueError(
                    "TensorRT INT8 build requires either an existing calibration cache_file "
                    "or both calibration_frames and preprocess_fn."
                )

            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)
            calib_frames = list(calibration_frames) if calibration_frames else []
            calib_batch = (
                max(min_batch, min(opt_batch, len(calib_frames))) if calib_frames else opt_batch
            )
            calibrator = EntropyCalibrator(
                calibration_frames=calib_frames,
                preprocess_fn=preprocess_fn if preprocess_fn is not None else _noop_preprocess,
                batch_size=calib_batch,
                input_shape=input_shape,
                cache_file=resolved_cache,
            )
            if hasattr(config, "set_calibration_profile"):
                calib_profile = builder.create_optimization_profile()
                calib_shape = (calib_batch, *input_shape)
                calib_profile.set_shape(
                    input_name,
                    min=calib_shape,
                    opt=calib_shape,
                    max=calib_shape,
                )
                config.set_calibration_profile(calib_profile)
            config.int8_calibrator = calibrator

        logger.info(
            "Building TensorRT %s engine from %s (profile batch=%d/%d/%d, shape=%s)",
            precision.upper(),
            src,
            min_batch,
            opt_batch,
            max_batch,
            input_shape,
        )
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(f"TensorRT failed to build serialized engine for {src}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(bytes(serialized_engine))
        return dest
    finally:
        if calibrator is not None:
            calibrator.close()
