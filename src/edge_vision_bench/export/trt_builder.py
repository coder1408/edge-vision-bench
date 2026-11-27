"""TensorRT 10.x and 11.x serialized engine builder supporting FP32, FP16, and INT8."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from edge_vision_bench.export.onnx_exporter import convert_onnx_to_fp16
from edge_vision_bench.export.trt_calibrator import EntropyCalibrator
from edge_vision_bench.schemas import PrecisionName

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
except ImportError:
    trt = None


def _noop_preprocess(path: Path) -> np.ndarray:
    raise RuntimeError(f"preprocess_fn was not provided while attempting to calibrate using {path}")


def quantize_onnx_qdq_int8(
    onnx_path: Path,
    output_path: Path,
    calibration_frames: list[Path],
    preprocess_fn: Callable[[Path], np.ndarray],
    input_shape: tuple[int, int, int],
) -> Path:
    """Quantize an FP32 ONNX graph to an explicit symmetric Q/DQ INT8 ONNX graph.

    Used for TensorRT 11+ strongly-typed explicit INT8 engine builds where legacy
    ``IInt8EntropyCalibrator2`` calibration has been removed.

    Args:
        onnx_path: Source FP32 ``.onnx`` model file.
        output_path: Destination Q/DQ INT8 ``.onnx`` file.
        calibration_frames: Representative image paths for entropy calibration.
        preprocess_fn: Function mapping an image path to a ``(C, H, W)`` or ``(1, C, H, W)``
            float32 array matching the model's expected preprocessing.
        input_shape: Expected spatial input shape ``(C, H, W)`` without batch dimension.

    Returns:
        Path to the verified Q/DQ INT8 ONNX graph.
    """
    src = Path(onnx_path)
    dest = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"ONNX model not found: {src}")
    if not calibration_frames:
        raise ValueError("calibration_frames must be non-empty for Q/DQ INT8 quantization")

    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    proto = onnx.load(str(src), load_external_data=False)
    input_name = proto.graph.input[0].name
    frames = [Path(frame) for frame in calibration_frames]

    class _FrameCalibrationReader(CalibrationDataReader):  # type: ignore[misc]
        def __init__(self) -> None:
            self._start = 0
            self._end = len(frames)
            self._index = 0

        def __len__(self) -> int:
            return len(frames)

        def set_range(self, start_index: int, end_index: int) -> None:
            self._start = start_index
            self._end = min(end_index, len(frames))
            self._index = start_index

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self._index >= self._end:
                return None
            frame_path = frames[self._index]
            self._index += 1
            arr = np.asarray(preprocess_fn(frame_path), dtype=np.float32)
            if arr.ndim == 3:
                if arr.shape != input_shape:
                    raise ValueError(
                        f"Preprocessed calibration frame {frame_path} has shape {arr.shape}, "
                        f"expected {input_shape}"
                    )
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 4 and arr.shape[0] == 1:
                if arr.shape[1:] != input_shape:
                    raise ValueError(
                        f"Preprocessed calibration frame {frame_path} has shape {arr.shape}, "
                        f"expected (1, {input_shape[0]}, {input_shape[1]}, {input_shape[2]})"
                    )
            else:
                raise ValueError(
                    f"Preprocessed calibration frame {frame_path} has unsupported shape "
                    f"{arr.shape}; expected {input_shape} or (1, *input_shape)"
                )
            return {input_name: np.ascontiguousarray(arr, dtype=np.float32)}

        def rewind(self) -> None:
            self._index = self._start

    calib_stride = max(d for d in range(1, min(8, len(frames)) + 1) if len(frames) % d == 0)
    available_providers = ort.get_available_providers()
    calibration_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers
        else ["CPUExecutionProvider"]
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Quantizing %s to explicit Q/DQ INT8 ONNX (%d calibration frames, stride=%d, providers=%s) "
        "-> %s",
        src,
        len(frames),
        calib_stride,
        calibration_providers,
        dest,
    )
    quantize_static(
        model_input=str(src),
        model_output=str(dest),
        calibration_data_reader=_FrameCalibrationReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Entropy,
        calibration_providers=calibration_providers,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "CalibStridedMinMax": calib_stride,
        },
    )
    onnx.checker.check_model(str(dest))
    return dest


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
    """Compile an ONNX model into a serialized TensorRT 10.x or 11.x ``.engine`` plan file.

    On TensorRT 10.x, FP16/INT8 engines use ``BuilderFlag.FP16``/``BuilderFlag.INT8`` and
    ``EntropyCalibrator``. On TensorRT 11.x (strongly-typed networks without legacy builder flags
    or ``IInt8EntropyCalibrator2``), FP16 parses an FP16 ONNX graph and INT8 parses an explicit
    symmetric Q/DQ INT8 ONNX graph.

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
        cache_file: Optional calibration cache file path for legacy TensorRT 10.x INT8 builds.
        workspace_mb: TensorRT builder workspace limit in MiB.

    Returns:
        Path to the serialized ``.engine`` artifact.
    """
    if trt is None:
        raise RuntimeError(
            "TensorRT is only available on CUDA GPU hosts. Install the `gpu` extra "
            "(`pip install -e '.[gpu]'`) on an NVIDIA GPU host with TensorRT 10.x/11.x."
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

    builder_flag_enum: Any = getattr(trt, "BuilderFlag", None)
    has_legacy_fp16 = builder_flag_enum is not None and hasattr(builder_flag_enum, "FP16")
    has_legacy_int8 = (
        builder_flag_enum is not None
        and hasattr(builder_flag_enum, "INT8")
        and hasattr(trt, "IInt8EntropyCalibrator2")
    )

    parse_src = src
    if precision == "fp16" and not has_legacy_fp16:
        if src.stem.endswith("_fp16"):
            parse_src = src
        else:
            fp16_path = src.with_name(f"{src.stem}_fp16.onnx")
            if not fp16_path.exists():
                convert_onnx_to_fp16(src, fp16_path)
            parse_src = fp16_path
    elif precision == "int8" and not has_legacy_int8:
        if src.stem.endswith("_int8_qdq"):
            parse_src = src
        else:
            qdq_path = src.with_name(f"{src.stem}_int8_qdq.onnx")
            if not qdq_path.exists():
                if not calibration_frames or preprocess_fn is None:
                    raise ValueError(
                        "TensorRT 11+ INT8 build requires either an existing Q/DQ ONNX model "
                        "or both calibration_frames and preprocess_fn."
                    )
                quantize_onnx_qdq_int8(
                    onnx_path=src,
                    output_path=qdq_path,
                    calibration_frames=list(calibration_frames),
                    preprocess_fn=preprocess_fn,
                    input_shape=input_shape,
                )
            parse_src = qdq_path

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)

    flags_enum: Any = getattr(trt, "NetworkDefinitionCreationFlag", None)
    explicit_batch = (
        1 << int(flags_enum.EXPLICIT_BATCH)
        if (flags_enum is not None and hasattr(flags_enum, "EXPLICIT_BATCH"))
        else 0
    )
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(parse_src)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"Failed to parse ONNX graph {parse_src}: {'; '.join(errors)}")

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
        if precision == "fp16" and has_legacy_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8" and has_legacy_int8:
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
            parse_src,
            min_batch,
            opt_batch,
            max_batch,
            input_shape,
        )
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(f"TensorRT failed to build serialized engine for {parse_src}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(bytes(serialized_engine))
        return dest
    finally:
        if calibrator is not None:
            calibrator.close()
