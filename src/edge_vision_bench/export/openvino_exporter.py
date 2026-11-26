"""OpenVINO IR (.xml/.bin) export supporting FP32, FP16, and NNCF INT8 quantization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from edge_vision_bench.schemas import PrecisionName

logger = logging.getLogger(__name__)


def _infer_static_input_shape(ov_model: Any) -> tuple[int, ...]:
    partial_shape = ov_model.inputs[0].get_partial_shape()
    dims: list[int] = []
    for dim in partial_shape:
        if dim.is_dynamic:
            dims.append(1)
        else:
            dims.append(int(dim.get_length()))
    return tuple(dims)


def _prepare_calibration_tensors(
    ov_model: Any,
    calibration_tensors: list[np.ndarray] | None,
) -> list[np.ndarray]:
    if not calibration_tensors:
        inferred_shape = _infer_static_input_shape(ov_model)
        logger.warning(
            "No calibration_tensors provided for OpenVINO INT8 quantization; "
            "falling back to 8 deterministic synthetic tensors of shape %s",
            inferred_shape,
        )
        rng = np.random.default_rng(42)
        return [rng.standard_normal(inferred_shape).astype(np.float32) for _ in range(8)]

    prepared: list[np.ndarray] = []
    for tensor in calibration_tensors:
        arr = np.ascontiguousarray(tensor, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        prepared.append(arr)
    return prepared


def export_to_openvino(
    onnx_path: str | Path,
    output_dir: str | Path,
    precision: PrecisionName = "fp32",
    calibration_tensors: list[np.ndarray] | None = None,
) -> Path:
    """Convert an ONNX graph to OpenVINO IR (``.xml`` + ``.bin``) in FP32, FP16, or INT8.

    Args:
        onnx_path: Source ``.onnx`` model file.
        output_dir: Directory where OpenVINO IR files will be written.
        precision: Target precision: ``"fp32"``, ``"fp16"``, or ``"int8"``.
        calibration_tensors: Representative NCHW float32 tensors for NNCF INT8 PTQ.
            When omitted for ``"int8"``, deterministic synthetic tensors are used with a warning.

    Returns:
        Path to the generated ``.xml`` model file.
    """
    src = Path(onnx_path)
    if not src.exists():
        raise FileNotFoundError(f"ONNX model not found: {src}")
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError(f"Unsupported OpenVINO precision: {precision!r}")

    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO is required for OpenVINO export. Install the `cpu` extra "
            "(`pip install -e '.[cpu]'`)."
        ) from exc

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    xml_path = dest_dir / f"{src.stem}_{precision}.xml"

    ov_model = ov.convert_model(str(src))

    if precision == "int8":
        try:
            import nncf
        except ImportError as exc:
            raise RuntimeError(
                "NNCF is required for OpenVINO INT8 quantization. Install the `cpu` extra "
                "(`pip install -e '.[cpu]'`)."
            ) from exc

        tensors = _prepare_calibration_tensors(ov_model, calibration_tensors)
        calibration_dataset = nncf.Dataset(tensors)
        quantized_model = nncf.quantize(
            ov_model,
            calibration_dataset,
            subset_size=min(len(tensors), 300),
        )
        ov.save_model(quantized_model, str(xml_path), compress_to_fp16=False)
    else:
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=(precision == "fp16"))

    return xml_path
