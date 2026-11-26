"""ONNX opset 17 export and PyTorch-vs-ONNXRuntime numerical parity verification."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from edge_vision_bench.schemas import ParityReport, TrackType

logger = logging.getLogger(__name__)


def _infer_model_device_and_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    first_param = next(model.parameters(), None)
    if first_param is not None:
        return first_param.device, first_param.dtype
    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return first_buffer.device, first_buffer.dtype
    return torch.device("cpu"), torch.float32


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int],
    opset_version: int = 17,
    simplify: bool = True,
) -> Path:
    """Export a PyTorch module to ONNX with dynamic batch axis and model verification.

    Args:
        model: PyTorch module to export.
        output_path: Destination ``.onnx`` path.
        input_shape: Representative 4D NCHW input shape ``(B, C, H, W)``.
        opset_version: Target ONNX opset version (defaults to 17).
        simplify: If True, attempts ``onnxsim.simplify`` when installed.

    Returns:
        Path to the verified ONNX file.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device, dtype = _infer_model_device_and_dtype(model)
    dummy_input = torch.randn(*input_shape, device=device, dtype=dtype)

    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ["images"],
        "output_names": ["output"],
        "dynamic_axes": {"images": {0: "batch"}, "output": {0: "batch"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(model, (dummy_input,), str(dest), **export_kwargs)

    onnx.checker.check_model(str(dest))

    if simplify:
        try:
            import onnxsim
        except ImportError:
            logger.warning(
                "onnxsim is not installed; keeping verified unsimplified ONNX model at %s",
                dest,
            )
        else:
            try:
                onnx_model = onnx.load(str(dest))
                simplified_model, check_ok = onnxsim.simplify(onnx_model)
                if check_ok:
                    onnx.save(simplified_model, str(dest))
                    onnx.checker.check_model(str(dest))
                else:
                    logger.warning(
                        "onnxsim validation returned False; keeping verified ONNX model at %s",
                        dest,
                    )
            except Exception as exc:
                logger.warning(
                    "onnxsim simplification failed (%s); keeping verified ONNX model at %s",
                    exc,
                    dest,
                )

    return dest


def convert_onnx_to_fp16(
    onnx_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Convert an FP32 ONNX graph to FP16 while preserving FP32 network I/O types.

    Args:
        onnx_path: Source FP32 ``.onnx`` model file.
        output_path: Destination FP16 ``.onnx`` model file.

    Returns:
        Path to the verified FP16 ONNX file.
    """
    src = Path(onnx_path)
    if not src.exists():
        raise FileNotFoundError(f"ONNX model not found: {src}")

    from onnxruntime.transformers import float16 as ort_float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    onnx_model = onnx.load(str(src))
    fp16_model = ort_float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    # Sort topologically so onnx.checker validates inserted Cast nodes around Slice/Resize ops.
    wrapped = OnnxModel(fp16_model)
    wrapped.topological_sort()

    onnx.save(wrapped.model, str(dest))
    onnx.checker.check_model(str(dest))
    return dest


def _batch_cosine_similarities(ref: np.ndarray, target: np.ndarray) -> list[float]:
    batch_size = ref.shape[0]
    ref_flat = ref.reshape(batch_size, -1).astype(np.float64, copy=False)
    tgt_flat = target.reshape(batch_size, -1).astype(np.float64, copy=False)
    similarities: list[float] = []
    for idx in range(batch_size):
        ref_norm = float(np.linalg.norm(ref_flat[idx]))
        tgt_norm = float(np.linalg.norm(tgt_flat[idx]))
        if ref_norm == 0.0 and tgt_norm == 0.0:
            similarities.append(1.0)
        elif ref_norm == 0.0 or tgt_norm == 0.0:
            similarities.append(0.0)
        else:
            sim = float(np.dot(ref_flat[idx], tgt_flat[idx]) / (ref_norm * tgt_norm))
            similarities.append(sim)
    return similarities


def verify_onnx_parity(
    pytorch_model: nn.Module,
    onnx_path: str | Path,
    input_shape: tuple[int, int, int, int],
    track: TrackType = "detection",
    model_name: str = "model",
    rtol: float = 1e-3,
    atol: float = 1e-4,
    min_cosine: float = 0.995,
    num_samples: int = 10,
    seed: int = 42,
    raise_on_fail: bool = True,
) -> ParityReport:
    """Verify numerical parity between a PyTorch reference model and an exported ONNX graph.

    Args:
        pytorch_model: Reference PyTorch module.
        onnx_path: Path to the exported ``.onnx`` file.
        input_shape: 4D NCHW tensor shape ``(B, C, H, W)`` for deterministic parity probes.
        track: ``"detection"`` (checks ``np.allclose``) or ``"embedding"`` (checks cosine).
        model_name: Human-readable model identifier recorded in the report.
        rtol: Relative tolerance for detection parity.
        atol: Absolute tolerance for detection parity.
        min_cosine: Minimum mean cosine similarity for embedding parity.
        num_samples: Number of deterministic random batches to compare.
        seed: Random seed for reproducible probe tensors.
        raise_on_fail: If True, raises ``AssertionError`` when parity fails.

    Returns:
        Structured :class:`ParityReport`.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    artifact = Path(onnx_path)
    if not artifact.exists():
        raise FileNotFoundError(f"ONNX artifact not found: {artifact}")

    pytorch_model.eval()
    device, dtype = _infer_model_device_and_dtype(pytorch_model)

    session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(seed)
    max_abs_diff = 0.0
    mean_abs_diffs: list[float] = []
    cosine_sims: list[float] = []
    all_samples_close = True

    with torch.no_grad():
        for _ in range(num_samples):
            sample_np = rng.standard_normal(input_shape).astype(np.float32)
            pt_input = torch.from_numpy(sample_np).to(device=device, dtype=dtype)
            pt_raw = pytorch_model(pt_input)
            if isinstance(pt_raw, (tuple, list)):
                pt_raw = pt_raw[0]
            pt_out = pt_raw.detach().cpu().numpy().astype(np.float32, copy=False)

            ort_raw = session.run(None, {input_name: sample_np})[0]
            ort_out = np.asarray(ort_raw, dtype=np.float32)

            if pt_out.shape != ort_out.shape:
                raise AssertionError(
                    f"Output shape mismatch for {model_name}: PyTorch {pt_out.shape} "
                    f"vs ONNX Runtime {ort_out.shape}"
                )

            diff = np.abs(pt_out - ort_out)
            max_abs_diff = max(max_abs_diff, float(diff.max()))
            mean_abs_diffs.append(float(diff.mean()))
            cosine_sims.extend(_batch_cosine_similarities(pt_out, ort_out))
            if not bool(np.allclose(pt_out, ort_out, rtol=rtol, atol=atol)):
                all_samples_close = False

    mean_abs_diff = float(np.mean(mean_abs_diffs))
    mean_cosine = float(np.mean(cosine_sims))

    if track == "detection":
        passed = all_samples_close
    elif track == "embedding":
        passed = mean_cosine >= min_cosine
    else:
        raise ValueError(f"Unsupported track type: {track!r}")

    report = ParityReport(
        model_name=model_name,
        reference_runtime="pytorch",
        target_runtime="onnxruntime",
        precision="fp32",
        passed=passed,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        cosine_similarity=mean_cosine,
        rtol=rtol,
        atol=atol,
        num_samples=num_samples,
    )

    if not passed and raise_on_fail:
        raise AssertionError(
            f"ONNX parity check failed for {model_name} (track={track}): "
            f"max_abs_diff={max_abs_diff:.6e}, mean_abs_diff={mean_abs_diff:.6e}, "
            f"cosine_similarity={mean_cosine:.6f} "
            f"(rtol={rtol}, atol={atol}, min_cosine={min_cosine})"
        )

    return report
