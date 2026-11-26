"""Typer CLI (`evb`) for exporting, verifying, benchmarking, serving, and reporting."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, cast

import cv2
import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table
from torch import nn

from edge_vision_bench.bench.datasets import (
    COCOSubsetDataset,
    MOTSequence,
    MOTSequenceDataset,
    VisDroneMOTDataset,
    discover_coco_paths,
    generate_synthetic_mot_sequence,
)
from edge_vision_bench.bench.evaluator import (
    evaluate_coco_map,
    evaluate_embedding_parity,
    evaluate_mot_metrics,
    render_pareto_markdown,
    save_benchmark_results,
)
from edge_vision_bench.bench.perturbations import get_default_ood_suite
from edge_vision_bench.bench.profiler import compute_artifact_size_mb, profile_engine
from edge_vision_bench.export.onnx_exporter import (
    convert_onnx_to_fp16,
    export_to_onnx,
    verify_onnx_parity,
)
from edge_vision_bench.export.openvino_exporter import export_to_openvino
from edge_vision_bench.export.trt_builder import build_trt_engine
from edge_vision_bench.models.dinov2 import load_dinov2
from edge_vision_bench.models.yolox import load_yolox
from edge_vision_bench.pipeline.preproc import dinov2_preprocess, letterbox_preprocess
from edge_vision_bench.runtimes.factory import create_engine
from edge_vision_bench.runtimes.openvino_engine import OpenVINORuntimeEngine
from edge_vision_bench.schemas import (
    BenchmarkResult,
    DetectionBox,
    ParityReport,
    PipelineConfig,
    PrecisionName,
    RuntimeName,
)

app = typer.Typer(
    name="evb",
    help="Edge Vision Bench: cross-runtime export, parity verification, benchmarking, and serving.",
    no_args_is_help=True,
)
console = Console()

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_VALID_RUNTIMES: frozenset[str] = frozenset(
    {"pytorch", "pytorch_compile", "onnxruntime", "tensorrt", "openvino"}
)
_VALID_PRECISIONS: frozenset[str] = frozenset({"fp32", "fp16", "int8"})


def _load_pytorch_model(cfg: PipelineConfig, pretrained: bool) -> nn.Module:
    if not pretrained:
        torch.manual_seed(42)
    if cfg.track == "detection":
        return load_yolox(
            variant=cfg.model_name,
            pretrained=pretrained,
            weights_dir=cfg.weights_dir,
            num_classes=cfg.num_classes,
        )
    if cfg.track == "embedding":
        return load_dinov2(
            variant=cfg.model_name,
            pretrained=pretrained,
            input_size=cfg.input_size,
        )
    raise typer.BadParameter(f"Unsupported track in config: {cfg.track!r}")


def _collect_calibration_images(calib_dir: Path | None, limit: int = 500) -> list[Path]:
    if calib_dir is None:
        return []
    if not calib_dir.exists():
        raise typer.BadParameter(f"Calibration directory does not exist: {calib_dir}")

    # Prefer strictly disjoint COCO calibration split when given a COCO dataset root.
    try:
        images_dir, ann_json = discover_coco_paths(calib_dir)
        coco_calib = COCOSubsetDataset(
            images_dir=images_dir,
            annotations_json=ann_json,
            split="calibration",
            calibration_size=limit,
            eval_size=limit,
            seed=42,
        )
        return coco_calib.image_paths[:limit]
    except FileNotFoundError:
        pass

    image_paths = sorted(
        p for p in calib_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    return image_paths[:limit]


def _make_preprocess_path_fn(cfg: PipelineConfig) -> Callable[[Path], np.ndarray]:
    def _preprocess(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read calibration image: {path}")
        if cfg.track == "detection":
            return letterbox_preprocess(img, input_size=cfg.input_size).tensor_nchw
        return dinov2_preprocess(img, input_size=cfg.input_size)

    return _preprocess


def _default_artifact_path(
    cfg: PipelineConfig,
    runtime: str,
    precision: str,
) -> Path | None:
    artifact_dir = Path(cfg.artifact_dir)
    if runtime in ("onnx", "onnxruntime"):
        if precision == "fp16":
            return artifact_dir / f"{cfg.model_name}_fp16.onnx"
        return artifact_dir / f"{cfg.model_name}.onnx"
    if runtime == "openvino":
        ov_sub = artifact_dir / "openvino" / f"{cfg.model_name}_{precision}.xml"
        if ov_sub.is_file():
            return ov_sub
        return artifact_dir / f"{cfg.model_name}_{precision}.xml"
    if runtime == "tensorrt":
        return artifact_dir / f"{cfg.model_name}_{precision}.engine"
    return None


def _print_parity_report(report: ParityReport) -> None:
    table = Table(title=f"Numerical Parity Report — {report.model_name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Reference -> Target", f"{report.reference_runtime} -> {report.target_runtime}")
    table.add_row("Precision", report.precision)
    table.add_row("Samples", str(report.num_samples))
    table.add_row("Max Abs Diff", f"{report.max_abs_diff:.6e}")
    table.add_row("Mean Abs Diff", f"{report.mean_abs_diff:.6e}")
    cos_str = f"{report.cosine_similarity:.6f}" if report.cosine_similarity is not None else "—"
    table.add_row("Cosine Similarity", cos_str)
    table.add_row("Tolerance (rtol / atol)", f"{report.rtol:.1e} / {report.atol:.1e}")
    status_str = "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]"
    table.add_row("Status", status_str)
    console.print(table)


def _print_benchmark_result(result: BenchmarkResult) -> None:
    table = Table(
        title=f"Benchmark Result — {result.model_name} ({result.runtime}/{result.precision})"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Hardware", result.hardware_target)
    table.add_row("Track / Batch", f"{result.track} / {result.batch_size}")
    table.add_row(
        "Compute Latency p50/p95/p99 (ms)",
        f"{result.p50_compute_ms:.2f} / {result.p95_compute_ms:.2f} / {result.p99_compute_ms:.2f}",
    )
    table.add_row(
        "Stages pre/h2d/d2h/post (ms)",
        f"{result.p50_preproc_ms:.2f} / {result.p50_h2d_ms:.2f} / "
        f"{result.p50_d2h_ms:.2f} / {result.p50_postproc_ms:.2f}",
    )
    table.add_row("End-to-End FPS", f"{result.end_to_end_fps:.2f}")
    table.add_row("Peak VRAM / RAM (MB)", f"{result.peak_vram_mb:.1f} / {result.peak_ram_mb:.1f}")
    table.add_row("Artifact Size (MB)", f"{result.artifact_size_mb:.2f}")
    if result.clean_map50_95 is not None:
        table.add_row(
            "Clean COCO mAP50:95 / mAP50",
            f"{result.clean_map50_95:.4f} / {result.clean_map50 or 0.0:.4f}",
        )
    if result.clean_mota is not None:
        table.add_row(
            "Clean MOTA / IDF1",
            f"{result.clean_mota:.4f} / {result.clean_idf1 or 0.0:.4f}",
        )
    if result.embedding_cosine_sim is not None:
        table.add_row("Clean Embedding Cosine Sim", f"{result.embedding_cosine_sim:.5f}")
    console.print(table)


@app.command("export")
def export_cmd(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML pipeline config.")],
    runtime: Annotated[
        str, typer.Option("--runtime", "-r", help="Target export runtime: onnx|openvino|tensorrt.")
    ],
    precision: Annotated[
        str, typer.Option("--precision", "-p", help="Target precision: fp32|fp16|int8.")
    ] = "fp32",
    pretrained: Annotated[
        bool,
        typer.Option(
            "--pretrained/--no-pretrained",
            help="Load pretrained weights before export.",
        ),
    ] = True,
    calib_dir: Annotated[
        Path | None,
        typer.Option("--calib-dir", help="Calibration images or COCO root for INT8 quantization."),
    ] = None,
    calib_samples: Annotated[
        int, typer.Option("--calib-samples", help="Maximum calibration images to use for INT8.")
    ] = 500,
    onnx_path: Annotated[
        Path | None,
        typer.Option("--onnx-path", help="Optional existing ONNX file for OpenVINO/TensorRT."),
    ] = None,
    simplify: Annotated[
        bool, typer.Option("--simplify/--no-simplify", help="Run onnxsim simplification on ONNX.")
    ] = True,
) -> None:
    """Export a configured vision model to ONNX, OpenVINO IR, or TensorRT .engine."""
    runtime_norm = runtime.strip().lower()
    precision_norm = precision.strip().lower()
    if runtime_norm not in {"onnx", "openvino", "tensorrt"}:
        raise typer.BadParameter("runtime must be one of: onnx, openvino, tensorrt")
    if precision_norm not in _VALID_PRECISIONS:
        raise typer.BadParameter("precision must be one of: fp32, fp16, int8")

    if runtime_norm == "onnx" and precision_norm == "int8":
        raise typer.BadParameter(
            "ONNX export supports fp32 and fp16. Use --runtime openvino or --runtime tensorrt "
            "for INT8 quantization."
        )

    cfg = PipelineConfig.from_yaml(config)
    prec_typed = cast(PrecisionName, precision_norm)
    input_h, input_w = cfg.input_size
    input_shape_4d = (1, 3, input_h, input_w)

    resolved_onnx = (
        onnx_path if onnx_path is not None else Path(cfg.artifact_dir) / f"{cfg.model_name}.onnx"
    )

    should_export_fp32_onnx = (
        runtime_norm == "onnx" and precision_norm == "fp32"
    ) or not resolved_onnx.is_file()
    if should_export_fp32_onnx:
        console.print(
            f"[bold]Loading PyTorch model[/bold] `{cfg.model_name}` (pretrained={pretrained})..."
        )
        pt_model = _load_pytorch_model(cfg, pretrained=pretrained)
        resolved_onnx = export_to_onnx(
            model=pt_model,
            output_path=resolved_onnx,
            input_shape=input_shape_4d,
            opset_version=cfg.opset_version,
            simplify=simplify,
        )
        console.print(
            f"[green]Exported ONNX (fp32):[/green] {resolved_onnx} "
            f"({compute_artifact_size_mb(resolved_onnx):.2f} MB)"
        )

    if runtime_norm == "onnx":
        if precision_norm == "fp16":
            fp16_onnx = Path(cfg.artifact_dir) / f"{cfg.model_name}_fp16.onnx"
            fp16_onnx = convert_onnx_to_fp16(onnx_path=resolved_onnx, output_path=fp16_onnx)
            console.print(
                f"[green]Exported ONNX (fp16):[/green] {fp16_onnx} "
                f"({compute_artifact_size_mb(fp16_onnx):.2f} MB)"
            )
        return

    calib_images = _collect_calibration_images(calib_dir, limit=calib_samples)
    preprocess_path = _make_preprocess_path_fn(cfg)

    if runtime_norm == "openvino":
        calib_tensors: list[np.ndarray] | None = None
        if prec_typed == "int8" and calib_images:
            console.print(
                f"Preparing {len(calib_images)} calibration tensors for OpenVINO NNCF INT8..."
            )
            calib_tensors = [preprocess_path(p) for p in calib_images]

        ov_out_dir = Path(cfg.artifact_dir) / "openvino"
        xml_path = export_to_openvino(
            onnx_path=resolved_onnx,
            output_dir=ov_out_dir,
            precision=prec_typed,
            calibration_tensors=calib_tensors,
        )
        console.print(
            f"[green]Exported OpenVINO IR ({precision_norm}):[/green] {xml_path} "
            f"({compute_artifact_size_mb(xml_path):.2f} MB)"
        )
        return

    # TensorRT build
    engine_path = Path(cfg.artifact_dir) / f"{cfg.model_name}_{precision_norm}.engine"
    cache_file = Path(cfg.artifact_dir) / f"{cfg.model_name}_int8.cache"
    built_path = build_trt_engine(
        onnx_path=resolved_onnx,
        engine_path=engine_path,
        precision=prec_typed,
        input_shape=(3, input_h, input_w),
        min_batch=cfg.min_batch,
        opt_batch=cfg.opt_batch,
        max_batch=cfg.max_batch,
        calibration_frames=calib_images if prec_typed == "int8" else None,
        preprocess_fn=preprocess_path if prec_typed == "int8" else None,
        cache_file=cache_file if prec_typed == "int8" else None,
    )
    console.print(
        f"[green]Built TensorRT engine ({precision_norm}):[/green] {built_path} "
        f"({compute_artifact_size_mb(built_path):.2f} MB)"
    )


def _verify_openvino_parity(
    pt_model: nn.Module,
    artifact: Path,
    cfg: PipelineConfig,
    precision: PrecisionName,
    num_samples: int,
    seed: int = 42,
) -> ParityReport:
    engine = OpenVINORuntimeEngine(artifact_path=artifact, precision=precision, device="cpu")
    pt_model.eval()
    rng = np.random.default_rng(seed)
    input_shape = (1, 3, cfg.input_size[0], cfg.input_size[1])

    max_abs_diff = 0.0
    mean_abs_diffs: list[float] = []
    cosine_sims: list[float] = []
    all_close = True

    try:
        with torch.inference_mode():
            for _ in range(num_samples):
                sample_np = rng.standard_normal(input_shape).astype(np.float32)
                pt_raw = pt_model(torch.from_numpy(sample_np))
                if isinstance(pt_raw, (tuple, list)):
                    pt_raw = pt_raw[0]
                pt_out = pt_raw.detach().cpu().numpy().astype(np.float32, copy=False)

                ov_outs, _ = engine.infer(sample_np)
                ov_out = np.asarray(ov_outs[0], dtype=np.float32)

                diff = np.abs(pt_out - ov_out)
                max_abs_diff = max(max_abs_diff, float(diff.max()))
                mean_abs_diffs.append(float(diff.mean()))
                cosine_sims.append(
                    evaluate_embedding_parity(
                        pt_out.reshape(pt_out.shape[0], -1),
                        ov_out.reshape(ov_out.shape[0], -1),
                    )
                )
                if not bool(
                    np.allclose(pt_out, ov_out, rtol=cfg.parity_rtol, atol=cfg.parity_atol)
                ):
                    all_close = False
    finally:
        engine.close()

    mean_abs = float(np.mean(mean_abs_diffs))
    mean_cos = float(np.mean(cosine_sims))
    passed = all_close if cfg.track == "detection" else mean_cos >= cfg.parity_min_cosine

    return ParityReport(
        model_name=cfg.model_name,
        reference_runtime="pytorch",
        target_runtime="openvino",
        precision=precision,
        passed=passed,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs,
        cosine_similarity=mean_cos,
        rtol=cfg.parity_rtol,
        atol=cfg.parity_atol,
        num_samples=num_samples,
    )


@app.command("verify-parity")
def verify_parity_cmd(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML pipeline config.")],
    artifact: Annotated[Path, typer.Option("--artifact", "-a", help="Path to exported artifact.")],
    runtime: Annotated[
        str, typer.Option("--runtime", "-r", help="Target runtime: onnxruntime|openvino.")
    ] = "onnxruntime",
    precision: Annotated[
        str, typer.Option("--precision", "-p", help="Artifact precision: fp32|fp16|int8.")
    ] = "fp32",
    pretrained: Annotated[
        bool,
        typer.Option(
            "--pretrained/--no-pretrained",
            help="Load pretrained weights for reference PyTorch model.",
        ),
    ] = True,
    num_samples: Annotated[
        int, typer.Option("--num-samples", help="Number of deterministic random inputs to compare.")
    ] = 10,
) -> None:
    """Verify numerical parity of an exported ONNX or OpenVINO model against PyTorch FP32."""
    runtime_norm = runtime.strip().lower()
    precision_norm = precision.strip().lower()
    if runtime_norm not in {"onnxruntime", "openvino"}:
        raise typer.BadParameter("runtime must be one of: onnxruntime, openvino")
    if precision_norm not in _VALID_PRECISIONS:
        raise typer.BadParameter("precision must be one of: fp32, fp16, int8")

    cfg = PipelineConfig.from_yaml(config)
    pt_model = _load_pytorch_model(cfg, pretrained=pretrained)

    if runtime_norm == "onnxruntime":
        report = verify_onnx_parity(
            pytorch_model=pt_model,
            onnx_path=artifact,
            input_shape=(1, 3, cfg.input_size[0], cfg.input_size[1]),
            track=cfg.track,
            model_name=cfg.model_name,
            rtol=cfg.parity_rtol,
            atol=cfg.parity_atol,
            min_cosine=cfg.parity_min_cosine,
            num_samples=num_samples,
            raise_on_fail=False,
        )
    else:
        report = _verify_openvino_parity(
            pt_model=pt_model,
            artifact=artifact,
            cfg=cfg,
            precision=cast(PrecisionName, precision_norm),
            num_samples=num_samples,
        )

    _print_parity_report(report)
    if not report.passed:
        raise typer.Exit(code=1)


def _preds_to_mot_dict(
    preds: list[list[DetectionBox]],
    *,
    is_visdrone: bool,
) -> dict[int, list[DetectionBox]]:
    out: dict[int, list[DetectionBox]] = {}
    for idx, boxes in enumerate(preds):
        out[idx] = VisDroneMOTDataset.remap_coco_detections(boxes) if is_visdrone else boxes
    return out


@app.command("bench")
def bench_cmd(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML pipeline config.")],
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            "-r",
            help="Runtime: pytorch|pytorch_compile|onnxruntime|openvino|tensorrt.",
        ),
    ],
    precision: Annotated[
        str, typer.Option("--precision", "-p", help="Precision: fp32|fp16|int8.")
    ] = "fp32",
    device: Annotated[str, typer.Option("--device", "-d", help="Device: cpu|cuda.")] = "cpu",
    artifact: Annotated[
        Path | None, typer.Option("--artifact", "-a", help="Optional serialized model path.")
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Inference batch size.")
    ] = 1,
    num_frames: Annotated[
        int, typer.Option("--num-frames", "-n", help="Maximum number of frames to benchmark.")
    ] = 100,
    warmup: Annotated[
        int | None, typer.Option("--warmup", help="Warmup runs (defaults to config.warmup_runs).")
    ] = None,
    ood: Annotated[
        bool, typer.Option("--ood/--no-ood", help="Run OOD robustness sweep (blur/occ/lowlight).")
    ] = False,
    coco_dir: Annotated[
        Path | None, typer.Option("--coco-dir", help="Optional COCO val2017 root for mAP eval.")
    ] = None,
    mot_dir: Annotated[
        Path | None,
        typer.Option("--mot-dir", help="Optional MOTChallenge / VisDrone-MOT root for MOT eval."),
    ] = None,
    pretrained: Annotated[
        bool,
        typer.Option(
            "--pretrained/--no-pretrained",
            help="Load pretrained weights when instantiating PyTorch models.",
        ),
    ] = True,
    output_json: Annotated[
        Path | None, typer.Option("--output-json", help="Destination JSON file for results.")
    ] = None,
    output_md: Annotated[
        Path | None, typer.Option("--output-md", help="Optional Markdown Pareto report path.")
    ] = None,
) -> None:
    """Profile a runtime engine across latency, memory, accuracy, and OOD robustness."""
    runtime_norm = runtime.strip().lower()
    precision_norm = precision.strip().lower()
    if runtime_norm not in _VALID_RUNTIMES:
        raise typer.BadParameter(f"runtime must be one of: {', '.join(sorted(_VALID_RUNTIMES))}")
    if precision_norm not in _VALID_PRECISIONS:
        raise typer.BadParameter("precision must be one of: fp32, fp16, int8")

    cfg = PipelineConfig.from_yaml(config)
    rt_typed = cast(RuntimeName, runtime_norm)
    prec_typed = cast(PrecisionName, precision_norm)

    pt_model: nn.Module | None = None
    resolved_artifact: Path | None = artifact
    if runtime_norm in ("pytorch", "pytorch_compile"):
        if resolved_artifact is None or not resolved_artifact.is_file():
            pt_model = _load_pytorch_model(cfg, pretrained=pretrained)
            resolved_artifact = None
    elif resolved_artifact is None:
        resolved_artifact = _default_artifact_path(cfg, runtime_norm, precision_norm)

    engine = create_engine(
        runtime=rt_typed,
        artifact_path=resolved_artifact if resolved_artifact is not None else "",
        precision=prec_typed,
        device=device,
        model=pt_model,
    )

    coco_dataset: COCOSubsetDataset | None = None
    coco_image_ids: list[int] = []
    coco_frames: list[np.ndarray] = []
    mot_seq: MOTSequence | None = None
    mot_frames: list[np.ndarray] = []
    mot_gt: dict[int, list[tuple[int, float, float, float, float]]] = {}

    if coco_dir is not None:
        images_dir, ann_json = discover_coco_paths(coco_dir)
        coco_dataset = COCOSubsetDataset(
            images_dir=images_dir,
            annotations_json=ann_json,
            split="eval",
            calibration_size=500,
            eval_size=max(500, num_frames),
            seed=42,
        )
        if len(coco_dataset) == 0:
            coco_dataset = COCOSubsetDataset(
                images_dir=images_dir,
                annotations_json=ann_json,
                split="all",
                seed=42,
            )
        coco_image_ids, coco_frames = coco_dataset.load_frames(limit=num_frames)

    if mot_dir is not None:
        mot_ds = MOTSequenceDataset(mot_dir)
        mot_seq = mot_ds[0]
        if mot_seq.is_visdrone:
            mot_seq = VisDroneMOTDataset(mot_dir, coco_compatible_only=True)[0]
        mot_frames, mot_gt = mot_seq.load_frames(limit=num_frames)

    if coco_dataset is not None:
        frames_bgr = coco_frames
        primary_is_mot = False
    elif mot_seq is not None:
        frames_bgr = mot_frames
        primary_is_mot = True
    else:
        frames_bgr, _ = generate_synthetic_mot_sequence(
            num_frames=num_frames,
            width=cfg.input_size[1],
            height=cfg.input_size[0],
            num_objects=4,
            seed=42,
        )
        primary_is_mot = False

    try:
        result, clean_preds = profile_engine(
            engine=engine,
            frames_bgr=frames_bgr,
            config=cfg,
            runtime_name=runtime_norm,
            precision=precision_norm,
            batch_size=batch_size,
            warmup_runs=warmup,
            artifact_path=resolved_artifact
            if resolved_artifact and resolved_artifact.exists()
            else None,
            enable_tracking=primary_is_mot,
        )

        ood_suite = get_default_ood_suite() if ood else {}

        # Track A: Detection & Tracking accuracy + OOD evaluation
        if cfg.track == "detection":
            det_preds = cast(list[list[DetectionBox]], clean_preds)

            if coco_dataset is not None:
                pred_map = dict(zip(coco_image_ids, det_preds, strict=True))
                clean_map, clean_map50 = evaluate_coco_map(
                    pred_map, coco_dataset.annotations_json, image_ids=coco_image_ids
                )
                result = result.model_copy(
                    update={
                        "clean_map50_95": round(clean_map, 4),
                        "clean_map50": round(clean_map50, 4),
                    }
                )

                if ood:
                    ood_updates: dict[str, float] = {}
                    for ood_key, field_name in (
                        ("blur", "ood_blur_map50_95"),
                        ("occlusion", "ood_occlusion_map50_95"),
                        ("lowlight", "ood_lowlight_map50_95"),
                    ):
                        _, ood_preds_raw = profile_engine(
                            engine=engine,
                            frames_bgr=coco_frames,
                            config=cfg,
                            runtime_name=runtime_norm,
                            precision=precision_norm,
                            batch_size=batch_size,
                            warmup_runs=0,
                            perturbation=ood_suite[ood_key],
                        )
                        ood_det_preds = cast(list[list[DetectionBox]], ood_preds_raw)
                        ood_map, _ = evaluate_coco_map(
                            dict(zip(coco_image_ids, ood_det_preds, strict=True)),
                            coco_dataset.annotations_json,
                            image_ids=coco_image_ids,
                        )
                        ood_updates[field_name] = round(ood_map, 4)
                    result = result.model_copy(update=ood_updates)

            if mot_seq is not None and mot_gt:
                if primary_is_mot:
                    mot_det_preds = det_preds
                else:
                    _, mot_preds_raw = profile_engine(
                        engine=engine,
                        frames_bgr=mot_frames,
                        config=cfg,
                        runtime_name=runtime_norm,
                        precision=precision_norm,
                        batch_size=batch_size,
                        warmup_runs=0,
                        enable_tracking=True,
                    )
                    mot_det_preds = cast(list[list[DetectionBox]], mot_preds_raw)

                pred_by_frame = _preds_to_mot_dict(mot_det_preds, is_visdrone=mot_seq.is_visdrone)
                clean_mota, clean_idf1 = evaluate_mot_metrics(mot_gt, pred_by_frame)
                updates: dict[str, float] = {
                    "clean_mota": round(clean_mota, 4),
                    "clean_idf1": round(clean_idf1, 4),
                }
                if ood and "blur" in ood_suite:
                    _, blur_preds_raw = profile_engine(
                        engine=engine,
                        frames_bgr=mot_frames,
                        config=cfg,
                        runtime_name=runtime_norm,
                        precision=precision_norm,
                        batch_size=batch_size,
                        warmup_runs=0,
                        perturbation=ood_suite["blur"],
                        enable_tracking=True,
                    )
                    blur_det_preds = cast(list[list[DetectionBox]], blur_preds_raw)
                    blur_mota, _ = evaluate_mot_metrics(
                        mot_gt,
                        _preds_to_mot_dict(blur_det_preds, is_visdrone=mot_seq.is_visdrone),
                    )
                    updates["ood_blur_mota"] = round(blur_mota, 4)
                result = result.model_copy(update=updates)

        # Track B: Embedding cosine similarity + OOD drift evaluation
        elif cfg.track == "embedding":
            target_embs = np.stack(cast(list[np.ndarray], clean_preds), axis=0)
            if runtime_norm == "pytorch" and precision_norm == "fp32":
                ref_embs = target_embs
            else:
                ref_model = _load_pytorch_model(cfg, pretrained=pretrained)
                ref_engine = create_engine(
                    runtime="pytorch",
                    artifact_path="",
                    precision="fp32",
                    device=device
                    if device.startswith("cuda") and torch.cuda.is_available()
                    else "cpu",
                    model=ref_model,
                )
                try:
                    _, ref_preds_raw = profile_engine(
                        engine=ref_engine,
                        frames_bgr=frames_bgr,
                        config=cfg,
                        runtime_name="pytorch",
                        precision="fp32",
                        batch_size=batch_size,
                        warmup_runs=0,
                    )
                    ref_embs = np.stack(cast(list[np.ndarray], ref_preds_raw), axis=0)
                finally:
                    ref_engine.close()

            clean_cos = evaluate_embedding_parity(ref_embs, target_embs)
            emb_updates: dict[str, float] = {"embedding_cosine_sim": round(clean_cos, 5)}

            if ood:
                for ood_key, field_name in (
                    ("blur", "ood_blur_cosine_sim"),
                    ("occlusion", "ood_occlusion_cosine_sim"),
                    ("lowlight", "ood_lowlight_cosine_sim"),
                ):
                    _, ood_emb_raw = profile_engine(
                        engine=engine,
                        frames_bgr=frames_bgr,
                        config=cfg,
                        runtime_name=runtime_norm,
                        precision=precision_norm,
                        batch_size=batch_size,
                        warmup_runs=0,
                        perturbation=ood_suite[ood_key],
                    )
                    ood_embs = np.stack(cast(list[np.ndarray], ood_emb_raw), axis=0)
                    emb_updates[field_name] = round(
                        evaluate_embedding_parity(ref_embs, ood_embs), 5
                    )

            result = result.model_copy(update=emb_updates)
    finally:
        engine.close()

    default_json = (
        Path(cfg.results_dir)
        / f"{cfg.model_name}_{runtime_norm}_{precision_norm}_{device.replace(':', '_')}.json"
    )
    dest_json = output_json if output_json is not None else default_json
    save_benchmark_results([result], json_path=dest_json, markdown_path=output_md)
    _print_benchmark_result(result)
    console.print(f"[green]Saved benchmark JSON:[/green] {dest_json}")
    if output_md is not None:
        console.print(f"[green]Saved Pareto Markdown:[/green] {output_md}")


@app.command("serve")
def serve_cmd(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML pipeline config.")],
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            "-r",
            help="Runtime: pytorch|pytorch_compile|onnxruntime|openvino|tensorrt.",
        ),
    ] = "onnxruntime",
    precision: Annotated[
        str, typer.Option("--precision", "-p", help="Precision: fp32|fp16|int8.")
    ] = "fp32",
    device: Annotated[str, typer.Option("--device", "-d", help="Device: cpu|cuda.")] = "cpu",
    artifact: Annotated[
        str, typer.Option("--artifact", "-a", help="Path to serialized model artifact.")
    ] = "",
    host: Annotated[str, typer.Option("--host", help="Bind host address.")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="Bind TCP port.")] = 8000,
) -> None:
    """Launch the FastAPI inference & Pareto reporting server via uvicorn."""
    import uvicorn

    from edge_vision_bench.api.app import create_app

    cfg = PipelineConfig.from_yaml(config)
    os.environ["EVB_CONFIG"] = str(config)
    os.environ["EVB_RUNTIME"] = runtime
    os.environ["EVB_PRECISION"] = precision
    os.environ["EVB_DEVICE"] = device
    os.environ["EVB_ARTIFACT"] = artifact
    os.environ["EVB_RESULTS_DIR"] = cfg.results_dir

    fastapi_app = create_app(
        config=cfg,
        active_runtime=runtime,
        active_precision=precision,
        device=device,
        results_dir=cfg.results_dir,
    )
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command("report")
def report_cmd(
    results_dir: Annotated[
        Path, typer.Option("--results-dir", help="Directory containing JSON benchmark outputs.")
    ] = Path("results"),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output Markdown file for unified Pareto report.")
    ] = Path("results/pareto_summary.md"),
) -> None:
    """Merge all JSON benchmark results in a directory into one Markdown Pareto summary."""
    if not results_dir.is_dir():
        raise typer.BadParameter(f"Results directory does not exist: {results_dir}")

    collected: list[BenchmarkResult] = []
    json_files = sorted(results_dir.rglob("*.json"))
    for json_file in json_files:
        try:
            payload: Any = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            console.print(f"[yellow]Skipping unreadable JSON {json_file}: {exc}[/yellow]")
            continue

        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                collected.append(BenchmarkResult.model_validate(item))
            except Exception as exc:
                console.print(
                    f"[yellow]Skipping invalid BenchmarkResult entry in {json_file}: {exc}[/yellow]"
                )

    markdown = render_pareto_markdown(collected)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    console.print(
        f"[green]Merged {len(collected)} benchmark result(s) from {len(json_files)} JSON file(s) "
        f"into {output}[/green]"
    )


if __name__ == "__main__":
    app()
