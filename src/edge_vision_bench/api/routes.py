"""HTTP route handlers for health, single-frame inference, sequence tracking, and Pareto queries."""

from __future__ import annotations

import io
import json
import logging
import re
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from pydantic import ValidationError

from edge_vision_bench.pipeline.postproc import postprocess_detections
from edge_vision_bench.pipeline.preproc import dinov2_preprocess, letterbox_preprocess
from edge_vision_bench.pipeline.tracker import ByteTracker
from edge_vision_bench.runtimes.base import BaseRuntimeEngine
from edge_vision_bench.runtimes.factory import list_available_runtimes
from edge_vision_bench.schemas import (
    BenchmarkResult,
    HealthResponse,
    InferenceResponse,
    PrecisionName,
    RuntimeName,
)

if TYPE_CHECKING:
    from edge_vision_bench.api.app import AppState

logger = logging.getLogger(__name__)

router = APIRouter()

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
)

_ASCENDING_METRICS: frozenset[str] = frozenset(
    {
        "p50_compute_ms",
        "p95_compute_ms",
        "p99_compute_ms",
        "p50_h2d_ms",
        "p50_d2h_ms",
        "p50_preproc_ms",
        "p50_postproc_ms",
        "peak_vram_mb",
        "peak_ram_mb",
        "artifact_size_mb",
    }
)

_DESCENDING_METRICS: frozenset[str] = frozenset(
    {
        "end_to_end_fps",
        "clean_map50_95",
        "clean_map50",
        "ood_blur_map50_95",
        "ood_occlusion_map50_95",
        "ood_lowlight_map50_95",
        "clean_mota",
        "clean_idf1",
        "ood_blur_mota",
        "embedding_cosine_sim",
        "ood_blur_cosine_sim",
        "ood_occlusion_cosine_sim",
        "ood_lowlight_cosine_sim",
    }
)

_VALID_METRICS: frozenset[str] = _ASCENDING_METRICS | _DESCENDING_METRICS


def _get_state(request: Request) -> AppState:
    return cast("AppState", request.app.state.evb)


def _parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return default


async def _resolve_inference_options(
    request: Request,
    runtime: str | None,
    precision: str | None,
    apply_homography: bool | None,
) -> tuple[str | None, str | None, bool]:
    resolved_runtime = runtime
    resolved_precision = precision
    resolved_homography = apply_homography

    if resolved_runtime is None or resolved_precision is None or resolved_homography is None:
        form = await request.form()
        if resolved_runtime is None:
            form_rt = form.get("runtime")
            if isinstance(form_rt, str) and form_rt.strip():
                resolved_runtime = form_rt.strip()
        if resolved_precision is None:
            form_prec = form.get("precision")
            if isinstance(form_prec, str) and form_prec.strip():
                resolved_precision = form_prec.strip()
        if resolved_homography is None:
            form_h = form.get("apply_homography")
            if form_h is not None:
                resolved_homography = _parse_bool(form_h, default=False)

    return resolved_runtime, resolved_precision, bool(resolved_homography)


def _resolve_engine(
    state: AppState,
    runtime: str | None,
    precision: str | None,
) -> tuple[str, str, BaseRuntimeEngine]:
    if not state.engines:
        raise HTTPException(
            status_code=503,
            detail="No inference engine is currently loaded.",
        )

    target_runtime = runtime or state.active_runtime
    target_precision = precision or state.active_precision
    engine = state.engines.get((target_runtime, target_precision))
    if engine is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Engine not registered for runtime={target_runtime!r}, "
                f"precision={target_precision!r}."
            ),
        )
    return target_runtime, target_precision, engine


def _decode_image_bytes(raw_bytes: bytes, label: str = "image") -> np.ndarray:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail=f"Empty payload for {label}.")
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupt image bytes for {label}.")
    return image_bgr


def _natural_sort_key(name: str) -> list[tuple[int, int, str]]:
    return [
        (0, int(part), "") if part.isdigit() else (1, 0, part.lower())
        for part in re.split(r"(\d+)", name)
    ]


def _extract_frames_from_zip(raw_bytes: bytes) -> list[np.ndarray]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw_bytes))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Corrupt or invalid zip archive.") from exc

    with archive as zf:
        members: list[zipfile.ZipInfo] = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            member_path = Path(info.filename)
            if "__MACOSX" in member_path.parts or member_path.name.startswith("."):
                continue
            if member_path.suffix.lower() in _IMAGE_EXTENSIONS:
                members.append(info)

        members.sort(key=lambda item: _natural_sort_key(item.filename))
        if not members:
            raise HTTPException(
                status_code=400,
                detail="Zip archive contains no supported image frames.",
            )

        frames: list[np.ndarray] = []
        for info in members:
            frame_bytes = zf.read(info)
            frames.append(_decode_image_bytes(frame_bytes, label=info.filename))
        return frames


def _load_benchmark_results(results_dir: Path) -> list[BenchmarkResult]:
    if not results_dir.is_dir():
        return []

    parsed: list[BenchmarkResult] = []
    for json_file in sorted(results_dir.rglob("*.json")):
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable benchmark JSON file: %s", json_file)
            continue

        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
            items = payload["results"]
        elif isinstance(payload, dict):
            items = [payload]
        else:
            continue

        for item in items:
            try:
                parsed.append(BenchmarkResult.model_validate(item))
            except ValidationError:
                logger.warning("Skipping invalid BenchmarkResult entry in %s", json_file)
    return parsed


@router.get("/healthz", response_model=HealthResponse)
def healthz(request: Request) -> HealthResponse:
    """Report service health, active runtime status, and available runtime backends."""
    state = _get_state(request)
    engine_loaded = (state.active_runtime, state.active_precision) in state.engines
    available = sorted(set(list_available_runtimes()) | {rt for rt, _ in state.engines})
    return HealthResponse(
        status="ok" if engine_loaded else "degraded",
        active_runtime=state.active_runtime,
        precision=state.active_precision,
        device=state.device,
        engine_loaded=engine_loaded,
        available_runtimes=available,
    )


@router.post("/v1/infer/image", response_model=InferenceResponse)
async def infer_image(
    request: Request,
    file: Annotated[UploadFile, File()],
    runtime: Annotated[str | None, Query()] = None,
    precision: Annotated[str | None, Query()] = None,
    apply_homography: Annotated[bool | None, Query()] = None,
) -> InferenceResponse:
    """Run single-image inference (detection or embedding) with latency breakdown."""
    state = _get_state(request)
    opt_runtime, opt_precision, opt_homography = await _resolve_inference_options(
        request=request,
        runtime=runtime,
        precision=precision,
        apply_homography=apply_homography,
    )
    target_runtime, target_precision, engine = _resolve_engine(state, opt_runtime, opt_precision)

    raw_bytes = await file.read()
    image_bgr = _decode_image_bytes(raw_bytes, label=file.filename or "file")

    t_total_start = time.perf_counter_ns()

    if state.config.track == "embedding":
        t_pre_start = time.perf_counter_ns()
        tensor_nchw = dinov2_preprocess(image_bgr, input_size=state.config.input_size)
        preproc_ms = (time.perf_counter_ns() - t_pre_start) / 1e6

        outputs, timing = engine.infer(tensor_nchw)
        if not outputs:
            raise HTTPException(status_code=500, detail="Runtime engine returned empty outputs.")

        t_post_start = time.perf_counter_ns()
        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        embedding_dim = int(embedding.shape[0])
        embedding_norm = float(np.linalg.norm(embedding))
        postproc_ms = (time.perf_counter_ns() - t_post_start) / 1e6

        total_ms = (time.perf_counter_ns() - t_total_start) / 1e6
        return InferenceResponse(
            runtime=cast(RuntimeName, target_runtime),
            precision=cast(PrecisionName, target_precision),
            detections=[],
            latency_breakdown_ms={
                "preproc": round(preproc_ms, 4),
                "h2d": round(timing.h2d_ms, 4),
                "compute": round(timing.compute_ms, 4),
                "d2h": round(timing.d2h_ms, 4),
                "postproc": round(postproc_ms, 4),
                "total": round(total_ms, 4),
            },
            embedding_dim=embedding_dim,
            embedding_norm=embedding_norm,
        )

    if opt_homography and state.homography is None:
        raise HTTPException(
            status_code=400,
            detail="Homography projection requested, but no homography_matrix is configured.",
        )

    t_pre_start = time.perf_counter_ns()
    lb = letterbox_preprocess(image_bgr, input_size=state.config.input_size)
    preproc_ms = (time.perf_counter_ns() - t_pre_start) / 1e6

    outputs, timing = engine.infer(lb.tensor_nchw)
    if not outputs:
        raise HTTPException(status_code=500, detail="Runtime engine returned empty outputs.")

    t_post_start = time.perf_counter_ns()
    detections = postprocess_detections(
        outputs[0],
        letterbox=lb,
        conf_threshold=state.config.conf_threshold,
        nms_threshold=state.config.nms_threshold,
        input_size=state.config.input_size,
    )
    if opt_homography and state.homography is not None:
        detections = state.homography.project_detections(detections)
    postproc_ms = (time.perf_counter_ns() - t_post_start) / 1e6

    total_ms = (time.perf_counter_ns() - t_total_start) / 1e6
    return InferenceResponse(
        runtime=cast(RuntimeName, target_runtime),
        precision=cast(PrecisionName, target_precision),
        detections=detections,
        latency_breakdown_ms={
            "preproc": round(preproc_ms, 4),
            "h2d": round(timing.h2d_ms, 4),
            "compute": round(timing.compute_ms, 4),
            "d2h": round(timing.d2h_ms, 4),
            "postproc": round(postproc_ms, 4),
            "total": round(total_ms, 4),
        },
    )


@router.post("/v1/infer/track-sequence", response_model=list[InferenceResponse])
async def infer_track_sequence(
    request: Request,
    files: Annotated[list[UploadFile] | None, File()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    runtime: Annotated[str | None, Query()] = None,
    precision: Annotated[str | None, Query()] = None,
    apply_homography: Annotated[bool | None, Query()] = None,
) -> list[InferenceResponse]:
    """Run detection + ByteTrack multi-object tracking on ordered frames or a zip archive."""
    state = _get_state(request)
    if state.config.track != "detection":
        raise HTTPException(
            status_code=400,
            detail="Sequence tracking is only supported for detection pipelines.",
        )

    opt_runtime, opt_precision, opt_homography = await _resolve_inference_options(
        request=request,
        runtime=runtime,
        precision=precision,
        apply_homography=apply_homography,
    )
    target_runtime, target_precision, engine = _resolve_engine(state, opt_runtime, opt_precision)

    if opt_homography and state.homography is None:
        raise HTTPException(
            status_code=400,
            detail="Homography projection requested, but no homography_matrix is configured.",
        )

    uploads: list[UploadFile] = []
    if files:
        uploads.extend(files)
    if file is not None:
        uploads.append(file)

    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="Provide one or more image files ('files') or a zip archive ('file'/'files').",
        )

    frames: list[np.ndarray]
    if len(uploads) == 1:
        single_upload = uploads[0]
        raw_bytes = await single_upload.read()
        is_zip = (single_upload.filename or "").lower().endswith(".zip") or zipfile.is_zipfile(
            io.BytesIO(raw_bytes)
        )
        if is_zip:
            frames = _extract_frames_from_zip(raw_bytes)
        else:
            frames = [_decode_image_bytes(raw_bytes, label=single_upload.filename or "file")]
    else:
        frames = []
        for idx, upload in enumerate(uploads):
            raw_bytes = await upload.read()
            frames.append(_decode_image_bytes(raw_bytes, label=upload.filename or f"frame_{idx}"))

    tracker = ByteTracker(
        high_thresh=state.config.tracker_high_thresh,
        low_thresh=state.config.tracker_low_thresh,
        match_thresh=state.config.tracker_match_thresh,
        track_buffer=state.config.tracker_track_buffer,
    )

    responses: list[InferenceResponse] = []
    for frame_index, image_bgr in enumerate(frames):
        t_total_start = time.perf_counter_ns()

        t_pre_start = time.perf_counter_ns()
        lb = letterbox_preprocess(image_bgr, input_size=state.config.input_size)
        preproc_ms = (time.perf_counter_ns() - t_pre_start) / 1e6

        outputs, timing = engine.infer(lb.tensor_nchw)
        if not outputs:
            raise HTTPException(status_code=500, detail="Runtime engine returned empty outputs.")

        t_post_start = time.perf_counter_ns()
        detections = postprocess_detections(
            outputs[0],
            letterbox=lb,
            conf_threshold=state.config.conf_threshold,
            nms_threshold=state.config.nms_threshold,
            input_size=state.config.input_size,
        )
        postproc_ms = (time.perf_counter_ns() - t_post_start) / 1e6

        t_track_start = time.perf_counter_ns()
        tracked = tracker.update(detections)
        if opt_homography and state.homography is not None:
            tracked = state.homography.project_detections(tracked)
        track_ms = (time.perf_counter_ns() - t_track_start) / 1e6

        total_ms = (time.perf_counter_ns() - t_total_start) / 1e6
        responses.append(
            InferenceResponse(
                runtime=cast(RuntimeName, target_runtime),
                precision=cast(PrecisionName, target_precision),
                detections=tracked,
                latency_breakdown_ms={
                    "preproc": round(preproc_ms, 4),
                    "h2d": round(timing.h2d_ms, 4),
                    "compute": round(timing.compute_ms, 4),
                    "d2h": round(timing.d2h_ms, 4),
                    "postproc": round(postproc_ms, 4),
                    "track": round(track_ms, 4),
                    "total": round(total_ms, 4),
                },
                frame_index=frame_index,
            )
        )

    return responses


@router.get("/v1/benchmarks/pareto", response_model=list[BenchmarkResult])
def get_benchmarks_pareto(
    request: Request,
    model: Annotated[str | None, Query()] = None,
    metric: Annotated[str, Query()] = "p95_compute_ms",
    track: Annotated[str | None, Query()] = None,
) -> list[BenchmarkResult]:
    """Read persisted benchmark JSON results, filter by model/track, and sort by metric."""
    if metric not in _VALID_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported metric {metric!r}. Allowed metrics: {sorted(_VALID_METRICS)}",
        )

    state = _get_state(request)
    results = _load_benchmark_results(state.results_dir)

    if model is not None:
        results = [item for item in results if item.model_name == model]
    if track is not None:
        results = [item for item in results if item.track == track]

    ascending = metric in _ASCENDING_METRICS

    def _sort_key(item: BenchmarkResult) -> tuple[bool, float]:
        raw_val = getattr(item, metric, None)
        if raw_val is None:
            return (True, float("inf"))
        numeric_val = float(raw_val)
        return (False, numeric_val if ascending else -numeric_val)

    results.sort(key=_sort_key)
    return results
