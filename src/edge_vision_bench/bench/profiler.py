"""Stage-decomposed latency profiler and NVML/psutil memory monitor."""

from __future__ import annotations

import contextlib
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from edge_vision_bench.bench.perturbations import OODPerturbation
from edge_vision_bench.pipeline.postproc import postprocess_detections
from edge_vision_bench.pipeline.preproc import (
    LetterboxResult,
    dinov2_preprocess,
    letterbox_preprocess,
)
from edge_vision_bench.pipeline.tracker import ByteTracker
from edge_vision_bench.runtimes.base import BaseRuntimeEngine
from edge_vision_bench.schemas import BenchmarkResult, DetectionBox, PipelineConfig


def _parse_cuda_device_id(device: str) -> int:
    if ":" in device:
        _, idx_str = device.split(":", 1)
        return int(idx_str)
    return 0


def detect_hardware_target(device: str) -> str:
    """Detect human-readable GPU or CPU hardware model string."""
    if device.lower().startswith("cuda"):
        device_id = _parse_cuda_device_id(device)
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                if name.strip():
                    return name.strip()
            finally:
                with contextlib.suppress(Exception):
                    pynvml.nvmlShutdown()
        except Exception:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                return str(torch.cuda.get_device_name(device_id))
        except Exception:
            pass

        return f"CUDA GPU ({device})"

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            with cpuinfo.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lower().startswith("model name") and ":" in line:
                        cpu_name = line.split(":", 1)[1].strip()
                        if cpu_name:
                            return cpu_name
        except OSError:
            pass

    proc = platform.processor().strip()
    if proc and proc.lower() not in {"x86_64", "amd64", "aarch64", "arm64"}:
        return proc
    return f"{platform.system()} {platform.machine()} CPU".strip()


def compute_artifact_size_mb(artifact_path: Path | str | None) -> float:
    """Compute serialized artifact footprint in MB (summing .xml + .bin for OpenVINO IR)."""
    if artifact_path is None:
        return 0.0
    path_str = str(artifact_path).strip()
    if not path_str or path_str == ".":
        return 0.0

    path = Path(path_str)
    if not path.exists():
        return 0.0

    if path.is_dir():
        total_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return float(total_bytes) / (1024.0 * 1024.0)

    total_bytes = path.stat().st_size
    if path.suffix.lower() == ".xml":
        bin_path = path.with_suffix(".bin")
        if bin_path.is_file():
            total_bytes += bin_path.stat().st_size
    return float(total_bytes) / (1024.0 * 1024.0)


class NVMLMemoryMonitor:
    """Context manager sampling peak GPU VRAM (via pynvml) and peak process RAM (via psutil)."""

    def __init__(self, device_id: int = 0, sample_interval_s: float = 0.02) -> None:
        self.device_id = device_id
        self.sample_interval_s = sample_interval_s
        self.peak_vram_mb: float = 0.0
        self.peak_ram_mb: float = 0.0
        self.gpu_name: str | None = None

        self._process = psutil.Process(os.getpid())
        self._pynvml: Any = None
        self._nvml_handle: Any = None
        self._baseline_gpu_used_bytes: int = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _init_nvml(self) -> None:
        try:
            import pynvml
        except ImportError:
            return

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            self.gpu_name = (
                raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            )
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self._baseline_gpu_used_bytes = int(mem_info.used)
            self._pynvml = pynvml
            self._nvml_handle = handle
        except Exception:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
            self._pynvml = None
            self._nvml_handle = None

    def sample(self) -> None:
        """Capture a single RAM and (if available) GPU VRAM sample."""
        try:
            rss_mb = float(self._process.memory_info().rss) / (1024.0 * 1024.0)
            if rss_mb > self.peak_ram_mb:
                self.peak_ram_mb = rss_mb
        except Exception:
            pass

        if self._pynvml is not None and self._nvml_handle is not None:
            vram_mb = 0.0
            pid = self._process.pid
            try:
                for proc in self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._nvml_handle):
                    if proc.pid == pid and proc.usedGpuMemory is not None:
                        vram_mb = max(vram_mb, float(proc.usedGpuMemory) / (1024.0 * 1024.0))
            except Exception:
                pass

            if vram_mb <= 0.0:
                try:
                    mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    delta = max(0, int(mem_info.used) - self._baseline_gpu_used_bytes)
                    vram_mb = float(delta) / (1024.0 * 1024.0)
                except Exception:
                    pass

            if vram_mb > self.peak_vram_mb:
                self.peak_vram_mb = vram_mb

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_s):
            self.sample()

    def __enter__(self) -> NVMLMemoryMonitor:
        self._init_nvml()
        self.sample()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.sample()
        if self._pynvml is not None:
            with contextlib.suppress(Exception):
                self._pynvml.nvmlShutdown()
            self._pynvml = None
            self._nvml_handle = None


@dataclass
class LatencyStats:
    """Collected per-batch latency distributions in milliseconds across pipeline stages."""

    preproc_ms: list[float] = field(default_factory=list)
    h2d_ms: list[float] = field(default_factory=list)
    compute_ms: list[float] = field(default_factory=list)
    d2h_ms: list[float] = field(default_factory=list)
    postproc_ms: list[float] = field(default_factory=list)
    track_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    total_frames: int = 0

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    @property
    def p50_compute_ms(self) -> float:
        return self._percentile(self.compute_ms, 50.0)

    @property
    def p95_compute_ms(self) -> float:
        return self._percentile(self.compute_ms, 95.0)

    @property
    def p99_compute_ms(self) -> float:
        return self._percentile(self.compute_ms, 99.0)

    @property
    def p50_h2d_ms(self) -> float:
        return self._percentile(self.h2d_ms, 50.0)

    @property
    def p50_d2h_ms(self) -> float:
        return self._percentile(self.d2h_ms, 50.0)

    @property
    def p50_preproc_ms(self) -> float:
        return self._percentile(self.preproc_ms, 50.0)

    @property
    def p50_postproc_ms(self) -> float:
        combined = [p + t for p, t in zip(self.postproc_ms, self.track_ms, strict=False)]
        return self._percentile(combined, 50.0)

    @property
    def p50_track_ms(self) -> float:
        return self._percentile(self.track_ms, 50.0)

    @property
    def end_to_end_fps(self) -> float:
        total_s = sum(self.total_ms) / 1000.0
        if total_s <= 0.0 or self.total_frames <= 0:
            return 0.0
        return float(self.total_frames) / total_s


def profile_engine(
    engine: BaseRuntimeEngine,
    frames_bgr: list[np.ndarray],
    config: PipelineConfig,
    runtime_name: str,
    precision: str,
    batch_size: int = 1,
    warmup_runs: int | None = None,
    artifact_path: Path | None = None,
    perturbation: OODPerturbation | None = None,
    enable_tracking: bool = False,
    seed: int = 42,
) -> tuple[BenchmarkResult, list[list[DetectionBox]] | list[np.ndarray]]:
    """Warm up and profile a runtime engine end-to-end across a sequence of BGR frames."""
    if not frames_bgr:
        raise ValueError("profile_engine requires at least one input frame.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    effective_warmup = config.warmup_runs if warmup_runs is None else warmup_runs
    input_h, input_w = config.input_size
    warmup_shape = (batch_size, 3, input_h, input_w)

    # Apply OOD perturbation ahead of timing so stage breakdown measures inference pipeline only.
    prepared_frames = (
        [perturbation.apply(frame, seed=seed + idx) for idx, frame in enumerate(frames_bgr)]
        if perturbation is not None
        else frames_bgr
    )

    device_id = (
        _parse_cuda_device_id(engine.device) if engine.device.lower().startswith("cuda") else 0
    )

    engine.warmup(warmup_shape, runs=effective_warmup)

    tracker: ByteTracker | None = None
    if enable_tracking and config.track == "detection":
        tracker = ByteTracker(
            high_thresh=config.tracker_high_thresh,
            low_thresh=config.tracker_low_thresh,
            match_thresh=config.tracker_match_thresh,
            track_buffer=config.tracker_track_buffer,
        )

    stats = LatencyStats(total_frames=len(prepared_frames))
    all_detections: list[list[DetectionBox]] = []
    all_embeddings: list[np.ndarray] = []

    with NVMLMemoryMonitor(device_id=device_id) as monitor:
        for batch_start in range(0, len(prepared_frames), batch_size):
            batch_frames = prepared_frames[batch_start : batch_start + batch_size]

            # 1. Preprocessing
            t_pre0 = time.perf_counter_ns()
            letterboxes: list[LetterboxResult] = []
            if config.track == "detection":
                tensors: list[np.ndarray] = []
                for frame in batch_frames:
                    lb = letterbox_preprocess(frame, input_size=config.input_size)
                    letterboxes.append(lb)
                    tensors.append(lb.tensor_nchw)
                batch_tensor = np.concatenate(tensors, axis=0)
            else:
                tensors = [
                    dinov2_preprocess(frame, input_size=config.input_size) for frame in batch_frames
                ]
                batch_tensor = np.concatenate(tensors, axis=0)
            t_pre1 = time.perf_counter_ns()
            preproc_ms = float(t_pre1 - t_pre0) / 1e6

            # 2. Device H2D + Compute + D2H
            outputs, timing = engine.infer(batch_tensor)

            # 3. Postprocessing & optional ByteTrack tracking
            track_ms = 0.0
            t_post0 = time.perf_counter_ns()
            if config.track == "detection":
                raw_out = outputs[0]
                for item_idx, lb in enumerate(letterboxes):
                    item_raw = raw_out[item_idx : item_idx + 1] if raw_out.ndim == 3 else raw_out
                    boxes = postprocess_detections(
                        item_raw,
                        letterbox=lb,
                        conf_threshold=config.conf_threshold,
                        nms_threshold=config.nms_threshold,
                        input_size=config.input_size,
                        already_decoded=True,
                    )
                    if tracker is not None:
                        t_tr0 = time.perf_counter_ns()
                        boxes = tracker.update(boxes)
                        t_tr1 = time.perf_counter_ns()
                        track_ms += float(t_tr1 - t_tr0) / 1e6
                    all_detections.append(boxes)
            else:
                emb_out = np.asarray(outputs[0], dtype=np.float32)
                if emb_out.ndim == 1:
                    emb_out = emb_out[None, :]
                for item_idx in range(len(batch_frames)):
                    vec = emb_out[item_idx].reshape(-1)
                    norm = float(np.linalg.norm(vec))
                    if norm > 0.0:
                        vec = vec / norm
                    all_embeddings.append(vec.astype(np.float32, copy=False))
            t_post1 = time.perf_counter_ns()

            post_total_ms = float(t_post1 - t_post0) / 1e6
            postproc_only_ms = max(0.0, post_total_ms - track_ms)
            total_batch_ms = (
                preproc_ms
                + timing.h2d_ms
                + timing.compute_ms
                + timing.d2h_ms
                + postproc_only_ms
                + track_ms
            )

            stats.preproc_ms.append(preproc_ms)
            stats.h2d_ms.append(timing.h2d_ms)
            stats.compute_ms.append(timing.compute_ms)
            stats.d2h_ms.append(timing.d2h_ms)
            stats.postproc_ms.append(postproc_only_ms)
            stats.track_ms.append(track_ms)
            stats.total_ms.append(total_batch_ms)
            monitor.sample()

        peak_vram_mb = max(monitor.peak_vram_mb, engine.get_vram_usage_mb())
        peak_ram_mb = max(monitor.peak_ram_mb, engine.get_ram_usage_mb())

    resolved_artifact = artifact_path if artifact_path is not None else engine.artifact_path
    hardware_target = (
        monitor.gpu_name
        if (engine.device.lower().startswith("cuda") and monitor.gpu_name)
        else detect_hardware_target(engine.device)
    )

    result = BenchmarkResult(
        model_name=config.model_name,
        track=config.track,
        runtime=runtime_name,
        precision=precision,
        hardware_target=hardware_target,
        batch_size=batch_size,
        p50_compute_ms=round(stats.p50_compute_ms, 4),
        p95_compute_ms=round(stats.p95_compute_ms, 4),
        p99_compute_ms=round(stats.p99_compute_ms, 4),
        p50_h2d_ms=round(stats.p50_h2d_ms, 4),
        p50_d2h_ms=round(stats.p50_d2h_ms, 4),
        p50_preproc_ms=round(stats.p50_preproc_ms, 4),
        p50_postproc_ms=round(stats.p50_postproc_ms, 4),
        end_to_end_fps=round(stats.end_to_end_fps, 2),
        peak_vram_mb=round(peak_vram_mb, 2),
        peak_ram_mb=round(peak_ram_mb, 2),
        artifact_size_mb=round(compute_artifact_size_mb(resolved_artifact), 3),
    )

    predictions: list[list[DetectionBox]] | list[np.ndarray] = (
        all_detections if config.track == "detection" else all_embeddings
    )
    return result, predictions
