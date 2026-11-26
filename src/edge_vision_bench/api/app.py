"""FastAPI application factory, state container, and lifespan management."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fastapi import FastAPI

from edge_vision_bench.api.routes import router
from edge_vision_bench.pipeline.homography import PlanarHomography
from edge_vision_bench.runtimes.base import BaseRuntimeEngine
from edge_vision_bench.runtimes.factory import create_engine
from edge_vision_bench.schemas import PipelineConfig, PrecisionName, RuntimeName

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Mutable runtime state container stored on ``app.state.evb``."""

    config: PipelineConfig
    engines: dict[tuple[str, str], BaseRuntimeEngine]
    active_runtime: str
    active_precision: str
    device: str
    homography: PlanarHomography | None
    results_dir: Path


def _build_homography(config: PipelineConfig) -> PlanarHomography | None:
    if config.homography_matrix is None:
        return None
    return PlanarHomography(config.homography_matrix)


def _load_config_from_env(fallback_model: str = "yolox-s") -> PipelineConfig:
    env_config = os.environ.get("EVB_CONFIG", "").strip()
    if env_config:
        cfg_path = Path(env_config)
        if cfg_path.is_file():
            return PipelineConfig.from_yaml(cfg_path)
        logger.warning("EVB_CONFIG points to non-existent file: %s", cfg_path)
    return PipelineConfig(model_name=fallback_model)


def create_app(
    config: PipelineConfig | None = None,
    engines: dict[tuple[str, str], BaseRuntimeEngine] | None = None,
    active_runtime: str = "onnxruntime",
    active_precision: str = "fp32",
    device: str = "cpu",
    results_dir: str | Path = "results",
) -> FastAPI:
    """Create and configure the FastAPI inference service.

    When ``engines`` is passed (including an empty dict), dependency injection is used
    and environment auto-loading is skipped. When ``engines`` is ``None``, the lifespan
    hook inspects ``EVB_CONFIG``, ``EVB_RUNTIME``, ``EVB_PRECISION``, ``EVB_DEVICE``,
    ``EVB_ARTIFACT``, and ``EVB_RESULTS_DIR`` to auto-load an inference engine on startup.
    """
    auto_load_from_env = engines is None
    config_was_explicit = config is not None

    if auto_load_from_env:
        resolved_config = config if config is not None else _load_config_from_env()
        resolved_runtime = os.environ.get("EVB_RUNTIME", active_runtime).strip() or active_runtime
        resolved_precision = (
            os.environ.get("EVB_PRECISION", active_precision).strip() or active_precision
        )
        resolved_device = os.environ.get("EVB_DEVICE", device).strip() or device
        resolved_results_dir = Path(
            os.environ.get("EVB_RESULTS_DIR", str(results_dir)).strip() or str(results_dir)
        )
        resolved_engines: dict[tuple[str, str], BaseRuntimeEngine] = {}
    else:
        resolved_config = config if config is not None else PipelineConfig(model_name="yolox-s")
        resolved_runtime = active_runtime
        resolved_precision = active_precision
        resolved_device = device
        resolved_results_dir = Path(results_dir)
        resolved_engines = dict(engines) if engines is not None else {}

    state = AppState(
        config=resolved_config,
        engines=resolved_engines,
        active_runtime=resolved_runtime,
        active_precision=resolved_precision,
        device=resolved_device,
        homography=_build_homography(resolved_config),
        results_dir=resolved_results_dir,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        managed_engines: list[BaseRuntimeEngine] = []
        app_state: AppState = application.state.evb

        if auto_load_from_env and not app_state.engines:
            if not config_was_explicit and os.environ.get("EVB_CONFIG", "").strip():
                try:
                    app_state.config = _load_config_from_env(app_state.config.model_name)
                    app_state.homography = _build_homography(app_state.config)
                except Exception:
                    logger.exception("Failed to reload PipelineConfig from EVB_CONFIG")

            env_runtime = os.environ.get("EVB_RUNTIME", "").strip()
            env_precision = os.environ.get("EVB_PRECISION", "").strip()
            env_device = os.environ.get("EVB_DEVICE", "").strip()
            env_results_dir = os.environ.get("EVB_RESULTS_DIR", "").strip()
            env_artifact = os.environ.get("EVB_ARTIFACT", "").strip()

            if env_runtime:
                app_state.active_runtime = env_runtime
            if env_precision:
                app_state.active_precision = env_precision
            if env_device:
                app_state.device = env_device
            if env_results_dir:
                app_state.results_dir = Path(env_results_dir)

            if env_artifact:
                try:
                    engine = create_engine(
                        runtime=cast(RuntimeName, app_state.active_runtime),
                        artifact_path=env_artifact,
                        precision=cast(PrecisionName, app_state.active_precision),
                        device=app_state.device,
                    )
                    warmup_env = os.environ.get("EVB_WARMUP_RUNS", "0").strip()
                    warmup_runs = int(warmup_env) if warmup_env.isdigit() else 0
                    if warmup_runs > 0:
                        in_h, in_w = app_state.config.input_size
                        engine.warmup((1, 3, in_h, in_w), runs=warmup_runs)
                    app_state.engines[(app_state.active_runtime, app_state.active_precision)] = (
                        engine
                    )
                    managed_engines.append(engine)
                    logger.info(
                        "Loaded engine runtime=%s precision=%s device=%s artifact=%s",
                        app_state.active_runtime,
                        app_state.active_precision,
                        app_state.device,
                        env_artifact,
                    )
                except Exception:
                    logger.exception(
                        "Failed to auto-load engine runtime=%s precision=%s artifact=%s",
                        app_state.active_runtime,
                        app_state.active_precision,
                        env_artifact,
                    )

        try:
            yield
        finally:
            for engine in managed_engines:
                try:
                    engine.close()
                except Exception:
                    logger.exception("Error while closing inference engine")

    application = FastAPI(
        title="Edge Vision Bench API",
        version="0.1.0",
        description=(
            "Cross-runtime edge computer vision inference, ByteTrack multi-object tracking, "
            "planar homography projection, and benchmark Pareto query service."
        ),
        lifespan=lifespan,
    )
    application.state.evb = state
    application.include_router(router)
    return application


app = create_app()
